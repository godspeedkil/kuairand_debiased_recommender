from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from krec.config import TwoTowerConfig
from krec.eval.metrics import top_k
from krec.models.base import Recommender
from krec.retrieval.inputs import encode_items, encode_users, history_snapshots
from krec.retrieval.model import TwoTower, in_batch_softmax_loss

# Scores held in memory at once when ranking the catalog (float32, ~400MB).
_MAX_SCORES = 100_000_000


class TwoTowerRetriever(Recommender):
    """
    Retrieve by dot product between a user's history and video content.

    Parameters
    ----------
    cfg : TwoTowerConfig
        Model and training settings.
    items : pd.DataFrame
        Item table, for video content.
    users : pd.DataFrame
        User table, for profiles.
    seed : int
        Seed for initialization and batch order.
    """

    name = "two_tower"

    def __init__(
        self, cfg: TwoTowerConfig, items: pd.DataFrame, users: pd.DataFrame, seed: int
    ):
        self.cfg, self.items, self.users, self.seed = cfg, items, users, seed

    def fit(self, interactions, catalog, as_of):
        cfg = self.cfg
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.catalog, self.as_of = catalog, as_of
        pos = interactions.loc[
            interactions[cfg.signal] == 1, ["user_id", "item_id", "date", "time_ms"]
        ]
        pos = pos.assign(item=catalog.index_of(pos.item_id.to_numpy()))
        self.positives = pos[pos.item >= 0]
        train = self.positives
        if cfg.max_examples is not None:
            train = train.iloc[-cfg.max_examples :]  # rows arrive in event order

        self.content = encode_items(
            self.items,
            catalog,
            train.item.to_numpy(),
            cfg.min_count,
            cfg.max_tags,
            cfg.id_min_count,
        )
        self.profile = encode_users(self.users)
        self._fields = {
            f: torch.from_numpy(v.astype(np.int64))
            for f, v in self.content.fields.items()
        }
        self._tags = torch.from_numpy(self.content.tags)

        # one history snapshot per (user, day), shared by that day's examples
        days = pd.to_datetime(train.date).to_numpy().astype("datetime64[D]")
        snaps, snap_of = np.unique(
            np.stack([train.user_id.to_numpy(), days.astype(np.int64)], axis=1),
            axis=0,
            return_inverse=True,
        )
        hist = history_snapshots(
            self.positives,
            snaps[:, 0],
            snaps[:, 1].astype("datetime64[D]"),
            cfg.max_history,
        )
        hist, snap_of = torch.from_numpy(hist), torch.from_numpy(snap_of.ravel())
        profile = {
            f: torch.from_numpy(v.astype(np.int64))
            for f, v in self.profile.rows(snaps[:, 0]).items()
        }
        item = torch.from_numpy(train.item.to_numpy().astype(np.int64))
        # log-Q: how often each video appears as a training positive
        counts = np.bincount(train.item.to_numpy(), minlength=catalog.n_items)
        log_q = torch.from_numpy(np.log(np.maximum(counts, 1) / max(len(train), 1)))

        self.model = TwoTower(
            self.content.sizes,
            self.profile.sizes,
            cfg.field_dim,
            cfg.user_field_dim,
            cfg.hidden,
            cfg.dim,
            cfg.history_grad,
        )
        # embedding tables get sparse gradients: only rows a batch touched update
        sparse, dense = self.model.parameter_groups()
        opts = [
            torch.optim.SparseAdam(sparse, lr=cfg.lr),
            torch.optim.Adam(dense, lr=cfg.lr),
        ]
        self.model.train()
        steps = -(-len(train) // cfg.batch_size)
        bar = tqdm(total=cfg.epochs * steps, desc="two_tower fit", leave=False)
        for _ in range(cfg.epochs):
            order = torch.from_numpy(rng.permutation(len(train)))
            for b in order.split(cfg.batch_size):
                s = snap_of[b]
                v = self._item_vectors(item[b])
                u = self.model.user_vectors(
                    *self._history_inputs(hist[s]),
                    {f: x[s] for f, x in profile.items()},
                )
                loss = in_batch_softmax_loss(
                    u, v, item[b], log_q[item[b]].float(), cfg.temperature
                )
                for opt in opts:
                    opt.zero_grad()
                loss.backward()
                for opt in opts:
                    opt.step()
                bar.update()
                bar.set_postfix(loss=f"{loss.item():.3f}")
        bar.close()

        self.model.eval()
        with torch.no_grad():
            self.item_emb = torch.cat(
                [
                    self._item_vectors(c)
                    for c in torch.arange(catalog.n_items).split(65_536)
                ]
            )
        return self

    def _item_vectors(self, idx: torch.Tensor) -> torch.Tensor:
        """Video tower output for catalog positions."""
        return self.model.item_vectors(
            {f: x[idx] for f, x in self._fields.items()}, self._tags[idx]
        )

    def _history_inputs(self, hist: torch.Tensor) -> tuple[dict, torch.Tensor]:
        """Content indices of history videos, 0 for empty (-1) slots."""
        known = hist >= 0
        h = hist.clamp(min=0)
        fields = {f: torch.where(known, x[h], 0) for f, x in self._fields.items()}
        return fields, self._tags[h] * known[..., None]

    def _user_vectors(self, user_ids: np.ndarray) -> torch.Tensor:
        """User tower output with histories as of the fit window's end."""
        user_ids = np.asarray(user_ids)
        days = np.full(len(user_ids), np.datetime64(self.as_of.date(), "D"))
        hist = torch.from_numpy(
            history_snapshots(self.positives, user_ids, days, self.cfg.max_history)
        )
        profile = {
            f: torch.from_numpy(v.astype(np.int64))
            for f, v in self.profile.rows(user_ids).items()
        }
        with torch.no_grad():
            return self.model.user_vectors(*self._history_inputs(hist), profile)

    def score_users(self, user_ids):
        return (self._user_vectors(user_ids) @ self.item_emb.T).numpy()

    def score_pairs(self, user_ids, item_ids, batch_users=2048):
        cols = self.catalog.index_of(np.asarray(item_ids))
        uniq, inv = np.unique(np.asarray(user_ids), return_inverse=True)
        u = self._user_vectors(uniq)
        known = cols >= 0
        out = np.full(len(cols), -np.inf)
        idx = torch.from_numpy(np.flatnonzero(known))
        rows = torch.from_numpy(inv[known].astype(np.int64))
        items = torch.from_numpy(cols[known].astype(np.int64))
        out[idx.numpy()] = (u[rows] * self.item_emb[items]).sum(dim=1).double().numpy()
        return out

    def recommend(self, user_ids, k, exclude, allowed, key, batch_users=2048):
        u = self._user_vectors(user_ids)
        blocked = torch.from_numpy(~allowed)
        chunk = max(1, min(batch_users, _MAX_SCORES // self.catalog.n_items))
        out = np.full((len(u), k), -1, dtype=np.int64)
        kk = min(k + 1, self.catalog.n_items)
        for start in range(0, len(u), chunk):
            with torch.no_grad():
                s = u[start : start + chunk] @ self.item_emb.T
            s[:, blocked] = float("-inf")
            for r, ex in enumerate(exclude[start : start + chunk]):
                s[r, torch.from_numpy(np.asarray(ex, dtype=np.int64))] = float("-inf")
            vals, idx = torch.topk(s, kk, dim=1)
            vals, idx = vals.numpy(), idx.numpy()
            # exact ties would need the shared key; fall back for those rows only
            tied = (vals[:, 1:] == vals[:, :-1]).any(axis=1)
            for r in np.flatnonzero(tied):
                out[start + r] = top_k(s[r : r + 1].double().numpy(), k, key=key)[0]
            fine = np.flatnonzero(~tied)
            top = np.where(np.isneginf(vals[fine, :k]), -1, idx[fine, :k])
            out[start + fine, : top.shape[1]] = top
        return out


def build_two_tower(cfg, items: pd.DataFrame, users: pd.DataFrame) -> TwoTowerRetriever:
    """
    Two-tower retriever from the experiment config.

    Parameters
    ----------
    cfg : Config
        Experiment config.
    items : pd.DataFrame
        Item table.
    users : pd.DataFrame
        User table.

    Returns
    -------
    TwoTowerRetriever
        Unfitted model.
    """
    return TwoTowerRetriever(cfg.two_tower, items, users, cfg.evaluation.seed)
