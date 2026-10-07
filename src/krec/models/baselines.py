"""Non-learned baselines. Every later model has to beat these on both test sets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from krec.config import BaselinesConfig
from krec.models.base import Recommender


def _ranked_head(
    scores: np.ndarray, key: np.ndarray, allowed: np.ndarray, n: int
) -> np.ndarray:
    """First n items ranked by (score desc, key asc).

    Parameters
    ----------
    scores : np.ndarray
        Score per catalog position.
    key : np.ndarray
        Tie-break order per catalog position.
    allowed : np.ndarray
        Boolean mask of positions that may be returned.
    n : int
        Number of positions to return.

    Returns
    -------
    np.ndarray
        Up to n catalog positions, best first.
    """
    idx = np.flatnonzero(allowed)
    n = min(n, len(idx))
    if n == 0:
        return idx
    s = scores[idx]
    kth = np.partition(s, len(s) - n)[len(s) - n]
    cand = idx[s >= kth]
    return cand[np.lexsort((key[cand], -scores[cand]))][:n]


def _first_k(order: np.ndarray, k: int, exclude: np.ndarray) -> np.ndarray:
    """First k entries of `order`.

    Parameters
    ----------
    order : np.ndarray
        Catalog positions.
    k : int
        List length.
    exclude : np.ndarray
        Positions to skip.

    Returns
    -------
    np.ndarray
        k positions, padded with -1 if `order` runs out.
    """
    head = order[: k + len(exclude)]
    head = head[~np.isin(head, exclude)][:k]
    return np.pad(head, (0, k - len(head)), constant_values=-1)


class RandomRecommender(Recommender):
    """Uniformly random scores.

    Parameters
    ----------
    seed : int, default 0
        Seed for the scores.
    """

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, interactions, catalog, as_of):
        self.catalog = catalog
        self._rng = np.random.default_rng(self.seed)
        return self

    def score_users(self, user_ids):
        return self._rng.random((len(user_ids), self.catalog.n_items))

    def score_pairs(self, user_ids, item_ids, batch_users=2048):
        cols = self.catalog.index_of(np.asarray(item_ids))
        return np.where(cols >= 0, self._rng.random(len(cols)), -np.inf)

    def recommend(self, user_ids, k, exclude, allowed, key, batch_users=2048):
        cand = np.flatnonzero(allowed)
        out = np.full((len(user_ids), k), -1, dtype=np.int64)
        for i, ex in enumerate(exclude):
            n = min(len(cand), k + len(ex))
            out[i] = _first_k(
                cand[self._rng.choice(len(cand), n, replace=False)], k, ex
            )
        return out


class MostPopular(Recommender):
    """Items ranked by their positives in the fit window.

    Parameters
    ----------
    signal : str, default "is_click"
        Label that counts as a positive.
    window_days : int, optional
        Count only the last N days before `as_of`, None for all days.
    """

    def __init__(self, signal: str = "is_click", window_days: int | None = None):
        self.signal, self.window_days = signal, window_days
        self.name = (
            "most_popular" if window_days is None else f"recent_popular_{window_days}d"
        )

    def fit(self, interactions, catalog, as_of):
        self.catalog = catalog
        df = interactions
        if self.window_days is not None:
            # Days [as_of - N, as_of - 1]: the N days right before the eval window.
            df = df[df.date >= (as_of - pd.Timedelta(days=self.window_days))]
        pos = df.loc[df[self.signal] == 1, "item_id"]
        counts = np.zeros(catalog.n_items)
        idx = catalog.index_of(pos.to_numpy())
        np.add.at(counts, idx[idx >= 0], 1.0)
        self.item_scores = counts
        return self

    def score_users(self, user_ids):
        return np.broadcast_to(
            self.item_scores, (len(user_ids), self.catalog.n_items)
        ).copy()

    def score_pairs(self, user_ids, item_ids, batch_users=2048):
        cols = self.catalog.index_of(item_ids)
        return np.where(cols >= 0, self.item_scores[np.maximum(cols, 0)], -np.inf)

    def recommend(self, user_ids, k, exclude, allowed, key, batch_users=2048):
        longest = max((len(ex) for ex in exclude), default=0)
        order = _ranked_head(self.item_scores, key, allowed, k + longest)
        return np.stack([_first_k(order, k, ex) for ex in exclude]).reshape(-1, k)


class ItemCooccurrence(Recommender):
    """Item-item collaborative filtering on co-clicks, with cosine normalization.

    Parameters
    ----------
    signal : str, default "is_click"
        Label that counts as a positive.
    top_neighbors : int, default 200
        Neighbours kept per item.
    shrinkage : float, default 10.0
        Added to the cosine denominator, damps similarities from few co-clicks.
    max_history : int, default 200
        Most recent positives per user used to fit and score.
    """

    name = "item_cooc"

    def __init__(
        self,
        signal: str = "is_click",
        top_neighbors: int = 200,
        shrinkage: float = 10.0,
        max_history: int = 200,
    ):
        self.signal, self.top_neighbors, self.shrinkage, self.max_history = (
            signal,
            top_neighbors,
            shrinkage,
            max_history,
        )

    def fit(self, interactions, catalog, as_of):
        self.catalog = catalog
        pos = interactions.loc[interactions[self.signal] == 1, ["user_id", "item_id"]]
        # rows arrive in event order (date, time_ms), so "last" means most recent
        pos = pos.drop_duplicates(keep="last")
        recent = pos.groupby("user_id").cumcount(ascending=False) < self.max_history
        pos = pos[recent.to_numpy()]
        cols = catalog.index_of(pos.item_id.to_numpy())
        pos = pos[cols >= 0]
        cols = cols[cols >= 0]
        self.user_index = pd.Index(np.unique(pos.user_id.to_numpy()))
        rows = self.user_index.get_indexer(pos.user_id.to_numpy())
        x = sp.csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(len(self.user_index), catalog.n_items),
        )
        self.history = x

        co = (x.T @ x).tocsr().astype(np.float64)
        co.setdiag(0.0)
        co.eliminate_zeros()
        n = np.asarray(x.sum(axis=0)).ravel()
        co = co.tocoo()
        vals = co.data / (np.sqrt(n[co.row] * n[co.col]) + self.shrinkage)
        sim = sp.csr_matrix((vals, (co.row, co.col)), shape=co.shape)
        self.sim = _keep_top_per_row(sim, self.top_neighbors)
        # break ties on user cold starts
        pop = n / max(n.max(), 1.0)
        self.pop_tiebreak = 1e-6 * pop
        return self

    def score_users(self, user_ids):
        rows = self.user_index.get_indexer(np.asarray(user_ids))
        out = np.tile(self.pop_tiebreak, (len(rows), 1))
        known = rows >= 0
        if known.any():
            out[known] += (self.history[rows[known]] @ self.sim).toarray()
        return out

    def recommend(self, user_ids, k, exclude, allowed, key, batch_users=1024):
        longest = max((len(ex) for ex in exclude), default=0)
        pop_order = np.empty(0, dtype=np.int64)
        n_allowed = int(allowed.sum())
        rows = self.user_index.get_indexer(np.asarray(user_ids))
        out = np.full((len(rows), k), -1, dtype=np.int64)
        empty = np.empty(0, dtype=np.int64)
        for start in range(0, len(rows), batch_users):
            block = rows[start : start + batch_users]
            known = block >= 0
            sparse = (self.history[block[known]] @ self.sim).tocsr()
            pos = np.cumsum(known) - 1  # row of each known user in `sparse`
            # the popularity head is shared by all batches; extend it only when a
            # batch needs a longer one (batch size is a memory knob, not a cost)
            need = k + longest + int(np.diff(sparse.indptr).max(initial=0))
            if len(pop_order) < min(need, n_allowed):
                pop_order = _ranked_head(self.pop_tiebreak, key, allowed, 2 * need)
            for j in range(len(block)):
                i = start + j
                if known[j]:
                    lo, hi = sparse.indptr[pos[j]], sparse.indptr[pos[j] + 1]
                    idx, val = sparse.indices[lo:hi], sparse.data[lo:hi]
                    keep = allowed[idx]
                    idx, val = idx[keep], val[keep]
                else:
                    idx, val = empty, np.empty(0)
                # enough popularity candidates to fill k after exclusions
                head = pop_order[: k + len(exclude[i]) + len(idx)]
                c, first = np.unique(np.concatenate([idx, head]), return_index=True)
                v = np.concatenate([val, np.zeros(len(head))])[first]
                total = v + self.pop_tiebreak[c]
                ok = ~np.isin(c, exclude[i])
                c, total = c[ok], total[ok]
                top = c[np.lexsort((key[c], -total))[:k]]
                out[i, : len(top)] = top
        return out

    def score_pairs(self, user_ids, item_ids, batch_users=4096):
        cols = self.catalog.index_of(np.asarray(item_ids))
        rows = self.user_index.get_indexer(np.asarray(user_ids))
        out = np.where(cols >= 0, self.pop_tiebreak[np.maximum(cols, 0)], -np.inf)
        known = np.flatnonzero((rows >= 0) & (cols >= 0))
        uniq, inv = np.unique(rows[known], return_inverse=True)
        for start in range(0, len(uniq), batch_users):
            block = uniq[start : start + batch_users]
            sel = (inv >= start) & (inv < start + len(block))
            scores = (self.history[block] @ self.sim).tocsr()  # sparse
            vals = scores[inv[sel] - start, cols[known[sel]]]
            out[known[sel]] += np.asarray(vals).ravel()
        return out


def _keep_top_per_row(m: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep the k largest entries in each row of a sparse matrix.

    Parameters
    ----------
    m : sp.csr_matrix
        Matrix to prune.
    k : int
        Entries kept per row.

    Returns
    -------
    sp.csr_matrix
        Same shape, at most k entries per row.
    """
    m = m.tocsr()
    rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
    order = np.lexsort((-m.data, rows))  # by row, then largest first
    # entries of row r sit at positions indptr[r]..indptr[r+1] of `order`
    rank = np.arange(len(order)) - m.indptr[rows[order]]
    keep = order[rank < k]
    return sp.csr_matrix((m.data[keep], (rows[keep], m.indices[keep])), shape=m.shape)


def build_baselines(cfg: BaselinesConfig, seed: int) -> list[Recommender]:
    """Instantiate the baselines enabled in the config.

    Parameters
    ----------
    cfg : BaselinesConfig
        Which baselines to build, with their parameters.
    seed : int
        Seed for the random baseline.

    Returns
    -------
    list of Recommender
        Unfitted models.
    """
    registry = {
        "most_popular": lambda p: MostPopular(**p),
        "recent_popular": lambda p: MostPopular(**p),
        "item_cooc": lambda p: ItemCooccurrence(**p),
        "random": lambda p: RandomRecommender(seed=seed, **p),
    }
    return [
        registry[name](params.model_dump()) for name, params in cfg.enabled().items()
    ]
