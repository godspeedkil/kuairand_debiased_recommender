"""Non-learned baselines. Every later model has to beat these on both test sets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from krec.config import BaselinesConfig
from krec.models.base import Recommender


class RandomRecommender(Recommender):
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, interactions, catalog, as_of):
        self.catalog = catalog
        self._rng = np.random.default_rng(self.seed)
        return self

    def score_users(self, user_ids):
        return self._rng.random((len(user_ids), self.catalog.n_items))


class MostPopular(Recommender):
    """Items ranked by how many positives they got in the fit window."""

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


class ItemCooccurrence(Recommender):
    """Item-item collaborative filtering on co-clicks, with cosine normalization."""

    name = "item_cooc"

    def __init__(
        self,
        signal: str = "is_click",
        top_neighbors: int = 200,
        shrinkage: float = 10.0,
    ):
        self.signal, self.top_neighbors, self.shrinkage = (
            signal,
            top_neighbors,
            shrinkage,
        )

    def fit(self, interactions, catalog, as_of):
        self.catalog = catalog
        pos = interactions.loc[interactions[self.signal] == 1, ["user_id", "item_id"]]
        pos = pos.drop_duplicates()
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


def _keep_top_per_row(m: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep only the k largest entries in each row of a sparse matrix."""
    m = m.tocsr()
    indptr, indices, data = [0], [], []
    for r in range(m.shape[0]):
        lo, hi = m.indptr[r], m.indptr[r + 1]
        d, c = m.data[lo:hi], m.indices[lo:hi]
        if len(d) > k:
            keep = np.argpartition(-d, k - 1)[:k]
            d, c = d[keep], c[keep]
        data.append(d)
        indices.append(c)
        indptr.append(indptr[-1] + len(d))
    return sp.csr_matrix(
        (
            np.concatenate(data) if data else [],
            np.concatenate(indices) if indices else [],
            indptr,
        ),
        shape=m.shape,
    )


def build_baselines(cfg: BaselinesConfig, seed: int) -> list[Recommender]:
    """Instantiate the baselines enabled in the config, in declaration order."""
    registry = {
        "most_popular": lambda p: MostPopular(**p),
        "recent_popular": lambda p: MostPopular(**p),
        "item_cooc": lambda p: ItemCooccurrence(**p),
        "random": lambda p: RandomRecommender(seed=seed, **p),
    }
    return [
        registry[name](params.model_dump()) for name, params in cfg.enabled().items()
    ]
