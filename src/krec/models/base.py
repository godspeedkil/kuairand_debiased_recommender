"""The interface every retrieval or ranking model implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ItemCatalog:
    """Dense item indexing shared by models and evaluation."""

    item_ids: np.ndarray  # (n_items,) catalog order
    category: np.ndarray  # (n_items,) primary tag, for diversity metrics
    upload_date: np.ndarray  # (n_items,) datetime64[D], NaT if unknown

    def __post_init__(self):
        self._index = pd.Index(self.item_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def index_of(self, item_ids) -> np.ndarray:
        """Catalog positions; -1 for unknown ids."""
        return self._index.get_indexer(np.asarray(item_ids))

    def available_mask(self, on_or_before: pd.Timestamp) -> np.ndarray:
        """Items that exist (were uploaded) by the given date."""
        up = self.upload_date
        return np.isnat(up) | (up <= np.datetime64(on_or_before.date(), "D"))

    @classmethod
    def build(cls, items: pd.DataFrame, interactions: pd.DataFrame) -> ItemCatalog:
        ids = np.union1d(items.item_id.to_numpy(), interactions.item_id.unique())
        meta = items.set_index("item_id").reindex(ids)
        tag = meta["tag"].astype("string").fillna("")
        primary = tag.str.split(",").str[0].replace("", "-1").fillna("-1")
        category = (
            pd.to_numeric(primary, errors="coerce")
            .fillna(-1)
            .astype(np.int64)
            .to_numpy()
        )
        upload = (
            pd.to_datetime(meta["upload_dt"], errors="coerce")
            .to_numpy()
            .astype("datetime64[D]")
        )
        return cls(item_ids=ids, category=category, upload_date=upload)


class Recommender(ABC):
    """Scores (user, item) pairs."""

    name: str = "recommender"
    catalog: ItemCatalog  # fixes the column order of score_users

    @abstractmethod
    def fit(
        self, interactions: pd.DataFrame, catalog: ItemCatalog, as_of: pd.Timestamp
    ) -> Recommender:
        """Fit on logs strictly before `as_of`."""

    @abstractmethod
    def score_users(self, user_ids: np.ndarray) -> np.ndarray:
        """(len(user_ids), catalog.n_items) scores in catalog order."""

    def score_pairs(
        self, user_ids: np.ndarray, item_ids: np.ndarray, batch_users: int = 2048
    ) -> np.ndarray:
        """Scores for explicit (user, item) pairs"""
        user_ids, item_ids = np.asarray(user_ids), np.asarray(item_ids)
        cols = self.catalog.index_of(item_ids)
        out = np.full(len(user_ids), -np.inf)
        uniq, inv = np.unique(user_ids, return_inverse=True)
        for start in range(0, len(uniq), batch_users):
            block = uniq[start : start + batch_users]
            scores = self.score_users(block)
            sel = (inv >= start) & (inv < start + len(block)) & (cols >= 0)
            out[sel] = scores[inv[sel] - start, cols[sel]]
        return out
