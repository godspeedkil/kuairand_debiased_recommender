"""The interface every retrieval or ranking model implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ItemCatalog:
    """Dense item indexing shared by models and evaluation.

    Attributes
    ----------
    item_ids : np.ndarray
        Item ids in catalog order, shape (n_items,).
    category : np.ndarray
        Primary tag per item, -1 if unknown.
    upload_date : np.ndarray
        Upload date per item (datetime64[D]), NaT if unknown.
    """

    item_ids: np.ndarray
    category: np.ndarray
    upload_date: np.ndarray

    def __post_init__(self):
        self._index = pd.Index(self.item_ids)

    @property
    def n_items(self) -> int:
        """Number of items in the catalog."""
        return len(self.item_ids)

    def index_of(self, item_ids) -> np.ndarray:
        """Catalog positions of item ids.

        Parameters
        ----------
        item_ids : array-like
            Item ids to look up.

        Returns
        -------
        np.ndarray
            Position of each id, -1 if missing.
        """
        return self._index.get_indexer(np.asarray(item_ids))

    def available_mask(self, on_or_before: pd.Timestamp) -> np.ndarray:
        """Items uploaded on or before a date.

        Parameters
        ----------
        on_or_before : pd.Timestamp
            Last allowed upload date.

        Returns
        -------
        np.ndarray
            Boolean mask over the catalog, unknown upload count as available.
        """
        up = self.upload_date
        return np.isnat(up) | (up <= np.datetime64(on_or_before.date(), "D"))

    @classmethod
    def build(cls, items: pd.DataFrame, interactions: pd.DataFrame) -> ItemCatalog:
        """Catalog of every item in the item table or the interactions.

        Parameters
        ----------
        items : pd.DataFrame
            Item table with `item_id`, `tag` and `upload_dt`.
        interactions : pd.DataFrame
            Logs whose items must also be in the catalog.

        Returns
        -------
        ItemCatalog
            Items sorted by id.
        """
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
    """Base class for models that score (user, item) pairs.

    Attributes
    ----------
    name : str
        Label used in results.
    catalog : ItemCatalog
        Set by `fit`.
    """

    name: str = "recommender"
    catalog: ItemCatalog

    @abstractmethod
    def fit(
        self, interactions: pd.DataFrame, catalog: ItemCatalog, as_of: pd.Timestamp
    ) -> Recommender:
        """Fit on logs strictly before `as_of`.

        Parameters
        ----------
        interactions : pd.DataFrame
            Fit-window logs, in event order (`date`, `time_ms`).
        catalog : ItemCatalog
            Catalog the model scores.
        as_of : pd.Timestamp
            Start of the eval window.

        Returns
        -------
        Recommender
            The fitted model.
        """

    @abstractmethod
    def score_users(self, user_ids: np.ndarray) -> np.ndarray:
        """Scores for every catalog item.

        Parameters
        ----------
        user_ids : np.ndarray
            Users to score.

        Returns
        -------
        np.ndarray
            Scores in catalog order, shape (len(user_ids), catalog.n_items).
        """

    def recommend(
        self,
        user_ids: np.ndarray,
        k: int,
        exclude: list[np.ndarray],
        allowed: np.ndarray,
        key: np.ndarray,
        batch_users: int = 2048,
    ) -> np.ndarray:
        """Top-k catalog positions per user.

        Parameters
        ----------
        user_ids : np.ndarray
            Users to recommend for.
        k : int
            List length.
        exclude : list[np.ndarray]
            Catalog positions never returned, one array per user.
        allowed : np.ndarray
            Boolean mask of catalog positions that may be returned.
        key : np.ndarray
            Tie-break order per catalog position.
        batch_users : int, default 2048
            Users scored at once.

        Returns
        -------
        np.ndarray
            Catalog positions, shape (len(user_ids), k), -1 padding on short lists.
        """
        from krec.eval.metrics import top_k

        out = []
        for start in range(0, len(user_ids), batch_users):
            sl = slice(start, start + batch_users)
            s = self.score_users(user_ids[sl]).astype(np.float64, copy=True)
            s[:, ~allowed] = -np.inf
            for r, ex in enumerate(exclude[sl]):
                s[r, ex] = -np.inf
            out.append(top_k(s, k, key=key))
        return np.concatenate(out) if out else np.empty((0, k), dtype=np.int64)

    def score_pairs(
        self, user_ids: np.ndarray, item_ids: np.ndarray, batch_users: int = 2048
    ) -> np.ndarray:
        """Scores for explicit (user, item) pairs.

        Parameters
        ----------
        user_ids : np.ndarray
            User of each pair.
        item_ids : np.ndarray
            Item of each pair.
        batch_users : int, default 2048
            Users scored at once.

        Returns
        -------
        np.ndarray
            Score per pair, -inf for items not in the catalog.
        """
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
