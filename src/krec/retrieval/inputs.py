"""
Inputs for the two-tower retriever: encoded video content, user profiles, and
point-in-time user histories.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from krec.models.base import ItemCatalog

# Duration bucket edges in seconds.
DURATION_EDGES_S = (7, 18, 30, 60, 120, 300, 600)
ITEM_FIELDS = ("author_id", "music_id", "music_type", "video_type", "upload_type")
USER_FIELDS = ("user_active_degree", *[f"onehot_feat{i}" for i in range(18)])


@dataclass
class Vocab:
    """Dense indices for raw values; 0 for unknown or rare values.

    Attributes
    ----------
    index : pd.Index
        Known values; value `index[i]` encodes to `i + 1`.
    """

    index: pd.Index

    @classmethod
    def fit(cls, values: pd.Series, min_count: int = 1) -> Vocab:
        """Keep values seen at least `min_count` times.

        Parameters
        ----------
        values : pd.Series
            Observed values, one per occurrence.
        min_count : int, default 1
            Minimum occurrences for a value to get its own index.

        Returns
        -------
        Vocab
            The fitted vocabulary.
        """
        counts = values.dropna().value_counts()
        return cls(pd.Index(counts.index[counts >= min_count]))

    def encode(self, values) -> np.ndarray:
        """Indices of values, 0 for unknown ones.

        Parameters
        ----------
        values : array-like
            Values to encode.

        Returns
        -------
        np.ndarray
            int64 index per value.
        """
        return self.index.get_indexer(pd.Index(values)) + 1

    def __len__(self) -> int:
        return len(self.index) + 1


@dataclass
class ItemContent:
    """Encoded content per catalog position.

    Attributes
    ----------
    fields : dict[str, np.ndarray]
        One index per item for each single-valued field.
    tags : np.ndarray
        Tag indices, shape (n_items, max_tags), 0 for padding.
    sizes : dict[str, int]
        Vocabulary size per field.
    """

    fields: dict[str, np.ndarray]
    tags: np.ndarray
    sizes: dict[str, int]


def duration_bucket(duration_ms: np.ndarray) -> np.ndarray:
    """Bucket durations at `DURATION_EDGES_S`.

    Parameters
    ----------
    duration_ms : np.ndarray
        Durations in milliseconds, NaN if unknown.

    Returns
    -------
    np.ndarray
        0 for unknown, then 1..len(edges) + 1 from shortest to longest.
    """
    d = np.asarray(duration_ms, dtype=float)
    b = np.searchsorted(np.asarray(DURATION_EDGES_S) * 1000.0, d, side="right") + 1
    return np.where(np.isnan(d), 0, b)


def _tag_matrix(tag: pd.Series, max_tags: int) -> np.ndarray:
    """Raw tag ids per item, at most `max_tags` each.

    Parameters
    ----------
    tag : pd.Series
        Comma-separated tags per item.
    max_tags : int
        Tags kept per item.

    Returns
    -------
    np.ndarray
        int64 tag ids, shape (n_items, max_tags), -1 padding.
    """
    # pyarrow splits in compiled code; pandas' .str.split is ~10x slower at 1K scale
    arr = pa.array(tag.astype("string").to_numpy(na_value=None), type=pa.string())
    lists = pc.list_slice(pc.split_pattern(arr, ","), 0, max_tags)
    flat = pc.utf8_trim_whitespace(pc.list_flatten(lists))
    rows = pc.list_parent_indices(lists).to_numpy()
    pos = np.arange(len(rows)) - np.searchsorted(rows, rows, side="left")
    numeric = pc.match_substring_regex(flat, r"^[0-9]+$").to_numpy(zero_copy_only=False)
    values = np.full(len(rows), -1, dtype=np.int64)
    values[numeric] = pc.cast(pc.filter(flat, numeric), pa.int64()).to_numpy()
    out = np.full((len(tag), max_tags), -1, dtype=np.int64)
    out[rows, pos] = values
    return out


def encode_items(
    items: pd.DataFrame,
    catalog: ItemCatalog,
    positive_items: np.ndarray,
    min_count: int = 1,
    max_tags: int = 4,
    id_min_count: int | None = None,
) -> ItemContent:
    """Encode every catalog item's content

    Vocabularies come from the items of training positives, so a value gets an
    embedding only if the model can learn it; everything else maps to 0.

    Parameters
    ----------
    items : pd.DataFrame
        Item table with `item_id`, the `ITEM_FIELDS`, `video_duration`, and `tag`.
    catalog : ItemCatalog
        Catalog whose positions are encoded.
    positive_items : np.ndarray
        Catalog position of each training positive, repeats included.
    min_count : int, default 1
        Minimum positives for a value to get its own index.
    max_tags : int, default 4
        Tags kept per item.
    id_min_count : int, optional
        Positives a video needs for its own id embedding, None for content only.

    Returns
    -------
    ItemContent
        Encoded fields, tags, and vocabulary sizes.
    """
    meta = items.set_index("item_id").reindex(catalog.item_ids)
    pos_meta = meta.iloc[positive_items]
    fields, sizes = {}, {}
    for f in ITEM_FIELDS:
        vocab = Vocab.fit(pos_meta[f], min_count)
        fields[f] = vocab.encode(meta[f])
        sizes[f] = len(vocab)
    if id_min_count is not None:
        ids = Vocab.fit(pd.Series(positive_items), id_min_count)
        fields["item"] = ids.encode(np.arange(catalog.n_items))
        sizes["item"] = len(ids)
    fields["duration"] = duration_bucket(meta["video_duration"].to_numpy())
    sizes["duration"] = len(DURATION_EDGES_S) + 2

    # parse once for the catalog
    raw = _tag_matrix(meta["tag"].reset_index(drop=True), max_tags)
    pos_tags = raw[positive_items].ravel()
    tag_vocab = Vocab.fit(pd.Series(pos_tags[pos_tags >= 0]), min_count)
    tags = np.where(raw >= 0, tag_vocab.encode(raw.ravel()).reshape(raw.shape), 0)
    sizes["tag"] = len(tag_vocab)
    return ItemContent(fields=fields, tags=tags, sizes=sizes)


@dataclass
class UserProfile:
    """Encoded profile fields per user.

    Attributes
    ----------
    index : pd.Index
        User ids, in row order of `fields`.
    fields : dict[str, np.ndarray]
        One index per user for each of `USER_FIELDS`.
    sizes : dict[str, int]
        Vocabulary size per field.
    """

    index: pd.Index
    fields: dict[str, np.ndarray]
    sizes: dict[str, int]

    def rows(self, user_ids: np.ndarray) -> dict[str, np.ndarray]:
        """Field indices for users, 0 for users not in the table.

        Parameters
        ----------
        user_ids : np.ndarray
            Users to look up.

        Returns
        -------
        dict[str, np.ndarray]
            One index per user for each field.
        """
        r = self.index.get_indexer(np.asarray(user_ids))
        return {
            f: np.where(r >= 0, v[np.maximum(r, 0)], 0) for f, v in self.fields.items()
        }


def encode_users(users: pd.DataFrame) -> UserProfile:
    """Encode the static profile of every user in the user table.

    Parameters
    ----------
    users : pd.DataFrame
        User table with `user_id` and the `USER_FIELDS`.

    Returns
    -------
    UserProfile
        Encoded fields and vocabulary sizes.
    """
    fields, sizes = {}, {}
    for f in USER_FIELDS:
        vocab = Vocab.fit(users[f])
        fields[f] = vocab.encode(users[f])
        sizes[f] = len(vocab)
    return UserProfile(pd.Index(users.user_id.to_numpy()), fields, sizes)


def history_snapshots(
    positives: pd.DataFrame, user_ids: np.ndarray, days: np.ndarray, max_len: int
) -> np.ndarray:
    """Each user's last `max_len` positives strictly before a day.

    Parameters
    ----------
    positives : pd.DataFrame
        Positive events with `user_id`, `item` (catalog position), `date`, and
        `time_ms`, in any order.
    user_ids : np.ndarray
        User of each snapshot.
    days : np.ndarray
        Snapshot day per user (datetime64-like); only earlier days count.
    max_len : int
        History length.

    Returns
    -------
    np.ndarray
        Catalog positions, shape (n_snapshots, max_len), oldest first and the most
        recent last, -1 padding on the left for short histories.
    """
    if positives.empty:
        return np.full((len(user_ids), max_len), -1, dtype=np.int64)
    p = positives.sort_values(["user_id", "date", "time_ms"], kind="stable")
    uid = p.user_id.to_numpy()
    pday = pd.to_datetime(p.date).to_numpy().astype("datetime64[D]").astype(np.int64)
    users, starts = np.unique(uid, return_index=True)
    rank = np.searchsorted(users, uid)
    # one sortable key per (user, day): user rank in the high bits, day in the low
    span = int(pday.max() - pday.min() + 2)
    base = pday.min()
    keys = rank * span + (pday - base)

    qday = pd.to_datetime(days).to_numpy().astype("datetime64[D]").astype(np.int64)
    qrank = np.searchsorted(users, user_ids)
    known = (qrank < len(users)) & (
        users[np.minimum(qrank, len(users) - 1)] == user_ids
    )
    qrank = np.minimum(qrank, len(users) - 1)
    qkey = qrank * span + np.clip(qday - base, 0, span - 1)
    end = np.searchsorted(keys, qkey, side="left")  # first event on or after the day
    start = starts[qrank]
    idx = end[:, None] - max_len + np.arange(max_len)
    valid = known[:, None] & (idx >= start[:, None]) & (idx < end[:, None])
    items = p["item"].to_numpy()
    return np.where(valid, items[np.clip(idx, 0, len(items) - 1)], -1)
