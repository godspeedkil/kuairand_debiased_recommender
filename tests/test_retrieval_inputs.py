import numpy as np
import pandas as pd
import pytest

from krec.data.tables import load_fit_window, load_items, load_users, split_start
from krec.models.base import ItemCatalog
from krec.retrieval.inputs import (
    DURATION_EDGES_S,
    Vocab,
    duration_bucket,
    encode_items,
    encode_users,
    history_snapshots,
)


@pytest.fixture(scope="module")
def fit(cfg):
    as_of = split_start(cfg, "val")
    df = load_fit_window(cfg, as_of)
    catalog = ItemCatalog.build(load_items(cfg), df)
    pos = df[df.is_click == 1].assign(item=lambda d: catalog.index_of(d.item_id))
    return df, catalog, pos


def test_vocab_maps_rare_and_unknown_values_to_zero():
    v = Vocab.fit(pd.Series(["a", "a", "b", None]), min_count=2)
    assert v.encode(["a", "b", "z"]).tolist() == [1, 0, 0]
    assert len(v) == 2


def test_duration_buckets_follow_the_edges():
    ms = np.array([np.nan, 1000, 7000, 7001, 18000, 10**7])
    assert duration_bucket(ms).tolist() == [0, 1, 2, 2, 3, len(DURATION_EDGES_S) + 1]


def test_item_content_is_encoded_from_training_positives(cfg, fit):
    df, catalog, pos = fit
    items = load_items(cfg)
    content = encode_items(items, catalog, pos.item.to_numpy(), max_tags=2)
    meta = items.set_index("item_id").reindex(catalog.item_ids)
    # the same author always gets the same index, and seen authors are known
    authors = content.fields["author_id"]
    seen = set(meta.author_id.iloc[pos.item].dropna())
    for a in list(seen)[:20]:
        assert len(set(authors[(meta.author_id == a).to_numpy()])) == 1
        assert authors[(meta.author_id == a).to_numpy()][0] > 0
    unseen = ~meta.author_id.isin(seen).to_numpy()
    assert (authors[unseen] == 0).all()
    # tags: at most max_tags, and an item with k tags has min(k, 2) nonzero slots
    assert content.tags.shape == (catalog.n_items, 2)
    n_tags = meta.tag.astype("string").str.split(",").str.len().fillna(0).to_numpy()
    filled = (content.tags > 0).sum(axis=1)
    assert (filled <= np.minimum(n_tags, 2)).all()
    assert content.sizes["tag"] > 1 and content.sizes["author_id"] == len(seen) + 1


def test_user_profile_rows_default_to_zero_for_unknown_users(cfg):
    users = load_users(cfg)
    profile = encode_users(users)
    rows = profile.rows(np.array([users.user_id.iloc[0], -99]))
    for f, v in rows.items():
        assert v[1] == 0 and v[0] == profile.fields[f][0]


def test_history_snapshots_match_a_brute_force_lookup(fit):
    df, catalog, pos = fit
    rng = np.random.default_rng(0)
    users = rng.choice(pos.user_id.unique(), 40)
    days = pd.to_datetime(rng.choice(pd.to_datetime(df.date).unique(), 40))
    users = np.append(users, -1)  # unknown user
    days = np.append(days, days[:1])
    got = history_snapshots(pos, users, days, max_len=5)
    ordered = pos.sort_values(["user_id", "date", "time_ms"], kind="stable")
    for row, u, d in zip(got, users, days, strict=True):
        before = ordered[(ordered.user_id == u) & (pd.to_datetime(ordered.date) < d)]
        want = before.item.to_numpy()[-5:]
        want = np.concatenate([np.full(5 - len(want), -1), want])
        assert row.tolist() == want.tolist()


def test_history_snapshots_exclude_the_same_day():
    pos = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "item": [10, 11, 12],
            "date": pd.to_datetime(["2022-04-01", "2022-04-02", "2022-04-02"]),
            "time_ms": [1, 2, 3],
        }
    )
    got = history_snapshots(
        pos, np.array([1, 1]), pd.to_datetime(["2022-04-02", "2022-04-03"]), 3
    )
    assert got.tolist() == [[-1, -1, 10], [10, 11, 12]]
    empty = history_snapshots(
        pos.iloc[:0], np.array([1]), pd.to_datetime(["2022-04-03"]), 2
    )
    assert empty.tolist() == [[-1, -1]]


def test_tag_parsing_handles_missing_blank_long_and_bad_values():
    from krec.retrieval.inputs import _tag_matrix

    tags = pd.Series(["3,1", None, "", "7, 8 ,9,10,11", "x,5"])
    assert _tag_matrix(tags, 4).tolist() == [
        [3, 1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [7, 8, 9, 10],
        [-1, 5, -1, -1],
    ]


def test_video_ids_get_an_index_only_with_enough_positives(cfg, fit):
    df, catalog, pos = fit
    counts = np.bincount(pos.item, minlength=catalog.n_items)
    content = encode_items(
        load_items(cfg), catalog, pos.item.to_numpy(), id_min_count=5
    )
    assert ((content.fields["item"] > 0) == (counts >= 5)).all()
    assert (
        "item" not in encode_items(load_items(cfg), catalog, pos.item.to_numpy()).fields
    )
