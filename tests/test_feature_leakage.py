"""The leakage tests. If any of these fail, no downstream number can be trusted.

1. Perturbation: rewriting every event on or after an example's date (labels,
   play times, extra fake rows) must not change a single feature value.
2. Sensitivity: rewriting events *before* that date must change features.
3. Oracle: features match a slow, obviously-correct pandas recomputation.
4. Blocklist: known after-the-fact columns never appear in the output.
5. Mutation: deliberately introducing a same-day leak makes test 1 fail, so
   the perturbation test is strong enough to catch the bug it exists for.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from krec.data.tables import load_eval_set
from krec.features.pipeline import get_historical_features

CUTOFF = pd.Timestamp("2022-05-01")


def _entities(cfg, n=300):
    ent = load_eval_set(cfg, "val_standard")
    ent = ent[ent.date == CUTOFF][["user_id", "item_id", "date", "is_click"]]
    return ent.sample(min(n, len(ent)), random_state=0).reset_index(drop=True)


def _with_modified_interactions(cfg, tmp_path, modify):
    """A config whose processed dir is a copy of cfg's with interactions modified."""
    new_dir = tmp_path / "processed"
    shutil.copytree(cfg.processed_dir, new_dir)
    df = pd.read_parquet(new_dir / "interactions.parquet")
    df["date"] = pd.to_datetime(df.date)
    modify(df).to_parquet(new_dir / "interactions.parquet", index=False)
    return cfg.override(dataset={"processed_dir": new_dir})


def _corrupt(df, mask, rng):
    df = df.copy()
    n = int(mask.sum())
    for col in ("is_click", "long_view", "is_like"):
        df.loc[mask, col] = rng.integers(0, 2, n).astype(df[col].dtype)
    df.loc[mask, "play_time_ms"] = rng.integers(0, 10**6, n)
    extra = df[mask].sample(frac=2.0, replace=True, random_state=1)  # fake traffic
    return pd.concat([df, extra], ignore_index=True)


def test_future_events_cannot_change_features(cfg, tmp_path):
    ent = _entities(cfg)
    before = get_historical_features(cfg, ent)
    rng = np.random.default_rng(0)
    cfg2 = _with_modified_interactions(
        cfg, tmp_path, lambda d: _corrupt(d, d.date >= CUTOFF, rng)
    )
    after = get_historical_features(cfg2, ent)
    pd.testing.assert_frame_equal(before, after)


def test_past_events_do_change_features(cfg, tmp_path):
    ent = _entities(cfg)
    before = get_historical_features(cfg, ent)
    rng = np.random.default_rng(0)
    day_before = CUTOFF - pd.Timedelta(days=1)
    cfg2 = _with_modified_interactions(
        cfg, tmp_path, lambda d: _corrupt(d, d.date == day_before, rng)
    )
    after = get_historical_features(cfg2, ent)
    changed = [
        c
        for c in before.columns
        if not before[c].equals(after[c]) and c.startswith(("item_", "user_", "ua_"))
    ]
    assert "item_impr_3d" in changed and "user_click_all" in changed


def test_features_match_bruteforce_oracle(cfg):
    ents = load_eval_set(cfg, "test_standard")[["user_id", "item_id", "date"]]
    ents = ents.sample(80, random_state=2).reset_index(drop=True)
    feats = get_historical_features(cfg, ents)

    log = pd.read_parquet(cfg.processed_dir / "interactions.parquet")
    log = log[log.source == "standard"].assign(date=lambda d: pd.to_datetime(d.date))
    items = pd.read_parquet(cfg.processed_dir / "items.parquet")[
        ["item_id", "author_id"]
    ]
    log = log.merge(items, on="item_id", how="left")
    author_of = items.set_index("item_id").author_id

    for r in feats.itertuples():
        d = pd.Timestamp(r.date)
        past = log[log.date < d]
        it, us = past[past.item_id == r.item_id], past[past.user_id == r.user_id]
        wk = us[us.date >= d - pd.Timedelta(days=7)]
        ua = us[us.author_id == author_of[r.item_id]]
        assert r.item_impr_all == len(it)
        assert r.item_click_all == it.is_click.sum()
        assert r.user_click_7d == wk.is_click.sum()
        assert r.user_impr_7d == len(wk)
        assert r.ua_impr_all == len(ua)
        assert r.ua_click_all == ua.is_click.sum()
        if len(it):
            assert r.item_days_since_last == (d - it.date.max()).days
            prior, k = past.is_click.mean(), cfg.features.ctr_prior_strength
            assert r.item_ctr_all == pytest.approx(
                (it.is_click.sum() + k * prior) / (len(it) + k)
            )
        else:
            assert pd.isna(r.item_days_since_last)


def test_blocklisted_columns_never_emitted(cfg):
    feats = get_historical_features(cfg, _entities(cfg, 20))
    for col in cfg.features.leaky_columns:
        assert not any(c == col or c.endswith("_" + col) for c in feats.columns)


def test_output_preserves_input_order_and_length(cfg):
    ent = _entities(cfg, 50).sample(frac=1.0, random_state=5).reset_index(drop=True)
    feats = get_historical_features(cfg, ent)
    assert len(feats) == len(ent)
    assert feats.user_id.tolist() == ent.user_id.tolist()
    assert feats.item_id.tolist() == ent.item_id.tolist()


def test_perturbation_test_catches_a_same_day_leak(cfg, tmp_path, monkeypatch):
    """Mutation check: make the ASOF join inclusive (>=) and the guard must fire."""
    from krec.features import pipeline

    def leaky_asof(view, alias, cutoff_sql, left="s"):
        on = " AND ".join(f"{left}.{k} = {alias}.{k}" for k in view.keys)
        return (
            f"ASOF LEFT JOIN cum_{view.name} {alias} "
            f"ON {on} AND ({cutoff_sql}) >= {alias}.date"
        )

    monkeypatch.setattr(pipeline, "_asof", leaky_asof)
    ent = _entities(cfg)
    before = get_historical_features(cfg, ent)
    rng = np.random.default_rng(0)
    cfg2 = _with_modified_interactions(
        cfg, tmp_path, lambda d: _corrupt(d, d.date >= CUTOFF, rng)
    )
    after = get_historical_features(cfg2, ent)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(before, after)
