import json

import pandas as pd
import pytest

from krec.config import load_config
from krec.data.ingest import DataContractError, ingest
from krec.data.tables import load_eval_set, load_fit_window, split_start


def _interactions(cfg):
    df = pd.read_parquet(cfg.processed_dir / "interactions.parquet")
    return df.assign(date=pd.to_datetime(df.date))


def test_splits_are_time_ordered_and_disjoint(cfg):
    df = _interactions(cfg)
    rng = df.groupby("split").date.agg(["min", "max"])
    assert rng.loc["train", "max"] < rng.loc["val", "min"]
    assert rng.loc["val", "max"] < rng.loc["test", "min"]
    assert set(df.split) == {"train", "val", "test"}


def test_random_logs_only_where_documented(cfg):
    df = _interactions(cfg)
    assert df[df.source == "random"].date.min() == pd.Timestamp("2022-04-22")


def test_fit_window_is_strictly_before_eval_set(cfg):
    for split in ("val", "test"):
        start = split_start(cfg, split)
        fit = load_fit_window(cfg, start)
        assert fit.date.max() < start
        assert set(fit.source) == {"standard"}
        assert load_eval_set(cfg, f"{split}_random").date.min() >= start


def test_leaky_item_columns_are_dropped(cfg):
    items = pd.read_parquet(cfg.processed_dir / "items.parquet")
    assert "visible_status" not in items.columns
    manifest = json.loads((cfg.processed_dir / "manifest.json").read_text())
    assert "visible_status" in manifest["dropped_leaky_item_columns"]


def test_contract_violation_is_caught(cfg, tmp_path):
    bad = load_config("configs/synthetic.yaml", root=tmp_path)
    bad.raw_dir.mkdir(parents=True)
    for f in cfg.raw_dir.iterdir():
        (bad.raw_dir / f.name).write_bytes(f.read_bytes())
    name = cfg.dataset.files.random_logs[0]
    logs = pd.read_csv(bad.raw_dir / name)
    logs.loc[0, "is_rand"] = 0  # a random-log row claiming to be policy traffic
    logs.to_csv(bad.raw_dir / name, index=False)
    with pytest.raises(DataContractError, match="is_rand"):
        ingest(bad)


def test_event_order_follows_date_when_clocks_disagree(cfg, tmp_path):
    late = load_config("configs/synthetic.yaml", root=tmp_path)
    late.raw_dir.mkdir(parents=True)
    for f in cfg.raw_dir.iterdir():
        (late.raw_dir / f.name).write_bytes(f.read_bytes())
    name = cfg.dataset.files.standard_logs[-1]
    logs = pd.read_csv(late.raw_dir / name)
    # like real KuaiRand: a row dated the next day (here into val) while later
    # rows of its own day remain in train
    day = logs[logs.date == 20220430].sort_values("time_ms")
    logs.loc[day.index[-10], "date"] = 20220501
    logs.to_csv(late.raw_dir / name, index=False)
    ingest(late)
    df = pd.read_parquet(late.processed_dir / "interactions.parquet")
    order = df.split.map({"train": 0, "val": 1, "test": 2})
    assert order.is_monotonic_increasing
    assert pd.to_datetime(df.date).is_monotonic_increasing
