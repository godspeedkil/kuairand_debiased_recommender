"""Read helpers for the processed tables.

Named evaluation sets are "<split>_<source>", e.g. `test_random`.

Allows for "as of" joins to make sure to respect sequential splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from krec.config import Config


@dataclass(frozen=True)
class EvalSet:
    name: str
    split: str
    source: str


def parse_eval_set(name: str) -> EvalSet:
    split, source = name.split("_", 1)
    if split not in ("train", "val", "test") or source not in ("standard", "random"):
        raise ValueError(f"bad eval set name {name!r}; expected e.g. 'test_random'")
    return EvalSet(name, split, source)


def _q(cfg: Config, sql: str, **params) -> pd.DataFrame:
    path = (cfg.processed_dir / "interactions.parquet").as_posix()
    with duckdb.connect() as con:
        con.execute(f"CREATE VIEW interactions AS SELECT * FROM read_parquet('{path}')")
        return con.execute(sql, params).df()


def split_start(cfg: Config, split: str) -> pd.Timestamp:
    return pd.Timestamp(cfg.split.start(split))


def load_eval_set(cfg: Config, name: str) -> pd.DataFrame:
    es = parse_eval_set(name)
    return _q(
        cfg,
        "SELECT * FROM interactions WHERE split = $s AND source = $src "
        "ORDER BY time_ms",
        s=es.split,
        src=es.source,
    )


def load_fit_window(
    cfg: Config, before: pd.Timestamp, sources=("standard",)
) -> pd.DataFrame:
    """All logs from `sources` with date strictly before `before`."""
    src = ", ".join(f"'{s}'" for s in sources)
    return _q(
        cfg,
        f"SELECT * FROM interactions WHERE date < $d AND source IN ({src}) "
        "ORDER BY time_ms",
        d=before.date(),
    )


def load_items(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.processed_dir / "items.parquet")


def load_users(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.processed_dir / "users.parquet")
