"""Raw KuaiRand CSVs -> validated parquet tables.

Outputs (in `processed_dir`):
  interactions.parquet  one row per impression, standard + random logs, with a
                        `source` column and a `split` column (train/val/test)
  users.parquet         user features
  items.parquet         basic video features with leaky columns dropped
  manifest.json         row counts per (source, split), date ranges, id counts
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from krec.config import Config
from krec.data.schema import (
    BINARY_LOG_COLUMNS,
    ITEM_COLUMNS,
    LOG_COLUMNS,
    USER_COLUMNS,
)


class DataContractError(ValueError):
    """Raised when raw data doesn't match the documented KuaiRand schema."""


def _csv(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_kuairand.sh (real data) or "
            "`krec synth` (synthetic data) first."
        )
    return f"read_csv('{path.as_posix()}', header=true, auto_detect=true)"


def _check_columns(
    con: duckdb.DuckDBPyConnection, rel: str, expected: list[str], name: str
):
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()]
    missing = [c for c in expected if c not in cols]
    if missing:
        raise DataContractError(f"{name}: missing columns {missing}")


def _check(con: duckdb.DuckDBPyConnection, sql: str, message: str) -> None:
    bad = con.execute(sql).fetchone()[0]
    if bad:
        raise DataContractError(f"{message} ({bad} offending rows)")


def ingest(cfg: Config) -> dict:
    raw, out = cfg.raw_dir, cfg.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    files = cfg.dataset.files
    con = duckdb.connect()

    # ---- logs ----
    parts = []
    for source, fnames in (
        ("standard", files.standard_logs),
        ("random", files.random_logs),
    ):
        for fname in fnames:
            rel = _csv(raw / fname)
            _check_columns(con, rel, LOG_COLUMNS, fname)
            parts.append(f"SELECT *, '{source}' AS source FROM {rel}")
    con.execute(f"CREATE TABLE logs AS {' UNION ALL BY NAME '.join(parts)}")

    for col in BINARY_LOG_COLUMNS:
        _check(
            con,
            f"SELECT count(*) FROM logs WHERE {col} NOT IN (0, 1)",
            f"logs.{col} must be binary",
        )
    _check(
        con,
        "SELECT count(*) FROM logs WHERE user_id IS NULL OR video_id IS NULL "
        "OR date IS NULL",
        "logs keys must be non-null",
    )
    _check(
        con,
        "SELECT count(*) FROM logs WHERE (source = 'random') <> (is_rand = 1)",
        "is_rand must be 1 exactly for rows from the random-exposure log",
    )

    s = cfg.split
    con.execute(f"""
        CREATE TABLE interactions AS
        SELECT
            user_id::BIGINT                                     AS user_id,
            video_id::BIGINT                                    AS item_id,
            strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE     AS date,
            time_ms::BIGINT                                     AS time_ms,
            tab::INTEGER                                        AS tab,
            source,
            CASE
              WHEN strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE
                   BETWEEN DATE '{s.train[0]}' AND DATE '{s.train[1]}' THEN 'train'
              WHEN strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE
                   BETWEEN DATE '{s.val[0]}' AND DATE '{s.val[1]}' THEN 'val'
              WHEN strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE
                   BETWEEN DATE '{s.test[0]}' AND DATE '{s.test[1]}' THEN 'test'
              ELSE 'unused'
            END                                                 AS split,
            is_click::TINYINT                                   AS is_click, 
            long_view::TINYINT                                  AS long_view,
            is_like::TINYINT                                    AS is_like, 
            is_follow::TINYINT                                  AS is_follow,
            is_comment::TINYINT                                 AS is_comment, 
            is_forward::TINYINT                                 AS is_forward,
            is_hate::TINYINT                                    AS is_hate, 
            is_profile_enter::TINYINT                           AS is_profile_enter,
            play_time_ms::BIGINT                                AS play_time_ms, 
            duration_ms::BIGINT                                 AS duration_ms
        FROM logs
        ORDER BY time_ms, user_id, item_id
    """)
    _check(
        con,
        "SELECT count(*) FROM interactions WHERE split = 'unused'",
        "every log row should fall inside the configured split ranges",
    )

    # ---- users / items ----
    users_rel = _csv(raw / files.users)
    _check_columns(con, users_rel, USER_COLUMNS, files.users)
    con.execute(f"CREATE TABLE users AS SELECT * FROM {users_rel}")

    items_rel = _csv(raw / files.items)
    _check_columns(con, items_rel, ITEM_COLUMNS, files.items)
    leaky = set(cfg.features.leaky_columns)
    keep = [c for c in ITEM_COLUMNS if c not in leaky]
    select = ", ".join(
        (
            "video_id::BIGINT AS item_id"
            if c == "video_id"
            else (
                "TRY_CAST(upload_dt AS DATE) AS upload_dt"
                if c == "upload_dt"
                else "CAST(tag AS VARCHAR) AS tag" if c == "tag" else c
            )
        )
        for c in keep
    )
    con.execute(f"CREATE TABLE items AS SELECT {select} FROM {items_rel}")
    _check(
        con,
        "SELECT count(*) - count(DISTINCT item_id) FROM items",
        "items.item_id must be unique",
    )
    _check(
        con,
        "SELECT count(*) - count(DISTINCT user_id) FROM users",
        "users.user_id must be unique",
    )
    # For now, just a soft check (might enforce later)
    items_missing_features = con.execute(
        "SELECT count(DISTINCT item_id) FROM interactions ANTI JOIN items USING (item_id)"
    ).fetchone()[0]

    for table in ("interactions", "users", "items"):
        con.execute(
            f"COPY {table} TO '{(out / f'{table}.parquet').as_posix()}' (FORMAT parquet)"
        )

    manifest = {
        "dataset": cfg.dataset.name,
        "counts": {
            f"{src}/{spl}": n
            for src, spl, n in con.execute(
                "SELECT source, split, count(*) FROM interactions GROUP BY ALL ORDER BY ALL"
            ).fetchall()
        },
        "n_users": con.execute(
            "SELECT count(DISTINCT user_id) FROM interactions"
        ).fetchone()[0],
        "n_items_logged": con.execute(
            "SELECT count(DISTINCT item_id) FROM interactions"
        ).fetchone()[0],
        "n_items_catalog": con.execute("SELECT count(*) FROM items").fetchone()[0],
        "date_range": [
            str(d)
            for d in con.execute(
                "SELECT min(date), max(date) FROM interactions"
            ).fetchone()
        ],
        "logged_items_missing_features": items_missing_features,
        "dropped_leaky_item_columns": sorted(leaky & set(ITEM_COLUMNS)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    con.close()
    return manifest
