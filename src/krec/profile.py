"""Profile a KuaiRand release into small summary tables, for cross-version EDA.

The "pool" is the set of videos that appear in a release's own random log
(KuaiRand's random-exposure candidate pool).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from krec.config import Config
from krec.data.schema import BINARY_LOG_COLUMNS

TABLES = (
    "files",
    "daily",
    "tabs",
    "user_activity",
    "attributes",
    "concentration",
    "pool",
    "clock",
    "users",
)
# display order of the bucket labels below.
BUCKET_ORDER = {
    "duration": ["0-7s", "7-18s", "18-60s", "1-3min", "3min+", "unknown"],
    "age_at_impression": [
        "0d",
        "1-7d",
        "8-30d",
        "31-180d",
        "181-365d",
        "365d+",
        "unknown",
    ],
}
# bucket edges in seconds. 7s and 18s are KuaiRand's thresholds for a valid
# play (`is_click` in the single-column UI) and for `long_view`.
_DURATION_SQL = """CASE
    WHEN duration_ms IS NULL THEN 'unknown'
    WHEN duration_ms <= 7000 THEN '0-7s'
    WHEN duration_ms <= 18000 THEN '7-18s'
    WHEN duration_ms <= 60000 THEN '18-60s'
    WHEN duration_ms <= 180000 THEN '1-3min'
    ELSE '3min+' END"""
_AGE_SQL = """CASE
    WHEN age_days IS NULL THEN 'unknown'
    WHEN age_days < 1 THEN '0d'
    WHEN age_days <= 7 THEN '1-7d'
    WHEN age_days <= 30 THEN '8-30d'
    WHEN age_days <= 180 THEN '31-180d'
    WHEN age_days <= 365 THEN '181-365d'
    ELSE '365d+' END"""


def profile_dir(cfg: Config) -> Path:
    return cfg.processed_dir / "profile"


def _connect(cfg: Config) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{cfg.features.duckdb_memory_limit}'")
    spill = (cfg.processed_dir / ".duckdb_tmp").as_posix()
    con.execute(f"SET temp_directory = '{spill}'")
    con.execute("SET preserve_insertion_order = false")  # lets big scans stream
    return con


def _csv(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the release with "
            "scripts/download_kuairand.sh first."
        )
    # everything as text, then TRY_CAST: a bad value becomes NULL and is counted
    return f"read_csv('{path.as_posix()}', header=true, all_varchar=true)"


def _stage(cfg: Config, con: duckdb.DuckDBPyConnection, out: Path) -> list[dict]:
    """One pass over every log file into a narrow, typed staging parquet."""
    f = cfg.dataset.files
    sources = [("standard", n) for n in f.standard_logs] + [
        ("random", n) for n in f.random_logs
    ]
    flags = ", ".join(f"TRY_CAST({c} AS TINYINT) AS {c}" for c in BINARY_LOG_COLUMNS)
    parts = [
        f"""SELECT {i} AS file_id, '{source}' AS source,
               TRY_CAST(user_id AS BIGINT) AS user_id,
               TRY_CAST(video_id AS BIGINT) AS video_id,
               TRY_CAST(date AS INTEGER) AS date,
               TRY_CAST(hourmin AS INTEGER) AS hourmin,
               TRY_CAST(time_ms AS BIGINT) AS time_ms,
               TRY_CAST(tab AS INTEGER) AS tab,
               TRY_CAST(play_time_ms AS BIGINT) AS play_time_ms,
               TRY_CAST(duration_ms AS BIGINT) AS duration_ms,
               {flags}
            FROM {_csv(cfg.raw_dir / name)}"""
        for i, (source, name) in enumerate(sources)
    ]
    con.execute(
        f"COPY ({' UNION ALL '.join(parts)}) TO '{out.as_posix()}' (FORMAT parquet)"
    )
    return [{"file_id": i, "file": n, "source": s} for i, (s, n) in enumerate(sources)]


def build_profile(cfg: Config, keep_staging: bool = False) -> Path:
    """Write the summary tables for one release to `profile_dir(cfg)`."""
    out = profile_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    staging = out / "_staging.parquet"
    t0 = time.perf_counter()

    def step(msg: str) -> None:
        print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)

    with _connect(cfg) as con:
        step(f"staging {cfg.dataset.name} logs")
        file_meta = _stage(cfg, con, staging)
        con.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet('{staging}')")
        con.register("file_meta", pd.DataFrame(file_meta))
        con.execute(
            "CREATE TABLE pool AS SELECT DISTINCT video_id FROM raw "
            "WHERE source = 'random' AND video_id IS NOT NULL"
        )
        # ts is time_ms in Beijing time; offset_days is 0 when it agrees with date
        con.execute("""
            CREATE VIEW s AS
            SELECT *, day - CAST(ts AS DATE) AS offset_days,
                   hour(ts) AS ts_hour,
                   hourmin // 100 AS hourmin_hour
            FROM (
                SELECT r.*, p.video_id IS NOT NULL AS in_pool,
                       strptime(CAST(r.date AS VARCHAR), '%Y%m%d')::DATE AS day,
                       epoch_ms(r.time_ms + 28800000) AS ts
                FROM raw r LEFT JOIN pool p ON r.video_id = p.video_id
            )
        """)

        def write(name: str, sql: str) -> None:
            step(f"writing {name}")
            con.execute(f"COPY ({sql}) TO '{(out / name).as_posix()}.parquet'")

        # contract checks per file. The date column is Beijing time, so it must
        # equal the date of time_ms shifted by +8h.
        nonbinary = ", ".join(
            f"sum(CAST(coalesce({c} NOT IN (0, 1), true) AS INT)) AS nonbinary_{c}"
            for c in BINARY_LOG_COLUMNS
        )
        write(
            "files",
            f"""
            SELECT m.file, m.source, count(*) AS rows,
                   min(date) AS min_date, max(date) AS max_date,
                   count(DISTINCT user_id) AS users,
                   min(user_id) AS min_user_id, max(user_id) AS max_user_id,
                   sum(CAST(user_id IS NULL OR video_id IS NULL OR date IS NULL
                            OR time_ms IS NULL AS INT)) AS null_or_unparseable_keys,
                   sum(CAST(is_rand <> CAST(m.source = 'random' AS TINYINT) AS INT))
                       AS is_rand_mismatch,
                   sum(CAST(date <> CAST(strftime(epoch_ms(time_ms + 28800000),
                            '%Y%m%d') AS INTEGER) AS INT)) AS date_time_mismatch,
                   {nonbinary}
            FROM raw JOIN file_meta m USING (file_id)
            GROUP BY m.file_id, m.file, m.source ORDER BY m.file_id
        """,
        )
        # per day: volumes, label sums, and two order-independent fingerprints of
        # the rows (sums of row hashes), one with IDs and one without.
        write(
            "daily",
            """
            SELECT day, source, in_pool, count(*) AS rows,
                   count(DISTINCT user_id) AS users, count(DISTINCT video_id) AS items,
                   sum(is_click) AS clicks, sum(long_view) AS long_views,
                   sum(is_like) AS likes,
                   CAST(sum(hash(time_ms, play_time_ms, duration_ms, is_click,
                                 long_view, is_like, tab)) AS VARCHAR) AS fp_rows,
                   CAST(sum(hash(user_id, video_id, time_ms)) AS VARCHAR) AS fp_ids
            FROM s GROUP BY ALL ORDER BY ALL
        """,
        )
        write(
            "tabs",
            """
            SELECT source, in_pool, tab, count(*) AS rows, sum(is_click) AS clicks,
                   sum(long_view) AS long_views
            FROM s GROUP BY ALL ORDER BY ALL
        """,
        )
        write(
            "user_activity",
            """
            SELECT user_id, file_id, source, in_pool, count(*) AS rows,
                   sum(is_click) AS clicks, sum(long_view) AS long_views,
                   sum(CAST(offset_days <> 0 AS INT)) AS date_mismatches
            FROM s GROUP BY ALL
        """,
        )
        # where date and time_ms disagree, and which of the two hourmin sides with
        write(
            "clock",
            """
            SELECT m.file, offset_days, ts_hour, hourmin_hour, count(*) AS rows
            FROM s JOIN file_meta m USING (file_id) GROUP BY ALL ORDER BY ALL
        """,
        )

        # exposure-weighted video attributes, pool vs not.
        items = _csv(cfg.raw_dir / cfg.dataset.files.items)
        con.execute(f"""
            CREATE TABLE items AS
            SELECT TRY_CAST(video_id AS BIGINT) AS video_id,
                   TRY_CAST(upload_dt AS DATE) AS upload_dt,
                   video_type, upload_type
            FROM {items}
        """)
        con.execute("""
            CREATE VIEW s_items AS
            SELECT s.*, i.upload_dt, s.day - i.upload_dt AS age_days, i.video_type,
                   i.upload_type
            FROM s LEFT JOIN items i ON s.video_id = i.video_id
        """)
        by = {
            "duration": _DURATION_SQL,
            "age_at_impression": _AGE_SQL,
            "video_type": "coalesce(video_type, 'unknown')",
            "upload_type": "coalesce(upload_type, 'unknown')",
        }
        write(
            "attributes",
            " UNION ALL ".join(f"""
            SELECT source, in_pool, '{name}' AS attribute, {expr} AS bucket,
                   count(*) AS rows, sum(is_click) AS clicks,
                   sum(long_view) AS long_views
            FROM s_items GROUP BY ALL""" for name, expr in by.items()),
        )

        # one row per pool video: upload date and when each log first shows it
        write(
            "pool",
            """
            SELECT video_id, any_value(upload_dt) AS upload_dt,
                   min(day) FILTER (source = 'standard') AS first_standard_day,
                   min(day) FILTER (source = 'random') AS first_random_day,
                   count(*) FILTER (source = 'standard') AS standard_rows,
                   count(*) FILTER (source = 'random') AS random_rows
            FROM s_items WHERE in_pool GROUP BY video_id ORDER BY video_id
        """,
        )

        # Gini and top-1% share over videos with at least one impression.
        write(
            "concentration",
            """
            WITH c AS (
                SELECT source, in_pool, video_id, count(*) AS n FROM s GROUP BY ALL
            ),
            r AS (
                SELECT *, row_number() OVER w AS i,
                       count(*) OVER (PARTITION BY source, in_pool) AS n_items,
                       sum(n) OVER (PARTITION BY source, in_pool) AS total,
                       row_number() OVER (PARTITION BY source, in_pool
                                          ORDER BY n DESC) AS rank_desc
                FROM c WINDOW w AS (PARTITION BY source, in_pool ORDER BY n)
            )
            SELECT source, in_pool, any_value(n_items) AS items_shown,
                   any_value(total) AS impressions,
                   sum((2 * i - n_items - 1) * n) / (any_value(n_items)
                       * any_value(total)) AS gini,
                   sum(CASE WHEN rank_desc <= greatest(1, n_items // 100)
                       THEN n ELSE 0 END) / any_value(total) AS top_1pct_share
            FROM r GROUP BY source, in_pool ORDER BY ALL
        """,
        )
        users = _csv(cfg.raw_dir / cfg.dataset.files.users)
        write(
            "users",
            f"""
            SELECT TRY_CAST(user_id AS BIGINT) AS user_id, user_active_degree,
                   TRY_CAST(register_days AS INTEGER) AS register_days,
                   TRY_CAST(follow_user_num AS INTEGER) AS follow_user_num,
                   TRY_CAST(fans_user_num AS INTEGER) AS fans_user_num,
                   TRY_CAST(is_video_author AS INTEGER) AS is_video_author
            FROM {users}
        """,
        )
        manifest = {
            "name": cfg.dataset.name,
            "variant": cfg.dataset.variant,
            "files": file_meta,
            "rows": con.execute("SELECT count(*) FROM raw").fetchone()[0],
            "pool_size": con.execute("SELECT count(*) FROM pool").fetchone()[0],
            "seconds": round(time.perf_counter() - t0, 1),
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if not keep_staging:
        staging.unlink()
    step(f"done: {manifest['rows']:,} rows, pool of {manifest['pool_size']:,}")
    return out


@dataclass
class Profile:
    """A release's summary tables, loaded from `profile_dir`."""

    name: str
    variant: str
    tables: dict[str, pd.DataFrame]

    def __getitem__(self, table: str) -> pd.DataFrame:
        return self.tables[table]

    @classmethod
    def load(cls, cfg: Config) -> Profile:
        d = profile_dir(cfg)
        manifest = json.loads((d / "manifest.json").read_text())
        tables = {t: pd.read_parquet(d / f"{t}.parquet") for t in TABLES}
        return cls(manifest["name"], manifest["variant"], tables)
