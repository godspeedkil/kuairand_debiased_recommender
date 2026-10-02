"""Point-in-time feature generation with DuckDB ASOF joins.

How it works (like Feast's `get_historical_features`):

1. Aggregate events per (entity key, day) into a daily table.
2. Turn that into running totals per entity: `cum(entity, day)` = totals over
   all days <= day. This table is sparse (only days with activity).
3. For an example (entity, D), ASOF-join the latest running total with
   day < D. That gives exact lifetime aggregates as of the start of day D.
4. A second ASOF join with day < D - w gives the totals as of D - w; the
   difference is the exact aggregate over the window [D - w, D - 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from krec.config import Config

# name -> SQL aggregate over events
AGGREGATES = {
    "impr": "count(*)",
    "click": "sum(is_click)",
    "lv": "sum(long_view)",
    "likes": "sum(is_like)",
    "play_ms": "sum(play_time_ms)",
    "dur_ms": "sum(duration_ms)",
}


@dataclass(frozen=True)
class FeatureView:
    """A group of features aggregated over the same entity key."""

    name: str  # also the column prefix, e.g. item_ctr_7d
    keys: tuple[str, ...]


VIEWS = (
    FeatureView("item", ("item_id",)),
    FeatureView("user", ("user_id",)),
    FeatureView("author", ("author_id",)),
    FeatureView("ua", ("user_id", "author_id")),  # user x author affinity
)
GLOBAL = FeatureView("global", ("_g",))  # constant key: totals over all events


class LeakageError(RuntimeError):
    """A feature would use information not available at prediction time."""


# Static columns copied as-is (prefixed item_ / user_).
ITEM_STATIC = [
    "video_type",
    "upload_type",
    "video_duration",
    "music_type",
    "server_width",
    "server_height",
]
USER_STATIC = [
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
    *[f"onehot_feat{i}" for i in range(18)],
]


def _connect(cfg: Config) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    # Cap DuckDB's memory and let it spill, so feature builds run on a laptop.
    con.execute(f"SET memory_limit = '{cfg.features.duckdb_memory_limit}'")
    spill = cfg.processed_dir / ".duckdb_tmp"
    con.execute(f"SET temp_directory = '{spill.as_posix()}'")
    pdir = cfg.processed_dir
    for t in ("interactions", "items", "users"):
        path = (pdir / f"{t}.parquet").as_posix()
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{path}')")
    return con


def _build_cumulative(con: duckdb.DuckDBPyConnection, sources: list[str]) -> None:
    src = ", ".join(f"'{s}'" for s in sources)
    # Standard logs only by default, so the random-exposure eval sets stay untouched.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE events AS
        SELECT i.* REPLACE (CAST(i.date AS DATE) AS date), it.author_id, 1 AS _g
        FROM interactions i LEFT JOIN items it USING (item_id)
        WHERE i.source IN ({src})
    """)
    for view in (*VIEWS, GLOBAL):
        keys = ", ".join(view.keys)
        aggs = ", ".join(
            f"CAST({sql} AS DOUBLE) AS {name}" for name, sql in AGGREGATES.items()
        )
        cums = ", ".join(f"sum({name}) OVER w AS {name}" for name in AGGREGATES)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE cum_{view.name} AS
            WITH daily AS (
                SELECT {keys}, date, {aggs}, 1.0 AS days
                FROM events
                WHERE {" AND ".join(f"{k} IS NOT NULL" for k in view.keys)}
                GROUP BY ALL
            )
            -- running totals: each row = everything up to and including `date`
            SELECT {keys}, date, {cums}, sum(days) OVER w AS days,
                   min(date) OVER w AS first_date
            FROM daily
            WINDOW w AS (PARTITION BY {keys} ORDER BY date
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        """)


def _asof(view: FeatureView, alias: str, cutoff_sql: str, left: str = "s") -> str:
    """Latest running total for the same entity with day strictly before the cutoff."""
    # The strict `>` is the whole leakage guarantee
    on = " AND ".join(f"{left}.{k} = {alias}.{k}" for k in view.keys)
    return (
        f"ASOF LEFT JOIN cum_{view.name} {alias} "
        f"ON {on} AND ({cutoff_sql}) > {alias}.date"
    )


def _view_features(view: FeatureView, windows: list[int], prior: float) -> list[str]:
    """SELECT expressions for one view's snapshot table.

    `a.<m>` = running total as of the start of the snapshot day `s._d`;
    `w<n>.<m>` = running total as of n days earlier; `g.*` = global rates.
    """
    p = view.name
    out = []

    def block(suffix: str, get) -> None:
        impr = get("impr")

        # Rates are smoothed toward the global rate: (hits + k * global) / (impr + k).
        def rate(hits: str, global_rate: str) -> str:
            return f"({hits} + {prior} * g.{global_rate}) / ({impr} + {prior})"

        play, dur = get("play_ms"), get("dur_ms")
        out.extend(
            [
                f"{impr} AS {p}_impr_{suffix}",
                f"{get('click')} AS {p}_click_{suffix}",
                f"{rate(get('click'), 'g_ctr')} AS {p}_ctr_{suffix}",
                f"{rate(get('lv'), 'g_lvr')} AS {p}_lvr_{suffix}",
                f"{rate(get('likes'), 'g_likr')} AS {p}_likr_{suffix}",
                f"CASE WHEN {dur} > 0 THEN {play} / {dur} END "
                f"AS {p}_play_ratio_{suffix}",
            ]
        )

    block("all", lambda m: f"coalesce(a.{m}, 0)")  # no history yet -> zeros
    for w in windows:
        # Window total = lifetime total now minus lifetime total w days ago.
        block(f"{w}d", lambda m, w=w: f"(coalesce(a.{m}, 0) - coalesce(w{w}.{m}, 0))")
    # NULL (not 0) when there is no history
    out.append(f"s._d - a.date AS {p}_days_since_last")
    out.append(f"s._d - a.first_date AS {p}_days_since_first")
    out.append(f"coalesce(a.days, 0) AS {p}_active_days_all")
    return out


def _build_snapshots(
    con: duckdb.DuckDBPyConnection, windows: list[int], prior: float
) -> None:
    """One row per distinct (entity keys, day) in the request, with its features.

    ASOF-joining distinct snapshots instead of every example keeps the joins
    small (items x days rather than impressions).
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE snap_global AS
        SELECT s._d,
               coalesce(a.click / nullif(a.impr, 0), 0) AS g_ctr,
               coalesce(a.lv    / nullif(a.impr, 0), 0) AS g_lvr,
               coalesce(a.likes / nullif(a.impr, 0), 0) AS g_likr
        FROM (SELECT DISTINCT _d, 1 AS _g FROM entity) s
        {_asof(GLOBAL, "a", "s._d")}
    """)
    for view in VIEWS:
        keys = ", ".join(view.keys)
        not_null = " AND ".join(f"{k} IS NOT NULL" for k in view.keys)
        joins = [_asof(view, "a", "s._d")]
        joins += [_asof(view, f"w{w}", f"s._d - {w}") for w in windows]
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE snap_{view.name} AS
            SELECT {", ".join(f"s.{k}" for k in view.keys)}, s._d,
                   {", ".join(_view_features(view, windows, prior))}
            FROM (SELECT DISTINCT {keys}, _d FROM entity WHERE {not_null}) s
            JOIN snap_global g ON g._d = s._d
            {" ".join(joins)}
        """)


def _prepare(
    cfg: Config, entity_df: pd.DataFrame, con: duckdb.DuckDBPyConnection
) -> str:
    """Build cumulative + snapshot tables for `entity_df`; return the final SELECT."""
    fcfg = cfg.features
    windows = fcfg.windows_days
    prior = fcfg.ctr_prior_strength
    _build_cumulative(con, fcfg.sources)

    ent = entity_df.copy()
    ent["_row"] = range(len(ent))  # restores input order at the end
    ent["date"] = pd.to_datetime(ent["date"]).dt.date
    con.register("entity_raw", ent)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE entity AS
        SELECT er.*, CAST(er.date AS DATE) AS _d, it.author_id
        FROM entity_raw er LEFT JOIN items it USING (item_id)
    """)
    con.unregister("entity_raw")
    _build_snapshots(con, windows, prior)

    leaky = set(fcfg.leaky_columns)
    static = [f"it.{c} AS item_{c}" for c in ITEM_STATIC if c not in leaky]
    static += [
        "e._d - it.upload_dt AS item_age_days",
        "TRY_CAST(split_part(it.tag, ',', 1) AS INTEGER) AS item_primary_tag",
    ]
    static += [f"us.{c} AS user_{c}" for c in USER_STATIC if c not in leaky]

    view_cols, view_joins = [], []
    for view in VIEWS:
        alias = f"v_{view.name}"
        view_cols.append(f"{alias}.* EXCLUDE ({', '.join(view.keys)}, _d)")
        on = " AND ".join(f"{alias}.{k} = e.{k}" for k in (*view.keys, "_d"))
        view_joins.append(f"LEFT JOIN snap_{view.name} {alias} ON {on}")

    input_cols = ", ".join(f'e."{c}"' for c in entity_df.columns)
    sql = f"""
        SELECT {input_cols}, e.author_id AS item_author_id,
               {", ".join(view_cols)},
               {", ".join(static)}
        FROM entity e
        {" ".join(view_joins)}
        LEFT JOIN items it ON it.item_id = e.item_id
        LEFT JOIN users us ON us.user_id = e.user_id
        ORDER BY e._row
    """
    # Check the output columns before running the query (DESCRIBE only plans it).
    cols = [r[0] for r in con.execute(f"DESCRIBE {sql}").fetchall()]
    bad = [c for c in cols if any(c == lc or c.endswith(f"_{lc}") for lc in leaky)]
    if bad:
        raise LeakageError(f"leaky columns reached the feature frame: {bad}")
    return sql


def get_historical_features(cfg: Config, entity_df: pd.DataFrame) -> pd.DataFrame:
    """Attach point-in-time features to `entity_df` (needs user_id, item_id, date).

    Output rows are in the same order as the input and all input columns are kept.
    """
    with _connect(cfg) as con:
        return con.execute(_prepare(cfg, entity_df, con)).df()


def write_historical_features(cfg: Config, entity_df: pd.DataFrame, path) -> int:
    """Same as `get_historical_features`, streamed straight to parquet, for memory optimization."""
    with _connect(cfg) as con:
        sql = _prepare(cfg, entity_df, con)
        con.execute(f"COPY ({sql}) TO '{Path(path).as_posix()}' (FORMAT parquet)")
        return con.execute("SELECT count(*) FROM entity").fetchone()[0]
