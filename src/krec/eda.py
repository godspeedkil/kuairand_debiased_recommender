"""EDA queries over the processed data, used by notebooks/01_eda.ipynb.

Each function returns a DataFrame (or a figure), so the notebook stays a thin
layer of narrative and the numbers themselves are unit-tested.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from krec.config import Config
from krec.eval.metrics import gini

# Restricts a query to days when both logs exist, so the standard and random
# rows compare the same users over the same days.
_OVERLAP = "date >= (SELECT min(date) FROM x WHERE source = 'random')"


def _query(cfg: Config, sql: str) -> pd.DataFrame:
    path = (cfg.processed_dir / "interactions.parquet").as_posix()
    with duckdb.connect() as con:
        con.execute(f"CREATE VIEW x AS SELECT * FROM read_parquet('{path}')")
        return con.execute(sql).df()


def overview(cfg: Config) -> pd.DataFrame:
    """Rows, users, items and date range per (source, split)."""
    return _query(
        cfg,
        """
        SELECT source, split, count(*) AS rows, count(DISTINCT user_id) AS users,
               count(DISTINCT item_id) AS items, min(date)::VARCHAR AS first_day,
               max(date)::VARCHAR AS last_day
        FROM x GROUP BY ALL ORDER BY source DESC, min(date)
    """,
    )


def feedback_rates(cfg: Config) -> pd.DataFrame:
    """Click, long-view and like rates: policy-chosen vs randomly shown items."""
    return _query(
        cfg,
        f"""
        SELECT source, count(*) AS impressions,
               avg(is_click) AS click_rate, avg(long_view) AS long_view_rate,
               avg(is_like) AS like_rate,
               avg(least(play_time_ms / nullif(duration_ms, 0), 3)) AS play_ratio
        FROM x WHERE {_OVERLAP} GROUP BY source ORDER BY source DESC
    """,
    ).set_index("source")


def item_exposure(cfg: Config) -> dict[str, np.ndarray]:
    """Impressions per item and source, over every item logged by either source.

    Items a source never showed are included as zeros, which is what makes the
    concentration numbers comparable between sources.
    """
    df = _query(
        cfg,
        f"""
        SELECT source, item_id, count(*) AS n FROM x WHERE {_OVERLAP} GROUP BY ALL
    """,
    )
    wide = df.pivot(index="item_id", columns="source", values="n").fillna(0)
    return {source: wide[source].to_numpy() for source in wide.columns}


def exposure_concentration(exposure: dict[str, np.ndarray]) -> pd.DataFrame:
    """Gini and the share of impressions going to the top 1% of items."""
    rows = []
    for source, counts in exposure.items():
        top = np.sort(counts)[::-1][: max(1, len(counts) // 100)]
        rows.append(
            {
                "source": source,
                "items_shown": int((counts > 0).sum()),
                "top_1pct_share": float(top.sum() / counts.sum()),
                "gini": gini(counts),
            }
        )
    return pd.DataFrame(rows).set_index("source").sort_index(ascending=False)


def lorenz_figure(exposure: dict[str, np.ndarray]) -> Figure:
    """Lorenz curves of item exposure, one line per source."""
    # Figure API rather than pyplot: no global state, no display backend needed.
    fig = Figure(figsize=(6, 4.5))
    ax = fig.subplots()
    for source, counts in sorted(exposure.items(), reverse=True):
        curve = np.concatenate([[0], np.cumsum(np.sort(counts)) / counts.sum()])
        ax.plot(
            np.linspace(0, 1, len(curve)),
            curve,
            label=f"{source} (Gini {gini(counts):.2f})",
        )
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--", label="perfect equality")
    ax.set_xlabel("Share of items (least to most exposed)")
    ax.set_ylabel("Share of impressions")
    ax.set_title("Exposure concentration: standard policy vs random")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def exposure_vs_quality(
    cfg: Config, min_random_impressions: int = 20, buckets: int = 5
) -> pd.DataFrame:
    """Does the policy's exposure track how much users like an item?

    Items are bucketed by how often the standard policy showed them, then
    compared on their click rate when shown at random. Only items with enough
    random impressions to measure a rate are kept.
    """
    return _query(
        cfg,
        f"""
        WITH std AS (
            SELECT item_id, count(*) AS std_impr FROM x
            WHERE source = 'standard' AND {_OVERLAP} GROUP BY item_id
        ),
        rnd AS (
            SELECT item_id, count(*) AS rnd_impr, avg(is_click) AS rnd_ctr
            FROM x WHERE source = 'random'
            GROUP BY item_id HAVING count(*) >= {int(min_random_impressions)}
        ),
        j AS (
            SELECT r.*, coalesce(s.std_impr, 0) AS std_impr
            FROM rnd r LEFT JOIN std s USING (item_id)
        ),
        b AS (
            SELECT *, ntile({int(buckets)}) OVER (ORDER BY std_impr) AS bucket
            FROM j
        )
        SELECT bucket AS standard_exposure_bucket, count(*) AS items,
               median(std_impr) AS median_standard_impr,
               avg(rnd_ctr) AS random_click_rate
        FROM b GROUP BY bucket ORDER BY bucket
    """,
    ).set_index("standard_exposure_bucket")


def sparsity(cfg: Config) -> pd.DataFrame:
    """Share of (user, item) pairs observed, and impressions per user."""
    return _query(
        cfg,
        """
        WITH u AS (SELECT source, user_id, count(*) AS n FROM x GROUP BY ALL),
        per_user AS (
            SELECT source, median(n) AS median_impr_per_user,
                   quantile_cont(n, 0.9) AS p90_impr_per_user
            FROM u GROUP BY source
        ),
        density AS (
            SELECT source,
                   count(DISTINCT (user_id, item_id))
                     / (count(DISTINCT user_id) * count(DISTINCT item_id)) AS density
            FROM x GROUP BY source
        )
        SELECT * FROM density JOIN per_user USING (source) ORDER BY source DESC
    """,
    ).set_index("source")


def daily_volume(cfg: Config) -> pd.DataFrame:
    """Impressions per day and source (one column per source, 0 where absent)."""
    df = _query(cfg, "SELECT date, source, count(*) AS n FROM x GROUP BY ALL")
    wide = df.pivot(index="date", columns="source", values="n").fillna(0)
    return wide.astype(int).sort_index()


def daily_volume_figure(volume: pd.DataFrame) -> Figure:
    """Line chart of `daily_volume`, one line per source."""
    fig = Figure(figsize=(7, 3.5))
    ax = fig.subplots()
    for source in sorted(volume.columns, reverse=True):
        ax.plot(volume.index, volume[source], marker=".", label=source)
    ax.set_ylabel("Impressions per day")
    ax.set_title("Daily log volume by source")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def tab_mix(cfg: Config) -> pd.DataFrame:
    """Share of impressions and click rate per app tab, by source.

    `is_click` means a tap in KuaiRand's two-column UI but a "valid play" in
    the single-column UI, so a different tab mix between sources would make
    their click rates not directly comparable. Same overlap window as
    `feedback_rates`.
    """
    df = _query(
        cfg,
        f"""
        SELECT tab, source, count(*) AS n, avg(is_click) AS click_rate
        FROM x WHERE {_OVERLAP} GROUP BY ALL
    """,
    )
    df["share"] = df.n / df.groupby("source").n.transform("sum")
    wide = df.pivot(index="tab", columns="source", values=["share", "click_rate"])
    wide.columns = [f"{source}_{metric}" for metric, source in wide.columns]
    order = [
        f"{s}_{m}" for m in ("share", "click_rate") for s in ("standard", "random")
    ]
    return wide[[c for c in order if c in wide.columns]].sort_index()


def within_user_rates(
    cfg: Config, labels: tuple[str, ...] = ("is_click", "long_view"), min_each: int = 5
) -> pd.DataFrame:
    """Standard vs random feedback rates compared within the same users.

    Only users with at least `min_each` impressions from each source in the
    overlap window count, and every user counts once. This removes the user-mix
    effect from `feedback_rates`, where heavy users dominate.
    """
    rates = ", ".join(f"avg({label}) AS {label}" for label in labels)
    per_user = _query(
        cfg,
        f"""
        SELECT user_id, source, count(*) AS n, {rates}
        FROM x WHERE {_OVERLAP} GROUP BY ALL
    """,
    )
    wide = per_user.pivot(index="user_id", columns="source")
    both = wide[(wide["n"] >= min_each).all(axis=1)]
    rows = []
    for label in labels:
        std, rnd = both[label]["standard"], both[label]["random"]
        rows.append(
            {
                "label": label,
                "users": len(both),
                "standard_rate": std.mean(),
                "random_rate": rnd.mean(),
                "mean_difference": (std - rnd).mean(),
                "share_users_higher_on_standard": (std > rnd).mean(),
            }
        )
    return pd.DataFrame(rows).set_index("label")
