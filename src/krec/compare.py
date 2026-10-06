"""Cross-version comparisons over `krec.profile` summary tables.

Used by notebooks/02_versions.ipynb. Every function takes `profiles`, a dict
of variant ("pure", "1k", "27k") -> Profile, and returns a DataFrame or a
figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from krec.profile import BUCKET_ORDER, Profile

Profiles = dict[str, Profile]


def _need(profiles: Profiles, *variants: str) -> None:
    missing = [v for v in variants if v not in profiles]
    if missing:
        raise KeyError(f"needs profiles for {missing}: run `krec profile` on them")


def _fp_total(series: pd.Series) -> str:
    """Combine per-day fingerprints (sums of row hashes) into one.

    Returned as text: the sums exceed 64 bits, which pandas can't store as ints.
    """
    return str(sum(int(x) for x in series))


def contracts(profiles: Profiles) -> pd.DataFrame:
    """Per-file contract checks; only checks that found a problem get a column."""
    df = pd.concat(p["files"].assign(version=v) for v, p in profiles.items()).set_index(
        ["version", "file"]
    )
    info = ["source", "rows", "min_date", "max_date"]
    checks = [
        c
        for c in df.columns
        if c.endswith(("_keys", "_mismatch")) or c.startswith("nonbinary_")
    ]
    failing = [c for c in checks if df[c].fillna(0).sum() > 0]
    return df[[*info, *failing]]


def random_log_identity(profiles: Profiles) -> pd.DataFrame:
    """Is each release's random log the same set of rows as 27K's?"""
    _need(profiles, "27k")
    ref = profiles["27k"]["daily"].query("source == 'random'")
    rows = []
    for v, p in profiles.items():
        d = p["daily"].query("source == 'random'")
        rows.append(
            {
                "version": v,
                "random_rows": int(d.rows.sum()),
                "same_rows_as_27k": _fp_total(d.fp_rows) == _fp_total(ref.fp_rows),
                "same_ids_as_27k": _fp_total(d.fp_ids) == _fp_total(ref.fp_ids),
            }
        )
    return pd.DataFrame(rows).set_index("version")


def pure_vs_full_pool(profiles: Profiles) -> pd.DataFrame:
    """Pure's rows per day and source vs 27K restricted to pool videos."""
    _need(profiles, "pure", "27k")
    keys = ["day", "source"]
    pure = (
        profiles["pure"]["daily"]
        .groupby(keys)
        .agg(pure_rows=("rows", "sum"), pure_fp=("fp_rows", _fp_total))
    )
    full = (
        profiles["27k"]["daily"]
        .query("in_pool")
        .groupby(keys)
        .agg(full_pool_rows=("rows", "sum"), full_fp=("fp_rows", _fp_total))
    )
    df = pure.join(full, how="outer").fillna({"pure_rows": 0, "full_pool_rows": 0})
    df["row_diff"] = df.pure_rows - df.full_pool_rows
    df["same_rows"] = df.pure_fp == df.full_fp
    return df.drop(columns=["pure_fp", "full_fp"])


def one_k_vs_full(profiles: Profiles) -> pd.DataFrame:
    """For 1K's users: do their per-source row counts equal 27K's?"""
    _need(profiles, "1k", "27k")
    # not split by pool: 1K's pool comes from its own random log, which covers
    # only part of 27K's pool (see pool_coverage)
    keys = ["user_id", "source"]
    k1 = profiles["1k"]["user_activity"].groupby(keys).rows.sum()
    users = k1.index.get_level_values("user_id").unique()
    full = profiles["27k"]["user_activity"]
    full = full[full.user_id.isin(users)].groupby(keys).rows.sum()
    joined = pd.concat({"1k": k1, "27k": full}, axis=1).fillna(0)
    per_user = (joined["1k"] == joined["27k"]).groupby("user_id").all()
    return pd.DataFrame(
        {
            "users_in_1k": [len(users)],
            "found_in_27k": [full.index.get_level_values("user_id").nunique()],
            "users_with_identical_counts": [int(per_user.sum())],
        }
    )


def pool_coverage(profiles: Profiles) -> pd.DataFrame:
    """Each release's pool (videos in its own random log) against 27K's."""
    _need(profiles, "27k")
    ref = set(profiles["27k"]["pool"].video_id)
    rows = []
    for v, p in profiles.items():
        pool = set(p["pool"].video_id)
        rows.append(
            {
                "version": v,
                "pool_videos": len(pool),
                "in_27k_pool": len(pool & ref),
                "not_in_27k_pool": len(pool - ref),
                "27k_pool_missing": len(ref - pool),
            }
        )
    return pd.DataFrame(rows).set_index("version")


def pool_upload_dates(profiles: Profiles, variant: str = "27k") -> pd.DataFrame:
    """Pool videos by upload date, with when the policy first showed them."""
    _need(profiles, variant)
    p = profiles[variant]["pool"].copy()
    for col in ("upload_dt", "first_standard_day", "first_random_day"):
        p[col] = pd.to_datetime(p[col])
    p["days_to_first_standard"] = (p.first_standard_day - p.upload_dt).dt.days
    return (
        p.groupby(p.upload_dt.dt.date, dropna=False)
        .agg(
            videos=("video_id", "size"),
            never_shown_by_policy=("first_standard_day", lambda d: d.isna().sum()),
            median_days_to_first_standard=("days_to_first_standard", "median"),
            standard_rows=("standard_rows", "sum"),
        )
        .rename_axis("upload_date")
    )


def daily_volume_by_scope(profiles: Profiles) -> pd.DataFrame:
    """Rows per day for each version, source and scope (all / pool / non-pool)."""
    frames = []
    for v, p in profiles.items():
        d = p["daily"]
        for scope, part in (
            ("all", d),
            ("pool", d[d.in_pool]),
            ("non-pool", d[~d.in_pool]),
        ):
            g = part.groupby(["day", "source"]).rows.sum().reset_index()
            frames.append(g.assign(version=v, scope=scope))
    return pd.concat(frames, ignore_index=True)


def daily_volume_table(volume: pd.DataFrame, source: str = "standard") -> pd.DataFrame:
    """`daily_volume_by_scope` as a table: one row per day, one column per line."""
    part = volume[volume.source == source]
    return part.pivot_table(
        index="day", columns=["version", "scope"], values="rows", aggfunc="sum"
    )


def daily_volume_figure(volume: pd.DataFrame, source: str = "standard") -> Figure:
    """Daily rows by version and scope."""
    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    for (v, scope), g in volume[volume.source == source].groupby(["version", "scope"]):
        if g.rows.sum() == 0:
            continue
        ax.plot(g.day, g.rows, marker=".", label=f"{v} · {scope}")
    ax.set_yscale("log")
    ax.set_ylabel("Rows per day (log scale)")
    ax.set_title(f"Daily {source}-log volume by version and scope")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def pool_share_per_user(profiles: Profiles, variant: str = "27k") -> pd.DataFrame:
    """Share of each user's standard impressions that land on pool videos."""
    _need(profiles, variant)
    a = profiles[variant]["user_activity"].query("source == 'standard'")
    wide = a.pivot_table(
        index="user_id", columns="in_pool", values="rows", aggfunc="sum", fill_value=0
    ).reindex(columns=[False, True], fill_value=0)
    share = wide[True] / wide.sum(axis=1)
    q = share.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return pd.DataFrame(
        {
            "users": [len(share)],
            "overall_share": [wide[True].sum() / wide.to_numpy().sum()],
            "mean_share": [share.mean()],
            **{f"p{int(k * 100)}": [v] for k, v in q.items()},
            "users_with_no_pool_rows": [int((wide[True] == 0).sum())],
        },
        index=[variant],
    )


def label_rates(profiles: Profiles) -> pd.DataFrame:
    """Click and long-view rates by version, source and pool membership."""
    frames = []
    for v, p in profiles.items():
        g = (
            p["daily"]
            .groupby(["source", "in_pool"])[["rows", "clicks", "long_views"]]
            .sum()
        )
        frames.append(g.assign(version=v))
    df = pd.concat(frames).reset_index().set_index(["version", "source", "in_pool"])
    df["click_rate"] = df.clicks / df.rows
    df["long_view_rate"] = df.long_views / df.rows
    return df[["rows", "click_rate", "long_view_rate"]]


def tab_mix(profiles: Profiles, variant: str = "27k") -> pd.DataFrame:
    """Share of impressions and click rate per tab, standard pool vs non-pool."""
    _need(profiles, variant)
    t = profiles[variant]["tabs"].copy()
    t["group"] = t.source + np.where(t.in_pool, " · pool", " · non-pool")
    t["share"] = t.rows / t.groupby("group").rows.transform("sum")
    t["click_rate"] = t.clicks / t.rows
    return t.pivot(index="tab", columns="group", values=["share", "click_rate"])


def pool_typicality(
    profiles: Profiles, attribute: str, variant: str = "27k"
) -> pd.DataFrame:
    """Standard impressions on pool vs non-pool videos, bucketed by an attribute."""
    _need(profiles, variant)
    a = profiles[variant]["attributes"]
    a = a[(a.attribute == attribute) & (a.source == "standard")].copy()
    a["group"] = np.where(a.in_pool, "pool", "non_pool")
    a["share"] = a.rows / a.groupby("group").rows.transform("sum")
    a["click_rate"] = a.clicks / a.rows
    a["long_view_rate"] = a.long_views / a.rows
    wide = a.pivot(
        index="bucket",
        columns="group",
        values=["share", "click_rate", "long_view_rate"],
    )
    wide.columns = [f"{group}_{metric}" for metric, group in wide.columns]
    order = [b for b in BUCKET_ORDER.get(attribute, []) if b in wide.index]
    return wide.loc[order] if order else wide.sort_index()


def concentration(profiles: Profiles) -> pd.DataFrame:
    """Exposure concentration per version."""
    return pd.concat(
        p["concentration"].assign(version=v) for v, p in profiles.items()
    ).set_index(["version", "source", "in_pool"])


def user_sample(profiles: Profiles) -> pd.DataFrame:
    """Are 1K's users a fair sample of 27K's?"""
    _need(profiles, "1k", "27k")
    rows = {}
    for v in ("27k", "1k"):
        act = (
            profiles[v]["user_activity"]
            .query("source == 'standard'")
            .groupby("user_id")
            .rows.sum()
        )
        users = profiles[v]["users"]
        degree = users.user_active_degree.value_counts(normalize=True)
        rows[v] = {
            "users": len(users),
            "median_standard_rows": act.median(),
            "p90_standard_rows": act.quantile(0.9),
            "median_register_days": users.register_days.median(),
            "share_video_authors": users.is_video_author.mean(),
            **{f"active_{k}": degree.get(k, 0.0) for k in sorted(degree.index)},
        }
    return pd.DataFrame(rows)


def clock_summary(profiles: Profiles) -> pd.DataFrame:
    """Per file: how often `date` disagrees with the date of `time_ms` (UTC+8)."""
    c = pd.concat(p["clock"].assign(version=v) for v, p in profiles.items())
    off = c[c.offset_days != 0]
    keys = ["version", "file"]
    total = c.groupby(keys, sort=False).rows.sum()
    bad = off.groupby(keys, sort=False).rows.sum()
    # offset = date minus the time_ms date, as a share of mismatched rows
    by_offset = off.assign(
        offset=np.select(
            [off.offset_days == -1, off.offset_days == 1],
            ["offset_-1", "offset_+1"],
            "offset_other",
        )
    ).pivot_table(index=keys, columns="offset", values="rows", aggfunc="sum")
    # hourmin is compared by hour: it may be rounded to the hour
    same_hour = off[off.hourmin_hour == off.ts_hour].groupby(keys).rows.sum()
    past_midnight = off[off.hourmin_hour < off.ts_hour].groupby(keys).rows.sum()
    out = pd.DataFrame({"rows": total, "mismatched": bad}).fillna({"mismatched": 0})
    out["mismatch_share"] = out.mismatched / out.rows
    # baseline over all rows: how often hourmin's hour equals time_ms's at all
    agree = c[c.hourmin_hour == c.ts_hour].groupby(keys).rows.sum()
    out["hourmin_hour_matches_overall"] = agree.reindex(out.index).fillna(0) / out.rows
    out = out.join(by_offset.div(out.mismatched, axis=0))
    # among mismatched rows: hourmin in time_ms's hour (date is the odd one out)
    # or past midnight (hourmin sides with date, so time_ms is the odd one out)
    for name, part in (
        ("hourmin_hour_matches_time_ms", same_hour),
        ("hourmin_past_midnight", past_midnight),
    ):
        out[name] = part.reindex(out.index).fillna(0) / out.mismatched
    return out


def clock_by_hour(profiles: Profiles, variant: str = "27k") -> pd.DataFrame:
    """Mismatch share by hour of time_ms (Beijing), one column per file."""
    _need(profiles, variant)
    c = profiles[variant]["clock"]
    c = c.assign(mismatched=np.where(c.offset_days != 0, c.rows, 0))
    g = c.groupby(["ts_hour", "file"], sort=False)[["rows", "mismatched"]].sum()
    return (g.mismatched / g.rows).unstack("file").sort_index()


def part_files(profiles: Profiles, variant: str = "27k") -> pd.DataFrame:
    """How a release's log files split users, and where clock mismatches sit."""
    _need(profiles, variant)
    p = profiles[variant]
    files = p["files"].reset_index(drop=True)
    files["file_id"] = files.index  # files are written in file_id order
    act = (
        p["user_activity"]
        .groupby(["file_id", "user_id"])
        .agg(rows=("rows", "sum"), mismatches=("date_mismatches", "sum"))
    )
    rows = []
    for f in files.itertuples():
        # siblings cover the same source and dates, like 27K's part1/part2
        sib = files[
            (files.source == f.source)
            & (files.min_date == f.min_date)
            & (files.file_id != f.file_id)
        ].file_id
        mine = act.loc[f.file_id]
        others = act[act.index.get_level_values("file_id").isin(sib)]
        other_users = set(others.index.get_level_values("user_id"))
        with_bad = mine[mine.mismatches > 0]
        rows.append(
            {
                "file": f.file,
                "users": f.users,
                "min_user_id": f.min_user_id,
                "max_user_id": f.max_user_id,
                # near 0 means the parts were split by user
                "users_also_in_sibling": (
                    mine.index.isin(other_users).sum() if len(sib) else np.nan
                ),
                "users_with_mismatches": len(with_bad),
                "mismatch_rate_among_those_users": (
                    with_bad.mismatches.sum() / with_bad.rows.sum()
                    if len(with_bad)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index("file")
