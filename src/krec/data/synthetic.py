"""Synthetic data with the exact KuaiRand-Pure file layout and schema. For
testing, CI, and running end-to-end pipeline checks.

The generator does the following:
* Adds exposure bias, by including the hype score (part of the standard policy).
  Weakly related to actual user popularity.
* Injects author affinity, by defining user latent vectors.
* Validates cold start items, since ~15% are uploaded during the log window,
  and thus can't be shown.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from krec.config import Config

START, END, RANDOM_START = date(2022, 4, 8), date(2022, 5, 8), date(2022, 4, 22)
_CST = timezone(timedelta(hours=8))  # Kuaishou logs are in Beijing time


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _labels(rng, logit, dur_ms):
    """Sample click, play time and the rule-based long_view for impressions."""
    n = len(logit)
    click = (rng.random(n) < _sigmoid(logit)).astype(np.int64)

    # Play times are chosen so is_click matches KuaiRand's "valid play" rule
    # (>= duration if <= 7s, else > 7s): clicks always qualify, skips never do.
    watched = np.maximum(dur_ms * rng.uniform(0.8, 2.0, n), np.minimum(dur_ms, 7001))
    skipped = rng.uniform(0.0, 0.9, n) * np.minimum(dur_ms, 7000)
    play = np.where(click == 1, watched, skipped).astype(np.int64)

    # Rule from the KuaiRand README.
    long_view = np.where(dur_ms <= 18000, play >= dur_ms, play >= 18000).astype(
        np.int64
    )
    like = click * (rng.random(n) < _sigmoid(logit - 3.0))  # rarer than a click

    # The remaining signals are constant-rate noise that only fills the schema.
    follow = click * (rng.random(n) < 0.01)
    comment = click * (rng.random(n) < 0.02)
    forward = click * (rng.random(n) < 0.005)
    hate = (1 - click) * (rng.random(n) < 0.002)
    profile = click * (rng.random(n) < 0.03)

    return dict(
        is_click=click,
        is_like=like.astype(np.int64),
        is_follow=follow.astype(np.int64),
        is_comment=comment.astype(np.int64),
        is_forward=forward.astype(np.int64),
        is_hate=hate.astype(np.int64),
        long_view=long_view,
        play_time_ms=play,
        duration_ms=dur_ms.astype(np.int64),
        profile_stay_time=(profile * rng.integers(1000, 30000, n)).astype(np.int64),
        comment_stay_time=(comment * rng.integers(1000, 60000, n)).astype(np.int64),
        is_profile_enter=profile.astype(np.int64),
    )


def generate(cfg: Config) -> Path:
    p = cfg.synthetic
    rng = np.random.default_rng(p.seed)
    n_users, n_items, n_authors, latent_dim = (
        p.n_users,
        p.n_items,
        p.n_authors,
        p.latent_dim,
    )

    # ---- latent world ----
    # 1/sqrt(d) keeps vector norms ~1.6 for any d; affinity std ends up ~0.9 at d=8.
    user_vec = rng.normal(0, 1 / np.sqrt(latent_dim), (n_users, latent_dim)) * 1.6
    activity = rng.lognormal(0, 0.5, n_users)
    activity /= activity.mean()
    author_vec = rng.normal(0, 1 / np.sqrt(latent_dim), (n_authors, latent_dim)) * 1.6
    item_author = rng.integers(0, n_authors, n_items)

    # Item = its author's vector + small noise, so same-author items look alike.
    item_vec = author_vec[item_author] + rng.normal(
        0, 0.5 / np.sqrt(latent_dim), (n_items, latent_dim)
    )
    quality = rng.normal(0, 0.7, n_items)
    hype = 0.4 * quality + rng.normal(0, 1.0, n_items)
    dur_ms = np.clip(rng.lognormal(np.log(20000), 0.8, n_items), 3000, 300000).astype(
        np.int64
    )

    n_new = int(0.15 * n_items)
    upload = np.array(
        [
            date(2021, 6, 1) + timedelta(days=int(x))
            for x in rng.integers(0, 300, n_items)
        ]
    )
    new_idx = rng.choice(n_items, n_new, replace=False)
    upload[new_idx] = [
        START + timedelta(days=int(x)) for x in rng.integers(1, 28, n_new)
    ]

    affinity = user_vec @ item_vec.T

    # Most items predate the logs; 15% are uploaded mid-window (cold start).
    click_logit = 1.2 * affinity + quality[None, :] - 0.8
    policy_logit = 1.5 * hype[None, :] + 1.0 * affinity

    # random exposure only draws from a candidate pool of items, like KuaiRand's
    # 7,583 pool videos.
    in_pool = np.ones(n_items, dtype=bool)
    if p.pool_fraction < 1:
        pool_rng = np.random.default_rng(p.seed + 1)
        n_pool = max(1, round(p.pool_fraction * n_items))
        in_pool[:] = False
        in_pool[pool_rng.choice(n_items, n_pool, replace=False)] = True

    # ---- logs, day by day ----
    rows = []
    day = START
    while day <= END:
        eligible = np.array([u <= day for u in upload])
        midnight_ms = int(
            datetime(day.year, day.month, day.day, tzinfo=_CST).timestamp() * 1000
        )
        for source, mean in (
            ("standard", p.mean_standard_per_user_day),
            ("random", p.mean_random_per_user_day),
        ):
            if source == "random" and day < RANDOM_START:
                continue
            allowed = eligible if source == "standard" else eligible & in_pool
            counts = np.minimum(rng.poisson(mean * activity), allowed.sum())

            # Gumbel-top-k: add Gumbel noise to the logits and take each user's top
            # `counts` items. Equivalent to sampling without replacement with
            # p ~ exp(logit), for all users in one sort. Zero logits = uniform.
            logits = (
                policy_logit if source == "standard" else np.zeros((n_users, n_items))
            )
            gumbel = logits + rng.gumbel(size=(n_users, n_items))
            gumbel[:, ~allowed] = -np.inf
            order = np.argsort(-gumbel, axis=1)

            # One row per impression: user u with counts=3 -> its ranks 0, 1, 2.
            u_idx = np.repeat(np.arange(n_users), counts)
            rank = (
                np.concatenate([np.arange(c) for c in counts])
                if counts.sum()
                else np.array([])
            )
            i_idx = order[u_idx, rank.astype(int)]
            n = len(u_idx)
            if n == 0:
                continue
            sec = rng.integers(0, 86400, n)

            # Feedback depends only on (user, item), not on which policy showed it.
            labels = _labels(rng, click_logit[u_idx, i_idx], dur_ms[i_idx])
            rows.append(
                pd.DataFrame(
                    {
                        "user_id": u_idx,
                        "video_id": i_idx,
                        "date": int(day.strftime("%Y%m%d")),
                        "hourmin": (sec // 3600) * 100 + (sec % 3600) // 60,
                        "time_ms": midnight_ms + sec * 1000 + rng.integers(0, 1000, n),
                        **labels,
                        "is_rand": int(source == "random"),
                        "tab": rng.choice(15, n, p=_tab_probs()),
                        "_source": source,
                    }
                )
            )
        day += timedelta(days=1)
    logs = pd.concat(rows, ignore_index=True).sort_values("time_ms", kind="stable")

    # ---- write files in the KuaiRand-Pure layout ----
    out = cfg.raw_dir
    out.mkdir(parents=True, exist_ok=True)
    files = cfg.dataset.files
    std, rnd = logs[logs._source == "standard"], logs[logs._source == "random"]
    first_half = std.date < 20220422
    cols = [c for c in logs.columns if c != "_source"]
    std[first_half][cols].to_csv(out / files.standard_logs[0], index=False)
    std[~first_half][cols].to_csv(out / files.standard_logs[1], index=False)
    rnd[cols].to_csv(out / files.random_logs[0], index=False)

    _users(rng, n_users).to_csv(out / files.users, index=False)
    items = pd.DataFrame(
        {
            "video_id": np.arange(n_items),
            "author_id": item_author,
            "video_type": np.where(rng.random(n_items) < 0.97, "NORMAL", "AD"),
            "upload_dt": [u.isoformat() for u in upload],
            "upload_type": rng.choice(
                ["ShortImport", "LongImport", "Web", "Kmovie"], n_items
            ),
            "visible_status": 1,  # leaky in the real data; present so ingest has to drop it
            "video_duration": dur_ms.astype(float),
            "server_width": 720,
            "server_height": 1280,
            "music_id": rng.integers(0, 10**9, n_items),
            "music_type": rng.integers(0, 10, n_items),
            "tag": [
                ",".join(map(str, sorted(set(rng.integers(0, 40, rng.integers(1, 3))))))
                for _ in range(n_items)
            ],
        }
    )
    items.to_csv(out / files.items, index=False)

    # Monthly aggregate (leaky). Written here for completeness, but never read.
    stats = std.groupby("video_id").agg(
        counts=("date", "nunique"),
        show_cnt=("is_click", "size"),
        play_cnt=("is_click", "sum"),
        like_cnt=("is_like", "sum"),
    )
    stats.reset_index().to_csv(out / "video_features_statistic_pure.csv", index=False)
    return out


def _tab_probs() -> np.ndarray:
    w = np.array([30, 25, 10, 8, 6, 5, 4, 3, 2, 2, 1, 1, 1, 1, 1], dtype=float)
    return w / w.sum()


def _users(rng, nu: int) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "user_id": np.arange(nu),
            "user_active_degree": rng.choice(
                ["high_active", "full_active", "middle_active", "UNKNOWN"],
                nu,
                p=[0.3, 0.4, 0.25, 0.05],
            ),
            "is_lowactive_period": rng.integers(0, 2, nu),
            "is_live_streamer": (rng.random(nu) < 0.05).astype(int),
            "is_video_author": (rng.random(nu) < 0.3).astype(int),
            "follow_user_num": rng.integers(0, 600, nu),
            "follow_user_num_range": "(0,10]",
            "fans_user_num": rng.integers(0, 2000, nu),
            "fans_user_num_range": "[1,10)",
            "friend_user_num": rng.integers(0, 300, nu),
            "friend_user_num_range": "[1,5)",
            "register_days": rng.integers(15, 3000, nu),
            "register_days_range": "730+",
        }
    )
    sizes = [2, 7, 50, 1471, 15, 34, 3, 118, 454, 7, 5, 5, 2, 2, 2, 2, 2, 2]
    for i, k in enumerate(sizes):
        df[f"onehot_feat{i}"] = rng.integers(0, k, nu)
    return df


def derive_variants(full: Config, pure: Config, k1: Config, n_users_1k: int) -> None:
    """Write Pure- and 1K-style datasets from a full (27K-style) synthetic one."""
    src, files = full.raw_dir, full.dataset.files
    std = pd.concat(pd.read_csv(src / f) for f in files.standard_logs)
    rnd = pd.concat(pd.read_csv(src / f) for f in files.random_logs)
    users = pd.read_csv(src / files.users)
    items = pd.read_csv(src / files.items)
    pool = set(rnd.video_id)

    rng = np.random.default_rng(full.synthetic.seed + 2)
    sample = set(rng.choice(users.user_id, n_users_1k, replace=False))
    in_1k = lambda df: df[df.user_id.isin(sample)]  # noqa: E731
    seen_1k = set(in_1k(std).video_id) | set(in_1k(rnd).video_id)

    for cfg, std_v, rnd_v, users_v, items_v in (
        (
            pure,
            std[std.video_id.isin(pool)],
            rnd,
            users,
            items[items.video_id.isin(pool)],
        ),
        (k1, in_1k(std), in_1k(rnd), in_1k(users), items[items.video_id.isin(seen_1k)]),
    ):
        out, f = cfg.raw_dir, cfg.dataset.files
        out.mkdir(parents=True, exist_ok=True)
        first_half = std_v.date < 20220422  # same file split as the real releases
        std_v[first_half].to_csv(out / f.standard_logs[0], index=False)
        std_v[~first_half].to_csv(out / f.standard_logs[-1], index=False)
        rnd_v.to_csv(out / f.random_logs[0], index=False)
        users_v.to_csv(out / f.users, index=False)
        items_v.to_csv(out / f.items, index=False)
