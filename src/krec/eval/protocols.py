"""
Evaluation protocols:
    - retrieval → For each eval user, rank the whole available catalog and compare the top K
        with the items the user engaged with in the eval window. Previous interactions are
        droppped.
    - ranking → For each user (or user-day), rank the items that were actually exposed in
        the eval window and compare with the observed labels. `*_standard` sets will include
        previous policy bias
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from krec.eval.metrics import (
    gini,
    grouped_ranking_metrics,
    intra_list_diversity,
    retrieval_metrics,
    top_k,
)
from krec.models.base import ItemCatalog, Recommender


def _positives_by_user(
    df: pd.DataFrame, label: str, catalog: ItemCatalog
) -> dict[int, np.ndarray]:
    """{user_id: sorted unique catalog positions of items with label == 1}."""
    pos = df.loc[df[label] == 1, ["user_id", "item_id"]]
    cols = catalog.index_of(pos.item_id.to_numpy())
    pairs = np.unique(
        np.stack([pos.user_id.to_numpy(), cols], axis=1)[cols >= 0], axis=0
    )
    if len(pairs) == 0:
        return {}
    users, starts = np.unique(pairs[:, 0], return_index=True)
    return dict(zip(users.tolist(), np.split(pairs[:, 1], starts[1:]), strict=True))


def evaluate_retrieval(
    model: Recommender,
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    catalog: ItemCatalog,
    labels: list[str],
    ks: list[int],
    users: str = "warm",
    exclude_fit_positives: bool = True,
    diversity_k: int = 10,
    batch_users: int = 2048,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    k_max = max(ks)
    # include items up to the end of eval windows
    window_end = pd.Timestamp(eval_df.date.max())
    available = catalog.available_mask(window_end)

    impr = np.zeros(catalog.n_items)
    idx = catalog.index_of(fit_df.item_id.to_numpy())
    np.add.at(impr, idx[idx >= 0], 1.0)
    p_item = (impr + 1.0) / (impr.sum() + catalog.n_items)
    self_info = -np.log2(p_item)

    warm = set(fit_df.user_id.unique())
    empty = np.empty(0, dtype=np.int64)
    results = {}
    for label in labels:
        targets = _positives_by_user(eval_df, label, catalog)
        seen = (
            _positives_by_user(fit_df, label, catalog)
            if exclude_fit_positives
            else None
        )
        user_ids, target_lists, excl_lists = [], [], []
        for user, target in targets.items():
            if users == "warm" and user not in warm:
                continue
            ex = seen.get(user, empty) if seen is not None else empty
            target = np.setdiff1d(target, ex)
            target = target[available[target]]
            if len(target) == 0:
                continue
            user_ids.append(user)
            target_lists.append(target)
            excl_lists.append(ex)
        if not user_ids:
            results[label] = {"n_users": 0}
            continue

        # TODO - need to vectorize
        per_user: dict[str, list[np.ndarray]] = {}
        rec_counts = {k: np.zeros(catalog.n_items) for k in ks}
        novelty = {k: [] for k in ks}
        ild = []
        user_arr = np.asarray(user_ids)
        for start in range(0, len(user_arr), batch_users):
            sl = slice(start, start + batch_users)
            scores = model.score_users(user_arr[sl]).astype(np.float64, copy=True)
            scores[:, ~available] = -np.inf
            tmask = np.zeros_like(scores, dtype=bool)
            for r, (target, ex) in enumerate(
                zip(target_lists[sl], excl_lists[sl], strict=True)
            ):
                tmask[r, target] = True
                scores[r, ex] = -np.inf  # already engaged with: not a recommendation
            top = top_k(scores, k_max, rng)
            hits = np.take_along_axis(tmask, top, axis=1)
            n_t = tmask.sum(axis=1)
            for m, v in retrieval_metrics(hits, n_t, ks).items():
                per_user.setdefault(m, []).append(v)
            for k in ks:
                np.add.at(rec_counts[k], top[:, :k].ravel(), 1.0)
                novelty[k].append(self_info[top[:, :k]].mean(axis=1))
            ild.append(intra_list_diversity(catalog.category[top[:, :diversity_k]]))

        res = {m: float(np.concatenate(v).mean()) for m, v in per_user.items()}
        n_avail = int(available.sum())
        for k in ks:
            res[f"coverage@{k}"] = float((rec_counts[k] > 0).sum() / n_avail)
            res[f"gini@{k}"] = gini(rec_counts[k][available])
            res[f"novelty@{k}"] = float(np.concatenate(novelty[k]).mean())
        res[f"ild@{diversity_k}"] = float(np.concatenate(ild).mean())
        res["n_users"] = len(user_ids)
        results[label] = res
    return results


def evaluate_ranking(
    model: Recommender,
    eval_df: pd.DataFrame,
    labels: list[str],
    ks: list[int],
    group_by: str = "user",
    gauc_weight: str = "impressions",
    seed: int = 0,
) -> dict:
    keys = ["user_id", "item_id"] + (["date"] if group_by == "user_date" else [])
    df = eval_df.groupby(keys, as_index=False)[labels].max()  # one row per exposure
    df["score"] = model.score_pairs(df.user_id.to_numpy(), df.item_id.to_numpy())
    df = df[np.isfinite(df.score)].copy()  # drop items the model can't score
    df["group"] = (
        df.user_id.astype(str) + "|" + df.date.astype(str)
        if group_by == "user_date"
        else df.user_id
    )
    out = {}
    for label in labels:
        rng = np.random.default_rng(seed)
        out[label] = grouped_ranking_metrics(
            df, "group", label, "score", ks, rng, gauc_weight
        )
        out[label]["n_impressions"] = int(len(df))
    return out
