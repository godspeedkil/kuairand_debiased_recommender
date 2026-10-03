"""
Relevant metrics
- per user:
    - top k
    - recall@k
    - hit@k
    - nDCG@k
    - intra-list diversity
- grouped (also user by default, could be user-day):
    - GAUC
    - nDCG@k
- others:
    - AUC
    - Gini coefficient
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def top_k(scores: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of the k highest scores per row, best first. `-inf` = excluded."""
    scores = np.asarray(scores, dtype=np.float64)
    k = min(k, scores.shape[1])
    # Shuffle the columns first. argpartition picks positions from the values
    # alone, so after a random shuffle every tied item is equally likely to land
    # in a chosen position.
    perm = rng.permutation(scores.shape[1])
    s = scores[:, perm]
    part = np.argpartition(-s, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(s, part, axis=1)
    order = np.lexsort((part, -vals), axis=1)  # by score, then shuffled position
    return perm[np.take_along_axis(part, order, axis=1)]


def _discounts(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, k + 2))


def retrieval_metrics(hits: np.ndarray, n_targets: np.ndarray, ks: list[int]) -> dict:
    """Per-user retrieval metrics from a hit matrix.

    hits: (n_users, K) bool, hits[u, r] = the item at rank r is a target.
    n_targets: (n_users,) number of reachable targets per user (> 0).
    Returns {metric@k: per-user array}.
    """
    hits = hits.astype(np.float64)
    disc = _discounts(hits.shape[1])
    ideal_cum = np.cumsum(disc)
    out = {}
    for k in ks:
        h = hits[:, :k]
        n_hit = h.sum(axis=1)
        out[f"recall@{k}"] = n_hit / n_targets
        out[f"hit@{k}"] = (n_hit > 0).astype(np.float64)
        dcg = (h * disc[: h.shape[1]]).sum(axis=1)
        # Best possible DCG: all of the user's targets at the top, at most k of them.
        ideal_len = np.minimum(n_targets, h.shape[1]).astype(int)
        out[f"ndcg@{k}"] = dcg / ideal_cum[ideal_len - 1]
    return out


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of non-negative counts (0 = equal, near 1 = concentrated)."""
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n, total = x.size, x.sum()
    if n == 0 or total == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * total))


def intra_list_diversity(topk_categories: np.ndarray) -> np.ndarray:
    """Share of item pairs in each list whose categories differ. (n, k) -> (n,)."""
    c = topk_categories
    k = c.shape[1]
    if k < 2:
        return np.zeros(len(c))
    diff = c[:, :, None] != c[:, None, :]
    return diff.sum(axis=(1, 2)) / (k * (k - 1))


# ---------------------------------------------------------------------------
# Impression-list ranking metrics
# ---------------------------------------------------------------------------


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC: P(random positive scores above random negative), ties count half."""
    labels = np.asarray(labels)
    n_pos = labels.sum()
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def grouped_ranking_metrics(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    score_col: str,
    ks: list[int],
    rng: np.random.Generator,
    gauc_weight: str = "impressions",
) -> dict:
    """AUC, GAUC and nDCG@k for ranking each group's impressions by score.

    GAUC averages per-group AUC over groups that have both a positive and a
    negative, weighted by impressions or uniformly.
    nDCG@k averages over groups with at least one positive.
    """
    d = pd.DataFrame(
        {
            "group": df[group_col].to_numpy(),
            "y": df[label_col].to_numpy().astype(np.int64),
            "score": df[score_col].to_numpy().astype(np.float64),
        }
    )
    out = {"auc": auc(d.y.to_numpy(), d.score.to_numpy())}

    # Per-group AUC, ties at average rank.
    d["r"] = d.groupby("group").score.rank(method="average")
    group = d.groupby("group").agg(n=("y", "size"), pos=("y", "sum"))
    group["rank_sum_pos"] = d[d.y == 1].groupby("group").r.sum()
    group["rank_sum_pos"] = group["rank_sum_pos"].fillna(0.0)
    group["neg"] = group.n - group.pos
    valid = group[
        (group.pos > 0) & (group.neg > 0)
    ]  # AUC is undefined without both classes
    group_auc = (valid.rank_sum_pos - valid.pos * (valid.pos + 1) / 2) / (
        valid.pos * valid.neg
    )
    w = valid.n if gauc_weight == "impressions" else pd.Series(1.0, index=valid.index)
    out["gauc"] = float((group_auc * w).sum() / w.sum()) if len(valid) else float("nan")
    out["gauc_groups"] = int(len(valid))

    # nDCG@k within groups. Shuffle rows first so the stable sort breaks score
    # ties at random rather than by original row order.
    d = d.iloc[rng.permutation(len(d))]
    d = d.sort_values(["group", "score"], ascending=[True, False], kind="stable")
    d["rank"] = d.groupby("group").cumcount()
    ideal = d.sort_values(["group", "y"], ascending=[True, False], kind="stable")
    ideal_rank = ideal.groupby("group").cumcount()
    has_pos = group.index[group.pos > 0]
    for k in ks:
        gain = np.where(d["rank"] < k, d.y / np.log2(d["rank"] + 2), 0.0)
        dcg = pd.Series(gain, index=d.index).groupby(d.group).sum()
        igain = np.where(ideal_rank < k, ideal.y / np.log2(ideal_rank + 2), 0.0)
        idcg = pd.Series(igain, index=ideal.index).groupby(ideal.group).sum()
        out[f"ndcg@{k}"] = (
            float((dcg[has_pos] / idcg[has_pos]).mean())
            if len(has_pos)
            else float("nan")
        )
    out["ndcg_groups"] = int(len(has_pos))
    return out
