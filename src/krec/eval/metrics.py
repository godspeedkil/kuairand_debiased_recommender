"""
Evaluation metrics.

- Per user: top k, recall@k, hit@k, NDCG@k, intra-list diversity.
- Per group (user by default, or user-day): GAUC, NDCG@k.
- Overall: AUC, Gini coefficient.
- Bootstrap confidence intervals over groups, for the grouped metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def top_k(
    scores: np.ndarray,
    k: int,
    rng: np.random.Generator | None = None,
    key: np.ndarray | None = None,
) -> np.ndarray:
    """Return indices of the k highest scores per row, best first.

    Equal scores are ordered by `key`, a random permutation unless given, so every
    tied item is equally likely to be picked.

    Parameters
    ----------
    scores : np.ndarray
        Scores, shape (n_rows, n_items), -inf for excluded items.
    k : int
        Indices per row.
    rng : np.random.Generator, optional
        Draws the tie-break key when `key` is not given.
    key : np.ndarray, optional
        Tie-break order per column.

    Returns
    -------
    np.ndarray
        Column indices, shape (n_rows, k); -1 where a row has fewer than k finite
        scores.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[1]
    k = min(k, n)
    if key is None:
        key = rng.permutation(n)
    # k-th largest per row; only items at or above it can make the list
    kth = np.partition(scores, n - k, axis=1)[:, n - k]
    out = np.empty((len(scores), k), dtype=np.int64)
    for r in range(len(scores)):
        cand = np.flatnonzero(scores[r] >= kth[r])
        top = cand[np.lexsort((key[cand], -scores[r, cand]))[:k]]
        out[r] = np.where(np.isneginf(scores[r, top]), -1, top)
    return out


def _discounts(k: int) -> np.ndarray:
    """DCG discounts for ranks 1..k.

    Parameters
    ----------
    k : int
        Number of ranks.

    Returns
    -------
    np.ndarray
        1 / log2(rank + 1) per rank.
    """
    return 1.0 / np.log2(np.arange(2, k + 2))


def retrieval_metrics(hits: np.ndarray, n_targets: np.ndarray, ks: list[int]) -> dict:
    """Per-user recall, hit rate, and NDCG at each cut-off.

    Parameters
    ----------
    hits : np.ndarray
        Bool, shape (n_users, K): whether the item at each rank is a target.
    n_targets : np.ndarray
        Reachable targets per user (> 0), shape (n_users,).
    ks : list of int
        Cut-offs.

    Returns
    -------
    dict[str, np.ndarray]
        `recall@k`, `hit@k` and `ndcg@k`, each a per-user array.
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
    """Gini coefficient of non-negative counts.

    Parameters
    ----------
    counts : np.ndarray
        Count per item.

    Returns
    -------
    float
        0 for equal counts, close to 1 when concentrated on few items.
    """
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n, total = x.size, x.sum()
    if n == 0 or total == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * total))


def intra_list_diversity(topk_categories: np.ndarray) -> np.ndarray:
    """Share of item pairs in each list whose categories differ.

    Parameters
    ----------
    topk_categories : np.ndarray
        Category of each listed item, shape (n_lists, k).

    Returns
    -------
    np.ndarray
        Diversity per list, shape (n_lists,).
    """
    c = topk_categories
    k = c.shape[1]
    if k < 2:
        return np.zeros(len(c))
    diff = c[:, :, None] != c[:, None, :]
    return diff.sum(axis=(1, 2)) / (k * (k - 1))


def bootstrap_ci(
    values: np.ndarray,
    weights: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> list[float]:
    """Percentile interval of a weighted mean, resampling groups with replacement.

    Parameters
    ----------
    values : np.ndarray
        Value per group.
    weights : np.ndarray
        Weight per group.
    n_boot : int
        Number of resamples.
    rng : np.random.Generator
        Source of the resamples.
    level : float, default 0.95
        Interval coverage.

    Returns
    -------
    list of float
        Lower and upper bound.
    """
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = (values[idx] * weights[idx]).sum(axis=1) / weights[idx].sum(axis=1)
    return np.quantile(means, [(1 - level) / 2, (1 + level) / 2]).tolist()


# ---------------------------------------------------------------------------
# Impression-list ranking metrics
# ---------------------------------------------------------------------------


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC, with ties counted as half.

    Parameters
    ----------
    labels : np.ndarray
        Binary labels.
    scores : np.ndarray
        Score per label.

    Returns
    -------
    float
        P(random positive scores above random negative), NaN if missing either
    """
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
    n_boot: int = 0,
) -> dict:
    """AUC, GAUC and NDCG@k for ranking each group's impressions by score.

    Parameters
    ----------
    df : pd.DataFrame
        One row per impression.
    group_col : str
        Column that defines the groups, e.g. user.
    label_col : str
        Binary label column.
    score_col : str
        Model score column.
    ks : list of int
        Cut-offs for NDCG.
    rng : np.random.Generator
        Breaks score ties in NDCG and draws bootstrap resamples.
    gauc_weight : {"impressions", "uniform"}, default "impressions"
        Weight of each group in GAUC.
    n_boot : int, default 0
        Bootstrap resamples for 95% intervals; 0 for none.

    Returns
    -------
    dict
        `auc`, `gauc`, `gauc_groups`, `ndcg@k`, `ndcg_groups`, `gauc_ci`,
        and `ndcg@k_ci`
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
    if n_boot and len(valid):
        out["gauc_ci"] = bootstrap_ci(group_auc.to_numpy(), w.to_numpy(), n_boot, rng)

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
        per_group = (dcg[has_pos] / idcg[has_pos]).to_numpy()
        out[f"ndcg@{k}"] = float(per_group.mean()) if len(has_pos) else float("nan")
        if n_boot and len(has_pos):
            ones = np.ones(len(per_group))
            out[f"ndcg@{k}_ci"] = bootstrap_ci(per_group, ones, n_boot, rng)
    out["ndcg_groups"] = int(len(has_pos))
    return out
