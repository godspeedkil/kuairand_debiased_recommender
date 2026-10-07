"""Evaluation protocols.

- Retrieval: for each eval user, rank the available catalog and compare the top K
  with the items they engaged with in the eval window. Items they already engaged
  with in the fit window are excluded.
- Ranking: for each user (or user-day), rank the items actually shown in the eval
  window and compare with the observed labels. On `*_standard` sets those items
  were chosen by the policy, so the scores carry its bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from krec.eval.metrics import (
    gini,
    grouped_ranking_metrics,
    intra_list_diversity,
    retrieval_metrics,
)
from krec.models.base import ItemCatalog, Recommender


def _positives_by_user(
    df: pd.DataFrame, label: str, catalog: ItemCatalog
) -> dict[int, np.ndarray]:
    """Each user's positives as catalog positions.

    Parameters
    ----------
    df : pd.DataFrame
        Logs with `user_id`, `item_id`, and the label column.
    label : str
        Column whose 1s count as positives.
    catalog : ItemCatalog
        Catalog for the positions, missing items are dropped.

    Returns
    -------
    dict of int to np.ndarray
        Sorted unique catalog positions per user id.
    """
    pos = df.loc[df[label] == 1, ["user_id", "item_id"]]
    cols = catalog.index_of(pos.item_id.to_numpy())
    known = cols >= 0
    u, users = pd.factorize(pos.user_id.to_numpy()[known])
    if len(users) == 0:
        return {}
    # one int64 per (user, item) pair, sorted and deduplicated: much faster
    # than np.unique over pairs
    codes = np.sort(u.astype(np.int64) * catalog.n_items + cols[known])
    codes = codes[np.r_[True, codes[1:] != codes[:-1]]]
    u, cols = np.divmod(codes, catalog.n_items)
    starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]])
    return dict(zip(users[u[starts]].tolist(), np.split(cols, starts[1:]), strict=True))


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
    max_dense_scores: int | None = None,
) -> dict:
    """Rank the catalog for each eval user and compare the top K with their positives.

    Parameters
    ----------
    model : Recommender
        Fitted model.
    fit_df : pd.DataFrame
        Fit-window logs: warm users, positives to exclude, popularity for novelty.
    eval_df : pd.DataFrame
        Eval-window logs, positives are targets.
    catalog : ItemCatalog
        Catalog the model ranks.
    labels : list[str]
        Labels that define positives.
    ks : list[int]
        Cut-offs for the metrics.
    users : {"warm", "all"}, default "warm"
        Evaluate only users seen in the fit window, or everyone.
    exclude_fit_positives : bool, default True
        Never recommend items the user was already positive on.
    diversity_k : int, default 10
        List length for intra-list diversity.
    batch_users : int, default 2048
        Users per batch, passed to `recommend`.
    seed : int, default 0
        Seed for the tie-break order.
    max_dense_scores : int, optional
        Refuse whole-catalog scoring above this many user x item scores.

    Returns
    -------
    dict
        Per label: recall, hit, NDCG, coverage, Gini and novelty at each k, ILD and
        `n_users`.

    Raises
    ------
    RuntimeError
        If the model can only score the whole catalog and that exceeds
        `max_dense_scores`.
    """
    rng = np.random.default_rng(seed)
    k_max = min(max(ks), catalog.n_items)
    # include items up to the end of eval windows
    window_end = pd.Timestamp(eval_df.date.max())
    available = catalog.available_mask(window_end)
    key = rng.permutation(catalog.n_items)  # one tie-break order for all users

    impr = np.zeros(catalog.n_items)
    idx = catalog.index_of(fit_df.item_id.to_numpy())
    np.add.at(impr, idx[idx >= 0], 1.0)
    p_item = (impr + 1.0) / (impr.sum() + catalog.n_items)
    self_info = -np.log2(p_item)

    warm = set(fit_df.user_id.unique())
    empty = np.empty(0, dtype=np.int64)
    dense = type(model).recommend is Recommender.recommend
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
        n_users = len(user_ids)
        if dense and max_dense_scores and n_users * catalog.n_items > max_dense_scores:
            raise RuntimeError(
                f"{model.name} has no recommend() of its own, and scoring "
                f"{n_users:,} users x {catalog.n_items:,} items exceeds "
                f"max_dense_scores={max_dense_scores:,}. Give it a recommend() "
                "that uses its structure (e.g. a nearest-neighbour index)."
            )

        top = model.recommend(
            np.asarray(user_ids), k_max, excl_lists, available, key, batch_users
        )
        valid = top >= 0
        # hits: look each recommended item up in a sparse user x item target matrix
        lens = np.fromiter((len(t) for t in target_lists), dtype=np.int64)
        targets_m = sp.csr_matrix(
            (
                np.ones(lens.sum(), dtype=bool),
                (np.repeat(np.arange(n_users), lens), np.concatenate(target_lists)),
            ),
            shape=(n_users, catalog.n_items),
        )
        rows = np.repeat(np.arange(n_users), k_max)[valid.ravel()]
        hits = np.zeros(top.shape, dtype=bool)
        hits[valid] = np.asarray(targets_m[rows, top[valid]]).ravel()

        res = {m: float(v.mean()) for m, v in retrieval_metrics(hits, lens, ks).items()}
        n_avail = int(available.sum())
        safe = np.where(valid, top, 0)
        for k in ks:
            counts = np.zeros(catalog.n_items)
            np.add.at(counts, top[:, :k][valid[:, :k]], 1.0)
            res[f"coverage@{k}"] = float((counts > 0).sum() / n_avail)
            res[f"gini@{k}"] = gini(counts[available])
            info = np.where(valid[:, :k], self_info[safe[:, :k]], 0.0).sum(axis=1)
            res[f"novelty@{k}"] = float(
                np.mean(info / np.maximum(valid[:, :k].sum(axis=1), 1))
            )
        # padded slots (tiny catalogs only) get a category of their own
        cats = np.where(valid, catalog.category[safe], -1 - np.arange(top.shape[1]))[
            :, :diversity_k
        ]
        res[f"ild@{diversity_k}"] = float(intra_list_diversity(cats).mean())
        res["n_users"] = n_users
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
    n_boot: int = 0,
) -> dict:
    """Rank each group's impressions by score and compare with the observed labels.

    Parameters
    ----------
    model : Recommender
        Fitted model.
    eval_df : pd.DataFrame
        Eval-window impressions with labels.
    labels : list[str]
        Labels to evaluate, separately.
    ks : list[int]
        Cut-offs for NDCG.
    group_by : {"user", "user_date"}, default "user"
        What makes one impression list.
    gauc_weight : {"impressions", "uniform"}, default "impressions"
        Weight of each group in GAUC.
    seed : int, default 0
        Seed for tie-breaking and bootstrap resamples.
    n_boot : int, default 0
        Bootstrap resamples for 95% intervals; 0 for none.

    Returns
    -------
    dict
        Per label: the output of `grouped_ranking_metrics` plus `n_impressions`.
    """
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
            df, "group", label, "score", ks, rng, gauc_weight, n_boot
        )
        out[label]["n_impressions"] = int(len(df))
    return out
