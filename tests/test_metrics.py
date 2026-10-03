import itertools

import numpy as np
import pandas as pd
import pytest

from krec.eval.metrics import (
    auc,
    gini,
    grouped_ranking_metrics,
    intra_list_diversity,
    retrieval_metrics,
    top_k,
)


def test_retrieval_metrics_hand_computed():
    # user 0: 2 targets, hits at ranks 0 and 2
    # user 1: 1 target, hit at rank 1
    hits = np.array([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=bool)
    m = retrieval_metrics(hits, np.array([2, 1]), ks=[1, 4])
    assert m["recall@1"].tolist() == [0.5, 0.0]
    assert m["recall@4"].tolist() == [1.0, 1.0]
    assert m["hit@1"].tolist() == [1.0, 0.0]
    idcg0 = 1 + 1 / np.log2(3)
    assert m["ndcg@4"][0] == pytest.approx((1 + 1 / np.log2(4)) / idcg0)
    assert m["ndcg@4"][1] == pytest.approx(1 / np.log2(3))


def test_ndcg_ideal_is_capped_at_k():
    # 5 targets but k=2 and both slots hit -> perfect score
    m = retrieval_metrics(np.array([[1, 1]], dtype=bool), np.array([5]), ks=[2])
    assert m["ndcg@2"][0] == pytest.approx(1.0)
    assert m["recall@2"][0] == pytest.approx(0.4)


def test_top_k_orders_and_respects_exclusion():
    rng = np.random.default_rng(0)
    s = np.array([[0.1, 0.9, 0.5, -np.inf, 0.7]])
    assert top_k(s, 3, rng).tolist() == [[1, 4, 2]]


def test_top_k_breaks_ties_uniformly_at_random():
    # 4 tied items, pick 1: each should win ~25% of the time.
    wins = np.bincount(
        [
            top_k(np.zeros((1, 4)), 1, np.random.default_rng(i))[0, 0]
            for i in range(4000)
        ],
        minlength=4,
    )
    assert np.all(np.abs(wins / 4000 - 0.25) < 0.03)


def test_top_k_never_reorders_tiny_but_real_differences():
    # important check to avoid artificial reordering (noise injection)
    s = np.zeros((1, 50))
    s[0, 37] = 1e-12
    for i in range(20):
        assert top_k(s, 1, np.random.default_rng(i))[0, 0] == 37


def test_top_k_matches_full_sort_without_ties():
    s = np.random.default_rng(3).random((5, 30))
    expected = np.argsort(-s, axis=1)[:, :7]
    assert (top_k(s, 7, np.random.default_rng(0)) == expected).all()


def test_auc_matches_pairwise_definition():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 60)
    s = rng.integers(0, 5, 60).astype(float)  # many ties
    pos, neg = s[y == 1], s[y == 0]
    pairs = [(p > n) + 0.5 * (p == n) for p, n in itertools.product(pos, neg)]
    assert auc(y, s) == pytest.approx(np.mean(pairs))


def test_gauc_skips_single_class_groups_and_weights_by_impressions():
    df = pd.DataFrame(
        {
            "group": [1, 1, 1, 2, 2, 3, 3],
            "y": [1, 0, 0, 0, 1, 1, 1],  # group 3 has no negatives
            "score": [0.9, 0.1, 0.2, 0.8, 0.3, 0.5, 0.4],
        }
    )
    m = grouped_ranking_metrics(
        df, "group", "y", "score", [2], np.random.default_rng(0)
    )
    # g1 AUC = 1 (3 rows), g2 AUC = 0 (2 rows)
    assert m["gauc"] == pytest.approx(3 / 5)
    assert m["gauc_groups"] == 2
    # NDCG@2: g1 = 1, g2 = (1/log2(3)) / 1, g3 = 1
    assert m["ndcg@2"] == pytest.approx((1 + 1 / np.log2(3) + 1) / 3)


def test_all_tied_scores_give_chance_level_auc():
    df = pd.DataFrame({"group": [1, 1, 1, 2, 2], "y": [1, 0, 0, 1, 0], "score": 0.0})
    m = grouped_ranking_metrics(
        df, "group", "y", "score", [1], np.random.default_rng(0)
    )
    assert m["auc"] == 0.5 and m["gauc"] == 0.5


def test_gini_extremes():
    assert gini(np.ones(10)) == pytest.approx(0.0)
    assert gini(np.r_[np.zeros(99), 1.0]) == pytest.approx(0.99)


def test_intra_list_diversity():
    cats = np.array([[1, 1, 1], [1, 2, 3], [1, 1, 2]])
    assert intra_list_diversity(cats).tolist() == pytest.approx([0.0, 1.0, 2 / 3])
