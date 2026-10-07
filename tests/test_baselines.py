import numpy as np
import pandas as pd
import pytest

from krec.data.tables import load_eval_set, load_fit_window, load_items, split_start
from krec.eval.protocols import evaluate_ranking, evaluate_retrieval
from krec.models.base import ItemCatalog
from krec.models.baselines import ItemCooccurrence, MostPopular, RandomRecommender
from krec.run import evaluate_models


@pytest.fixture(scope="module")
def fitted(cfg):
    as_of = split_start(cfg, "test")
    fit = load_fit_window(cfg, as_of)
    catalog = ItemCatalog.build(load_items(cfg), fit)
    return fit, catalog, as_of


def test_most_popular_scores_are_click_counts(fitted):
    fit, catalog, as_of = fitted
    m = MostPopular().fit(fit, catalog, as_of)
    counts = fit[fit.is_click == 1].item_id.value_counts()
    top = counts.index[0]
    assert m.item_scores[catalog.index_of([top])[0]] == counts.iloc[0]
    assert m.item_scores.sum() == fit.is_click.sum()


def test_recent_popular_only_uses_the_window(fitted):
    fit, catalog, as_of = fitted
    m = MostPopular(window_days=3).fit(fit, catalog, as_of)
    recent = fit[(fit.date >= as_of - pd.Timedelta(days=3)) & (fit.is_click == 1)]
    assert m.item_scores.sum() == len(recent)


def test_cooccurrence_prefers_items_coclicked_with_history(fitted):
    fit, catalog, as_of = fitted
    m = ItemCooccurrence(top_neighbors=50).fit(fit, catalog, as_of)
    user = fit[fit.is_click == 1].user_id.value_counts().index[0]
    s = m.score_users(np.array([user]))[0]
    assert s.shape == (catalog.n_items,)
    assert np.isfinite(s).all() and s.max() > 1e-3
    # unknown user falls back to popularity
    cold = m.score_users(np.array([-12345]))[0]
    assert cold.max() < 1e-5 and cold.std() > 0


def test_score_pairs_matches_score_users(fitted):
    fit, catalog, as_of = fitted
    m = ItemCooccurrence().fit(fit, catalog, as_of)
    users = fit.user_id.unique()[:5]
    items = catalog.item_ids[:7]
    uu, ii = np.meshgrid(users, items, indexing="ij")
    pairs = m.score_pairs(uu.ravel(), ii.ravel()).reshape(uu.shape)
    np.testing.assert_allclose(pairs, m.score_users(users)[:, :7])


def test_random_recommender_hits_chance_level(cfg, fitted):
    fit, catalog, as_of = fitted
    ev = load_eval_set(cfg, "test_standard")
    res = evaluate_retrieval(
        RandomRecommender(seed=1).fit(fit, catalog, as_of),
        fit,
        ev,
        catalog,
        ["is_click"],
        [20],
        seed=0,
    )["is_click"]
    # expected recall@k for random ranking = k / (#candidates)
    assert 0.08 < res["recall@20"] < 0.3
    rk = evaluate_ranking(
        RandomRecommender(seed=1).fit(fit, catalog, as_of),
        load_eval_set(cfg, "test_random"),
        ["is_click"],
        [5],
    )["is_click"]
    assert abs(rk["auc"] - 0.5) < 0.06


def test_popularity_beats_random_on_standard_test(cfg):
    from krec.models.baselines import build_baselines

    cfg2 = cfg.override(
        evaluation={"eval_sets": ["test_standard"], "ks": [20], "headline_k": 20},
        baselines={"recent_popular": None, "item_cooc": None},
    )
    res = evaluate_models(cfg2, lambda: build_baselines(cfg2.baselines, seed=0), "t")
    r = res["eval_sets"]["test_standard"]["models"]
    assert (
        r["most_popular"]["retrieval"]["is_click"]["recall@20"]
        > r["random"]["retrieval"]["is_click"]["recall@20"]
    )


def _recommend_inputs(fit, catalog, n_users=40, seed=0):
    """Users, per-user exclusions (their fit positives) and an availability mask
    with a few items pretended to be uploaded in the future."""
    rng = np.random.default_rng(seed)
    users = fit.user_id.unique()[:n_users]
    pos = fit[fit.is_click == 1].groupby("user_id").item_id.unique()
    exclude = [
        catalog.index_of(pos[u]) if u in pos.index else np.empty(0, dtype=np.int64)
        for u in users
    ]
    allowed = np.ones(catalog.n_items, dtype=bool)
    allowed[rng.choice(catalog.n_items, 5, replace=False)] = False
    return np.append(users, -12345), exclude + [np.empty(0, dtype=np.int64)], allowed


@pytest.mark.parametrize(
    "make",
    [
        lambda: MostPopular(),
        lambda: MostPopular(window_days=3),
        lambda: ItemCooccurrence(top_neighbors=20),
    ],
    ids=["most_popular", "recent_popular", "item_cooc"],
)
def test_recommend_matches_scoring_the_whole_catalog(fitted, make):
    from krec.models.base import Recommender

    fit, catalog, as_of = fitted
    m = make().fit(fit, catalog, as_of)
    users, exclude, allowed = _recommend_inputs(fit, catalog)
    key = np.random.default_rng(3).permutation(catalog.n_items)
    dense = Recommender.recommend(m, users, 30, exclude, allowed, key)
    for batch in (2048, 7):  # small batches exercise the shared popularity head
        fast = m.recommend(users, 30, exclude, allowed, key, batch_users=batch)
        np.testing.assert_array_equal(fast, dense)


@pytest.mark.parametrize(
    "make",
    [
        lambda: MostPopular(),
        lambda: ItemCooccurrence(),
        lambda: RandomRecommender(seed=0),
    ],
    ids=["most_popular", "item_cooc", "random"],
)
def test_recommend_never_returns_excluded_or_unreleased_items(fitted, make):
    fit, catalog, as_of = fitted
    m = make().fit(fit, catalog, as_of)
    users, exclude, allowed = _recommend_inputs(fit, catalog)
    key = np.random.default_rng(0).permutation(catalog.n_items)
    top = m.recommend(users, 50, exclude, allowed, key)
    for row, ex in zip(top, exclude, strict=True):
        row = row[row >= 0]
        assert len(set(row)) == len(row)
        assert not set(row) & set(ex)
        assert allowed[row].all()
        # the list is only short when too few items are left
        assert len(row) == min(
            50, allowed.sum() - len(set(ex) & set(np.flatnonzero(allowed)))
        )


def test_dense_scoring_is_refused_above_the_limit(cfg, fitted):
    from krec.models.base import Recommender

    class DenseOnly(Recommender):
        name = "dense_only"

        def fit(self, interactions, catalog, as_of):
            self.catalog = catalog
            return self

        def score_users(self, user_ids):
            return np.zeros((len(user_ids), self.catalog.n_items))

    fit, catalog, as_of = fitted
    ev = load_eval_set(cfg, "test_standard")
    m = DenseOnly().fit(fit, catalog, as_of)
    with pytest.raises(RuntimeError, match="max_dense_scores"):
        evaluate_retrieval(m, fit, ev, catalog, ["is_click"], [20], max_dense_scores=10)
    evaluate_retrieval(m, fit, ev, catalog, ["is_click"], [20])  # no limit: runs


def test_retrieval_can_keep_fit_positives(cfg, fitted):
    fit, catalog, as_of = fitted
    ev = load_eval_set(cfg, "test_standard")
    m = MostPopular().fit(fit, catalog, as_of)
    kept = evaluate_retrieval(
        m, fit, ev, catalog, ["is_click"], [20], exclude_fit_positives=False
    )["is_click"]
    dropped = evaluate_retrieval(m, fit, ev, catalog, ["is_click"], [20])["is_click"]
    # re-watched items become reachable targets, so more users qualify
    assert kept["n_users"] >= dropped["n_users"] > 0


def test_cooccurrence_keeps_only_the_most_recent_positives(fitted):
    fit, catalog, as_of = fitted
    m = ItemCooccurrence(max_history=2).fit(fit, catalog, as_of)
    pos = fit[fit.is_click == 1].drop_duplicates(["user_id", "item_id"], keep="last")
    user = pos.user_id.value_counts().index[0]  # has more than 2 positives
    want = catalog.index_of(pos[pos.user_id == user].item_id.to_numpy()[-2:])
    row = m.history[m.user_index.get_loc(user)]
    assert sorted(row.indices) == sorted(want)


def test_cooccurrence_pairs_handle_unknown_users_and_items(fitted):
    fit, catalog, as_of = fitted
    m = ItemCooccurrence().fit(fit, catalog, as_of)
    item = catalog.item_ids[0]
    s = m.score_pairs(np.array([-1, -1]), np.array([item, -999]))
    assert s[0] == m.pop_tiebreak[0] and s[1] == -np.inf


def test_keep_top_per_row_matches_a_dense_sort():
    from scipy import sparse

    from krec.models.baselines import _keep_top_per_row

    m = sparse.random(40, 30, density=0.4, random_state=0, format="csr")
    kept = _keep_top_per_row(m, 3).toarray()
    dense = m.toarray()
    for r in range(40):
        nz = np.flatnonzero(dense[r])
        top = nz[np.argsort(-dense[r, nz])[:3]]
        assert sorted(np.flatnonzero(kept[r])) == sorted(top)
        np.testing.assert_allclose(kept[r, top], dense[r, top])


def test_random_pairs_skip_unknown_items(fitted):
    fit, catalog, as_of = fitted
    m = RandomRecommender(seed=0).fit(fit, catalog, as_of)
    s = m.score_pairs(np.array([1, 1]), np.array([catalog.item_ids[0], -999]))
    assert 0 <= s[0] < 1 and s[1] == -np.inf


def test_random_eval_sets_get_ranking_only_with_intervals(cfg):
    from krec.models.baselines import build_baselines
    from krec.run import results_markdown

    cfg2 = cfg.override(
        evaluation={
            "eval_sets": ["test_standard", "test_random"],
            "ks": [20],
            "headline_k": 20,
            "bootstrap_samples": 50,
        },
        baselines={"recent_popular": None, "item_cooc": None},
    )
    res = evaluate_models(cfg2, lambda: build_baselines(cfg2.baselines, seed=0), "t")
    std = res["eval_sets"]["test_standard"]["models"]["most_popular"]
    rnd = res["eval_sets"]["test_random"]["models"]["most_popular"]
    assert "retrieval" in std and "retrieval" not in rnd
    rk = rnd["ranking"]["is_click"]
    lo, hi = rk["gauc_ci"]
    assert lo <= rk["gauc"] <= hi
    assert "Recall@20" not in results_markdown(res, 20).split("## test_random")[1]
