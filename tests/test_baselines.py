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
    ev = load_eval_set(cfg, "test_random")
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
        RandomRecommender(seed=1).fit(fit, catalog, as_of), ev, ["is_click"], [5]
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


def test_retrieval_never_ranks_seen_or_unreleased_items(cfg, fitted, monkeypatch):
    """Capture the score matrix the protocol ranks and check what was masked."""
    from dataclasses import replace

    from krec.eval import metrics, protocols

    fit, catalog, as_of = fitted
    captured = []

    def spy(scores, k, rng):
        captured.append(scores.copy())
        return metrics.top_k(scores, k, rng)

    monkeypatch.setattr(protocols, "top_k", spy)

    # pretend the first 5 items are uploaded far in the future
    upload = catalog.upload_date.copy()
    upload[:5] = np.datetime64("2099-01-01")
    catalog = replace(catalog, upload_date=upload)

    # one user with clicks in the fit window and new clicks in the eval window
    ev = load_eval_set(cfg, "test_standard")
    seen = fit[fit.is_click == 1].groupby("user_id").item_id.unique()
    new = ev[ev.is_click == 1].groupby("user_id").item_id.unique()
    user = next(u for u in new.index if u in seen.index and set(new[u]) - set(seen[u]))

    m = MostPopular().fit(fit, catalog, as_of)
    evaluate_retrieval(m, fit, ev[ev.user_id == user], catalog, ["is_click"], [20])
    (row,) = np.concatenate(captured)
    assert np.isneginf(row[:5]).all()
    assert np.isneginf(row[catalog.index_of(seen[user])]).all()
    assert np.isfinite(row).sum() == catalog.n_items - len(
        set(range(5)) | set(catalog.index_of(seen[user]))
    )


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
