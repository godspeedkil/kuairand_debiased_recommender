import json

import numpy as np
import pytest

from krec.data.tables import (
    load_eval_set,
    load_fit_window,
    load_items,
    load_users,
    split_start,
)
from krec.eval.protocols import evaluate_ranking, evaluate_retrieval
from krec.models.base import ItemCatalog, Recommender
from krec.retrieval.twotower import build_two_tower


@pytest.fixture(scope="module")
def fitted(cfg):
    as_of = split_start(cfg, "test")
    fit = load_fit_window(cfg, as_of)
    items, users = load_items(cfg), load_users(cfg)
    catalog = ItemCatalog.build(items, fit)
    model = build_two_tower(cfg, items, users).fit(fit, catalog, as_of)
    return model, fit, catalog, as_of


def test_recommend_matches_scoring_the_whole_catalog(fitted):
    model, fit, catalog, _ = fitted
    users = np.append(fit.user_id.unique()[:30], -5)  # incl. an unknown user
    pos = fit[fit.is_click == 1].groupby("user_id").item_id.unique()
    exclude = [
        catalog.index_of(pos[u]) if u in pos.index else np.empty(0, dtype=np.int64)
        for u in users
    ]
    allowed = np.ones(catalog.n_items, dtype=bool)
    allowed[:7] = False
    key = np.random.default_rng(0).permutation(catalog.n_items)
    fast = model.recommend(users, 25, exclude, allowed, key, batch_users=4)
    dense = Recommender.recommend(model, users, 25, exclude, allowed, key)
    np.testing.assert_array_equal(fast, dense)
    for row, ex in zip(fast, exclude, strict=True):
        assert not set(row[row >= 0]) & (set(ex) | set(range(7)))


def test_pair_scores_match_catalog_scores(fitted):
    model, fit, catalog, _ = fitted
    users = fit.user_id.unique()[:4]
    uu, ii = np.meshgrid(users, catalog.item_ids[:9], indexing="ij")
    pairs = model.score_pairs(uu.ravel(), ii.ravel()).reshape(uu.shape)
    np.testing.assert_allclose(pairs, model.score_users(users)[:, :9], rtol=1e-5)
    assert np.isneginf(model.score_pairs(users[:1], np.array([-999])))[0]


def test_same_seed_gives_the_same_model(cfg, fitted):
    model, fit, catalog, as_of = fitted
    again = build_two_tower(cfg, load_items(cfg), load_users(cfg)).fit(
        fit, catalog, as_of
    )
    assert np.array_equal(again.item_emb.numpy(), model.item_emb.numpy())


def test_two_tower_beats_random_on_both_protocols(cfg, fitted):
    from krec.models.baselines import RandomRecommender

    model, fit, catalog, as_of = fitted
    rand = RandomRecommender(seed=0).fit(fit, catalog, as_of)
    std, rnd = load_eval_set(cfg, "test_standard"), load_eval_set(cfg, "test_random")
    recall = {
        m.name: evaluate_retrieval(m, fit, std, catalog, ["is_click"], [20])[
            "is_click"
        ]["recall@20"]
        for m in (model, rand)
    }
    gauc = {
        m.name: evaluate_ranking(m, rnd, ["is_click"], [5])["is_click"]["gauc"]
        for m in (model, rand)
    }
    assert recall["two_tower"] > 1.5 * recall["random"]
    assert gauc["two_tower"] > gauc["random"] + 0.05


def test_retrieve_command_reports_two_tower_next_to_baselines(cfg):
    from krec.cli import COMMANDS

    quick = cfg.override(two_tower={"epochs": 1})  # only checks the report
    COMMANDS["retrieve"](quick)
    path = quick.reports_path / "retrieval" / "results.json"
    models = json.loads(path.read_text())["eval_sets"]["test_random"]["models"]
    assert {"two_tower", "most_popular", "item_cooc"} <= set(models)
