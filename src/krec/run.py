"""Experiment runners: evaluate a set of models on every configured eval set."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from krec.config import Config
from krec.data.tables import (
    load_eval_set,
    load_fit_window,
    load_items,
    parse_eval_set,
    split_start,
)
from krec.eval.protocols import evaluate_ranking, evaluate_retrieval
from krec.models.base import ItemCatalog, Recommender


def evaluate_models(cfg: Config, make_models, tag: str) -> dict:
    """Fit fresh models as of each eval set's start date and evaluate them."""
    ev = cfg.evaluation
    items = load_items(cfg)
    results: dict = {"config": cfg.dataset.name, "tag": tag, "eval_sets": {}}
    fitted_cache: dict[str, tuple[pd.DataFrame, list[Recommender], ItemCatalog]] = {}

    for name in ev.eval_sets:
        es = parse_eval_set(name)
        as_of = split_start(cfg, es.split)
        if es.split not in fitted_cache:
            fit_df = load_fit_window(cfg, as_of)
            catalog = ItemCatalog.build(items, fit_df)
            models = [m.fit(fit_df, catalog, as_of) for m in make_models()]
            fitted_cache[es.split] = (fit_df, models, catalog)
        fit_df, models, catalog = fitted_cache[es.split]
        eval_df = load_eval_set(cfg, name)

        per_model = {}
        for m in models:
            t0 = time.perf_counter()
            per_model[m.name] = {
                "retrieval": evaluate_retrieval(
                    m,
                    fit_df,
                    eval_df,
                    catalog,
                    ev.labels,
                    ev.ks,
                    users=ev.retrieval.users,
                    exclude_fit_positives=ev.retrieval.exclude_fit_positives,
                    diversity_k=ev.retrieval.diversity_k,
                    batch_users=ev.retrieval.batch_users,
                    seed=ev.seed,
                ),
                "ranking": evaluate_ranking(
                    m,
                    eval_df,
                    ev.labels,
                    ev.ranking_ks,
                    group_by=ev.ranking.group_by,
                    gauc_weight=ev.ranking.gauc_weight,
                    seed=ev.seed,
                ),
                "eval_seconds": round(time.perf_counter() - t0, 2),
            }
        results["eval_sets"][name] = {
            "fit_window_end": str((as_of - pd.Timedelta(days=1)).date()),
            "fit_rows": int(len(fit_df)),
            "eval_rows": int(len(eval_df)),
            "models": per_model,
        }
    return results


def write_results(results: dict, out_dir: Path, headline_k: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "results.md").write_text(results_markdown(results, headline_k))
    return out_dir / "results.md"


def _metric_key(metrics: dict, prefix: str) -> str:
    """The `prefix@k` key with the largest k (e.g. 'ndcg@10' from ndcg@5, ndcg@10)."""
    return max(
        (m for m in metrics if m.startswith(prefix + "@")),
        key=lambda m: int(m.split("@")[1]),
    )


def results_markdown(results: dict, k: int) -> str:
    """Compact tables: retrieval at the headline K, ranking on the impression lists."""
    lines = [f"# Results: {results['tag']} ({results['config']})", ""]
    for name, block in results["eval_sets"].items():
        lines += [
            f"## {name}",
            f"Fit on standard logs through {block['fit_window_end']} "
            f"({block['fit_rows']:,} rows); {block['eval_rows']:,} eval rows.",
            "",
        ]
        labels = next(iter(block["models"].values()))["retrieval"].keys()
        for label in labels:
            sample = next(iter(block["models"].values()))
            ild = _metric_key(sample["retrieval"][label], "ild")
            rnk = _metric_key(sample["ranking"][label], "ndcg")
            lines += [
                f"**Label: `{label}`**",
                "",
                f"| model | Recall@{k} | NDCG@{k} | Coverage@{k} | Novelty@{k} | "
                f"{ild.upper()} | AUC | GAUC | {rnk.upper()} (impr.) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for model, r in block["models"].items():
                rt, rk = r["retrieval"][label], r["ranking"][label]
                if rt.get("n_users", 0) == 0:
                    continue
                lines.append(
                    f"| {model} | {rt[f'recall@{k}']:.4f} | {rt[f'ndcg@{k}']:.4f} | "
                    f"{rt[f'coverage@{k}']:.3f} | {rt[f'novelty@{k}']:.2f} | "
                    f"{rt[ild]:.3f} | {rk['auc']:.4f} | {rk['gauc']:.4f} | "
                    f"{rk[rnk]:.4f} |"
                )
            first = next(iter(block["models"].values()))
            lines += [
                "",
                f"Retrieval users: {first['retrieval'][label]['n_users']:,}; "
                f"GAUC groups: {first['ranking'][label]['gauc_groups']:,}.",
                "",
            ]
    return "\n".join(lines)
