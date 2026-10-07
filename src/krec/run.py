"""Experiment runners: evaluate a set of models on every configured eval set."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

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


def _log(msg: str, t0: float) -> None:
    """Print a timed status line above any progress bar.

    Parameters
    ----------
    msg : str
        What finished.
    t0 : float
        `time.perf_counter()` when it started.
    """
    tqdm.write(f"{msg} ({time.perf_counter() - t0:.1f}s)")


def evaluate_models(cfg: Config, make_models, tag: str) -> dict:
    """Fit fresh models as of each eval set's start date and evaluate them.

    Parameters
    ----------
    cfg : Config
        Experiment config: data paths, eval sets, and protocol settings.
    make_models : callable
        Returns new, unfitted models.
    tag : str
        Name of this run in the results.

    Returns
    -------
    dict
        Per eval set: fit window end, row counts, and each model's metrics.
    """
    ev = cfg.evaluation
    items = load_items(cfg)
    results: dict = {"config": cfg.dataset.name, "tag": tag, "eval_sets": {}}
    fitted_cache: dict[str, tuple[pd.DataFrame, list[Recommender], ItemCatalog]] = {}
    # one step per (eval set, model, protocol)
    steps_per_model = sum(
        2 if parse_eval_set(n).source == "standard" else 1 for n in ev.eval_sets
    )
    bar = None

    for name in ev.eval_sets:
        es = parse_eval_set(name)
        as_of = split_start(cfg, es.split)
        if es.split not in fitted_cache:
            t0 = time.perf_counter()
            fit_df = load_fit_window(cfg, as_of)
            _log(f"{es.split}: loaded fit window, {len(fit_df):,} rows", t0)
            t0 = time.perf_counter()
            catalog = ItemCatalog.build(items, fit_df)
            _log(f"{es.split}: built catalog, {catalog.n_items:,} items", t0)
            models = []
            for m in make_models():
                t0 = time.perf_counter()
                models.append(m.fit(fit_df, catalog, as_of))
                _log(f"{es.split}: fit {m.name}", t0)
            fitted_cache[es.split] = (fit_df, models, catalog)
        fit_df, models, catalog = fitted_cache[es.split]
        t0 = time.perf_counter()
        eval_df = load_eval_set(cfg, name)
        _log(f"{name}: loaded {len(eval_df):,} eval rows", t0)
        if bar is None:
            bar = tqdm(total=steps_per_model * len(models), desc=tag, unit="eval")

        per_model = {}
        for m in models:
            t0 = time.perf_counter()
            r = {}
            # Retrieval only on standard logs
            if es.source == "standard":
                bar.set_postfix_str(f"{name} · {m.name} · retrieval")
                r["retrieval"] = evaluate_retrieval(
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
                    max_dense_scores=ev.retrieval.max_dense_scores,
                )
                bar.update()
            bar.set_postfix_str(f"{name} · {m.name} · ranking")
            r["ranking"] = evaluate_ranking(
                m,
                eval_df,
                ev.labels,
                ev.ranking_ks,
                group_by=ev.ranking.group_by,
                gauc_weight=ev.ranking.gauc_weight,
                seed=ev.seed,
                n_boot=ev.bootstrap_samples,
            )
            bar.update()
            r["eval_seconds"] = round(time.perf_counter() - t0, 2)
            per_model[m.name] = r
        results["eval_sets"][name] = {
            "fit_window_end": str((as_of - pd.Timedelta(days=1)).date()),
            "fit_rows": int(len(fit_df)),
            "eval_rows": int(len(eval_df)),
            "models": per_model,
        }
    if bar is not None:
        bar.close()
    return results


def write_results(results: dict, out_dir: Path, headline_k: int) -> Path:
    """Write results as JSON and markdown tables.

    Parameters
    ----------
    results : dict
        Output of `evaluate_models`.
    out_dir : Path
        Folder to write to.
    headline_k : int
        Retrieval cut-off shown in the tables.

    Returns
    -------
    Path
        Path of the markdown file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "results.md").write_text(results_markdown(results, headline_k))
    return out_dir / "results.md"


def _metric_key(metrics: dict, prefix: str) -> str:
    """Find the `prefix@k` key with the largest k.

    Parameters
    ----------
    metrics : dict
        Metrics keyed like `ndcg@10`.
    prefix : str
        Metric name, e.g. "ndcg".

    Returns
    -------
    str
        E.g. "ndcg@10" from ndcg@5 and ndcg@10.
    """
    return max(
        (
            m
            for m in metrics
            if m.startswith(prefix + "@") and m[len(prefix) + 1 :].isdigit()
        ),
        key=lambda m: int(m.split("@")[1]),
    )


def _with_ci(metrics: dict, key: str, digits: int = 4, ci_digits: int = 3) -> str:
    """Format a metric with its interval, if one was computed.

    Parameters
    ----------
    metrics : dict
        Metrics, possibly with `<key>_ci`.
    key : str
        Metric to format.
    digits : int, default 4
        Decimals for the value.
    ci_digits : int, default 3
        Decimals for the interval.

    Returns
    -------
    str
        "0.5951 (0.590–0.600)", or "0.5951" without an interval.
    """
    ci = metrics.get(f"{key}_ci")
    v = f"{metrics[key]:.{digits}f}"
    d = ci_digits
    return f"{v} ({ci[0]:.{d}f}–{ci[1]:.{d}f})" if ci else v


def _table(header: list[list[str]], rows: list[list[str]], markdown: bool) -> list[str]:
    """Pad a table so its columns line up: first column left, the rest right.

    Parameters
    ----------
    header : list[list[str]]
        Header lines, each with one cell per column.
    rows : list[list[str]]
        Body rows, one cell per column.
    markdown : bool
        Markdown table if True, plain text with a dashed rule otherwise.

    Returns
    -------
    list[str]
        Lines of the table.
    """
    widths = [max(len(r[i]) for r in [*header, *rows]) for i in range(len(header[0]))]

    def line(cells: list[str]) -> str:
        padded = [
            c.ljust(w) if i == 0 else c.rjust(w)
            for i, (c, w) in enumerate(zip(cells, widths, strict=True))
        ]
        return f"| {' | '.join(padded)} |" if markdown else "  ".join(padded)

    if markdown:
        rule = [":" + "-" * (widths[0] - 1)] + ["-" * (w - 1) + ":" for w in widths[1:]]
        return [line(header[0]), line(rule), *map(line, rows)]
    rule = "  ".join("-" * w for w in widths)
    return [*map(line, header), rule, *map(line, rows)]


def results_markdown(results: dict, k: int) -> str:
    """Results as markdown tables, per eval set and label.

    Parameters
    ----------
    results : dict
        Output of `evaluate_models`.
    k : int
        Retrieval cut-off to show.

    Returns
    -------
    str
        Markdown text.
    """
    lines = [f"# Results: {results['tag']} ({results['config']})", ""]
    for name, block in results["eval_sets"].items():
        lines += [
            f"## {name}",
            f"Fit on standard logs through {block['fit_window_end']} "
            f"({block['fit_rows']:,} rows), {block['eval_rows']:,} eval rows.",
            "",
        ]
        sample = next(iter(block["models"].values()))
        has_retrieval = "retrieval" in sample
        for label in sample["ranking"]:
            rnk = _metric_key(sample["ranking"][label], "ndcg")
            header = ["model"]
            if has_retrieval:
                ild = _metric_key(sample["retrieval"][label], "ild")
                header += [
                    f"Recall@{k}",
                    f"NDCG@{k}",
                    f"Coverage@{k}",
                    f"Novelty@{k}",
                    ild.upper(),
                ]
            header += ["AUC", "GAUC", f"{rnk.upper()} (impr.)"]
            rows = []
            for model, r in block["models"].items():
                row = [model]
                if has_retrieval:
                    rt = r["retrieval"][label]
                    if rt.get("n_users", 0) == 0:
                        continue
                    row += [
                        f"{rt[f'recall@{k}']:.4f}",
                        f"{rt[f'ndcg@{k}']:.4f}",
                        f"{rt[f'coverage@{k}']:.3f}",
                        f"{rt[f'novelty@{k}']:.2f}",
                        f"{rt[ild]:.3f}",
                    ]
                rk = r["ranking"][label]
                row += [f"{rk['auc']:.4f}", _with_ci(rk, "gauc"), _with_ci(rk, rnk)]
                rows.append(row)
            users = (
                f"Retrieval users: {sample['retrieval'][label]['n_users']:,}, "
                if has_retrieval
                else ""
            )
            lines += [
                f"**Label: `{label}`**",
                "",
                *_table([header], rows, markdown=True),
                "",
                f"{users}GAUC groups: {sample['ranking'][label]['gauc_groups']:,}.",
                "",
            ]
    return "\n".join(lines)


def results_summary(results: dict, k: int) -> str:
    """Headline numbers per label as aligned plain-text tables, for the terminal.

    Recall@k on standard eval sets, GAUC with its 95% interval on random ones.

    Parameters
    ----------
    results : dict
        Output of `evaluate_models`.
    k : int
        Retrieval cut-off to show.

    Returns
    -------
    str
        Plain text, one table per label.
    """
    sets = results["eval_sets"]
    first = next(iter(sets.values()))
    models = list(first["models"])
    labels = list(next(iter(first["models"].values()))["ranking"])
    out = []
    for label in labels:
        names, metrics, rows = ["model"], [""], [[m] for m in models]
        for name, block in sets.items():
            names.append(name)
            retrieval = "retrieval" in next(iter(block["models"].values()))
            metrics.append(f"Recall@{k}" if retrieval else "GAUC (95% CI)")
            for row, m in zip(rows, models, strict=True):
                r = block["models"][m]
                row.append(
                    f"{r['retrieval'][label].get(f'recall@{k}', float('nan')):.4f}"
                    if retrieval
                    else _with_ci(r["ranking"][label], "gauc", digits=3)
                )
        out += [
            f"{results['tag']} ({results['config']}), label {label}",
            *_table([names, metrics], rows, markdown=False),
            "",
        ]
    return "\n".join(out)
