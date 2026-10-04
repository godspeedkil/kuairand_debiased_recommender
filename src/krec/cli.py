"""Command line entry point: `krec <command> --config configs/pure.yaml`."""

from __future__ import annotations

import argparse
import json
import time

from krec.config import load_config


def cmd_synth(cfg) -> None:
    from krec.data.synthetic import generate

    print(f"wrote synthetic raw files to {generate(cfg)}")


def cmd_ingest(cfg) -> None:
    from krec.data.ingest import ingest

    print(json.dumps(ingest(cfg), indent=2))


def cmd_features(cfg) -> None:
    """Materialize point-in-time feature frames for every split/source."""
    from krec.data.tables import load_eval_set
    from krec.features.pipeline import write_historical_features

    out = cfg.processed_dir / "features"
    out.mkdir(parents=True, exist_ok=True)
    keep = [
        "user_id",
        "item_id",
        "date",
        "time_ms",
        "tab",
        "source",
        *cfg.evaluation.labels,
        "is_like",
        "play_time_ms",
        "duration_ms",
    ]
    for name in (
        "train_standard",
        "train_random",
        "val_standard",
        "val_random",
        "test_standard",
        "test_random",
    ):
        t0 = time.perf_counter()
        ent = load_eval_set(cfg, name)[keep]  # entity rows + labels, carried through
        if ent.empty:
            continue
        n = write_historical_features(cfg, ent, out / f"{name}.parquet")
        del ent
        print(f"{name}: {n:,} rows ({time.perf_counter() - t0:.1f}s)")


def cmd_baselines(cfg) -> None:
    from krec.models.baselines import build_baselines
    from krec.run import evaluate_models, write_results

    seed = cfg.evaluation.seed
    results = evaluate_models(
        cfg, lambda: build_baselines(cfg.baselines, seed), "baselines"
    )
    path = write_results(
        results, cfg.reports_path / "baselines", cfg.evaluation.headline_k
    )
    print(path.read_text())


COMMANDS = {
    "synth": cmd_synth,
    "ingest": cmd_ingest,
    "features": cmd_features,
    "baselines": cmd_baselines,
}
# What `krec all` runs, in order. `synth` is separate: real data is downloaded.
# EDA lives in notebooks/01_eda.ipynb, which reads what `ingest` writes.
PIPELINE = ["ingest", "features", "baselines"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="krec", description=__doc__)
    ap.add_argument("command", choices=[*COMMANDS, "all"])
    ap.add_argument("--config", default="configs/pure.yaml")
    ap.add_argument(
        "--root",
        default=None,
        help="folder that data/ and reports/ paths resolve against (default: repo)",
    )
    args = ap.parse_args(argv)
    cfg = load_config(args.config, root=args.root)
    for step in PIPELINE if args.command == "all" else [args.command]:
        print(f"== {step} ==")
        COMMANDS[step](cfg)


if __name__ == "__main__":
    main()
