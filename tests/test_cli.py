import json

import pandas as pd
import pytest
import yaml
from conftest import SMALL

from krec.cli import COMMANDS, PIPELINE, main
from krec.config import REPO_ROOT
from krec.data.tables import load_eval_set


def test_features_cover_every_eval_set_row(cfg):
    COMMANDS["features"](cfg)
    for name in ("train_standard", "val_random", "test_standard", "test_random"):
        feats = pd.read_parquet(cfg.processed_dir / "features" / f"{name}.parquet")
        assert len(feats) == len(load_eval_set(cfg, name))


def test_baselines_report_every_eval_set_and_model(cfg):
    COMMANDS["baselines"](cfg)
    results = json.loads((cfg.reports_path / "baselines" / "results.json").read_text())
    assert list(results["eval_sets"]) == cfg.evaluation.eval_sets
    for block in results["eval_sets"].values():
        assert set(block["models"]) == {
            "most_popular",
            "recent_popular_3d",
            "item_cooc",
            "random",
        }


def test_main_runs_full_pipeline_end_to_end(tmp_path, capsys):
    # the config is small enough to keep this test light
    small = tmp_path / "small.yaml"
    small.write_text(
        yaml.safe_dump(
            {"inherits": str(REPO_ROOT / "configs/synthetic.yaml"), "synthetic": SMALL}
        )
    )
    args = ["--config", str(small), "--root", str(tmp_path)]
    main(["synth", *args])
    main(["all", *args])
    printed = capsys.readouterr().out
    assert all(f"== {step} ==" in printed for step in PIPELINE)
    assert (tmp_path / "reports/synthetic/baselines/results.md").exists()


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["train"])
