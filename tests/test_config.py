import pydantic
import pytest
import yaml

from krec.config import REPO_ROOT, load_config


def test_child_config_inherits_and_overrides(tmp_path):
    cfg = load_config("configs/synthetic.yaml", root=tmp_path)
    # inherited from pure.yaml
    assert [str(d) for d in cfg.split.test] == ["2022-05-05", "2022-05-08"]
    assert "visible_status" in cfg.features.leaky_columns
    # overridden by synthetic.yaml
    assert cfg.dataset.name == "synthetic"
    assert cfg.dataset.files.users == "user_features_pure.csv"


def test_paths_resolve_against_root_and_kwargs_override(tmp_path):
    cfg = load_config("configs/synthetic.yaml", root=tmp_path, synthetic={"n_users": 7})
    assert cfg.raw_dir == tmp_path / "data/raw/synthetic/data"
    assert cfg.synthetic.n_users == 7
    assert cfg.synthetic.n_items == 400


def test_typo_in_key_fails_at_load_with_its_path():
    with pytest.raises(pydantic.ValidationError, match=r"features\.window_days"):
        load_config("configs/pure.yaml", features={"window_days": [3]})


def test_invalid_values_fail_at_load():
    with pytest.raises(pydantic.ValidationError, match="eval_sets"):
        load_config("configs/pure.yaml", evaluation={"eval_sets": ["test_rand"]})
    with pytest.raises(pydantic.ValidationError, match="labels"):
        load_config("configs/pure.yaml", evaluation={"labels": ["clicks"]})
    with pytest.raises(pydantic.ValidationError, match="windows_days"):
        load_config("configs/pure.yaml", features={"windows_days": [7, 7]})
    with pytest.raises(pydantic.ValidationError, match="headline_k"):
        load_config("configs/pure.yaml", evaluation={"headline_k": 20})


def test_overlapping_splits_are_rejected():
    with pytest.raises(pydantic.ValidationError, match="val must end before test"):
        load_config("configs/pure.yaml", split={"val": ["2022-05-01", "2022-05-05"]})
    with pytest.raises(pydantic.ValidationError, match="start .* is after end"):
        load_config("configs/pure.yaml", split={"test": ["2022-05-08", "2022-05-05"]})


def test_missing_section_is_rejected(tmp_path):
    raw = yaml.safe_load((REPO_ROOT / "configs/pure.yaml").read_text())
    del raw["split"]
    path = tmp_path / "no_split.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(pydantic.ValidationError, match="split"):
        load_config(path)


def test_config_is_frozen_and_override_revalidates():
    cfg = load_config("configs/pure.yaml")
    with pytest.raises(pydantic.ValidationError):
        cfg.evaluation.seed = 3
    assert cfg.override(evaluation={"seed": 3}).evaluation.seed == 3
    assert cfg.evaluation.seed == 17  # original untouched
    with pytest.raises(pydantic.ValidationError):
        cfg.override(evaluation={"ks": [0]})


def test_baselines_are_typed_and_ordered():
    cfg = load_config("configs/pure.yaml")
    assert list(cfg.baselines.enabled()) == [
        "most_popular",
        "recent_popular",
        "item_cooc",
        "random",
    ]
    with pytest.raises(pydantic.ValidationError, match="recent_popular.window_days"):
        load_config(
            "configs/pure.yaml", baselines={"recent_popular": {"window_days": None}}
        )
