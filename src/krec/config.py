"""
YAML files are parsed into the Pydantic models below.

A config file may name one parent with `inherits: other.yaml`. The child is
deep-merged over the parent before validation.
"""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

Source = Literal["standard", "random"]
SplitName = Literal["train", "val", "test"]
EvalSetName = Literal[
    "train_standard",
    "train_random",
    "val_standard",
    "val_random",
    "test_standard",
    "test_random",
]
# Binary feedback columns that can serve as labels
Label = Literal[
    "is_click",
    "long_view",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


class DatasetFiles(_Strict):
    standard_logs: list[str] = Field(min_length=1)
    random_logs: list[str] = Field(min_length=1)
    users: str
    items: str


class DatasetConfig(_Strict):
    name: str
    # Which KuaiRand release this is
    variant: Literal["pure", "1k", "27k"] = "pure"
    raw_dir: Path
    processed_dir: Path
    files: DatasetFiles


class SplitConfig(_Strict):
    """Inclusive [start, end] date ranges. YAML: `train: ["2022-04-08", "2022-04-30"]`."""

    train: tuple[date, date]
    val: tuple[date, date]
    test: tuple[date, date]

    @model_validator(mode="after")
    def _ordered_and_disjoint(self) -> SplitConfig:
        for name in ("train", "val", "test"):
            start, end = getattr(self, name)
            if start > end:
                raise ValueError(f"split.{name}: start {start} is after end {end}")
        if not self.train[1] < self.val[0]:
            raise ValueError("split: train must end before val starts")
        if not self.val[1] < self.test[0]:
            raise ValueError("split: val must end before test starts")
        return self

    def start(self, split: SplitName) -> date:
        return getattr(self, split)[0]


class FeaturesConfig(_Strict):
    sources: list[Source] = Field(min_length=1)
    windows_days: list[PositiveInt] = Field(min_length=1)
    ctr_prior_strength: float = Field(gt=0)
    leaky_columns: list[str]
    duckdb_memory_limit: str = Field(pattern=r"^\d+(\.\d+)?\s*(KB|MB|GB|TB)$")

    @field_validator("windows_days")
    @classmethod
    def _unique_windows(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError("windows_days must not repeat")
        return sorted(v)


class RetrievalConfig(_Strict):
    users: Literal["warm", "all"]
    exclude_fit_positives: bool
    diversity_k: PositiveInt
    batch_users: PositiveInt
    # cap on users x catalog for models that can only score the whole catalog
    max_dense_scores: PositiveInt


class RankingConfig(_Strict):
    group_by: Literal["user", "user_date"]
    gauc_weight: Literal["impressions", "uniform"]


class EvaluationConfig(_Strict):
    labels: list[Label] = Field(min_length=1)
    ks: list[PositiveInt] = Field(min_length=1)
    headline_k: PositiveInt  # the K shown in the results table; must be one of `ks`
    ranking_ks: list[PositiveInt] = Field(min_length=1)
    eval_sets: list[EvalSetName] = Field(min_length=1)
    retrieval: RetrievalConfig
    ranking: RankingConfig
    bootstrap_samples: int = Field(ge=0)  # 0 = no confidence intervals
    seed: int

    @model_validator(mode="after")
    def _headline_in_ks(self) -> EvaluationConfig:
        if self.headline_k not in self.ks:
            raise ValueError(
                f"headline_k={self.headline_k} must be one of ks={self.ks}"
            )
        return self


class PopularityParams(_Strict):
    signal: Label = "is_click"
    window_days: PositiveInt | None = None


class RecentPopularityParams(PopularityParams):
    window_days: PositiveInt  # required here


class ItemCoocParams(_Strict):
    signal: Label = "is_click"
    top_neighbors: PositiveInt
    shrinkage: float = Field(ge=0)
    max_history: PositiveInt  # most recent positives per user


class RandomParams(_Strict):
    pass


class BaselinesConfig(_Strict):
    """Which baselines to run. Omit a key to skip that model."""

    most_popular: PopularityParams | None = None
    recent_popular: RecentPopularityParams | None = None
    item_cooc: ItemCoocParams | None = None
    random: RandomParams | None = None

    def enabled(self) -> dict[str, BaseModel]:
        """Configured baselines in declaration order."""
        return {
            name: p
            for name in type(self).model_fields
            if (p := getattr(self, name)) is not None
        }


class SyntheticConfig(_Strict):
    n_users: PositiveInt
    n_items: PositiveInt
    n_authors: PositiveInt
    latent_dim: PositiveInt
    mean_standard_per_user_day: float = Field(gt=0)
    mean_random_per_user_day: float = Field(gt=0)
    seed: int
    # Share of items in the random-exposure candidate pool (1.0 = all items).
    pool_fraction: float = Field(default=1.0, gt=0, le=1)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class Config(_Strict):
    dataset: DatasetConfig
    split: SplitConfig
    features: FeaturesConfig
    evaluation: EvaluationConfig
    baselines: BaselinesConfig
    reports_dir: Path
    synthetic: SyntheticConfig | None = None  # only for the synthetic dataset
    root: Path = REPO_ROOT  # relative paths resolve against this

    def path(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    @property
    def raw_dir(self) -> Path:
        return self.path(self.dataset.raw_dir)

    @property
    def processed_dir(self) -> Path:
        return self.path(self.dataset.processed_dir)

    @property
    def reports_path(self) -> Path:
        return self.path(self.reports_dir)

    def override(self, **overrides: Any) -> Config:
        """A new, re-validated config with `overrides` deep-merged in."""
        data = _deep_merge(self.model_dump(mode="python"), overrides)
        return Config.model_validate(data)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_with_parents(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("inherits", None)
    if parent is None:
        return raw
    return _deep_merge(_read_with_parents(path.parent / parent), raw)


def load_config(
    path: str | Path, root: str | Path | None = None, **overrides: Any
) -> Config:
    """Load, merge (`inherits`, then keyword overrides) and validate a config.

    `root` is the folder relative paths resolve against (default: the repo).
    Keyword overrides use the same nesting as the YAML, e.g.
    `load_config("configs/synthetic.yaml", synthetic={"n_users": 100})`.
    """
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    data = _deep_merge(_read_with_parents(path), overrides)
    if root is not None:
        data["root"] = Path(root)
    return Config.model_validate(data)
