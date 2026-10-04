import os

import nbformat
import numpy as np
import pandas as pd
import pytest
from nbclient import NotebookClient

from krec import eda
from krec.config import REPO_ROOT


@pytest.fixture(scope="module")
def logs(cfg):
    df = pd.read_parquet(cfg.processed_dir / "interactions.parquet")
    return df.assign(date=pd.to_datetime(df.date))


def _overlap(logs):
    return logs[logs.date >= logs[logs.source == "random"].date.min()]


def test_overview_counts_match_the_logs(cfg, logs):
    got = eda.overview(cfg).set_index(["source", "split"])
    want = logs.groupby(["source", "split"]).size()
    assert got["rows"].sort_index().tolist() == want.sort_index().tolist()


def test_feedback_rates_use_only_days_both_logs_cover(cfg, logs):
    got = eda.feedback_rates(cfg)
    want = _overlap(logs).groupby("source").is_click.mean()
    assert got.click_rate.to_dict() == pytest.approx(want.to_dict())


def test_item_exposure_counts_unshown_items_as_zero(cfg, logs):
    exposure = eda.item_exposure(cfg)
    overlap = _overlap(logs)
    n_items = overlap.item_id.nunique()
    for source, counts in exposure.items():
        assert len(counts) == n_items
        assert counts.sum() == (overlap.source == source).sum()


def test_concentration_and_lorenz_on_known_input():
    exposure = {"standard": np.r_[np.zeros(99), 100.0], "random": np.ones(100)}
    table = eda.exposure_concentration(exposure)
    assert table.loc["standard", "top_1pct_share"] == 1.0
    assert table.loc["random", "gini"] == pytest.approx(0.0)
    fig = eda.lorenz_figure(exposure)
    for line in fig.axes[0].lines[:2]:  # each curve runs from (0, 0) to (1, 1)
        x, y = line.get_data()
        assert (x[0], y[0], x[-1], y[-1]) == (0, 0, 1, 1)


def test_exposure_vs_quality_buckets_items(cfg):
    table = eda.exposure_vs_quality(cfg, min_random_impressions=5, buckets=3)
    assert list(table.index) == [1, 2, 3]
    assert table["random_click_rate"].between(0, 1).all()


def test_sparsity_density_is_a_share(cfg):
    assert eda.sparsity(cfg).density.between(0, 1).all()


def test_daily_volume_matches_the_logs(cfg, logs):
    volume = eda.daily_volume(cfg)
    want = logs.groupby([logs.date.dt.date, "source"]).size().unstack(fill_value=0)
    assert volume.to_numpy().sum() == len(logs)
    assert (volume["random"].to_numpy() == want["random"].to_numpy()).all()
    assert len(eda.daily_volume_figure(volume).axes[0].lines) == 2


def test_tab_mix_shares_sum_to_one_per_source(cfg, logs):
    mix = eda.tab_mix(cfg)
    assert mix.standard_share.sum() == pytest.approx(1.0)
    assert mix.random_share.sum() == pytest.approx(1.0)
    overlap = _overlap(logs)
    tab0 = overlap[(overlap.tab == 0) & (overlap.source == "random")]
    assert mix.loc[0, "random_click_rate"] == pytest.approx(tab0.is_click.mean())


def test_within_user_rates_weight_each_user_once(cfg, logs):
    got = eda.within_user_rates(cfg, labels=("is_click",), min_each=3)
    overlap = _overlap(logs)
    per_user = overlap.groupby(["user_id", "source"]).is_click.agg(["size", "mean"])
    wide = per_user.unstack()
    both = wide[(wide["size"] >= 3).all(axis=1)]
    row = got.loc["is_click"]
    assert row.users == len(both)
    assert row.standard_rate == pytest.approx(both["mean"]["standard"].mean())
    assert row.mean_difference == pytest.approx(
        (both["mean"]["standard"] - both["mean"]["random"]).mean()
    )


def test_eda_notebook_runs_end_to_end(cfg, monkeypatch):
    """Execute the notebook on the synthetic data, so it can't silently break."""
    monkeypatch.setenv("KREC_CONFIG", "configs/synthetic.yaml")
    monkeypatch.setenv("KREC_ROOT", str(cfg.root))
    nb = nbformat.read(REPO_ROOT / "notebooks/01_eda.ipynb", as_version=4)
    NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    ).execute(env=dict(os.environ))
    outputs = [
        out for cell in nb.cells if cell.cell_type == "code" for out in cell.outputs
    ]
    assert not [out for out in outputs if out.output_type == "error"]
    # The Lorenz chart must render as an image, not just a "<Figure ...>" repr.
    assert any("image/png" in out.get("data", {}) for out in outputs)
