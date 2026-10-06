"""Cross-version profiling, on a synthetic 27K with Pure and 1K derived from it."""

import os

import nbformat
import pandas as pd
import pytest
import yaml
from nbclient import NotebookClient

from krec import compare
from krec.config import REPO_ROOT, load_config
from krec.data.synthetic import derive_variants, generate
from krec.eval.metrics import gini
from krec.profile import Profile, build_profile

FULL_FILES = {
    "standard_logs": ["std_a.csv", "std_b.csv"],
    "random_logs": ["rnd.csv"],
    "users": "users.csv",
    "items": "items.csv",
}
SYNTH = {"n_users": 120, "n_items": 150, "n_authors": 30, "pool_fraction": 0.3}


def _write_config(root, variant, base, extra):
    path = root / f"{variant}.yaml"
    body = {"inherits": str(REPO_ROOT / "configs" / base), **extra}
    body["dataset"] = {
        "name": f"syn_{variant}",
        "variant": variant,
        "raw_dir": f"raw/{variant}",
        "processed_dir": f"processed/{variant}",
        **body.get("dataset", {}),
    }
    path.write_text(yaml.safe_dump(body))
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("versions")
    synth = {**load_config("configs/synthetic.yaml").synthetic.model_dump(), **SYNTH}
    paths = {
        "27k": _write_config(
            root,
            "27k",
            "27k.yaml",
            {"dataset": {"files": FULL_FILES}, "synthetic": synth},
        ),
        "pure": _write_config(root, "pure", "pure.yaml", {}),
        "1k": _write_config(root, "1k", "1k.yaml", {}),
    }
    cfgs = {v: load_config(p, root=root) for v, p in paths.items()}
    generate(cfgs["27k"])
    derive_variants(cfgs["27k"], cfgs["pure"], cfgs["1k"], n_users_1k=30)
    dropped = _drop_from_1k_random_log(cfgs)
    for cfg in cfgs.values():
        build_profile(cfg)
    profiles = {v: Profile.load(cfg) for v, cfg in cfgs.items()}
    return {
        "root": root,
        "paths": paths,
        "cfgs": cfgs,
        "profiles": profiles,
        "dropped": dropped,
    }


def _drop_from_1k_random_log(cfgs):
    """Leave a pool video out of 1K's random log, as in the real 1K."""
    # real 1K's random log covers 7,388 of 7,583 pool videos. Drop 1K users'
    # random rows for a video they saw in standard logs, in every release, so
    # it stays in 27K's pool through other users
    k1 = cfgs["1k"]
    users = set(pd.read_csv(k1.raw_dir / k1.dataset.files.users).user_id)
    std = pd.concat(pd.read_csv(k1.raw_dir / n) for n in k1.dataset.files.standard_logs)
    full = cfgs["27k"]
    rnd = pd.read_csv(full.raw_dir / full.dataset.files.random_logs[0])
    others = set(rnd[~rnd.user_id.isin(users)].video_id)
    video = sorted(others & set(std.video_id))[0]
    for cfg in cfgs.values():
        path = cfg.raw_dir / cfg.dataset.files.random_logs[0]
        df = pd.read_csv(path)
        df[~((df.video_id == video) & df.user_id.isin(users))].to_csv(path, index=False)
    return video


def _raw_logs(cfg):
    f = cfg.dataset.files
    parts = [
        pd.read_csv(cfg.raw_dir / n).assign(source=s)
        for s, names in (("standard", f.standard_logs), ("random", f.random_logs))
        for n in names
    ]
    return pd.concat(parts, ignore_index=True)


def test_clean_releases_have_no_contract_violations(world):
    table = compare.contracts(world["profiles"])
    assert list(table.columns) == ["source", "rows", "min_date", "max_date"]


def test_violations_are_counted_not_raised(world, tmp_path):
    src = world["cfgs"]["pure"]
    cfg = src.override(
        root=tmp_path, dataset={"raw_dir": "raw", "processed_dir": "processed"}
    )
    cfg.raw_dir.mkdir()
    for f in src.raw_dir.iterdir():
        (cfg.raw_dir / f.name).write_bytes(f.read_bytes())
    name = cfg.dataset.files.random_logs[0]
    logs = pd.read_csv(cfg.raw_dir / name)
    logs.loc[0, "is_rand"] = 0  # claims to be policy traffic
    logs["is_like"] = logs["is_like"].astype(object)
    logs.loc[1, "is_like"] = "x"  # unparseable
    logs.loc[2, "date"] = 20220101  # disagrees with time_ms
    logs.to_csv(cfg.raw_dir / name, index=False)
    build_profile(cfg)
    row = Profile.load(cfg)["files"].set_index("file").loc[name]
    assert row.is_rand_mismatch == 1
    assert row.date_time_mismatch == 1
    assert row.nonbinary_is_like == 1  # missing or unparseable counts too
    assert row.rows == len(logs)


def test_pure_is_exactly_27k_filtered_to_the_pool(world):
    p = world["profiles"]
    table = compare.pure_vs_full_pool(p)
    assert table.same_rows.all() and (table.row_diff == 0).all()
    identity = compare.random_log_identity(p)
    assert identity.loc["pure", "same_rows_as_27k"]
    assert identity.loc["pure", "same_ids_as_27k"]


def test_reindexed_ids_still_match_on_rows(world, tmp_path):
    """A release with the same rows but renumbered videos: rows match, IDs don't."""
    src = world["cfgs"]["pure"]
    out = tmp_path / "raw"
    out.mkdir()
    for f in src.raw_dir.iterdir():
        df = pd.read_csv(f)
        if "video_id" in df:
            df["video_id"] = df["video_id"] + 10_000
        df.to_csv(out / f.name, index=False)
    cfg = src.override(root=tmp_path, dataset={"raw_dir": out})
    build_profile(cfg)
    profiles = {**world["profiles"], "pure": Profile.load(cfg)}
    identity = compare.random_log_identity(profiles)
    assert identity.loc["pure", "same_rows_as_27k"]
    assert not identity.loc["pure", "same_ids_as_27k"]


def test_a_missing_day_in_pure_is_caught(world):
    p = dict(world["profiles"])
    pure = p["pure"]
    daily = pure["daily"]
    first = daily[daily.source == "standard"].day.min()
    trimmed = daily[~((daily.day == first) & (daily.source == "standard"))]
    p["pure"] = Profile(pure.name, pure.variant, {**pure.tables, "daily": trimmed})
    table = compare.pure_vs_full_pool(p)
    assert not table.loc[(first, "standard"), "same_rows"]
    assert table.loc[(first, "standard"), "row_diff"] < 0


def test_1k_matches_27k_user_by_user(world):
    row = compare.one_k_vs_full(world["profiles"]).iloc[0]
    assert row.users_in_1k == row.found_in_27k == row.users_with_identical_counts == 30


def test_pool_coverage_shows_what_1k_is_missing(world):
    table = compare.pool_coverage(world["profiles"])
    assert table.loc["pure", "27k_pool_missing"] == 0
    assert table.loc["1k", "27k_pool_missing"] >= 1
    assert table.loc["1k", "not_in_27k_pool"] == 0
    assert world["dropped"] not in set(world["profiles"]["1k"]["pool"].video_id)


def test_pool_upload_dates_match_the_items_file(world):
    cfg = world["cfgs"]["27k"]
    items = pd.read_csv(cfg.raw_dir / cfg.dataset.files.items)
    pool = world["profiles"]["27k"]["pool"]
    want = items.set_index("video_id").upload_dt.loc[pool.video_id]
    got = pd.to_datetime(pool.upload_dt).dt.strftime("%Y-%m-%d")
    assert got.tolist() == pd.to_datetime(want).dt.strftime("%Y-%m-%d").tolist()
    table = compare.pool_upload_dates(world["profiles"])
    assert table.videos.sum() == len(pool)


def test_clock_checks_find_a_shifted_date(world, tmp_path):
    src = world["cfgs"]["pure"]
    cfg = src.override(
        root=tmp_path, dataset={"raw_dir": "raw", "processed_dir": "processed"}
    )
    cfg.raw_dir.mkdir()
    for f in src.raw_dir.iterdir():
        (cfg.raw_dir / f.name).write_bytes(f.read_bytes())
    name = cfg.dataset.files.standard_logs[0]
    logs = pd.read_csv(cfg.raw_dir / name)
    shifted = logs.index[:3]
    days = pd.to_datetime(logs.loc[shifted, "date"].astype(str))
    logs.loc[shifted, "date"] = (
        (days + pd.Timedelta(days=1)).dt.strftime("%Y%m%d").astype(int)
    )
    logs.to_csv(cfg.raw_dir / name, index=False)
    build_profile(cfg)
    profiles = {"pure": Profile.load(cfg)}
    summary = compare.clock_summary(profiles).loc[("pure", name)]
    assert summary.mismatched == 3
    assert summary["offset_+1"] == 1.0
    assert summary.hourmin_hour_matches_time_ms == 1.0
    assert summary.hourmin_past_midnight == 0.0
    assert summary.hourmin_hour_matches_overall == 1.0
    hours = compare.clock_by_hour(profiles, "pure")[name]
    assert hours.sum() > 0
    parts = compare.part_files(profiles, "pure").loc[name]
    assert parts.users_with_mismatches == logs.loc[shifted, "user_id"].nunique()


def test_clean_releases_have_no_clock_mismatches(world):
    summary = compare.clock_summary(world["profiles"])
    assert (summary.mismatched == 0).all()


def test_part_files_tell_a_split_by_user_from_a_split_by_rows():
    files = pd.DataFrame(
        {
            "file": ["by_user_1", "by_user_2", "by_rows_1", "by_rows_2"],
            "source": "standard",
            "min_date": [1, 1, 2, 2],  # same dates -> sibling parts
            "users": 2,
            "min_user_id": [0, 2, 0, 0],
            "max_user_id": [1, 3, 1, 1],
        }
    )
    act = pd.DataFrame(
        {
            "file_id": [0, 0, 1, 1, 2, 2, 3, 3],
            "user_id": [0, 1, 2, 3, 0, 1, 0, 1],
            "rows": 10,
            "date_mismatches": [0, 0, 0, 0, 0, 0, 5, 0],
        }
    )
    p = Profile("x", "27k", {"files": files, "user_activity": act})
    parts = compare.part_files({"27k": p})
    assert parts.users_also_in_sibling.tolist() == [0, 0, 2, 2]
    assert parts.loc["by_rows_2", "users_with_mismatches"] == 1
    assert parts.loc["by_rows_2", "mismatch_rate_among_those_users"] == 0.5


def test_pool_share_matches_the_raw_logs(world):
    cfg = world["cfgs"]["27k"]
    logs = _raw_logs(cfg)
    pool = set(logs[logs.source == "random"].video_id)
    std = logs[logs.source == "standard"].assign(
        in_pool=lambda d: d.video_id.isin(pool)
    )
    want = std.groupby("user_id").in_pool.mean()
    got = compare.pool_share_per_user(world["profiles"])
    assert got.loc["27k", "overall_share"] == pytest.approx(std.in_pool.mean())
    assert got.loc["27k", "p50"] == pytest.approx(want.quantile(0.5))


def test_label_rates_and_concentration_match_the_raw_logs(world):
    cfg = world["cfgs"]["27k"]
    logs = _raw_logs(cfg)
    rates = compare.label_rates(world["profiles"])
    rnd = logs[logs.source == "random"]
    assert rates.loc[("27k", "random", True), "click_rate"] == pytest.approx(
        rnd.is_click.mean()
    )
    conc = compare.concentration(world["profiles"])
    counts = rnd.video_id.value_counts().to_numpy()
    assert conc.loc[("27k", "random", True), "gini"] == pytest.approx(gini(counts))


def test_pool_typicality_shares_sum_to_one(world):
    for attribute in ("duration", "age_at_impression", "video_type"):
        table = compare.pool_typicality(world["profiles"], attribute)
        assert table.pool_share.sum() == pytest.approx(1.0)
        assert table.non_pool_share.sum() == pytest.approx(1.0)


def test_daily_volume_covers_every_scope(world):
    volume = compare.daily_volume_by_scope(world["profiles"])
    full = volume[(volume.version == "27k") & (volume.source == "standard")]
    by_scope = full.groupby("scope").rows.sum()
    assert by_scope["all"] == by_scope["pool"] + by_scope["non-pool"]
    fig = compare.daily_volume_figure(volume)
    assert len(fig.axes[0].lines) >= 3


def test_user_sample_compares_both_versions(world):
    table = compare.user_sample(world["profiles"])
    assert list(table.columns) == ["27k", "1k"]
    assert table.loc["users", "1k"] == 30


def test_versions_notebook_runs_end_to_end(world, monkeypatch):
    configs = ",".join(str(world["paths"][v]) for v in ("27k", "pure", "1k"))
    monkeypatch.setenv("KREC_VERSION_CONFIGS", configs)
    monkeypatch.setenv("KREC_ROOT", str(world["root"]))
    nb = nbformat.read(REPO_ROOT / "notebooks/02_versions.ipynb", as_version=4)
    NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    ).execute(env=dict(os.environ))
    outputs = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs]
    assert not [o for o in outputs if o.output_type == "error"]
    assert any("image/png" in o.get("data", {}) for o in outputs)
