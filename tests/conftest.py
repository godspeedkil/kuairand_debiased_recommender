from __future__ import annotations

import pytest

from krec.config import load_config
from krec.data.ingest import ingest
from krec.data.synthetic import generate

SMALL = {"n_users": 150, "n_items": 120, "n_authors": 40,
         "mean_standard_per_user_day": 5.0, "mean_random_per_user_day": 2.0, "seed": 3}


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    """A tiny synthetic dataset, generated and ingested once per test session."""
    root = tmp_path_factory.mktemp("krec")
    c = load_config("configs/synthetic.yaml", root=root, synthetic=SMALL)
    generate(c)
    ingest(c)
    return c
