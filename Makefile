# Dev tasks only. The pipeline itself runs through the `krec` command
# (see README): krec ingest | features | baselines | all, krec synth.
PY ?= python

.PHONY: help install format lint test check

help:     ## List targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

install:  ## Install the package with dev tools
	$(PY) -m pip install -e ".[dev]"

format:   ## Apply Black and Ruff's safe fixes
	black src tests
	ruff check --fix src tests

lint:     ## Ruff + Black check (no changes)
	ruff check src tests
	black --check src tests

test:     ## Run the test suite (includes executing the EDA notebook)
	$(PY) -m pytest

check: lint test  ## Everything CI runs
