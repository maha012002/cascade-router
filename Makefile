PY ?= python

.PHONY: setup data eda lint format test test-unit embed train eval serve

setup:            ## install pinned deps + the package in editable mode
	$(PY) -m pip install -r requirements-dev.txt -e .

data:             ## download RouterBench at the pinned revision and build tidy Parquet
	$(PY) -m router.download --config configs/data.yaml
	$(PY) -m router.data --config configs/data.yaml

eda:              ## re-execute the EDA notebook in place (needs `make data`)
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

test:             ## all tests (integration tests skip if data is missing)
	$(PY) -m pytest

test-unit:        ## tests that need no downloaded data (what CI runs)
	$(PY) -m pytest -m "not integration"

embed train eval serve:
	@echo "'make $@' is not implemented yet (see milestones M1-M4)" && exit 1
