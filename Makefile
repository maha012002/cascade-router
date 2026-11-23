PY ?= python

.PHONY: setup data splits baselines eda lint format test test-unit mlflow-ui embed train eval serve

setup:            ## install pinned deps + the package in editable mode
	$(PY) -m pip install -r requirements-dev.txt -e .

data:             ## download RouterBench at the pinned revision and build tidy Parquet
	$(PY) -m router.download --config configs/data.yaml
	$(PY) -m router.data --config configs/data.yaml

splits:           ## near-duplicate groups + ID / leave-one-benchmark-out splits
	$(PY) -m router.splits --config configs/splits.yaml

baselines:        ## evaluate baselines + oracle on every split (logs to MLflow)
	$(PY) -m router.run_baselines --config configs/eval_baselines.yaml

mlflow-ui:        ## browse runs at http://127.0.0.1:5000
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

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

eval: baselines   ## learned routers join this target in M2

embed train serve:
	@echo "'make $@' is not implemented yet (see milestones M2-M4)" && exit 1
