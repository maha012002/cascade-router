PY ?= python

.PHONY: setup data splits baselines eda lint format test test-unit mlflow-ui embed train calibrate errors eval artifact serve bench docker-build docker-run

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

embed:            ## MiniLM embeddings + handcrafted features (CPU, RAM-gated)
	$(PY) -m router.features --config configs/features.yaml

train:            ## train + evaluate learned routers on every split (RAM-gated)
	$(PY) -m router.run_routers --config configs/routers.yaml

task-aware:       ## M6: task-aware zero router vs zero_router / LightGBM
	$(PY) -m router.run_task_aware --config configs/task_aware.yaml

bootstrap:        ## M6: paired bootstrap 95% CIs (needs artifacts/predictions)
	$(PY) -m router.run_bootstrap --config configs/bootstrap.yaml

calibrate:        ## ECE before/after isotonic + temperature; routing impact
	$(PY) -m router.run_calibration --config configs/calibration.yaml

errors:           ## router vs oracle error analysis (source/length/domain)
	$(PY) -m router.run_error_analysis --config configs/error_analysis.yaml

eval: baselines train calibrate errors

artifact:         ## train LightGBM on ID and write artifacts/serve/<version>
	$(PY) -m router.serve.export_artifact --config configs/serve.yaml

serve:            ## run FastAPI locally (needs `make artifact` once)
	$(PY) -m uvicorn router.serve.app:app --host 127.0.0.1 --port 8000

bench:            ## CPU latency p50/p95, n=1000 (needs artifact)
	$(PY) scripts/bench_latency.py --config configs/serve.yaml --n 1000 --warmup 20

docker-build:     ## build the serving image (needs artifact on disk)
	docker build -t cascade-router:latest .

docker-run:       ## one-command run on :8000
	docker compose up --build
