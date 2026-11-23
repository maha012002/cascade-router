"""API contract tests with a tiny synthetic serving artifact (no MiniLM download)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from router.models import CostModel, LightGBMPredictor
from router.serve.app import create_app
from router.serve.artifact import features_for_prompt, save_artifact
from router.serve.runtime import RouterRuntime


@pytest.fixture
def client(tmp_path):
    """Throwaway LightGBM bundle + TestClient (embed mocked; no Hub download)."""
    rng = np.random.default_rng(0)
    n, d, m = 120, 20, 3
    X = rng.normal(size=(n, d)).astype(np.float32)
    Y = 1 / (1 + np.exp(-X[:, :m]))
    pred = LightGBMPredictor(num_boost_round=20, seed=0)
    pred.models_ = []
    for j in range(m):
        ds = lgb.Dataset(X, label=Y[:, j])
        pred.models_.append(
            lgb.train(
                {"objective": "cross_entropy", "verbosity": -1, "num_threads": 1, "seed": 0},
                ds,
                num_boost_round=20,
            )
        )

    chars = rng.integers(50, 500, size=n).astype(float)
    costs = np.column_stack([chars * c for c in (1e-7, 5e-7, 2e-6)])
    cost_model = CostModel().fit(chars, costs)
    path = tmp_path / "bundle"
    save_artifact(
        path,
        predictor=pred,
        cost_model=cost_model,
        models=["cheap", "mid", "expensive"],
        meta={
            "router_version": "test-v0",
            "default_lambda": 10.0,
            "embedding_model": "fake",
            "embedding_revision": "0" * 40,
            "n_features": d,
        },
    )

    rt = RouterRuntime(path, enable_cache=True, embed_fn=lambda p: np.zeros(d, dtype=np.float32))

    def _predict(prompt: str):
        x = np.zeros((1, d), dtype=np.float32)
        return rt.predictor.predict(x)[0], rt.cost_model.predict(np.array([len(prompt)]))[0]

    rt.predict_quality_and_cost = _predict  # type: ignore[method-assign]
    with TestClient(create_app(runtime=rt)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["router_version"] == "test-v0"
    assert body["n_models"] == 3


def test_route_contract(client):
    r = client.post("/route", json={"prompt": "What is 2+2?", "lambda": 10.0})
    assert r.status_code == 200
    body = r.json()
    assert body["chosen_model"] in {"cheap", "mid", "expensive"}
    assert set(body["per_model_p_correct"]) == {"cheap", "mid", "expensive"}
    assert set(body["expected_cost"]) == {"cheap", "mid", "expensive"}
    assert body["router_version"] == "test-v0"
    assert body["lambda_used"] == 10.0
    assert body["budget_used"] is None


def test_empty_prompt_422(client):
    assert client.post("/route", json={"prompt": ""}).status_code == 422


def test_lambda_and_budget_together_422(client):
    r = client.post("/route", json={"prompt": "hi", "lambda": 1.0, "budget": 0.01})
    assert r.status_code == 422


def test_budget_route(client):
    r = client.post("/route", json={"prompt": "code please", "budget": 1e-9})
    assert r.status_code == 200
    assert r.json()["budget_used"] == 1e-9
    assert r.json()["lambda_used"] is None


def test_cache_hit(client):
    payload = {"prompt": "cache me", "lambda": 5.0}
    a = client.post("/route", json=payload).json()
    b = client.post("/route", json=payload).json()
    assert a["chosen_model"] == b["chosen_model"]
    assert a["cached"] is False
    assert b["cached"] is True


def test_features_for_prompt_width():
    emb = np.zeros(384, dtype=np.float32)
    x = features_for_prompt("def foo():\n  return 1", emb)
    assert x.shape == (1, 384 + 9)
