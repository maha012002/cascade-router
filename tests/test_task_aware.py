import numpy as np
import pandas as pd
import pytest

from router.baselines import fit_task_aware_hulls, task_aware_zero_router_curve


def _toy_train() -> pd.DataFrame:
    """Two sources with different best cheap models."""
    rows = []
    # source alpha: cheap model a is weak, mid b is strong
    for i in range(20):
        for model, q, c in [("a", 0.2, 0.001), ("b", 0.8, 0.002), ("c", 0.85, 0.01)]:
            rows.append(
                {
                    "prompt_id": f"alpha-{i}",
                    "source": "alpha",
                    "model": model,
                    "quality": q,
                    "cost": c,
                }
            )
    # source beta: cheap a is already strong
    for i in range(20):
        for model, q, c in [("a", 0.7, 0.001), ("b", 0.72, 0.002), ("c", 0.9, 0.01)]:
            rows.append(
                {
                    "prompt_id": f"beta-{i}",
                    "source": "beta",
                    "model": model,
                    "quality": q,
                    "cost": c,
                }
            )
    return pd.DataFrame(rows)


def test_fit_task_aware_hulls_are_per_source():
    train = _toy_train()
    hulls, means, global_hull, global_means = fit_task_aware_hulls(train)
    assert set(hulls) == {"alpha", "beta"}
    assert "a" in hulls["beta"]  # cheap+strong should sit on beta hull
    assert len(global_hull) >= 2
    assert list(global_means.index) == ["a", "b", "c"]


def test_lobo_held_out_source_falls_back_to_global():
    train = _toy_train()
    hulls, means, global_hull, global_means = fit_task_aware_hulls(train)
    # Remove gamma from train hulls (simulate LOBO): test prompts labeled gamma.
    models = ["a", "b", "c"]
    n = 6
    quality = np.array([[0.2, 0.5, 0.9]] * n, dtype=float)
    cost = np.array([[0.001, 0.002, 0.01]] * n, dtype=float)
    sources = ["gamma"] * n  # never seen in train
    curve, n_fallback = task_aware_zero_router_curve(
        quality,
        cost,
        sources,
        models,
        hulls,
        means,
        global_hull,
        global_means,
        lambdas=[0.0, 100.0, float("inf")],
    )
    assert n_fallback == n
    assert curve.ndim == 2 and curve.shape[0] >= 1


def test_task_aware_uses_source_hull_not_global_when_present():
    train = _toy_train()
    hulls, means, global_hull, global_means = fit_task_aware_hulls(train)
    models = ["a", "b", "c"]
    # One alpha prompt: at λ=0 should prefer high train quality on alpha hull.
    quality = np.array([[0.1, 0.5, 0.99]], dtype=float)
    cost = np.array([[0.001, 0.002, 0.01]], dtype=float)
    curve, n_fallback = task_aware_zero_router_curve(
        quality,
        cost,
        ["alpha"],
        models,
        hulls,
        means,
        global_hull,
        global_means,
        lambdas=[0.0],
    )
    assert n_fallback == 0
    # Realised quality at λ=0 equals the test quality of the train-best hull model on alpha.
    alpha_best = means["alpha"]["quality"].idxmax()
    expected_q = quality[0, models.index(alpha_best)]
    assert curve[-1, 1] == pytest.approx(expected_q)
