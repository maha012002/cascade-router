"""Unit tests for paired bootstrap helpers (no downloaded data)."""

import numpy as np

from router.bootstrap import curve_zero_router, paired_bootstrap, reference_points


def test_reference_points_and_zero_curve():
    quality = np.array([[0.2, 0.8], [0.3, 0.7], [0.1, 0.9]], dtype=float)
    cost = np.array([[0.001, 0.01], [0.001, 0.01], [0.001, 0.01]], dtype=float)
    models = ["cheap", "best"]
    c_lo, c_hi, budgets, target = reference_points(
        quality, cost, models, "cheap", "best", {"b50": 0.5}, 0.95
    )
    assert c_lo == 0.001
    assert c_hi == 0.01
    assert budgets["b50"] == 0.005
    assert abs(target - 0.95 * quality[:, 1].mean()) < 1e-12
    curve = curve_zero_router(quality, cost, models, ["cheap", "best"])
    assert curve.shape[1] == 2
    assert curve[0, 0] <= curve[-1, 0]


def test_paired_bootstrap_contains_zero_for_identical_methods():
    rng = np.random.default_rng(0)
    n, m = 40, 3
    quality = rng.random((n, m))
    cost = 0.001 + 0.01 * np.arange(m)[None, :] + 0.0001 * rng.random((n, m))
    models = [f"m{i}" for i in range(m)]
    p_hat = quality.copy()
    cost_est = cost.copy()
    methods = {
        "lightgbm": {"kind": "predictions", "p_hat": p_hat, "cost_est": cost_est},
        "twin": {"kind": "predictions", "p_hat": p_hat.copy(), "cost_est": cost_est.copy()},
        "zero_router": {"kind": "zero_router", "hull_models": models},
    }
    res = paired_bootstrap(
        quality=quality,
        cost=cost,
        models=models,
        cheapest=models[0],
        best=models[-1],
        budget_fracs={"b50": 0.5},
        target_frac=0.95,
        lambdas=[0.0, 100.0, float("inf")],
        methods=methods,
        n_boot=50,
        seed=1,
    )
    delta = res[
        (res["contrast"] == "lightgbm_minus_twin") & (res["metric"] == "auc")
    ].iloc[0]
    assert abs(delta["mean"]) < 1e-12
    assert delta["contains_zero"] is True
