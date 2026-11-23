import numpy as np
import pandas as pd
import pytest

from router.baselines import (
    mixture_curve,
    oracle_curve,
    pick_best,
    pick_cheapest,
    pick_hull_models,
)
from router.evaluate import operating_point, quality_at_cost, sweep, upper_hull
from router.policy import route

LAMBDAS = [0.0, 1.0, 10.0, 100.0, 1e3, 1e4, np.inf]


def _random_problem(seed: int, n: int = 60, m: int = 5, graded: bool = False):
    rng = np.random.default_rng(seed)
    base_cost = np.sort(rng.uniform(1e-4, 5e-3, size=m))
    cost = base_cost * rng.uniform(0.5, 1.5, size=(n, m))
    p_correct = np.linspace(0.3, 0.8, m)
    if graded:
        quality = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=(n, m))
    else:
        quality = (rng.random((n, m)) < p_correct).astype(float)
    return quality, cost, rng


def _oracle_dominates(oracle: np.ndarray, cost: float, quality: float) -> bool:
    return quality <= quality_at_cost(oracle, cost) + 1e-12


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("graded", [False, True])
def test_oracle_beats_every_router_at_equal_cost(seed, graded):
    quality, cost, rng = _random_problem(seed, graded=graded)
    oracle = oracle_curve(quality, cost)

    for _ in range(200):  # arbitrary deterministic routers
        c, q = operating_point(rng.integers(0, quality.shape[1], len(quality)), quality, cost)
        assert _oracle_dominates(oracle, c, q)

    for _ in range(20):  # lambda sweeps over noisy predictions
        p_hat = np.clip(quality + rng.normal(0, 0.4, quality.shape), 0, 1)
        pts = sweep(p_hat, cost.mean(axis=0), quality, cost, LAMBDAS)
        for c, q in zip(pts["cost"], pts["quality"], strict=True):
            assert _oracle_dominates(oracle, c, q)


@pytest.mark.parametrize("seed", range(3))
def test_oracle_curve_matches_policy_with_true_labels(seed):
    """Routing on TRUE quality/cost at any lambda lands exactly on the oracle curve."""
    quality, cost, _ = _random_problem(seed)
    oracle = oracle_curve(quality, cost)
    for lam in LAMBDAS:
        c, q = operating_point(route(quality, cost, lam), quality, cost)
        assert q == pytest.approx(float(quality_at_cost(oracle, c)), abs=1e-9)


def test_oracle_endpoints():
    quality, cost, _ = _random_problem(0)
    oracle = oracle_curve(quality, cost)
    assert oracle[0, 0] == pytest.approx(cost.min(axis=1).mean())
    assert oracle[-1, 1] == pytest.approx(quality.max(axis=1).mean())


def test_pick_cheapest_and_best_use_given_table():
    train = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "cost": [1.0, 1.0, 5.0, 5.0],
            "quality": [0.2, 0.4, 0.9, 0.7],
        }
    )
    assert pick_cheapest(train) == "a"
    assert pick_best(train) == "b"


def test_pick_hull_models_uses_train_points():
    train = pd.DataFrame(
        {
            "model": ["a", "b", "c", "d"],
            "cost": [1.0, 2.0, 3.0, 4.0],
            "quality": [0.2, 0.25, 0.8, 0.7],  # b below the a->c line, d dominated by c
        }
    )
    assert pick_hull_models(train) == ["a", "c"]


def test_random_mixture_is_straight_line_between_two_models():
    points = pd.DataFrame({"model": ["a", "b", "c"], "cost": [1, 3, 2], "quality": [0.2, 0.6, 0.1]})
    curve = mixture_curve(points, ["a", "b"])
    np.testing.assert_allclose(curve, upper_hull(np.array([1, 3]), np.array([0.2, 0.6])))
    assert quality_at_cost(curve, 2.0) == pytest.approx(0.4)
