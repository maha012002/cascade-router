import numpy as np
import pytest

from router.evaluate import (
    area_under_curve,
    cost_to_reach,
    operating_point,
    quality_at_cost,
    summarize,
    upper_hull,
)


def test_upper_hull_drops_dominated_and_concave_violations():
    costs = np.array([1.0, 2.0, 3.0, 4.0, 2.5])
    quals = np.array([0.0, 0.2, 0.9, 0.8, 0.1])
    # (2, .2) lies below the line (1,0)->(3,.9); (4, .8) and (2.5, .1) are dominated.
    np.testing.assert_allclose(upper_hull(costs, quals), [[1.0, 0.0], [3.0, 0.9]])


def test_upper_hull_single_point():
    np.testing.assert_allclose(upper_hull(np.array([2.0]), np.array([0.5])), [[2.0, 0.5]])


def test_quality_at_cost_interpolates_and_clamps():
    curve = np.array([[1.0, 0.2], [3.0, 0.6]])
    assert quality_at_cost(curve, 2.0) == pytest.approx(0.4)
    assert quality_at_cost(curve, 0.5) == 0.0  # cannot operate below cheapest point
    assert quality_at_cost(curve, 10.0) == pytest.approx(0.6)  # spending less is allowed


def test_area_under_curve_matches_hand_computation():
    curve = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert area_under_curve(curve, 0.0, 1.0) == pytest.approx(0.5, abs=1e-6)
    flat = np.array([[0.0, 0.7]])
    assert area_under_curve(flat, 0.0, 2.0) == pytest.approx(0.7, abs=1e-6)


def test_cost_to_reach():
    curve = np.array([[1.0, 0.2], [3.0, 0.6]])
    assert cost_to_reach(curve, 0.4) == pytest.approx(2.0)
    assert cost_to_reach(curve, 0.1) == pytest.approx(1.0)
    assert np.isnan(cost_to_reach(curve, 0.9))


def test_operating_point():
    q = np.array([[0.0, 1.0], [1.0, 0.0]])
    c = np.array([[1.0, 3.0], [1.0, 3.0]])
    assert operating_point(np.array([1, 0]), q, c) == (pytest.approx(2.0), pytest.approx(1.0))


def test_summarize_keys():
    out = summarize(np.array([[1.0, 0.2], [3.0, 0.6]]), 1.0, 3.0, {"b": 2.0}, 0.5)
    assert set(out) == {
        "auc",
        "quality_at_b",
        "cost_to_target",
        "cost_to_target_frac_of_best",
        "max_quality",
    }
