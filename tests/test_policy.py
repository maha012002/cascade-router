import numpy as np
import pytest

from router.policy import route

P_HAT = np.array([[0.2, 0.9, 0.6], [0.5, 0.4, 0.3]])
COST = np.array([0.001, 0.010, 0.003])  # model 0 cheapest, model 1 most expensive


def test_lambda_zero_picks_highest_predicted_quality():
    np.testing.assert_array_equal(route(P_HAT, COST, 0.0), P_HAT.argmax(axis=1))


@pytest.mark.parametrize("lam", [np.inf, 1e9])
def test_lambda_huge_picks_cheapest(lam):
    np.testing.assert_array_equal(route(P_HAT, COST, lam), [0, 0])


def test_per_prompt_cost_matrix_is_respected():
    cost = np.array([[0.010, 0.001, 0.003], [0.001, 0.010, 0.003]])
    np.testing.assert_array_equal(route(P_HAT, cost, np.inf), [1, 0])


def test_tie_breaks_to_cheaper_model():
    p_hat = np.array([[0.7, 0.7, 0.7]])
    assert route(p_hat, COST, 0.0)[0] == 0


def test_inf_tie_on_cost_breaks_to_higher_p_hat():
    p_hat = np.array([[0.1, 0.8]])
    assert route(p_hat, np.array([0.002, 0.002]), np.inf)[0] == 1


def test_intermediate_lambda_trades_off():
    # utility at lam=50: [0.2-0.05, 0.9-0.5, 0.6-0.15] = [0.15, 0.40, 0.45] -> model 2
    assert route(P_HAT[:1], COST, 50.0)[0] == 2


def test_negative_lambda_rejected():
    with pytest.raises(ValueError):
        route(P_HAT, COST, -1.0)
