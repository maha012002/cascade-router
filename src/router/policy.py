"""Utility-based routing: pick m*(x) = argmax_m [ p_hat(x, m) - lambda * cost(x, m) ]."""

import numpy as np


def route(p_hat: np.ndarray, cost: np.ndarray, lam: float) -> np.ndarray:
    """Return the chosen model index per prompt.

    Args:
        p_hat: (n_prompts, n_models) predicted quality / P(correct).
        cost:  (n_prompts, n_models) or (n_models,) estimated cost in USD.
        lam:   quality-per-dollar trade-off, >= 0. ``np.inf`` means "cost is all that matters".

    Ties are broken towards the cheaper model (and, for lam = inf, towards higher p_hat).
    """
    if lam < 0:
        raise ValueError("lambda must be >= 0")
    p_hat = np.asarray(p_hat, dtype=float)
    cost = np.broadcast_to(np.asarray(cost, dtype=float), p_hat.shape)

    if np.isinf(lam):
        primary, tiebreak = -cost, p_hat
    else:
        primary, tiebreak = p_hat - lam * cost, -cost

    is_best = np.isclose(primary, primary.max(axis=1, keepdims=True), rtol=0.0, atol=1e-12)
    return np.where(is_best, tiebreak, -np.inf).argmax(axis=1)
