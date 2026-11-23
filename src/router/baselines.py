"""Non-learned reference routers. Each returns operating points (cost, quality) on test data.

Model choices that need statistics (cheapest / best) are made on TRAIN, never on test.
"""

import numpy as np
import pandas as pd

from router.evaluate import upper_hull, upper_hull_indices


def single_model_points(models: list[str], quality: np.ndarray, cost: np.ndarray) -> pd.DataFrame:
    """One operating point per model: always route to that model."""
    return pd.DataFrame(
        {"model": models, "cost": cost.mean(axis=0), "quality": quality.mean(axis=0)}
    )


def pick_cheapest(train: pd.DataFrame) -> str:
    """Model with the lowest mean cost on the training prompts."""
    return str(train.groupby("model")["cost"].mean().idxmin())


def pick_best(train: pd.DataFrame) -> str:
    """Model with the highest mean quality on the training prompts."""
    return str(train.groupby("model")["quality"].mean().idxmax())


def pick_hull_models(train: pd.DataFrame) -> list[str]:
    """Models on the upper cost-quality hull of TRAIN single-model points (zero-router set)."""
    pts = train.groupby("model")[["cost", "quality"]].mean()
    idx = upper_hull_indices(pts["cost"].to_numpy(), pts["quality"].to_numpy())
    return pts.index[idx].tolist()


def mixture_curve(points: pd.DataFrame, models: list[str]) -> np.ndarray:
    """Randomly route to one of ``models``: the hull of their single-model points.

    With two models this is the straight 'random mixture' line; with all models it is the
    'zero router' (best prompt-agnostic strategy).
    """
    sel = points[points["model"].isin(models)]
    return upper_hull(sel["cost"].to_numpy(), sel["quality"].to_numpy())


def oracle_curve(quality: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Exact cost-quality frontier of a router that knows every true label.

    Each prompt starts at its cheapest model; its upper hull defines 'upgrade' steps with
    gain per dollar dq/dc. Taking all prompts' steps in decreasing dq/dc order is optimal
    (greedy is exact because each prompt's steps are concave), so the cumulative sums trace
    the frontier. Endpoints: lambda -> inf (all cheapest) and lambda = 0 (all best).
    """
    n = len(quality)
    start_cost = start_quality = 0.0
    steps = []  # (slope, d_cost, d_quality)
    for q_row, c_row in zip(quality, cost, strict=True):
        hull = upper_hull(c_row, q_row)
        start_cost += hull[0, 0]
        start_quality += hull[0, 1]
        d = np.diff(hull, axis=0)
        steps.extend(zip(d[:, 1] / d[:, 0], d[:, 0], d[:, 1], strict=True))

    if not steps:
        return np.array([[start_cost / n, start_quality / n]])
    steps_arr = np.array(sorted(steps, key=lambda s: -s[0]))
    costs = (start_cost + np.cumsum(steps_arr[:, 1])) / n
    qualities = (start_quality + np.cumsum(steps_arr[:, 2])) / n
    return np.vstack([[start_cost / n, start_quality / n], np.column_stack([costs, qualities])])
