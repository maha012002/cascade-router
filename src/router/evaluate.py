"""Cost-quality evaluation.

Every method is reduced to a set of operating points (mean cost, mean quality) on the test set.
Its curve is the upper concave hull of those points: randomly mixing two operating points
achieves any point on the segment between them, so the hull is what the method can deliver.

Metrics (all on that curve):
    auc               mean quality over cost in [c_lo, c_hi] (linear USD), i.e. normalised area
    quality@budget    quality at a cost budget
    cost_to_target    cheapest cost reaching a target quality (also as a fraction of c_hi)
Below a curve's cheapest point the method cannot operate; quality there counts as 0.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from router.policy import route


def operating_point(
    choices: np.ndarray, quality: np.ndarray, cost: np.ndarray
) -> tuple[float, float]:
    """Mean realised (cost, quality) when prompt i is sent to model ``choices[i]``."""
    rows = np.arange(len(choices))
    return float(cost[rows, choices].mean()), float(quality[rows, choices].mean())


def sweep(
    p_hat: np.ndarray,
    cost_estimate: np.ndarray,
    quality: np.ndarray,
    cost: np.ndarray,
    lambdas: list[float],
) -> pd.DataFrame:
    """Route with each lambda (using predictions + estimated cost), score with true labels."""
    rows = []
    for lam in lambdas:
        c, q = operating_point(route(p_hat, cost_estimate, lam), quality, cost)
        rows.append({"lam": lam, "cost": c, "quality": q})
    return pd.DataFrame(rows)


def upper_hull_indices(costs: np.ndarray, qualities: np.ndarray) -> list[int]:
    """Indices of the upper concave hull, from the cheapest point to the best-quality point.

    Returned in cost-ascending order; qualities along the hull are strictly increasing.
    """
    costs, qualities = np.asarray(costs, float), np.asarray(qualities, float)
    order = np.lexsort((-qualities, costs))  # cost asc, then quality desc

    hull: list[int] = []
    for i in order:
        if hull and qualities[i] <= qualities[hull[-1]]:
            continue  # dominated: costs at least as much, no better quality
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            cross = (costs[b] - costs[a]) * (qualities[i] - qualities[a]) - (
                qualities[b] - qualities[a]
            ) * (costs[i] - costs[a])
            if cross >= 0:  # b is on or below the line a -> i
                hull.pop()
            else:
                break
        hull.append(int(i))
    return hull


def upper_hull(costs: np.ndarray, qualities: np.ndarray) -> np.ndarray:
    """Upper concave hull as an (k, 2) array of (cost, quality)."""
    idx = upper_hull_indices(costs, qualities)
    return np.column_stack([np.asarray(costs, float)[idx], np.asarray(qualities, float)[idx]])


def quality_at_cost(curve: np.ndarray, budget: float | np.ndarray) -> np.ndarray:
    """Interpolated quality at a cost budget: 0 below the curve, flat beyond its last point.

    Budgets within float rounding of the curve's first point count as on the curve.
    """
    c0 = curve[0, 0]
    budget = np.where(np.isclose(budget, c0, rtol=1e-9, atol=0.0), c0, budget)
    return np.interp(budget, curve[:, 0], curve[:, 1], left=0.0, right=curve[-1, 1])


def area_under_curve(curve: np.ndarray, c_lo: float, c_hi: float, n_grid: int = 2001) -> float:
    """Mean quality over cost in [c_lo, c_hi] (area normalised by the cost range)."""
    grid = np.linspace(c_lo, c_hi, n_grid)
    return float(np.trapezoid(quality_at_cost(curve, grid), grid) / (c_hi - c_lo))


def cost_to_reach(curve: np.ndarray, target: float) -> float:
    """Lowest cost at which the curve reaches ``target`` quality (NaN if it never does)."""
    if curve[-1, 1] < target:
        return float("nan")
    if curve[0, 1] >= target:
        return float(curve[0, 0])
    i = int(np.argmax(curve[:, 1] >= target))
    (c0, q0), (c1, q1) = curve[i - 1], curve[i]
    return float(c0 + (target - q0) * (c1 - c0) / (q1 - q0))


def summarize(
    curve: np.ndarray,
    c_lo: float,
    c_hi: float,
    budgets: dict[str, float],
    target_quality: float,
) -> dict[str, float]:
    """All scalar metrics for one curve."""
    out = {"auc": area_under_curve(curve, c_lo, c_hi)}
    for name, budget in budgets.items():
        out[f"quality_at_{name}"] = float(quality_at_cost(curve, budget))
    out["cost_to_target"] = cost_to_reach(curve, target_quality)
    out["cost_to_target_frac_of_best"] = out["cost_to_target"] / c_hi
    out["max_quality"] = float(curve[-1, 1])
    return out


def plot_curves(
    curves: dict[str, np.ndarray],
    single_models: pd.DataFrame,
    title: str,
    path: Path,
    budgets: dict[str, float] | None = None,
) -> None:
    """Cost (log x) vs quality: one line per method, one dot per single model."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, curve in curves.items():
        if len(curve) == 1:
            ax.scatter(curve[:, 0], curve[:, 1], label=name, s=120, marker="D", zorder=2)
        else:
            ax.plot(curve[:, 0], curve[:, 1], label=name)
    ax.scatter(single_models["cost"], single_models["quality"], color="black", s=18, zorder=3)
    for _, r in single_models.iterrows():
        ax.annotate(
            r["model"].split("/")[-1],
            (r["cost"], r["quality"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    for name, b in (budgets or {}).items():
        ax.axvline(b, color="grey", linestyle=":", linewidth=1)
        ax.text(b, ax.get_ylim()[0], f" {name}", fontsize=7, color="grey", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("mean cost per prompt (USD, log scale)")
    ax.set_ylabel("mean quality")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
