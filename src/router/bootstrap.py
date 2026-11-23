"""Paired bootstrap CIs for cost–quality metrics (same resampled prompts per method)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from router.evaluate import summarize, upper_hull


def _sweep_means(
    p_hat: np.ndarray,
    cost_est: np.ndarray,
    quality: np.ndarray,
    cost: np.ndarray,
    lambdas: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean (cost, quality) per lambda; same tie-break as ``policy.route``, fewer temporaries."""
    p_hat = np.asarray(p_hat, dtype=float)
    cost_est = np.asarray(cost_est, dtype=float)
    rows = np.arange(len(quality))
    costs = np.empty(len(lambdas), dtype=float)
    quals = np.empty(len(lambdas), dtype=float)
    # Scale for a tiny secondary key so argmax matches route()'s isclose+tiebreak.
    tb_scale = 1e-12 / (np.max(np.abs(p_hat)) + np.max(np.abs(cost_est)) + 1.0)
    for i, lam in enumerate(lambdas):
        if np.isinf(lam):
            score = -cost_est + tb_scale * p_hat
        else:
            score = p_hat - lam * cost_est - tb_scale * cost_est
        ch = score.argmax(axis=1)
        costs[i] = cost[rows, ch].mean()
        quals[i] = quality[rows, ch].mean()
    return costs, quals


def reference_points(
    quality: np.ndarray,
    cost: np.ndarray,
    models: list[str],
    cheapest: str,
    best: str,
    budget_fracs: dict[str, float],
    target_frac: float,
) -> tuple[float, float, dict[str, float], float]:
    """c_lo / c_hi / budgets / target on a (possibly resampled) prompt set."""
    ci, bi = models.index(cheapest), models.index(best)
    c_lo = float(cost[:, ci].mean())
    c_hi = float(cost[:, bi].mean())
    budgets = {k: f * c_hi for k, f in budget_fracs.items()}
    target = target_frac * float(quality[:, bi].mean())
    return c_lo, c_hi, budgets, target


def curve_zero_router(
    quality: np.ndarray, cost: np.ndarray, models: list[str], hull_models: list[str]
) -> np.ndarray:
    """Prompt-agnostic zero router: hull of hull-models' mean (cost, quality) on this sample."""
    idx = [models.index(m) for m in hull_models]
    return upper_hull(cost[:, idx].mean(axis=0), quality[:, idx].mean(axis=0))


def curve_from_predictions(
    p_hat: np.ndarray,
    cost_est: np.ndarray,
    quality: np.ndarray,
    cost: np.ndarray,
    lambdas: list[float],
) -> np.ndarray:
    costs, quals = _sweep_means(p_hat, cost_est, quality, cost, lambdas)
    return upper_hull(costs, quals)


def metrics_pair(
    curve: np.ndarray, c_lo: float, c_hi: float, budgets: dict[str, float], target: float
) -> tuple[float, float]:
    """Return (auc, cost_to_target_frac_of_best)."""
    s = summarize(curve, c_lo, c_hi, budgets, target)
    return float(s["auc"]), float(s["cost_to_target_frac_of_best"])


def paired_bootstrap(
    *,
    quality: np.ndarray,
    cost: np.ndarray,
    models: list[str],
    cheapest: str,
    best: str,
    budget_fracs: dict[str, float],
    target_frac: float,
    lambdas: list[float],
    methods: dict[str, dict],
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired bootstrap of AUC and cost-to-95% (as fraction of always-best cost).

    ``methods`` maps name -> spec:
      {"kind": "predictions", "p_hat": ..., "cost_est": ...}
      {"kind": "zero_router", "hull_models": [...]}
    """
    rng = np.random.default_rng(seed)
    n = len(quality)
    names = list(methods)
    auc_s = {m: np.empty(n_boot) for m in names}
    c95_s = {m: np.empty(n_boot) for m in names}

    for b in range(n_boot):
        if b and b % 100 == 0:
            print(f"   ... boot {b}/{n_boot}", flush=True)
        idx = rng.integers(0, n, size=n)
        q_b, c_b = quality[idx], cost[idx]
        c_lo, c_hi, budgets, target = reference_points(
            q_b, c_b, models, cheapest, best, budget_fracs, target_frac
        )
        for name, spec in methods.items():
            if spec["kind"] == "predictions":
                curve = curve_from_predictions(
                    spec["p_hat"][idx], spec["cost_est"][idx], q_b, c_b, lambdas
                )
            elif spec["kind"] == "zero_router":
                curve = curve_zero_router(q_b, c_b, models, spec["hull_models"])
            else:
                raise ValueError(spec["kind"])
            auc, c95 = metrics_pair(curve, c_lo, c_hi, budgets, target)
            auc_s[name][b] = auc
            c95_s[name][b] = c95

    rows = []
    for name in names:
        rows.append(_ci_row(name, "auc", auc_s[name], None))
        rows.append(_ci_row(name, "cost_to_target_frac_of_best", c95_s[name], None))
    if "lightgbm" in auc_s:
        for name in names:
            if name == "lightgbm":
                continue
            rows.append(
                _ci_row(f"lightgbm_minus_{name}", "auc", auc_s["lightgbm"] - auc_s[name], name)
            )
            rows.append(
                _ci_row(
                    f"lightgbm_minus_{name}",
                    "cost_to_target_frac_of_best",
                    c95_s["lightgbm"] - c95_s[name],
                    name,
                )
            )
    return pd.DataFrame(rows)


def _ci_row(label: str, metric: str, samples: np.ndarray, vs: str | None) -> dict:
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        lo = hi = mean = float("nan")
    else:
        lo, hi = np.percentile(finite, [2.5, 97.5])
        mean = float(finite.mean())
        lo, hi = float(lo), float(hi)
    return {
        "contrast": label,
        "vs_method": vs,
        "metric": metric,
        "mean": mean,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "n_boot_finite": int(len(finite)),
        "contains_zero": bool(lo <= 0.0 <= hi) if vs is not None and np.isfinite(lo) else None,
    }
