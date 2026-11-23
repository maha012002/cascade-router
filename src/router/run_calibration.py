"""Fit val calibrators, measure ECE, and check whether calibration moves the cost-quality curve."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from router.calibration import (
    IsotonicCalibrator,
    TemperatureCalibrator,
    expected_calibration_error,
    reliability_table,
)
from router.config import PROJECT_ROOT, load_config
from router.data import load_outcomes, to_matrices
from router.evaluate import plot_curves, summarize, sweep, upper_hull
from router.resources import limit_threads, require_free_ram
from router.run_baselines import prepare_split
from router.run_routers import lambda_grid
from router.splits import load_splits
from router.tracking import init_mlflow

limit_threads(2)

CALIBRATORS = {
    "isotonic": IsotonicCalibrator,
    "temperature": TemperatureCalibrator,
}


def load_pred(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/calibration.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    base_cfg = load_config(cfg["baseline_config"])
    require_free_ram(1.5, "calibration")

    outcomes, splits = load_outcomes(), load_splits()
    lambdas = lambda_grid(cfg["lambda_grid"])
    init_mlflow(**cfg["mlflow"])
    pred_dir = PROJECT_ROOT / cfg["predictions_dir"]
    fig_dir = PROJECT_ROOT / cfg["outputs"]["figure_dir"]
    n_bins = cfg["ece_bins"]

    rows = []
    for split in cfg["splits"]:
        ev = prepare_split(outcomes, splits, split, base_cfg)
        y_val, _, _ = to_matrices(outcomes, ev.ids["val"])
        y_test, cost_test, models = to_matrices(outcomes, ev.ids["test"])

        for router in cfg["routers"]:
            path = pred_dir / f"{split}__{router}.npz"
            if not path.exists():
                print(f"skip missing {path.name}")
                continue
            pred = load_pred(path)
            assert list(pred["models"]) == models
            p_val, p_test = pred["p_val"].astype(float), pred["p_test"].astype(float)
            cost_est = pred["cost_est_test"].astype(float)

            ece_val = expected_calibration_error(y_val, p_val, n_bins)
            ece_test = expected_calibration_error(y_test, p_test, n_bins)
            raw_pts = sweep(p_test, cost_est, y_test, cost_test, lambdas)
            raw_curve = upper_hull(raw_pts["cost"].to_numpy(), raw_pts["quality"].to_numpy())
            raw_metrics = summarize(raw_curve, ev.c_lo, ev.c_hi, ev.budgets, ev.target)

            rows.append(
                {
                    "split": split,
                    "router": router,
                    "calibrator": "none",
                    "ece_val": ece_val,
                    "ece_test": ece_test,
                    **{f"route_{k}": v for k, v in raw_metrics.items()},
                }
            )

            calibrated_curves = {"uncalibrated": raw_curve}
            for name in cfg["methods"]:
                cal = CALIBRATORS[name]().fit(p_val, y_val)
                p_val_c, p_test_c = cal.transform(p_val), cal.transform(p_test)
                pts = sweep(p_test_c, cost_est, y_test, cost_test, lambdas)
                curve = upper_hull(pts["cost"].to_numpy(), pts["quality"].to_numpy())
                metrics = summarize(curve, ev.c_lo, ev.c_hi, ev.budgets, ev.target)
                calibrated_curves[name] = curve
                rows.append(
                    {
                        "split": split,
                        "router": router,
                        "calibrator": name,
                        "ece_val": expected_calibration_error(y_val, p_val_c, n_bins),
                        "ece_test": expected_calibration_error(y_test, p_test_c, n_bins),
                        **{f"route_{k}": v for k, v in metrics.items()},
                        "temps": (
                            np.round(cal.temperatures_, 3).tolist()
                            if name == "temperature"
                            else None
                        ),
                    }
                )

            # Reliability diagram for the uncalibrated + isotonic on the ID lightgbm only.
            if split == "id" and router == "lightgbm":
                _plot_reliability(
                    y_test,
                    p_test,
                    IsotonicCalibrator().fit(p_val, y_val).transform(p_test),
                    n_bins,
                    fig_dir / "m3_reliability_id_lightgbm.png",
                )
                shown = {k: ev.curves[k] for k in ["zero_router", "oracle"]}
                plot_curves(
                    {**shown, **calibrated_curves},
                    ev.points,
                    f"Calibration effect - {router} / {split}",
                    fig_dir / f"m3_calibrated_{router}_{split}.png",
                    ev.budgets,
                )

            with mlflow.start_run(run_name=f"cal_{router}_{split}"):
                mlflow.set_tags({"milestone": "M3", "kind": "calibration", "router": router})
                mlflow.log_params({"router": router, "split": split})
                mlflow.log_metrics(
                    {
                        "ece_val_raw": ece_val,
                        "ece_test_raw": ece_test,
                        "auc_raw": raw_metrics["auc"],
                    }
                )

    results = pd.DataFrame(rows)
    out = PROJECT_ROOT / cfg["outputs"]["results_csv"]
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    cols = [
        "split",
        "router",
        "calibrator",
        "ece_val",
        "ece_test",
        "route_auc",
        "route_quality_at_b25",
        "route_cost_to_target_frac_of_best",
    ]
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(results[cols].round(4).to_string(index=False))


def _plot_reliability(
    y_true: np.ndarray,
    p_raw: np.ndarray,
    p_cal: np.ndarray,
    n_bins: int,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    for name, p in [("uncalibrated", p_raw), ("isotonic", p_cal)]:
        tab = reliability_table(y_true, p, n_bins)
        ax.plot(
            [r["mean_pred"] for r in tab],
            [r["mean_true"] for r in tab],
            marker="o",
            label=name,
        )
    ax.set_xlabel("mean predicted quality")
    ax.set_ylabel("mean realised quality")
    ax.set_title("Reliability diagram (ID test, LightGBM, all model cells)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
