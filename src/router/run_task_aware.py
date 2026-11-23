"""Evaluate the task-aware zero router on every split; compare to zero_router and LightGBM."""

from __future__ import annotations

import argparse

import pandas as pd

from router.baselines import fit_task_aware_hulls, task_aware_zero_router_curve
from router.config import PROJECT_ROOT, load_config
from router.data import load_outcomes
from router.resources import limit_threads
from router.run_baselines import prepare_split
from router.run_routers import lambda_grid
from router.splits import load_splits

limit_threads(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task_aware.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    base_cfg = load_config(cfg["baseline_config"])
    routers_cfg = load_config(cfg["routers_config"])
    lambdas = lambda_grid(routers_cfg["lambda_grid"])

    outcomes, splits = load_outcomes(), load_splits()
    # Optional: LightGBM metrics from the committed M2 table for side-by-side printout.
    m2_path = PROJECT_ROOT / "docs/results/m2_results.csv"
    m2 = pd.read_csv(m2_path) if m2_path.exists() else None

    rows = []
    for split in base_cfg["splits"]:
        ev = prepare_split(outcomes, splits, split, base_cfg)
        train = outcomes[outcomes["prompt_id"].isin(ev.ids["train"])]
        hulls, means, global_hull, global_means = fit_task_aware_hulls(train)

        sources = (
            outcomes.drop_duplicates("prompt_id")
            .set_index("prompt_id")
            .loc[ev.ids["test"], "source"]
            .astype(str)
            .tolist()
        )
        curve, n_fallback = task_aware_zero_router_curve(
            ev.quality,
            ev.cost,
            sources,
            ev.models,
            hulls,
            means,
            global_hull,
            global_means,
            lambdas,
        )
        metrics = ev.metrics(curve)
        held_out = split.removeprefix("lobo_") if split.startswith("lobo_") else None
        row = {
            "split": split,
            "method": "task_aware_zero_router",
            **metrics,
            "n_test": len(ev.quality),
            "n_fallback_global": n_fallback,
            "frac_fallback_global": n_fallback / len(ev.quality),
            "n_source_hulls": len(hulls),
            "held_out_source": held_out,
            "held_out_used_global_fallback": bool(held_out is not None and held_out not in hulls),
        }
        rows.append(row)

        # Also record the existing zero_router / lightgbm from prepare_split / m2 for the table.
        zr = ev.metrics(ev.curves["zero_router"])
        rows.append(
            {
                "split": split,
                "method": "zero_router",
                **zr,
                "n_test": len(ev.quality),
                "n_fallback_global": 0,
                "frac_fallback_global": 0.0,
                "n_source_hulls": len(hulls),
                "held_out_source": held_out,
                "held_out_used_global_fallback": False,
            }
        )
        if m2 is not None:
            hit = m2[(m2["split"] == split) & (m2["method"] == "lightgbm")]
            if len(hit) == 1:
                r = hit.iloc[0]
                rows.append(
                    {
                        "split": split,
                        "method": "lightgbm",
                        "auc": float(r["auc"]),
                        "quality_at_b10": float(r["quality_at_b10"]),
                        "quality_at_b25": float(r["quality_at_b25"]),
                        "quality_at_b50": float(r["quality_at_b50"]),
                        "cost_to_target": float(r["cost_to_target"]),
                        "cost_to_target_frac_of_best": float(r["cost_to_target_frac_of_best"]),
                        "max_quality": float(r["max_quality"]),
                        "n_test": int(r["n_test"]),
                        "n_fallback_global": 0,
                        "frac_fallback_global": 0.0,
                        "n_source_hulls": len(hulls),
                        "held_out_source": held_out,
                        "held_out_used_global_fallback": False,
                    }
                )

        print(
            f"{split}: task_aware auc={metrics['auc']:.4f} | "
            f"zero_router auc={zr['auc']:.4f} | "
            f"fallback={n_fallback}/{len(ev.quality)} "
            f"(held_out_missing_hull={row['held_out_used_global_fallback']})"
        )

    results = pd.DataFrame(rows)
    out = PROJECT_ROOT / cfg["outputs"]["results_csv"]
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)

    shown = [
        "split",
        "method",
        "auc",
        "quality_at_b25",
        "cost_to_target_frac_of_best",
        "frac_fallback_global",
        "held_out_used_global_fallback",
    ]
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(results[shown].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
