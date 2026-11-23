"""Evaluate all baselines on every split; write a results table, plots, and MLflow runs."""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from router.baselines import (
    mixture_curve,
    oracle_curve,
    pick_best,
    pick_cheapest,
    pick_hull_models,
    single_model_points,
)
from router.config import PROJECT_ROOT, load_config
from router.data import load_outcomes, to_matrices
from router.evaluate import plot_curves, summarize, upper_hull
from router.splits import load_splits
from router.tracking import init_mlflow

METRICS = [
    "auc",
    "quality_at_b10",
    "quality_at_b25",
    "quality_at_b50",
    "cost_to_target",
    "cost_to_target_frac_of_best",
    "max_quality",
]


def baseline_curves(
    train: pd.DataFrame, points: pd.DataFrame
) -> tuple[dict[str, np.ndarray], str, str]:
    """Curves for every prompt-agnostic baseline; cheapest/best are chosen on train."""
    cheapest, best = pick_cheapest(train), pick_best(train)
    by_model = points.set_index("model")

    def single(m: str) -> np.ndarray:
        return upper_hull(by_model.loc[[m], "cost"].values, by_model.loc[[m], "quality"].values)

    curves = {
        "always_cheapest": single(cheapest),
        "always_best": single(best),
        "random_mixture": mixture_curve(points, [cheapest, best]),
        "zero_router": mixture_curve(points, pick_hull_models(train)),
    }
    return curves, cheapest, best


def evaluate_split(
    outcomes: pd.DataFrame, splits: pd.DataFrame, split: str, cfg: dict
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, dict[str, float]]:
    """Metrics table, curves, single-model points and budgets for one split."""
    s = splits[splits["split"] == split]
    train_ids = s.loc[s["fold"] == "train", "prompt_id"]
    test_ids = s.loc[s["fold"] == cfg["eval_fold"], "prompt_id"].sort_values().tolist()

    train = outcomes[outcomes["prompt_id"].isin(train_ids)]
    quality, cost, models = to_matrices(outcomes, test_ids)
    points = single_model_points(models, quality, cost)

    curves, cheapest, best = baseline_curves(train, points)
    curves["oracle"] = oracle_curve(quality, cost)

    by_model = points.set_index("model")
    c_lo, c_hi = float(by_model.loc[cheapest, "cost"]), float(by_model.loc[best, "cost"])
    budgets = {k: f * c_hi for k, f in cfg["budgets_frac_of_best_cost"].items()}
    target = cfg["target_frac_of_best_quality"] * float(by_model.loc[best, "quality"])

    table = pd.DataFrame(
        [
            {"split": split, "method": name, **summarize(curve, c_lo, c_hi, budgets, target)}
            for name, curve in curves.items()
        ]
    )
    table["n_test"] = len(test_ids)
    table["cheapest_model"], table["best_model"] = cheapest, best
    return table, curves, points, budgets


def log_run(split: str, table: pd.DataFrame, artifacts: list[Path]) -> None:
    """One MLflow run per split; metrics are named '<method>.<metric>'."""
    first = table.iloc[0]
    with mlflow.start_run(run_name=f"baselines_{split}"):
        mlflow.set_tags({"milestone": "M1", "kind": "baselines", "split": split})
        mlflow.log_params(
            {
                "split": split,
                "n_test": int(first["n_test"]),
                "cheapest_model": first["cheapest_model"],
                "best_model": first["best_model"],
            }
        )
        for _, r in table.iterrows():
            for k in METRICS:
                if pd.notna(r[k]):
                    mlflow.log_metric(f"{r['method']}.{k}", float(r[k]))
        for path in artifacts:
            mlflow.log_artifact(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_baselines.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    outcomes, splits = load_outcomes(), load_splits()
    init_mlflow(**cfg["mlflow"])
    fig_dir = PROJECT_ROOT / cfg["outputs"]["figure_dir"]

    tables = []
    for split in cfg["splits"]:
        table, curves, points, budgets = evaluate_split(outcomes, splits, split, cfg)
        n_test = int(table["n_test"].iloc[0])
        fig_path = fig_dir / f"m1_baselines_{split}.png"
        plot_curves(curves, points, f"Baselines - {split} (n_test={n_test})", fig_path, budgets)
        log_run(split, table, [fig_path, PROJECT_ROOT / args.config])
        tables.append(table)

    results = pd.concat(tables, ignore_index=True)
    out = PROJECT_ROOT / cfg["outputs"]["results_csv"]
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)

    shown = ["split", "method", "auc", *METRICS[1:4], "cost_to_target_frac_of_best", "max_quality"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(results[shown].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
