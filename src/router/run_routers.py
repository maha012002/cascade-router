"""Train learned routers on every split, route with the utility policy, and compare to baselines.

Per split and router: tune on val (soft log-loss), predict test, sweep lambda with *estimated*
costs, score with *realised* costs/quality, log to MLflow, save val/test predictions for M3.
"""

import argparse
import gc
import inspect
import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from router.config import PROJECT_ROOT, load_config
from router.data import load_outcomes, load_prompts, to_matrices
from router.evaluate import plot_curves, sweep, upper_hull
from router.features import HANDCRAFTED_COLUMNS, load_features
from router.models import PREDICTORS, CostModel, soft_log_loss
from router.resources import limit_threads, require_free_ram
from router.run_baselines import SplitEval, baseline_table, prepare_split
from router.splits import load_splits
from router.tracking import init_mlflow

limit_threads(2)

REFERENCE_METHODS = ["always_best", "random_mixture", "zero_router", "oracle"]


def lambda_grid(spec: dict) -> list[float]:
    return [0.0, *np.geomspace(spec["min"], spec["max"], spec["n"]).tolist(), float("inf")]


def feature_matrix(features: pd.DataFrame, ids: list[str], inputs: str) -> np.ndarray:
    emb_cols = [c for c in features.columns if c.startswith("e") and c[1:].isdigit()]
    cols = emb_cols if inputs == "embeddings" else emb_cols + HANDCRAFTED_COLUMNS
    return features.loc[ids, cols].to_numpy(dtype=np.float32)


def build_predictor(name: str, params: dict, seed: int):
    cls = PREDICTORS[name]
    if "seed" in inspect.signature(cls).parameters:
        params = {**params, "seed": seed}
    return cls(**params)


def fit_and_select(name: str, spec: dict, data: dict, seed: int) -> tuple:
    """Try each grid entry, keep the one with the lowest val soft log-loss."""
    best = None
    for params in spec["grid"]:
        start = time.perf_counter()
        model = build_predictor(name, params, seed).fit(
            data["X_train"], data["Y_train"], data["X_val"], data["Y_val"]
        )
        p_val = model.predict(data["X_val"])
        loss = soft_log_loss(data["Y_val"], p_val)
        elapsed = time.perf_counter() - start
        print(f"    {name} {params}: val log-loss {loss:.4f} ({elapsed:.1f}s)")
        if best is None or loss < best[2]:
            best = (model, params, loss, p_val, elapsed)
    return best


def run_split(
    ev: SplitEval,
    outcomes: pd.DataFrame,
    features: pd.DataFrame,
    n_chars: pd.Series,
    cfg: dict,
    lambdas: list[float],
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Train + evaluate every router on one split. Returns result rows and curves."""
    Y_train, cost_train, models = to_matrices(outcomes, ev.ids["train"])
    Y_val, _, _ = to_matrices(outcomes, ev.ids["val"])
    assert models == ev.models, "model column order must match the evaluation matrices"

    cost_model = CostModel().fit(n_chars.loc[ev.ids["train"]].to_numpy(), cost_train)
    cost_est = cost_model.predict(n_chars.loc[ev.ids["test"]].to_numpy())
    cost_r2 = float(r2_score(ev.cost, cost_est, multioutput="uniform_average"))

    rows, curves = [], {}
    pred_dir = PROJECT_ROOT / "artifacts" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in cfg["routers"].items():
        data = {
            "X_train": feature_matrix(features, ev.ids["train"], spec["inputs"]),
            "Y_train": Y_train,
            "X_val": feature_matrix(features, ev.ids["val"], spec["inputs"]),
            "Y_val": Y_val,
        }
        model, params, val_loss, p_val, fit_s = fit_and_select(name, spec, data, cfg["seed"])
        p_test = model.predict(feature_matrix(features, ev.ids["test"], spec["inputs"]))

        points = sweep(p_test, cost_est, ev.quality, ev.cost, lambdas)
        curve = upper_hull(points["cost"].to_numpy(), points["quality"].to_numpy())
        curves[name] = curve
        row = {
            "split": ev.split,
            "method": name,
            **ev.metrics(curve),
            "val_log_loss": val_loss,
            "test_log_loss": soft_log_loss(ev.quality, p_test),
            "fit_seconds": fit_s,
            "params": str(params),
            "cost_model_r2": cost_r2,
            "n_test": len(ev.quality),
        }
        rows.append(row)

        np.savez_compressed(
            pred_dir / f"{ev.split}__{name}.npz",
            models=np.array(models),
            val_ids=np.array(ev.ids["val"]),
            test_ids=np.array(ev.ids["test"]),
            p_val=p_val,
            p_test=p_test,
            cost_est_test=cost_est,
        )
        log_router_run(ev.split, name, spec, params, row, points)
        del model, data
        gc.collect()
    return rows, curves


def log_router_run(
    split: str, name: str, spec: dict, params: dict, row: dict, points: pd.DataFrame
) -> None:
    with mlflow.start_run(run_name=f"{name}_{split}"):
        mlflow.set_tags({"milestone": "M2", "kind": "router", "router": name, "split": split})
        mlflow.log_params({"router": name, "split": split, "inputs": spec["inputs"], **params})
        for k, v in row.items():
            if isinstance(v, int | float) and not isinstance(v, bool) and np.isfinite(v):
                mlflow.log_metric(k, float(v))
        mlflow.log_table(points.replace([np.inf], 1e300), artifact_file="lambda_sweep.json")


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Per method: ID AUC, mean LOBO AUC, and share of the zero-router -> oracle gap closed."""
    wide = results.pivot(index="split", columns="method", values="auc")
    gap = wide["oracle"] - wide["zero_router"]
    closed = wide.sub(wide["zero_router"], axis=0).div(gap, axis=0)
    lobo = [s for s in wide.index if s.startswith("lobo_")]
    return pd.DataFrame(
        {
            "auc_id": wide.loc["id"],
            "auc_lobo_mean": wide.loc[lobo].mean(),
            "gap_closed_id": closed.loc["id"],
            "gap_closed_lobo_mean": closed.loc[lobo].mean(),
        }
    ).sort_values("auc_id", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/routers.yaml")
    parser.add_argument("--splits", nargs="*", help="override the split list")
    args = parser.parse_args()
    cfg = load_config(args.config)
    base_cfg = load_config(cfg["baseline_config"])
    np.random.seed(cfg["seed"])
    require_free_ram(cfg.get("min_free_ram_gb", 2.5), "router training")

    outcomes, splits = load_outcomes(), load_splits()
    features = load_features(cfg["features_config"]).set_index("prompt_id")
    prompts = load_prompts().set_index("prompt_id")
    n_chars = prompts["prompt"].str.len()
    lambdas = lambda_grid(cfg["lambda_grid"])
    init_mlflow(**base_cfg["mlflow"])
    fig_dir = PROJECT_ROOT / cfg["outputs"]["figure_dir"]

    all_rows = []
    for split in args.splits or base_cfg["splits"]:
        print(f"== {split}")
        ev = prepare_split(outcomes, splits, split, base_cfg)
        rows, curves = run_split(ev, outcomes, features, n_chars, cfg, lambdas)
        ref = baseline_table(ev)
        all_rows += rows + ref[ref["method"].isin(REFERENCE_METHODS)].to_dict("records")

        shown = {k: ev.curves[k] for k in ["random_mixture", "zero_router", "oracle"]}
        plot_curves(
            {**shown, **curves},
            ev.points,
            f"Learned routers - {split} (n_test={len(ev.quality)})",
            fig_dir / f"m2_routers_{split}.png",
            ev.budgets,
        )

    results = pd.DataFrame(all_rows)
    out = PROJECT_ROOT / cfg["outputs"]["results_csv"]
    out.parent.mkdir(parents=True, exist_ok=True)
    # Merge with any previous rows for other splits so partial --splits runs don't wipe them.
    if out.exists():
        prev = pd.read_csv(out)
        kept = prev[~prev["split"].isin(results["split"])]
        results = pd.concat([kept, results], ignore_index=True)
    results.to_csv(out, index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        cols = [
            "split",
            "method",
            "auc",
            "quality_at_b10",
            "quality_at_b25",
            "cost_to_target_frac_of_best",
            "val_log_loss",
            "test_log_loss",
            "params",
        ]
        print(results[cols].round(4).to_string(index=False))
        if results["split"].nunique() > 1 and "id" in set(results["split"]):
            summary = summarize_results(results)
            summary.to_csv(PROJECT_ROOT / cfg["outputs"]["summary_csv"])
            print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
