"""Paired bootstrap CIs for AUC and cost-to-95% across methods (M6 step 2)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from router.baselines import fit_task_aware_hulls, pick_hull_models, task_aware_scores
from router.bootstrap import paired_bootstrap
from router.config import PROJECT_ROOT, load_config
from router.data import load_outcomes
from router.resources import limit_threads
from router.run_baselines import prepare_split
from router.run_routers import lambda_grid
from router.splits import load_splits

limit_threads(2)

PRED_DIR = PROJECT_ROOT / "artifacts" / "predictions"


def _load_learned(split: str, method: str) -> tuple[np.ndarray, np.ndarray]:
    path = PRED_DIR / f"{split}__{method}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path)
    return z["p_test"].astype(float), z["cost_est_test"].astype(float)


def run(cfg: dict) -> pd.DataFrame:
    base_cfg = load_config(cfg["baseline_config"])
    routers_cfg = load_config(cfg["routers_config"])
    lambdas = lambda_grid(routers_cfg["lambda_grid"])
    budget_fracs = {k: float(v) for k, v in base_cfg["budgets_frac_of_best_cost"].items()}
    target_frac = float(base_cfg["target_frac_of_best_quality"])
    n_boot = int(cfg["bootstrap"]["n_resamples"])
    seed = int(cfg["bootstrap"]["seed"])
    methods_wanted = list(cfg["bootstrap"]["methods"])
    splits = list(cfg["bootstrap"].get("splits") or base_cfg["splits"])

    outcomes, splits_df = load_outcomes(), load_splits()
    frames: list[pd.DataFrame] = []

    for split in splits:
        print(f"== bootstrap {split} (n_boot={n_boot})", flush=True)
        ev = prepare_split(outcomes, splits_df, split, base_cfg)
        train = outcomes[outcomes["prompt_id"].isin(ev.ids["train"])]
        sources = (
            outcomes.drop_duplicates("prompt_id")
            .set_index("prompt_id")
            .loc[ev.ids["test"], "source"]
            .astype(str)
            .tolist()
        )

        methods: dict = {}
        if "zero_router" in methods_wanted:
            methods["zero_router"] = {
                "kind": "zero_router",
                "hull_models": pick_hull_models(train),
            }
        if "task_aware" in methods_wanted:
            hulls, means, g_hull, g_means = fit_task_aware_hulls(train)
            p_hat, cost_est, _, n_fb = task_aware_scores(
                sources, ev.models, hulls, means, g_hull, g_means
            )
            methods["task_aware"] = {
                "kind": "predictions",
                "p_hat": p_hat,
                "cost_est": cost_est,
            }
            print(f"   task_aware fallback={n_fb}/{len(sources)}", flush=True)
        for name in methods_wanted:
            if name in ("zero_router", "task_aware"):
                continue
            p_hat, cost_est = _load_learned(split, name)
            if len(p_hat) != len(ev.quality):
                raise ValueError(f"{split}/{name}: pred n={len(p_hat)} != test n={len(ev.quality)}")
            methods[name] = {"kind": "predictions", "p_hat": p_hat, "cost_est": cost_est}

        res = paired_bootstrap(
            quality=ev.quality,
            cost=ev.cost,
            models=ev.models,
            cheapest=ev.cheapest,
            best=ev.best,
            budget_fracs=budget_fracs,
            target_frac=target_frac,
            lambdas=lambdas,
            methods=methods,
            n_boot=n_boot,
            seed=seed,
        )
        res.insert(0, "split", split)
        frames.append(res)
        # Print ID-style contrasts first for review.
        show = res[res["contrast"].str.startswith("lightgbm") | (res["vs_method"].isna())]
        print(show.to_string(index=False), flush=True)

    table = pd.concat(frames, ignore_index=True)
    out = PROJECT_ROOT / cfg["outputs"]["results_csv"]
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"wrote {out}", flush=True)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bootstrap.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
