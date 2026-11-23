"""Where the router loses to the oracle: by source, length, domain; costliest mistakes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from router.config import PROJECT_ROOT, load_config
from router.data import load_outcomes, load_prompts, to_matrices
from router.policy import route
from router.resources import limit_threads, require_free_ram
from router.run_baselines import prepare_split
from router.run_routers import lambda_grid
from router.splits import load_splits

limit_threads(2)


def domain_map(cfg: dict) -> dict[str, str]:
    out = {}
    for domain, sources in cfg["domains"].items():
        for s in sources:
            out[s] = domain
    return out


def pick_lambda(
    p_hat: np.ndarray,
    cost_est: np.ndarray,
    cost_true: np.ndarray,
    lambdas: list[float],
    target_cost: float,
) -> float:
    """Lambda whose realised mean cost is closest to ``target_cost``."""
    best_lam, best_gap = lambdas[0], float("inf")
    for lam in lambdas:
        choices = route(p_hat, cost_est, lam)
        mean_c = float(cost_true[np.arange(len(choices)), choices].mean())
        gap = abs(mean_c - target_cost)
        if gap < best_gap:
            best_lam, best_gap = lam, gap
    return float(best_lam)


def analyse_split(
    split: str,
    router: str,
    cfg: dict,
    outcomes: pd.DataFrame,
    splits: pd.DataFrame,
    prompts: pd.DataFrame,
    base_cfg: dict,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Per-prompt router vs oracle comparison at a matched cost budget."""
    pred_path = PROJECT_ROOT / cfg["predictions_dir"] / f"{split}__{router}.npz"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    raw = np.load(pred_path, allow_pickle=True)
    ev = prepare_split(outcomes, splits, split, base_cfg)
    y, cost, models = to_matrices(outcomes, ev.ids["test"])
    assert list(raw["models"]) == models
    p_hat = raw["p_test"].astype(float)
    cost_est = raw["cost_est_test"].astype(float)

    lambdas = lambda_grid(cfg["lambda_grid"])
    target_cost = cfg["budget_frac_of_best"] * ev.c_hi
    lam = pick_lambda(p_hat, cost_est, cost, lambdas, target_cost)

    router_m = route(p_hat, cost_est, lam)
    oracle_m = route(y, cost, lam)  # oracle uses true quality + true cost at the same λ
    rows = np.arange(len(router_m))

    meta = prompts.set_index("prompt_id").loc[ev.ids["test"]]
    domains = domain_map(cfg)
    sources = (
        outcomes.drop_duplicates("prompt_id").set_index("prompt_id").loc[ev.ids["test"], "source"]
    )

    table = pd.DataFrame(
        {
            "prompt_id": ev.ids["test"],
            "split": split,
            "router": router,
            "lambda": lam,
            "source": sources.to_numpy(),
            "domain": sources.map(lambda s: domains.get(s, "other")).to_numpy(),
            "n_chars": meta["prompt"].str.len().to_numpy(),
            "router_model": np.asarray(models)[router_m],
            "oracle_model": np.asarray(models)[oracle_m],
            "router_quality": y[rows, router_m],
            "oracle_quality": y[rows, oracle_m],
            "router_cost": cost[rows, router_m],
            "oracle_cost": cost[rows, oracle_m],
        }
    )
    table["quality_gap"] = table["oracle_quality"] - table["router_quality"]
    table["cost_gap"] = table["router_cost"] - table["oracle_cost"]
    table["wrong_route"] = table["router_model"] != table["oracle_model"]
    table["length_bin"] = pd.qcut(
        table["n_chars"],
        q=4,
        labels=["p0-25", "p25-50", "p50-75", "p75-100"],
        duplicates="drop",
    )

    def agg(df: pd.DataFrame, key: str) -> pd.DataFrame:
        g = df.groupby(key, observed=True).agg(
            n=("prompt_id", "size"),
            frac_wrong=("wrong_route", "mean"),
            mean_quality_gap=("quality_gap", "mean"),
            mean_router_quality=("router_quality", "mean"),
            mean_oracle_quality=("oracle_quality", "mean"),
            mean_extra_cost=("cost_gap", "mean"),
        )
        g.insert(0, "split", split)
        g.insert(1, "router", router)
        return g.reset_index()

    summaries = {
        "source": agg(table, "source"),
        "length": agg(table, "length_bin"),
        "domain": agg(table, "domain"),
    }
    # Costliest mistakes: wrong route AND we spent more AND got worse or equal quality.
    costly = table[table["wrong_route"] & (table["cost_gap"] > 0) & (table["quality_gap"] >= 0)]
    costly = costly.sort_values("cost_gap", ascending=False)
    return table, {**summaries, "costly": costly}


def _csv_block(df: pd.DataFrame) -> str:
    """Plain CSV fenced block (no tabulate dependency)."""
    return "```csv\n" + df.round(6).to_csv(index=False).rstrip() + "\n```"


def write_summary_md(
    path: Path,
    by_source: pd.DataFrame,
    by_domain: pd.DataFrame,
    by_length: pd.DataFrame,
    costly: pd.DataFrame,
    id_lam: float,
) -> None:
    id_src = by_source[by_source["split"] == "id"].sort_values("mean_quality_gap", ascending=False)
    id_dom = by_domain[by_domain["split"] == "id"]
    top = costly.head(15)[
        [
            "prompt_id",
            "source",
            "router_model",
            "oracle_model",
            "router_quality",
            "oracle_quality",
            "cost_gap",
        ]
    ]
    lines = [
        "# M3 error analysis (measured)",
        "",
        "Router: **lightgbm** (and knn for contrast). Operating point: λ chosen so mean cost "
        f"≈ 25% of always-best cost. On the ID split that λ was **{id_lam:.4g}**.",
        "",
        "Oracle here = route with *true* quality and *true* cost at the same λ.",
        "",
        "## Quality gap by source (ID, lightgbm)",
        "",
        _csv_block(id_src),
        "",
        "## Quality gap by domain (ID, lightgbm)",
        "",
        _csv_block(id_dom),
        "",
        "## Quality gap by prompt-length quartile (ID, lightgbm)",
        "",
        _csv_block(by_length[by_length["split"] == "id"]),
        "",
        "## Costliest wrong routes (ID, lightgbm): spent more, quality ≤ oracle",
        "",
        "Top 15 by extra USD cost:",
        "",
        _csv_block(top),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/error_analysis.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    base_cfg = load_config(cfg["baseline_config"])
    require_free_ram(1.5, "error analysis")

    outcomes, splits = load_outcomes(), load_splits()
    prompts = load_prompts()
    routers = [cfg["router"], cfg["compare_router"]]

    all_source, all_length, all_domain, all_costly = [], [], [], []
    id_lam = float("nan")
    for split in cfg["splits"]:
        for router in routers:
            try:
                table, parts = analyse_split(
                    split, router, cfg, outcomes, splits, prompts, base_cfg
                )
            except FileNotFoundError as exc:
                print(f"skip {exc}")
                continue
            all_source.append(parts["source"])
            all_length.append(parts["length"])
            all_domain.append(parts["domain"])
            top = parts["costly"].head(50).copy()
            all_costly.append(top)
            if split == "id" and router == cfg["router"]:
                id_lam = float(table["lambda"].iloc[0])
                _plot_gap_by_source(parts["source"], PROJECT_ROOT / cfg["outputs"]["figure_dir"])

    by_source = pd.concat(all_source, ignore_index=True)
    by_length = pd.concat(all_length, ignore_index=True)
    by_domain = pd.concat(all_domain, ignore_index=True)
    costly = pd.concat(all_costly, ignore_index=True)

    out = cfg["outputs"]
    for frame, key in [
        (by_source, "by_source_csv"),
        (by_length, "by_length_csv"),
        (by_domain, "by_domain_csv"),
        (costly, "costly_mistakes_csv"),
    ]:
        path = PROJECT_ROOT / out[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    write_summary_md(
        PROJECT_ROOT / out["summary_md"],
        by_source[by_source["router"] == cfg["router"]],
        by_domain[by_domain["router"] == cfg["router"]],
        by_length[by_length["router"] == cfg["router"]],
        costly[(costly["router"] == cfg["router"]) & (costly["split"] == "id")],
        id_lam,
    )
    print(f"wrote {out['summary_md']}")
    print(
        by_source[(by_source.split == "id") & (by_source.router == cfg["router"])]
        .round(4)
        .to_string(index=False)
    )


def _plot_gap_by_source(id_source: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    df = id_source.sort_values("mean_quality_gap", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df["source"], df["mean_quality_gap"])
    ax.set_xlabel("mean (oracle quality − router quality)")
    ax.set_title("LightGBM vs oracle at ~b25 cost (ID test)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "m3_quality_gap_by_source_id.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
