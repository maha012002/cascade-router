"""Train the chosen router on the ID split and write the serving artifact."""

from __future__ import annotations

import argparse

from router.config import PROJECT_ROOT, load_config
from router.data import load_outcomes, load_prompts, to_matrices
from router.features import load_features
from router.models import CostModel, LightGBMPredictor
from router.resources import limit_threads, require_free_ram
from router.run_routers import feature_matrix
from router.serve.artifact import save_artifact
from router.splits import load_splits

limit_threads(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/serve.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    require_free_ram(2.0, "artifact export")

    feat_cfg = load_config(cfg["features_config"])
    routers_cfg = load_config(cfg["routers_config"])
    spec = routers_cfg["routers"][cfg["router"]]
    assert spec["inputs"] == "all", "serving artifact expects embeddings + handcrafted"

    outcomes, splits = load_outcomes(), load_splits()
    features = load_features(cfg["features_config"]).set_index("prompt_id")
    prompts = load_prompts().set_index("prompt_id")
    s = splits[splits["split"] == cfg["split"]]
    train_ids = sorted(s.loc[s["fold"] == "train", "prompt_id"])
    val_ids = sorted(s.loc[s["fold"] == "val", "prompt_id"])

    y_train, cost_train, models = to_matrices(outcomes, train_ids)
    y_val, _, _ = to_matrices(outcomes, val_ids)
    x_train = feature_matrix(features, train_ids, "all")
    x_val = feature_matrix(features, val_ids, "all")

    predictor = LightGBMPredictor(seed=cfg["seed"]).fit(x_train, y_train, x_val, y_val)
    n_chars = prompts.loc[train_ids, "prompt"].str.len().to_numpy()
    cost_model = CostModel().fit(n_chars, cost_train)

    out = PROJECT_ROOT / cfg["artifact_dir"]
    save_artifact(
        out,
        predictor=predictor,
        cost_model=cost_model,
        models=models,
        meta={
            "router_version": cfg["router_version"],
            "router": cfg["router"],
            "split": cfg["split"],
            "seed": cfg["seed"],
            "default_lambda": cfg["default_lambda"],
            "embedding_model": feat_cfg["embedding"]["model_name"],
            "embedding_revision": feat_cfg["embedding"]["revision"],
            "n_features": int(x_train.shape[1]),
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "best_iterations": [int(b.best_iteration) for b in predictor.models_],
        },
    )
    print(f"wrote {out} ({len(models)} models, {x_train.shape[1]} features)")


if __name__ == "__main__":
    main()
