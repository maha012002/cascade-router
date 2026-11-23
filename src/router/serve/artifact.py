"""Save / load the deployable router bundle (LightGBM + cost model + metadata)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from router.models import CostModel, LightGBMPredictor

META_NAME = "meta.json"
PREDICTOR_NAME = "predictor.joblib"
COST_NAME = "cost_model.joblib"


def save_artifact(
    path: Path,
    *,
    predictor: LightGBMPredictor,
    cost_model: CostModel,
    models: list[str],
    meta: dict[str, Any],
) -> Path:
    """Write the serving bundle. MiniLM weights are *not* included (loaded by name at runtime)."""
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, path / PREDICTOR_NAME)
    joblib.dump(cost_model, path / COST_NAME)
    payload = {
        **meta,
        "models": models,
        "n_models": len(models),
    }
    (path / META_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_artifact(path: Path) -> dict[str, Any]:
    """Load predictor, cost model, and metadata from ``path``."""
    path = Path(path)
    meta = json.loads((path / META_NAME).read_text(encoding="utf-8"))
    predictor: LightGBMPredictor = joblib.load(path / PREDICTOR_NAME)
    cost_model: CostModel = joblib.load(path / COST_NAME)
    return {
        "meta": meta,
        "predictor": predictor,
        "cost_model": cost_model,
        "models": meta["models"],
    }


def features_for_prompt(
    prompt: str,
    embedding: np.ndarray,
) -> np.ndarray:
    """Build the (1, d) feature row: 384-d embedding + handcrafted columns."""
    from router.features import HANDCRAFTED_COLUMNS, handcrafted_features

    hand = handcrafted_features([prompt])[HANDCRAFTED_COLUMNS].to_numpy(dtype=np.float32)
    emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    return np.hstack([emb, hand])
