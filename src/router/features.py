"""Prompt features: cached sentence embeddings + simple handcrafted signals.

Embeddings are computed once for every prompt and written to Parquet. Downstream
training only joins on ``prompt_id``; it never re-runs the encoder.

Resource safety: thread caps, RAM gate, small batches, encode in chunks, free the
model before writing Parquet.
"""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd

from router.config import PROJECT_ROOT, load_config
from router.data import load_prompts
from router.resources import limit_threads, require_free_ram

HANDCRAFTED_COLUMNS = [
    "log_chars",
    "log_words",
    "log_lines",
    "log_numbers",
    "digit_ratio",
    "non_ascii_ratio",
    "has_code",
    "has_choices",
    "has_question_mark",
]

_CODE_RE = re.compile(
    r"```|def\s+\w+\(|class\s+\w+|import\s+\w+|from\s+\w+\s+import|;\s*$|{\s*$",
    re.MULTILINE,
)
_CHOICE_RE = re.compile(r"\b[A-D]\)|\b[A-D]\.")


def handcrafted_features(texts: list[str] | pd.Series) -> pd.DataFrame:
    """Cheap signals that embeddings may under-weight (length, code, digits, MCQ, etc.)."""
    series = pd.Series(list(texts), dtype="string").fillna("")
    n_chars = series.str.len().astype(float)
    n_words = series.str.split().str.len().astype(float)
    n_lines = series.str.count("\n").astype(float) + 1.0
    n_numbers = series.str.count(r"\d+").astype(float)
    n_digits = series.str.count(r"\d").astype(float)
    n_non_ascii = series.map(lambda t: float(sum(ord(c) > 127 for c in t)))
    return pd.DataFrame(
        {
            "log_chars": np.log1p(n_chars),
            "log_words": np.log1p(n_words),
            "log_lines": np.log1p(n_lines),
            "log_numbers": np.log1p(n_numbers),
            "digit_ratio": np.where(n_chars > 0, n_digits / n_chars, 0.0),
            "non_ascii_ratio": np.where(n_chars > 0, n_non_ascii / n_chars, 0.0),
            "has_code": series.map(lambda t: float(bool(_CODE_RE.search(t)))),
            "has_choices": series.map(lambda t: float(bool(_CHOICE_RE.search(t)))),
            "has_question_mark": series.str.contains(r"\?", regex=True).astype(float),
        }
    )


def embed_texts(
    texts: list[str],
    model_name: str,
    revision: str,
    batch_size: int,
    normalize: bool,
    seed: int,
    chunk_size: int = 2000,
) -> np.ndarray:
    """Encode texts on CPU in chunks so peak RAM stays bounded."""
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "2")))
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SentenceTransformer(model_name, revision=revision, device="cpu")
    parts: list[np.ndarray] = []
    try:
        for start in range(0, len(texts), chunk_size):
            chunk = texts[start : start + chunk_size]
            vec = model.encode(
                chunk,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            ).astype(np.float32, copy=False)
            parts.append(vec)
            print(f"  embedded {min(start + chunk_size, len(texts))}/{len(texts)}")
    finally:
        # Drop the model before we allocate the combined DataFrame + Parquet write.
        del model
        gc.collect()

    return np.vstack(parts)


def build_embeddings(cfg: dict) -> pd.DataFrame:
    """Embed every prompt -> ``embeddings.parquet`` with columns prompt_id, e0, e1, ..."""
    prompts = load_prompts().sort_values("prompt_id").reset_index(drop=True)
    emb_cfg = cfg["embedding"]
    vectors = embed_texts(
        prompts["prompt"].tolist(),
        model_name=emb_cfg["model_name"],
        revision=emb_cfg["revision"],
        batch_size=emb_cfg["batch_size"],
        normalize=emb_cfg["normalize"],
        seed=cfg["seed"],
        chunk_size=emb_cfg.get("chunk_size", 2000),
    )
    cols = {f"e{i}": vectors[:, i] for i in range(vectors.shape[1])}
    out = pd.DataFrame({"prompt_id": prompts["prompt_id"].astype("string"), **cols})
    path = PROJECT_ROOT / cfg["paths"]["embeddings"]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    del vectors
    gc.collect()
    return out


def build_handcrafted(cfg: dict) -> pd.DataFrame:
    """Compute handcrafted features for every prompt -> ``features.parquet``."""
    prompts = load_prompts().sort_values("prompt_id").reset_index(drop=True)
    feats = handcrafted_features(prompts["prompt"])
    out = pd.concat([prompts[["prompt_id"]].astype("string"), feats], axis=1)
    path = PROJECT_ROOT / cfg["paths"]["features"]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def load_features(config_path: str | Path = "configs/features.yaml") -> pd.DataFrame:
    """Join embeddings + handcrafted into one table (one row per prompt_id)."""
    cfg = load_config(config_path)
    emb = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["embeddings"])
    hand = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["features"])
    out = emb.merge(hand, on="prompt_id", validate="one_to_one")
    if out.isna().any().any():
        raise ValueError("NaNs after joining embeddings and handcrafted features")
    return out


def load_feature_matrix(
    prompt_ids: list[str] | pd.Index,
    embeddings_path: str | Path = "data/processed/embeddings.parquet",
    handcrafted_path: str | Path = "data/processed/features.parquet",
) -> np.ndarray:
    """Return an (n_prompts, n_features) matrix in ``prompt_ids`` order."""
    emb = pd.read_parquet(PROJECT_ROOT / embeddings_path).set_index("prompt_id")
    hand = pd.read_parquet(PROJECT_ROOT / handcrafted_path).set_index("prompt_id")
    emb, hand = emb.reindex(prompt_ids), hand.reindex(prompt_ids)
    if emb.isna().any().any() or hand.isna().any().any():
        raise ValueError("Missing features for some prompt_ids; re-run `make embed`")
    return np.hstack([emb.to_numpy(dtype=np.float32), hand.to_numpy(dtype=np.float32)])


def main() -> None:
    limit_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/features.yaml")
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="only rebuild handcrafted features (keeps existing embeddings.parquet)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.skip_embeddings:
        emb_path = PROJECT_ROOT / cfg["paths"]["embeddings"]
        if not emb_path.exists():
            raise FileNotFoundError(f"{emb_path} missing; run without --skip-embeddings")
        emb = pd.read_parquet(emb_path)
    else:
        require_free_ram(cfg.get("min_free_ram_gb", 3.0), "embedding")
        emb = build_embeddings(cfg)
    hand = build_handcrafted(cfg)
    print(
        f"embeddings: {emb.shape[0]} x {emb.shape[1] - 1} | "
        f"handcrafted: {hand.shape[0]} x {len(HANDCRAFTED_COLUMNS)}"
    )


if __name__ == "__main__":
    main()
