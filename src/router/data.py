"""Load RouterBench's wide pickle and normalise it into tidy Parquet tables.

Raw layout (one row per prompt):
    sample_id, prompt, eval_name,
    <model>                 -> quality score in [0, 1]
    <model>|model_response  -> generated text (not used)
    <model>|total_cost      -> USD cost of that call
    oracle_model_to_route_to (not used; we compute our own oracle)

Outputs:
    outcomes.parquet  long table, one row per (prompt_id, model)
    prompts.parquet   one row per prompt_id with the parsed prompt text
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from router.config import PROJECT_ROOT, load_config

COST_SUFFIX = "|total_cost"

OUTCOME_COLUMNS = ["prompt_id", "source", "eval_name", "model", "quality", "cost"]
PROMPT_COLUMNS = ["prompt_id", "source", "eval_name", "prompt"]

# eval_name -> coarse benchmark family; anything unmatched becomes "misc".
_EXACT_SOURCES = {
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "arc-challenge": "arc_challenge",
    "mbpp": "mbpp",
    "grade-school-math": "gsm8k",
}
_PREFIX_SOURCES = {"mmlu-": "mmlu", "mtbench": "mtbench"}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read a RouterBench pickle. Only use on files from the pinned, trusted revision:
    unpickling can execute arbitrary code."""
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame in {path}, got {type(df).__name__}")
    return df


def detect_models(columns: list[str]) -> list[str]:
    """Model names are the columns that have a matching '<model>|total_cost' column."""
    models = [c.removesuffix(COST_SUFFIX) for c in columns if c.endswith(COST_SUFFIX)]
    missing = [m for m in models if m not in columns]
    if missing:
        raise ValueError(f"Cost column without quality column for: {missing}")
    return sorted(models)


def source_from_eval_name(eval_name: str) -> str:
    """Map a fine-grained eval_name (86 of them) to a benchmark family."""
    if eval_name in _EXACT_SOURCES:
        return _EXACT_SOURCES[eval_name]
    for prefix, source in _PREFIX_SOURCES.items():
        if eval_name.startswith(prefix):
            return source
    return "misc"


def parse_prompt(raw: str) -> str:
    """Prompts are stored as a stringified list of parts, e.g. "['instruction', 'question']".
    Join the parts with blank lines; fall back to the raw string if it isn't a list."""
    try:
        parts = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw
    if isinstance(parts, list) and all(isinstance(p, str) for p in parts):
        return "\n\n".join(parts)
    return raw


def to_outcomes(wide: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Wide (one column per model) -> long (one row per prompt_id x model)."""
    ids = wide[["sample_id", "eval_name"]].rename(columns={"sample_id": "prompt_id"})
    ids["source"] = ids["eval_name"].map(source_from_eval_name)

    quality = wide[models].set_axis(models, axis=1)
    cost = wide[[m + COST_SUFFIX for m in models]].set_axis(models, axis=1)

    long_q = quality.assign(prompt_id=ids["prompt_id"]).melt(
        id_vars="prompt_id", var_name="model", value_name="quality"
    )
    long_c = cost.assign(prompt_id=ids["prompt_id"]).melt(
        id_vars="prompt_id", var_name="model", value_name="cost"
    )
    out = long_q.merge(long_c, on=["prompt_id", "model"], validate="one_to_one")
    out = out.merge(ids, on="prompt_id", validate="many_to_one")

    out = out[OUTCOME_COLUMNS].astype(
        {
            "prompt_id": "string",
            "source": "string",
            "eval_name": "string",
            "model": "string",
            "quality": "float64",
            "cost": "float64",
        }
    )
    return out.sort_values(["prompt_id", "model"], ignore_index=True)


def to_prompts(wide: pd.DataFrame) -> pd.DataFrame:
    """One row per prompt with parsed text."""
    out = pd.DataFrame(
        {
            "prompt_id": wide["sample_id"],
            "source": wide["eval_name"].map(source_from_eval_name),
            "eval_name": wide["eval_name"],
            "prompt": wide["prompt"].map(parse_prompt),
        }
    ).astype("string")
    return out.sort_values("prompt_id", ignore_index=True)


def validate_outcomes(df: pd.DataFrame) -> None:
    """Raise if the tidy table breaks any invariant the rest of the pipeline relies on."""
    if list(df.columns) != OUTCOME_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    if df.duplicated(["prompt_id", "model"]).any():
        raise ValueError("Duplicate (prompt_id, model) rows")
    if df[OUTCOME_COLUMNS].isna().any().any():
        raise ValueError(f"Nulls found:\n{df.isna().sum()}")
    if not df["quality"].between(0.0, 1.0).all():
        raise ValueError("quality outside [0, 1]")
    if not (df["cost"] > 0).all():
        raise ValueError("Non-positive cost")
    per_prompt = df.groupby("prompt_id")["model"].nunique()
    if per_prompt.nunique() != 1:
        raise ValueError("Not every prompt has an outcome for every model")


def build(raw_path: Path, processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: raw pickle -> validated outcomes + prompts Parquet files."""
    wide = load_raw(raw_path)
    if not wide["sample_id"].is_unique:
        raise ValueError(f"sample_id is not unique in {raw_path}")
    models = detect_models(list(wide.columns))
    outcomes = to_outcomes(wide, models)
    validate_outcomes(outcomes)
    prompts = to_prompts(wide)

    processed_dir.mkdir(parents=True, exist_ok=True)
    outcomes.to_parquet(processed_dir / "outcomes.parquet", index=False)
    prompts.to_parquet(processed_dir / "prompts.parquet", index=False)
    return outcomes, prompts


def to_matrices(
    outcomes: pd.DataFrame, prompt_ids: list[str] | pd.Index
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(quality, cost) as (n_prompts, n_models) arrays in ``prompt_ids`` order, plus model names."""
    sub = outcomes[outcomes["prompt_id"].isin(prompt_ids)]
    quality = sub.pivot(index="prompt_id", columns="model", values="quality")
    cost = sub.pivot(index="prompt_id", columns="model", values="cost")
    quality, cost = quality.reindex(prompt_ids), cost.reindex(prompt_ids)
    if quality.isna().any().any() or cost.isna().any().any():
        raise ValueError("Missing outcomes for some requested prompt_ids")
    return quality.to_numpy(), cost[quality.columns].to_numpy(), list(quality.columns)


def load_outcomes(processed_dir: str | Path = "data/processed") -> pd.DataFrame:
    """Load the tidy outcomes table written by ``build``."""
    return pd.read_parquet(PROJECT_ROOT / processed_dir / "outcomes.parquet")


def load_prompts(processed_dir: str | Path = "data/processed") -> pd.DataFrame:
    """Load the prompts table written by ``build``."""
    return pd.read_parquet(PROJECT_ROOT / processed_dir / "prompts.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_path = PROJECT_ROOT / cfg["paths"]["raw_dir"] / cfg["primary_file"]
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    outcomes, prompts = build(raw_path, processed_dir)
    print(
        f"outcomes: {len(outcomes)} rows | prompts: {len(prompts)} | "
        f"models: {outcomes['model'].nunique()} | sources: {outcomes['source'].nunique()}"
    )


if __name__ == "__main__":
    main()
