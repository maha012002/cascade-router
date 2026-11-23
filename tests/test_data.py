import pandas as pd
import pytest

from router.config import PROJECT_ROOT
from router.data import (
    OUTCOME_COLUMNS,
    build,
    detect_models,
    load_outcomes,
    load_prompts,
    parse_prompt,
    source_from_eval_name,
    to_outcomes,
    validate_outcomes,
)

EXPECTED_DTYPES = {
    "prompt_id": "string",
    "source": "string",
    "eval_name": "string",
    "model": "string",
    "quality": "float64",
    "cost": "float64",
}


def _check_schema(df: pd.DataFrame) -> None:
    assert list(df.columns) == OUTCOME_COLUMNS
    assert {c: str(t) for c, t in df.dtypes.items()} == EXPECTED_DTYPES
    assert not df.duplicated(["prompt_id", "model"]).any()


# ---------- unit tests (synthetic data, always run) ----------


def test_detect_models_uses_cost_suffix(wide_raw):
    assert detect_models(list(wide_raw.columns)) == ["model-a", "model-b"]


@pytest.mark.parametrize(
    ("eval_name", "source"),
    [
        ("mmlu-anatomy", "mmlu"),
        ("grade-school-math", "gsm8k"),
        ("mtbench-math", "mtbench"),
        ("arc-challenge", "arc_challenge"),
        ("chinese_zodiac", "misc"),
    ],
)
def test_source_mapping(eval_name, source):
    assert source_from_eval_name(eval_name) == source


def test_parse_prompt():
    assert parse_prompt("['a', 'b']") == "a\n\nb"
    assert parse_prompt("['only']") == "only"
    assert parse_prompt("plain text") == "plain text"


def test_to_outcomes_shape_and_schema(wide_raw):
    out = to_outcomes(wide_raw, detect_models(list(wide_raw.columns)))
    assert len(out) == 3 * 2
    _check_schema(out)
    validate_outcomes(out)


def test_to_outcomes_keeps_quality_and_cost_aligned(wide_raw):
    out = to_outcomes(wide_raw, detect_models(list(wide_raw.columns)))
    row = out.query("prompt_id == 'grade-school-math.dev.0' and model == 'model-a'").iloc[0]
    assert row["quality"] == 0.5
    assert row["cost"] == 0.0002
    assert row["source"] == "gsm8k"


def test_validate_rejects_duplicates(wide_raw):
    out = to_outcomes(wide_raw, detect_models(list(wide_raw.columns)))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_outcomes(pd.concat([out, out.iloc[[0]]], ignore_index=True))


def test_validate_rejects_nonpositive_cost(wide_raw):
    out = to_outcomes(wide_raw, detect_models(list(wide_raw.columns)))
    out.loc[0, "cost"] = 0.0
    with pytest.raises(ValueError, match="cost"):
        validate_outcomes(out)


def test_build_roundtrips_through_parquet(wide_raw, tmp_path):
    raw_path = tmp_path / "raw.pkl"
    wide_raw.to_pickle(raw_path)
    build(raw_path, tmp_path)
    _check_schema(pd.read_parquet(tmp_path / "outcomes.parquet"))


# ---------- integration tests (real data, skipped if not built) ----------

_HAS_DATA = (PROJECT_ROOT / "data/processed/outcomes.parquet").exists()
needs_data = pytest.mark.skipif(not _HAS_DATA, reason="run `make data` first")


@pytest.mark.integration
@needs_data
def test_real_outcomes_schema():
    df = load_outcomes()
    _check_schema(df)
    validate_outcomes(df)


@pytest.mark.integration
@needs_data
def test_real_outcomes_cover_every_prompt_model_pair():
    df = load_outcomes()
    prompts = load_prompts()
    assert len(df) == len(prompts) * df["model"].nunique()
    assert set(df["prompt_id"]) == set(prompts["prompt_id"])
