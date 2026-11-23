import pandas as pd
import pytest

MODELS = ["model-b", "model-a"]


@pytest.fixture
def wide_raw() -> pd.DataFrame:
    """Tiny frame mimicking RouterBench's wide pickle layout (3 prompts x 2 models)."""
    return pd.DataFrame(
        {
            "sample_id": ["mmlu-anatomy.val.0", "grade-school-math.dev.0", "chinese_poem.dev.1"],
            "prompt": ["['What is a bone?']", "['Solve:', '2+2']", "not a list"],
            "eval_name": ["mmlu-anatomy", "grade-school-math", "chinese_poem"],
            "model-b": [1.0, 0.75, 0.0],
            "model-a": [0.0, 0.5, 1.0],
            "model-b|model_response": ["x", "y", "z"],
            "model-a|model_response": ["x", "y", "z"],
            "model-b|total_cost": [0.002, 0.003, 0.001],
            "model-a|total_cost": [0.0001, 0.0002, 0.0003],
            "oracle_model_to_route_to": ["model-b", "model-b", "model-a"],
        }
    )
