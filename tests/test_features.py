import numpy as np
import pytest

from router.features import HANDCRAFTED_COLUMNS, handcrafted_features
from router.resources import free_ram_gb, require_free_ram


def test_handcrafted_columns_and_shapes():
    out = handcrafted_features(["hello 123", "def foo():\n    return 1"])
    assert list(out.columns) == HANDCRAFTED_COLUMNS
    assert len(out) == 2
    assert out.loc[1, "has_code"] == 1.0
    assert out.loc[0, "digit_ratio"] == pytest.approx(3 / 9)  # "hello 123"
    assert out.loc[0, "has_question_mark"] == 0.0
    assert out.loc[1, "log_words"] > 0


def test_handcrafted_empty_safe():
    out = handcrafted_features([""])
    assert out.loc[0, "digit_ratio"] == 0.0
    assert np.isfinite(out.loc[0, "log_chars"])


def test_require_free_ram_blocks_when_threshold_impossible():
    with pytest.raises(SystemExit, match="Refusing"):
        require_free_ram(10_000.0, "unit-test")


def test_free_ram_gb_returns_positive():
    free = free_ram_gb()
    assert free is None or free > 0
