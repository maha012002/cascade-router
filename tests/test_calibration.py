import numpy as np
import pytest

from router.calibration import (
    IsotonicCalibrator,
    TemperatureCalibrator,
    expected_calibration_error,
)


def test_ece_zero_when_perfectly_calibrated():
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.5])
    assert expected_calibration_error(y, y, n_bins=6) == pytest.approx(0.0, abs=1e-12)


def test_ece_positive_when_overconfident():
    y = np.array([0.0, 0.0, 0.0, 1.0])
    p = np.array([0.9, 0.9, 0.9, 0.9])
    assert expected_calibration_error(y, p, n_bins=5) > 0.3


def test_isotonic_reduces_ece_on_biased_scores():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 1, size=(200, 3))
    # Systematically over-confident predictions.
    p = np.clip(y * 0.5 + 0.4, 0, 1)
    cal = IsotonicCalibrator().fit(p, y)
    p_cal = cal.transform(p)
    assert expected_calibration_error(y, p_cal) < expected_calibration_error(y, p)


def test_temperature_learns_t_and_stays_in_unit_interval():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=(150, 2)).astype(float)
    p = np.clip(y * 0.7 + 0.2 + rng.normal(0, 0.05, size=y.shape), 0.01, 0.99)
    cal = TemperatureCalibrator().fit(p, y)
    out = cal.transform(p)
    assert out.shape == p.shape
    assert np.all((out >= 0) & (out <= 1))
    assert np.all(cal.temperatures_ > 0)
