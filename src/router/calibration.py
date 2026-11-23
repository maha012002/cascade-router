"""Calibrate per-model predicted quality and measure Expected Calibration Error (ECE).

Fits are always on validation predictions. Soft labels (graded quality in [0, 1]) are
supported: ECE compares mean predicted quality to mean realised quality in each bin.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """ECE over flattened (prompt, model) pairs. Bins by predicted quality in [0, 1]."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.clip(np.asarray(y_prob, dtype=float).ravel(), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        # Last bin is closed on the right so 1.0 is included.
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def reliability_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> list[dict[str, float]]:
    """Per-bin counts for reliability diagrams."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.clip(np.asarray(y_prob, dtype=float).ravel(), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": float(mask.sum()),
                "mean_pred": float(y_prob[mask].mean()),
                "mean_true": float(y_true[mask].mean()),
            }
        )
    return rows


class IsotonicCalibrator:
    """One isotonic regression per model column, fit on validation (p_hat, quality)."""

    def fit(self, p_val: np.ndarray, y_val: np.ndarray) -> IsotonicCalibrator:
        self.models_ = []
        for j in range(p_val.shape[1]):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(p_val[:, j], y_val[:, j])
            self.models_.append(iso)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict(p[:, j]) for j, m in enumerate(self.models_)])


class TemperatureCalibrator:
    """One temperature T_j > 0 per model: sigmoid(logit(p) / T). Soft-label NLL on val."""

    def __init__(self, grid: np.ndarray | None = None):
        self.grid = grid if grid is not None else np.geomspace(0.25, 4.0, 40)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p) - np.log(1 - p)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _nll(y: np.ndarray, p: np.ndarray) -> float:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit(self, p_val: np.ndarray, y_val: np.ndarray) -> TemperatureCalibrator:
        logits = self._logit(p_val)
        temps = []
        for j in range(p_val.shape[1]):
            best_t, best_loss = 1.0, self._nll(y_val[:, j], p_val[:, j])
            for t in self.grid:
                loss = self._nll(y_val[:, j], self._sigmoid(logits[:, j] / t))
                if loss < best_loss:
                    best_t, best_loss = float(t), loss
            temps.append(best_t)
        self.temperatures_ = np.asarray(temps, dtype=float)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        logits = self._logit(p)
        return self._sigmoid(logits / self.temperatures_)
