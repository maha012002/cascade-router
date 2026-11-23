import numpy as np
import pytest

from router.models import (
    KNNPredictor,
    LightGBMPredictor,
    LogisticPredictor,
    MLPPredictor,
    soft_log_loss,
)
from router.policy import route


def _toy(seed: int = 0, n: int = 80, d: int = 8, m: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    # Soft labels correlated with a linear projection of X.
    logits = X[:, :m]
    Y = 1 / (1 + np.exp(-logits))
    return X, Y, rng


def test_soft_log_loss_perfect_and_worst():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    assert soft_log_loss(y, y) == pytest.approx(0.0, abs=1e-5)
    assert soft_log_loss(y, 1 - y) > 1.0


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (KNNPredictor, {"k": 5}),
        (LogisticPredictor, {"C": 1.0, "max_iter": 500}),
        (LightGBMPredictor, {"num_boost_round": 30, "seed": 0}),
        (MLPPredictor, {"hidden": 16, "max_epochs": 5, "patience": 5, "batch_size": 32, "seed": 0}),
    ],
)
def test_predictor_fit_predict_shape_and_range(cls, kwargs):
    X, Y, _ = _toy()
    X_val, Y_val = X[:20], Y[:20]
    model = cls(**kwargs).fit(X, Y, X_val, Y_val)
    p = model.predict(X_val)
    assert p.shape == Y_val.shape
    assert np.all((p >= 0) & (p <= 1))


def test_knn_predicts_training_label_for_identical_point():
    X, Y, _ = _toy()
    model = KNNPredictor(k=1).fit(X, Y)
    p = model.predict(X[:1])
    np.testing.assert_allclose(p[0], Y[0], atol=1e-9)


def test_policy_with_predictor_outputs():
    X, Y, rng = _toy()
    model = LogisticPredictor(C=1.0, max_iter=500).fit(X, Y, X[:20], Y[:20])
    p = model.predict(X[:10])
    cost = np.abs(rng.normal(size=p.shape)) * 0.001 + 1e-4
    choices = route(p, cost.mean(axis=0), lam=10.0)
    assert choices.shape == (10,)
    assert set(choices).issubset(set(range(p.shape[1])))
