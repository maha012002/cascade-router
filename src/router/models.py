"""Per-model quality predictors: X (n, d) -> p_hat (n, n_models) in [0, 1].

Target Y is the dataset's quality in [0, 1] (binary for most sources, graded for GSM8K /
MT-Bench). Every model minimises soft-label cross-entropy, so on binary sources p_hat is
P(correct) and on graded sources it is the expected grade.
"""

from __future__ import annotations

import numpy as np
import torch
from lightgbm import Booster, Dataset, early_stopping
from lightgbm import train as lgb_train
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

EPS = 1e-6


def soft_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary cross-entropy with soft targets y in [0, 1]."""
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class KNNPredictor:
    """Average quality of the k most similar training prompts (cosine on embeddings)."""

    def __init__(self, k: int = 50):
        self.k = k

    def fit(self, X, Y, X_val=None, Y_val=None) -> KNNPredictor:
        self.nn_ = NearestNeighbors(
            n_neighbors=self.k, metric="cosine", algorithm="brute", n_jobs=1
        )
        self.nn_.fit(X)
        self.Y_ = np.asarray(Y, dtype=float)
        return self

    def predict(self, X) -> np.ndarray:
        idx = self.nn_.kneighbors(X, return_distance=False)
        return self.Y_[idx].mean(axis=1)


class LogisticPredictor:
    """One L2-regularised logistic regression per LLM.

    Soft labels: each row appears twice, as y=1 with weight q and y=0 with weight 1-q, which
    gives exactly the soft-label cross-entropy.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 2000):
        self.C, self.max_iter = C, max_iter

    def fit(self, X, Y, X_val=None, Y_val=None) -> LogisticPredictor:
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        X2 = np.vstack([Xs, Xs])
        self.models_ = []
        for q in np.asarray(Y, dtype=float).T:
            y2 = np.r_[np.ones_like(q), np.zeros_like(q)]
            w2 = np.r_[q, 1 - q]
            keep = w2 > 0
            clf = LogisticRegression(C=self.C, max_iter=self.max_iter)
            self.models_.append(clf.fit(X2[keep], y2[keep], sample_weight=w2[keep]))
        return self

    def predict(self, X) -> np.ndarray:
        Xs = self.scaler_.transform(X)
        return np.column_stack([m.predict_proba(Xs)[:, 1] for m in self.models_])


class LightGBMPredictor:
    """One gradient-boosted tree model per LLM, cross-entropy objective, early-stopped on val."""

    def __init__(self, params: dict | None = None, num_boost_round: int = 2000, seed: int = 0):
        self.params = {
            "objective": "cross_entropy",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.5,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": seed,
            "deterministic": True,
            "num_threads": 2,  # keep the laptop responsive
            **(params or {}),
        }
        self.num_boost_round = num_boost_round

    def fit(self, X, Y, X_val, Y_val) -> LightGBMPredictor:
        self.models_: list[Booster] = []
        for j in range(Y.shape[1]):
            train_set = Dataset(X, label=Y[:, j], free_raw_data=False)
            val_set = Dataset(X_val, label=Y_val[:, j], reference=train_set)
            booster = lgb_train(
                self.params,
                train_set,
                num_boost_round=self.num_boost_round,
                valid_sets=[val_set],
                callbacks=[early_stopping(50, verbose=False)],
            )
            self.models_.append(booster)
        return self

    def predict(self, X) -> np.ndarray:
        return np.column_stack([b.predict(X, num_iteration=b.best_iteration) for b in self.models_])


class MLPPredictor:
    """Small multi-task MLP: shared hidden layer, one sigmoid output per LLM."""

    def __init__(
        self,
        hidden: int = 256,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 100,
        patience: int = 5,
        seed: int = 0,
    ):
        self.hidden, self.dropout, self.lr = hidden, dropout, lr
        self.weight_decay, self.batch_size = weight_decay, batch_size
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed

    def _tensor(self, X) -> torch.Tensor:
        return torch.as_tensor(self.scaler_.transform(X), dtype=torch.float32)

    def fit(self, X, Y, X_val, Y_val) -> MLPPredictor:
        torch.set_num_threads(2)
        torch.manual_seed(self.seed)
        self.scaler_ = StandardScaler().fit(X)
        Xt, Yt = self._tensor(X), torch.as_tensor(Y, dtype=torch.float32)
        Xv, Yv = self._tensor(X_val), torch.as_tensor(Y_val, dtype=torch.float32)

        self.net_ = torch.nn.Sequential(
            torch.nn.Linear(Xt.shape[1], self.hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden, Yt.shape[1]),
        )
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()  # accepts soft targets in [0, 1]
        gen = torch.Generator().manual_seed(self.seed)

        best_loss, best_state, bad_epochs = float("inf"), None, 0
        for _ in range(self.max_epochs):
            self.net_.train()
            for idx in torch.randperm(len(Xt), generator=gen).split(self.batch_size):
                opt.zero_grad()
                loss_fn(self.net_(Xt[idx]), Yt[idx]).backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net_(Xv), Yv).item()
            if val_loss < best_loss - 1e-5:
                best_loss, bad_epochs = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        return self

    def predict(self, X) -> np.ndarray:
        self.net_.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net_(self._tensor(X))).numpy().astype(float)


class CostModel:
    """Per-LLM linear model: cost ~ a + b * prompt_chars, fit on train (realised costs).

    At routing time the answer (output tokens) is unknown, but prompt length is known.
    """

    def fit(self, n_chars: np.ndarray, cost: np.ndarray) -> CostModel:
        x = np.asarray(n_chars, dtype=float).reshape(-1, 1)
        self.models_ = [LinearRegression().fit(x, c) for c in np.asarray(cost).T]
        self.floor_ = np.asarray(cost).min(axis=0)
        return self

    def predict(self, n_chars: np.ndarray) -> np.ndarray:
        x = np.asarray(n_chars, dtype=float).reshape(-1, 1)
        pred = np.column_stack([m.predict(x) for m in self.models_])
        return np.maximum(pred, self.floor_)  # never predict below the cheapest seen call


PREDICTORS = {
    "knn": KNNPredictor,
    "logreg": LogisticPredictor,
    "lightgbm": LightGBMPredictor,
    "mlp": MLPPredictor,
}
