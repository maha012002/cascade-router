"""In-process router: embed prompt -> predict -> utility policy."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from router.policy import route
from router.serve.artifact import features_for_prompt, load_artifact


@dataclass
class RouteResult:
    chosen_model: str
    per_model_p_correct: dict[str, float]
    expected_cost: dict[str, float]
    router_version: str
    lambda_used: float
    cached: bool = False


class PromptCache:
    """Optional Redis cache; falls back to an in-process dict. Disabled when enable=False."""

    def __init__(self, enable: bool, redis_url: str | None, ttl_seconds: int):
        self.enable = enable
        self.ttl = ttl_seconds
        self._mem: dict[str, dict[str, Any]] = {}
        self._redis = None
        if enable and redis_url:
            try:
                import redis

                client = redis.Redis.from_url(redis_url, socket_connect_timeout=0.5)
                client.ping()
                self._redis = client
            except Exception:
                self._redis = None  # silent fallback

    @staticmethod
    def key(prompt: str, lam: float) -> str:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return f"cascade-router:{digest}:{lam:.8g}"

    def get(self, prompt: str, lam: float) -> dict[str, Any] | None:
        if not self.enable:
            return None
        k = self.key(prompt, lam)
        if self._redis is not None:
            import json

            raw = self._redis.get(k)
            return json.loads(raw) if raw else None
        return self._mem.get(k)

    def set(self, prompt: str, lam: float, value: dict[str, Any]) -> None:
        if not self.enable:
            return
        k = self.key(prompt, lam)
        if self._redis is not None:
            import json

            self._redis.setex(k, self.ttl, json.dumps(value))
        else:
            self._mem[k] = value


class RouterRuntime:
    """Loads the artifact + sentence encoder once; routes individual prompts."""

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        enable_cache: bool = True,
        cache_ttl_seconds: int = 3600,
        redis_url: str | None = None,
        device: str = "cpu",
        embed_fn=None,
    ):
        bundle = load_artifact(Path(artifact_dir))
        self.meta = bundle["meta"]
        self.predictor = bundle["predictor"]
        self.cost_model = bundle["cost_model"]
        self.models: list[str] = list(bundle["models"])
        self.version = str(self.meta["router_version"])
        self.default_lambda = float(self.meta.get("default_lambda", 40.0))
        self.cache = PromptCache(
            enable_cache, redis_url or os.environ.get("CACHE_REDIS_URL"), cache_ttl_seconds
        )
        self._encoder = None
        self._device = device
        self._emb_name = self.meta["embedding_model"]
        self._emb_revision = self.meta["embedding_revision"]
        self._embed_fn = embed_fn  # optional override for tests (skips MiniLM load)

    def _ensure_encoder(self):
        if self._embed_fn is not None:
            return None
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                self._emb_name, revision=self._emb_revision, device=self._device
            )
        return self._encoder

    def embed(self, prompt: str) -> np.ndarray:
        if self._embed_fn is not None:
            return np.asarray(self._embed_fn(prompt), dtype=np.float32)
        enc = self._ensure_encoder()
        return enc.encode(
            [prompt],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

    def predict_quality_and_cost(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        emb = self.embed(prompt)
        x = features_for_prompt(prompt, emb)
        p = self.predictor.predict(x)[0]
        cost = self.cost_model.predict(np.array([len(prompt)]))[0]
        return p, cost

    def route_prompt(self, prompt: str, lam: float | None = None) -> RouteResult:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be non-empty")
        lam_used = self.default_lambda if lam is None else float(lam)
        if lam_used < 0:
            raise ValueError("lambda must be >= 0")

        cached = self.cache.get(prompt, lam_used)
        if cached is not None:
            return RouteResult(**cached, cached=True)

        p, cost = self.predict_quality_and_cost(prompt)
        choice = int(route(p.reshape(1, -1), cost.reshape(1, -1), lam_used)[0])
        result = RouteResult(
            chosen_model=self.models[choice],
            per_model_p_correct={m: float(p[i]) for i, m in enumerate(self.models)},
            expected_cost={m: float(cost[i]) for i, m in enumerate(self.models)},
            router_version=self.version,
            lambda_used=lam_used,
            cached=False,
        )
        self.cache.set(
            prompt,
            lam_used,
            {
                "chosen_model": result.chosen_model,
                "per_model_p_correct": result.per_model_p_correct,
                "expected_cost": result.expected_cost,
                "router_version": result.router_version,
                "lambda_used": result.lambda_used,
            },
        )
        return result
