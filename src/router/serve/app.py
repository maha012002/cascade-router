"""FastAPI entrypoint: load the router once at startup, serve /health and /route."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from router.config import PROJECT_ROOT, load_config
from router.resources import limit_threads
from router.serve.runtime import RouterRuntime

limit_threads(2)

_RUNTIME: RouterRuntime | None = None


class RouteRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    lambda_: float | None = Field(default=None, alias="lambda", ge=0.0)
    budget: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def at_most_one_tradeoff(self) -> RouteRequest:
        if self.lambda_ is not None and self.budget is not None:
            raise ValueError("provide at most one of 'lambda' or 'budget'")
        return self


class RouteResponse(BaseModel):
    chosen_model: str
    per_model_p_correct: dict[str, float]
    expected_cost: dict[str, float]
    router_version: str
    lambda_used: float | None
    budget_used: float | None
    cached: bool


class HealthResponse(BaseModel):
    status: str
    router_version: str
    n_models: int
    cache_enabled: bool


def _budget_route(p: np.ndarray, cost: np.ndarray, models: list[str], budget: float) -> str:
    """Pick the highest-p model with expected cost <= budget; else the cheapest."""
    feasible = [i for i, c in enumerate(cost) if c <= budget]
    if not feasible:
        return models[int(np.argmin(cost))]
    best = max(feasible, key=lambda i: (p[i], -cost[i]))
    return models[best]


def get_runtime() -> RouterRuntime:
    if _RUNTIME is None:
        raise HTTPException(status_code=503, detail="router not loaded")
    return _RUNTIME


def create_app(runtime: RouterRuntime | None = None) -> FastAPI:
    """Build the API. If ``runtime`` is given (tests), skip disk/Hub loading."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _RUNTIME
        if runtime is not None:
            _RUNTIME = runtime
        else:
            cfg_path = os.environ.get("SERVE_CONFIG", "configs/serve.yaml")
            cfg = load_config(cfg_path)
            artifact = PROJECT_ROOT / cfg["artifact_dir"]
            if not (artifact / "meta.json").exists():
                raise RuntimeError(
                    f"serving artifact missing at {artifact}; "
                    "run: python -m router.serve.export_artifact"
                )
            _RUNTIME = RouterRuntime(
                artifact,
                enable_cache=bool(cfg.get("enable_cache", True)),
                cache_ttl_seconds=int(cfg.get("cache_ttl_seconds", 3600)),
                redis_url=os.environ.get("CACHE_REDIS_URL"),
                device=os.environ.get("EMBED_DEVICE", "cpu"),
            )
            _RUNTIME._ensure_encoder()
        yield
        _RUNTIME = None

    api = FastAPI(title="cascade-router", version="0.0.1", lifespan=lifespan)

    @api.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        rt = get_runtime()
        return HealthResponse(
            status="ok",
            router_version=rt.version,
            n_models=len(rt.models),
            cache_enabled=rt.cache.enable,
        )

    @api.post("/route", response_model=RouteResponse)
    def route_endpoint(body: RouteRequest) -> RouteResponse:
        rt = get_runtime()
        prompt = body.prompt

        if body.budget is not None:
            p, cost = rt.predict_quality_and_cost(prompt)
            chosen = _budget_route(p, cost, rt.models, body.budget)
            return RouteResponse(
                chosen_model=chosen,
                per_model_p_correct={m: float(p[i]) for i, m in enumerate(rt.models)},
                expected_cost={m: float(cost[i]) for i, m in enumerate(rt.models)},
                router_version=rt.version,
                lambda_used=None,
                budget_used=body.budget,
                cached=False,
            )

        try:
            result = rt.route_prompt(prompt, lam=body.lambda_)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return RouteResponse(
            chosen_model=result.chosen_model,
            per_model_p_correct=result.per_model_p_correct,
            expected_cost=result.expected_cost,
            router_version=result.router_version,
            lambda_used=result.lambda_used,
            budget_used=None,
            cached=result.cached,
        )

    return api


app = create_app()
