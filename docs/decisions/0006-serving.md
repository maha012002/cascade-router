# ADR 0006: Serving design

- Status: accepted (M4)
- Date: 2026-11-24

## Decision

- Ship **LightGBM-ID** as `artifacts/serve/lightgbm-id-v1/` (boosters + cost model + meta).
  MiniLM weights are **not** in the bundle; the API loads
  `sentence-transformers/all-MiniLM-L6-v2` at the pinned revision on startup.
- FastAPI process loads the artifact **once** in the lifespan hook.
- `POST /route` accepts `lambda` **or** `budget` (not both). Budget = highest-`p` model with
  expected cost ≤ budget, else cheapest.
- Cache: in-process dict by default; Redis if `CACHE_REDIS_URL` is set and reachable.
- Measured CPU latency (n=1000, cache off, 20 warmup): **p50 = 18.7 ms, p95 = 39.1 ms**,
  mean 22.8 ms. (Earlier n=30 spot check was p50 16.3 / p95 18.6; the larger sample includes
  longer prompts and a heavier tail.)
