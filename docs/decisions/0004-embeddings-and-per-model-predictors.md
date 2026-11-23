# ADR 0004: Embeddings, training target, and per-model predictors

- Status: accepted (M2)
- Date: 2026-11-24

## Context

M2 needs a prompt representation and a way to turn that into a routing decision under
`m* = argmax_m [q̂(x,m) − λ · ĉ(x,m)]`.

## Decisions

### 1. Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

- 384-d, CPU-friendly, free Hub weights, pinned revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Cached once to `data/processed/embeddings_minilm.parquet` (never re-encoded during train).
- Handcrafted features (length, code, digits, MCQ markers, non-ASCII ratio, …) are concatenated
  for logreg / LightGBM / MLP; kNN uses embeddings only (cosine).

### 2. Training target: expected quality in [0, 1] (soft labels)

- Use the dataset's `quality` as-is (binary or graded).
- Train with soft-label binary cross-entropy so `q̂` is P(correct) on binary sources and an
  expected grade on GSM8K / MT-Bench.
- Matches the evaluation metric (mean quality). Avoids an arbitrary correctness threshold.

### 3. Predict per-model quality, not "best model"

Predicting only `argmax_m quality` cannot trade off cost: at high λ we want the cheapest model
that is *good enough*, not the best model. Per-model scores plug directly into the utility policy
and the λ sweep that traces the cost–quality curve.

### 4. Cost at routing time: prompt-length linear model

True per-prompt costs include output tokens (unknown before generation). We fit
`cost ≈ a + b · n_chars` per LLM on train and use that estimate in the policy. Scoring still uses
realised RouterBench costs.

## Measured outcome (do not invent; from `artifacts/m2_summary.csv`)

| Method | AUC (ID) | Mean AUC (LOBO) | Gap closed vs zero→oracle (ID) |
|---|---|---|---|
| LightGBM | 0.757 | 0.683 | 25% |
| kNN | 0.756 | 0.687 | 25% |
| MLP | 0.756 | 0.677 | 24% |
| logreg | 0.753 | 0.666 | 23% |
| zero_router | 0.706 | 0.660 | 0% |
| oracle | 0.908 | 0.898 | 100% |

On Winogrande-held-out, every learned router is **below** the zero router (AUC ≈ 0.69–0.70 vs
0.716): lexical near-duplicates dominate that benchmark, so prompt-agnostic model ranking wins.
Honest limitation for M3 error analysis.

## Resource safety

Embedding/training refuse to start below a free-RAM threshold; OpenMP/LightGBM/torch capped to
2 threads; embeddings encoded in chunks with the model freed before Parquet write.
