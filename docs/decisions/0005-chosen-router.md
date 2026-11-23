# ADR 0005: Chosen router for serving and reporting

- Status: accepted (M3); robustness note added (M6)
- Date: 2026-11-24

## Context

M2 trained four per-model predictors (kNN, logistic regression, LightGBM, MLP). We need one
default router for the FastAPI service (M4) and for the portfolio write-up.

## Decision

**Primary router: LightGBM** (embeddings + 9 handcrafted features, soft-label cross-entropy,
early-stopped on val).

Reasons (point estimates from `docs/results/m2_summary.csv`):

| Method | AUC (ID) | Mean AUC (LOBO) |
|---|---|---|
| LightGBM | **0.757** | 0.683 |
| kNN | 0.756 | **0.687** |
| MLP | 0.756 | 0.677 |
| logreg | 0.753 | 0.666 |
| zero_router | 0.706 | 0.660 |

- Best *point-estimate* in-distribution AUC and tied for largest ID gap closed vs the zero
  router (~25%). See **Robustness (M6)** for which of those gaps survive bootstrap CIs.
- Small artifact (boosters), fast CPU inference, no embedding index to ship beyond the feature
  vector (unlike kNN, which must store the train set).
- kNN remains the strongest *mean*-LOBO method by a thin margin (+0.004); we keep its
  predictions for analysis but do not serve it by default.

## Calibration (M3)

On the ID test set, LightGBM ECE is already **0.0087**. Isotonic / temperature cut val ECE to ~0
but **do not improve** routing AUC (0.757 → 0.754 / 0.757). We therefore **serve uncalibrated**
scores. On some LOBO folds temperature helps a little (e.g. HellaSwag); that is noted as an
optional flag, not the default.

## Robustness (M6)

Two checks after M5: a **task-aware zero router** (per-source train hull; LOBO held-out → global
fallback) and **paired bootstrap** (1,000 resamples, seed 42) of AUC and cost-to-95%/c_best,
plus LightGBM − other differences. Tables: `docs/results/m6_task_aware.csv`,
`docs/results/m6_bootstrap.csv`.

### What the CIs weaken

- **ID LightGBM vs task-aware is not significant.** Point estimates 0.757 vs 0.754; bootstrap
  ΔAUC mean +0.003, 95% CI **[−0.0003, +0.007]** (contains 0). Cost-to-95% Δ likewise contains 0.
  Most of the ID gain over the prompt-agnostic zero router is explained by knowing the benchmark
  family, not by prompt-level features.
- **ID LightGBM vs knn / mlp AUC gaps contain 0** (Δknn [−0.002, +0.005], Δmlp [−0.002, +0.005]).
  Only **vs logreg** excludes 0 among learned peers (Δ [+0.001, +0.009]).
- Saying “LightGBM is the best learned router on ID” is therefore **over-precise**. Prefer:
  “LightGBM matches the best ID point estimate among learned routers; differences vs kNN/MLP
  are within sampling noise.”

### What still holds

- **ID LightGBM beats the prompt-agnostic zero router.** ΔAUC +0.051, CI **[+0.046, +0.056]**;
  cost-to-95% Δ **[−0.523, −0.475]** (both exclude 0).
- **LOBO is fold-dependent.** vs zero_router (= task_aware under 100% global fallback): AUC CI
  excludes 0 in LightGBM’s favour on mmlu / gsm8k / mbpp / misc; contains 0 on hellaswag /
  arc_challenge; **excludes 0 against LightGBM on winogrande** (Δ [−0.033, −0.003]).
- **Serving choice unchanged:** LightGBM remains the default because (1) it still maximises the
  ID point estimate, (2) the artifact is small and CPU-fast vs kNN, and (3) the zero-router gain
  on ID is robust. We do **not** claim a significant ID edge over a source-aware hull.

## Related ADRs

- Embedding model + per-model formulation: `0004`
- Split protocol: `0002`
- Evaluation protocol: `0003`
