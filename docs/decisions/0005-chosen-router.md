# ADR 0005: Chosen router for serving and reporting

- Status: accepted (M3)
- Date: 2026-11-24

## Context

M2 trained four per-model predictors (kNN, logistic regression, LightGBM, MLP). We need one
default router for the FastAPI service (M4) and for the portfolio write-up.

## Decision

**Primary router: LightGBM** (embeddings + 9 handcrafted features, soft-label cross-entropy,
early-stopped on val).

Reasons (measured on `docs/results/m2_summary.csv`):

| Method | AUC (ID) | Mean AUC (LOBO) |
|---|---|---|
| LightGBM | **0.757** | 0.683 |
| kNN | 0.756 | **0.687** |
| MLP | 0.756 | 0.677 |
| logreg | 0.753 | 0.666 |
| zero_router | 0.706 | 0.660 |

- Best in-distribution AUC and tied for largest ID gap closed vs the zero router (~25%).
- Small artifact (boosters), fast CPU inference, no embedding index to ship beyond the feature
  vector (unlike kNN, which must store the train set).
- kNN remains the strongest *mean*-LOBO method by a thin margin (+0.004); we keep its
  predictions for analysis but do not serve it by default.

## Calibration (M3)

On the ID test set, LightGBM ECE is already **0.0087**. Isotonic / temperature cut val ECE to ~0
but **do not improve** routing AUC (0.757 → 0.754 / 0.757). We therefore **serve uncalibrated**
scores. On some LOBO folds temperature helps a little (e.g. HellaSwag); that is noted as an
optional flag, not the default.

## Related ADRs

- Embedding model + per-model formulation: `0004`
- Split protocol: `0002`
- Evaluation protocol: `0003`
