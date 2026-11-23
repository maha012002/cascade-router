# Work log

| Date | What I did | The number | What's next |
|---|---|---|---|
| 2026-11-23 | M0: repo skeleton, pinned download, tidy Parquet, tests, CI, EDA, data card | 36,497 prompts × 11 models; Mistral-7B already matches the best model on 31.2% of prompts; 3.6% unsolved | M1: splits, baselines, eval harness |
| 2026-11-23 | M1: near-dup groups (TF-IDF, cos ≥ 0.7), ID + 7 leave-one-benchmark-out splits, baselines, exact oracle, metrics, MLflow (SQLite) | ID test AUC: zero router 0.706, oracle 0.908, random mixture 0.547; fixed a test-leak in the zero-router baseline | M2: embeddings + learned routers; decide target for graded labels |
| 2026-11-24 | M2: MiniLM embeddings cached; kNN/logreg/LightGBM/MLP per-model predictors; ID+7 LOBO evals; RAM gates | ID AUC: LightGBM 0.757 > zero 0.706; LOBO mean kNN 0.687; Winogrande OOD learned < zero router | M3: calibration + error analysis |
| 2026-11-24 | M3: ECE + isotonic/temperature; error analysis vs oracle; chose LightGBM | ID LGBM ECE 0.0087, cal does not raise AUC (0.757→0.754); biggest quality gaps: winogrande/mmlu; costliest mistakes: GPT-4 on CRT/MT-Bench | M4: FastAPI serve |
| 2026-11-24 | M4: FastAPI /health+/route, LightGBM artifact, optional Redis, Docker, latency bench | CPU latency p50 16.3 ms / p95 18.6 ms; offline-online parity passed | M5: documentation |
| 2026-11-24 | M5: full README (problem→reproduce) with measured-only tables, links to ADRs/results/figures | README documents ID LGBM AUC 0.757, latency p50 16.3 ms, calibration no-op | Done — commit/push when ready |
