# Work log

| Date | What I did | The number | What's next |
|---|---|---|---|
| 2026-11-23 | M0: repo skeleton, pinned download, tidy Parquet, tests, CI, EDA, data card | 36,497 prompts × 11 models; Mistral-7B already matches the best model on 31.2% of prompts; 3.6% unsolved | M1: splits, baselines, eval harness |
| 2026-11-23 | M1: near-dup groups (TF-IDF, cos ≥ 0.7), ID + 7 leave-one-benchmark-out splits, baselines, exact oracle, metrics, MLflow (SQLite) | ID test AUC: zero router 0.706, oracle 0.908, random mixture 0.547; fixed a test-leak in the zero-router baseline | M2: embeddings + learned routers; decide target for graded labels |
