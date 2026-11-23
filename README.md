# cascade-router

A learned router that decides, per query, which LLM (from cheap to expensive) to use, maximising
answer quality per dollar. Evaluated offline on RouterBench precomputed LLM outputs, served via a
FastAPI endpoint.

> Status: **M1 (splits, baselines, evaluation harness)** complete. Full README is written in M5.
> See [docs/DATA_CARD.md](docs/DATA_CARD.md) and [docs/decisions/](docs/decisions/).

## Quickstart

```bash
python -m venv .venv          # Python 3.11
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
make setup                    # pip install -r requirements-dev.txt -e .
make data                     # download RouterBench @ pinned revision -> data/processed/*.parquet
make splits                   # near-duplicate groups + ID / leave-one-benchmark-out splits
make baselines                # baselines + oracle on every split -> docs/figures, MLflow
make mlflow-ui                # browse runs (SQLite store: mlflow.db)
make test                     # unit + integration tests
make eda                      # re-execute notebooks/01_eda.ipynb
```

## Data and acknowledgements

This project uses the [RouterBench](https://huggingface.co/datasets/withmartian/routerbench)
dataset by Martian (Hu et al., 2024, [arXiv:2403.12031](https://arxiv.org/abs/2403.12031)),
which bundles MMLU, HellaSwag, GSM8K, ARC, WinoGrande, MBPP, MT-Bench and other tasks. The data is
**not** redistributed here; `make data` downloads it. Full references: [NOTICE.md](NOTICE.md).

## License

Copyright (c) 2026 maha012002. **All rights reserved.** The source is public for viewing and
evaluation only. No permission is granted to copy, modify, or reuse it without written
permission. See [LICENSE](LICENSE).

## Running without make

No `make` on Windows? Each target is one or two commands in the `Makefile`, e.g.
`python -m router.download` then `python -m router.data`.
