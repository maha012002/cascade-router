# cascade-router

A learned router that decides, per query, which LLM (from cheap to expensive) to use, maximising
answer quality per dollar. Evaluated offline on RouterBench precomputed LLM outputs, served via a
FastAPI endpoint.

> Status: **M0 (setup + data understanding)** complete. Full README is written in M5.
> See [docs/DATA_CARD.md](docs/DATA_CARD.md).

## Quickstart

```bash
python -m venv .venv          # Python 3.11
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
make setup                    # pip install -r requirements-dev.txt -e .
make data                     # download RouterBench @ pinned revision -> data/processed/*.parquet
make test                     # unit + integration tests
make eda                      # re-execute notebooks/01_eda.ipynb
```

No `make` on Windows? Each target is one or two commands in the `Makefile`, e.g.
`python -m router.download` then `python -m router.data`.
