# CPU-only serving image for cascade-router.
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt pyproject.toml README.md LICENSE NOTICE.md ./
COPY src ./src
COPY configs ./configs
COPY artifacts/serve ./artifacts/serve

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir fastapi uvicorn[standard] redis httpx

ENV SERVE_CONFIG=configs/serve.yaml
ENV EMBED_DEVICE=cpu
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

EXPOSE 8000
CMD ["uvicorn", "router.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
