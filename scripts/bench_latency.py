"""Measure CPU routing latency (embed + predict + policy) for p50/p95."""

from __future__ import annotations

import argparse
import time

import numpy as np

from router.config import PROJECT_ROOT, load_config
from router.data import load_prompts
from router.resources import limit_threads
from router.serve.runtime import RouterRuntime

limit_threads(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/serve.yaml")
    parser.add_argument("--n", type=int, default=50, help="number of timed requests")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    cfg = load_config(args.config)
    artifact = PROJECT_ROOT / cfg["artifact_dir"]
    rt = RouterRuntime(artifact, enable_cache=False)
    rt._ensure_encoder()

    prompts = load_prompts()["prompt"].head(args.n + args.warmup).tolist()
    for p in prompts[: args.warmup]:
        rt.route_prompt(p, lam=cfg["default_lambda"])

    times = []
    for p in prompts[args.warmup : args.warmup + args.n]:
        t0 = time.perf_counter()
        rt.route_prompt(p, lam=cfg["default_lambda"])
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times)
    print(
        f"n={len(arr)}  p50={np.percentile(arr, 50):.1f} ms  "
        f"p95={np.percentile(arr, 95):.1f} ms  mean={arr.mean():.1f} ms"
    )


if __name__ == "__main__":
    main()
