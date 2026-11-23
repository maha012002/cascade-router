# ADR 0003: Evaluation protocol (cost-quality curves)

- Status: accepted (M1)
- Date: 2026-11-23

## Decision

- **Quality** of a routing decision = the dataset's `quality` for the chosen model (binary for
  most sources, graded for GSM8K / MT-Bench / part of misc). Reported as a mean over test prompts.
- **Cost** = the dataset's realised `total_cost` of the chosen model, in USD.
- A method yields operating points (one per λ for learned routers). Its **curve** is the upper
  concave hull of those points, because randomly mixing two operating points achieves every point
  on the segment between them. All methods get the same treatment.
- **Baselines choose models on train only**: always-cheapest (lowest train mean cost),
  always-best (highest train mean quality), random mixture of those two, and the zero router
  (mixture over the models on the *train* single-model hull, evaluated on test).
- **Oracle**: exact frontier computed by greedy merging of per-prompt upgrade steps sorted by
  quality gained per dollar (exact because each prompt's hull is concave). It is an upper bound
  on every router at every cost (tested).
- **Metrics** on each curve, with c_lo / c_hi = test cost of always-cheapest / always-best:
  - `auc`: mean quality over cost in [c_lo, c_hi], linear USD axis;
  - `quality_at_b10/b25/b50`: quality at 10/25/50% of always-best's cost;
  - `cost_to_target`: cheapest cost reaching 95% of always-best's test quality
    (also as a fraction of c_hi).
- Below a curve's cheapest point a method cannot operate and scores 0; beyond its last point it
  stays flat (spending less than the budget is allowed).

## Consequences

- Single-point methods (always-best) get ~0 AUC and 0 quality at budgets below their cost. That
  is intended: they cannot meet those budgets.
- Linear-USD AUC weights the expensive region (GPT-4/Claude, ~10x pricier) heavily, so the
  budget metrics and cost-to-target are reported alongside AUC.
- A first version built the zero-router hull from **test** model means, which leaked test
  information into a baseline (e.g. it knew Claude v2 beats GPT-4 on GSM8K). Fixed to use train.
