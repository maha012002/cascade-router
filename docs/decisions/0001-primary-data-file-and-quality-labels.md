# ADR 0001: Primary data file and how quality labels are stored

- Status: accepted (M0)
- Date: 2026-11-23

## Context

RouterBench ships 0-shot and 5-shot pickles. They share the same prompt ids, and labels differ only on
the multiple-choice sources. The 5-shot file has 14 duplicated ids and 14 all-NaN rows. Quality
is binary for most sources but graded for GSM8K, MT-Bench and part of `misc`.

## Decision

1. Use `routerbench_0shot.pkl` as the single source of truth for the pipeline.
2. Store `quality` exactly as provided (float in [0, 1]); do not binarise in `data.py`.
   How graded labels become a training target is decided in M1/M2, not at load time.
3. `data.py` never filters rows. Dropping smoke tests or duplicate prompts belongs in `splits.py`.

## Consequences

- Clean, fully validated input (no nulls, unique ids).
- "P(correct)" is only well defined on binary sources; for graded sources the router predicts an
  expected quality instead. This must be stated whenever results are reported.
- 5-shot results are not used; mixing them would double-count prompts.
