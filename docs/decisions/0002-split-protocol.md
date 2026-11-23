# ADR 0002: Split protocol and near-duplicate handling

- Status: accepted (M1)
- Date: 2026-11-23

## Context

A router learns "prompts like this are solved by model m". If a near-copy of a test prompt is
in train, a kNN/embedding router effectively looks up the test label. RouterBench contains
such copies: Winogrande twin sentences (one word changed, answer flipped), several MMLU
questions on the same reading passage, ARC questions with shuffled options, and 16 exact
duplicate texts (15 of 176 duplicate (text, model) pairs even carry different labels).

## Decision

1. **Near-duplicate groups.** TF-IDF on word 1–2-grams, dropping terms that occur in > 1% of
   prompts (removes shared templates such as the GSM8K few-shot prefix and the
   "Print only a single choice ..." suffix). Cosine ≥ 0.7 links two prompts; groups are the
   connected components. The threshold was chosen by reading sampled pairs per similarity band:
   0.5–0.7 were "same template, different question"; ≥ 0.7 were true near-duplicates.
2. **Split by group, never by prompt.** No group straddles folds (`check_no_leakage`).
3. **In-distribution (`id`)**: 70/15/15 train/val/test, stratified by each group's majority source.
4. **Leave-one-benchmark-out (`lobo_<source>`)**: all prompts of one source are test; train/val
   (85/15) come from other sources, minus every group that contains a held-out prompt.
   MT-Bench (80 prompts) is never a test fold, only training data.
5. `test-match` (3-row smoke test) is excluded. Seed 42.

## Consequences

- 34,987 groups over 36,494 prompts; 2,276 prompts sit in a multi-prompt group.
- Transitive linking over-groups some small template-heavy evals (largest group: 282 of 285
  `bias_detection` prompts; 56 Chinese riddles). This is the safe direction: it cannot leak,
  it only makes those evals land wholly in one fold of the `id` split.
- TF-IDF is lexical: paraphrases with different wording are not caught. An embedding-based
  check can be added once embeddings exist (M2).
