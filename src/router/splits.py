"""Leakage-safe train/val/test splits.

Near-duplicate prompts (cosine >= threshold on TF-IDF) are linked into groups via connected
components. Every split assigns whole groups, so no group straddles train and test.

Two protocols:
    id           random in-distribution split, stratified by source
    lobo_<src>   leave-one-benchmark-out: <src> is the test fold; train/val come from all
                 other sources, minus any group that contains a <src> prompt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from router.config import PROJECT_ROOT, load_config
from router.data import load_prompts

SPLIT_COLUMNS = ["prompt_id", "split", "fold"]


def near_duplicate_pairs(
    texts: list[str],
    threshold: float,
    max_df: float,
    ngram_range: tuple[int, int] = (1, 2),
    chunk_size: int = 1000,
) -> np.ndarray:
    """Return an (k, 2) array of index pairs (i < j) whose TF-IDF cosine is >= threshold."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        max_df=max_df,
        sublinear_tf=True,
        dtype=np.float32,
    )
    X = vectorizer.fit_transform(texts)  # rows are L2-normalised, so X @ X.T is cosine
    pairs = []
    for start in range(0, X.shape[0], chunk_size):
        sims = (X[start : start + chunk_size] @ X.T).toarray()  # dense chunk: bounded memory
        rows, cols = np.nonzero(sims >= threshold)
        rows += start
        keep = rows < cols
        pairs.append(np.column_stack([rows[keep], cols[keep]]))
    return np.vstack(pairs) if pairs else np.empty((0, 2), dtype=int)


def group_ids(n: int, pairs: np.ndarray) -> np.ndarray:
    """Connected components of the near-duplicate graph -> one group id per item."""
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def assign_groups(prompts: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add a ``dup_group`` column. Exact duplicates always share a group (cosine = 1)."""
    nd = cfg["near_duplicates"]
    pairs = near_duplicate_pairs(
        prompts["prompt"].tolist(),
        threshold=nd["threshold"],
        max_df=nd["max_df"],
        ngram_range=tuple(nd["ngram_range"]),
    )
    return prompts.assign(dup_group=group_ids(len(prompts), pairs))


def _split_groups(
    groups: pd.DataFrame, fractions: dict[str, float], rng: np.random.Generator
) -> pd.Series:
    """Shuffle groups, then fill folds in order until each reaches its share of *prompts*.

    ``groups`` has one row per group with a ``size`` column. Returns fold label per group.
    """
    order = rng.permutation(len(groups))
    cum_share = groups["size"].to_numpy()[order].cumsum() / groups["size"].sum()
    edges = np.cumsum(list(fractions.values()))
    fold_idx = np.searchsorted(edges, cum_share, side="left").clip(max=len(edges) - 1)
    labels = np.empty(len(groups), dtype=object)
    labels[order] = np.array(list(fractions.keys()), dtype=object)[fold_idx]
    return pd.Series(labels, index=groups.index)


def _group_table(prompts: pd.DataFrame) -> pd.DataFrame:
    """One row per dup_group: its size and its majority source."""
    return prompts.groupby("dup_group").agg(
        size=("prompt_id", "size"),
        source=("source", lambda s: s.mode().iloc[0]),
    )


def in_distribution_split(
    prompts: pd.DataFrame, fractions: dict[str, float], seed: int
) -> pd.DataFrame:
    """Group-aware random split, stratified by each group's majority source."""
    rng = np.random.default_rng(seed)
    groups = _group_table(prompts)
    fold_by_group = pd.concat(
        [_split_groups(g, fractions, rng) for _, g in groups.groupby("source", sort=True)]
    )
    return pd.DataFrame(
        {
            "prompt_id": prompts["prompt_id"],
            "split": "id",
            "fold": prompts["dup_group"].map(fold_by_group).to_numpy(),
        }
    )


def leave_one_benchmark_out(
    prompts: pd.DataFrame, held_out: str, val_fraction: float, seed: int
) -> pd.DataFrame:
    """Test = all prompts of ``held_out``. Groups touching it are removed from train/val."""
    is_test = prompts["source"] == held_out
    contaminated = set(prompts.loc[is_test, "dup_group"])
    pool = prompts[~is_test & ~prompts["dup_group"].isin(contaminated)]

    rng = np.random.default_rng(seed)
    fractions = {"train": 1.0 - val_fraction, "val": val_fraction}
    fold_by_group = _split_groups(_group_table(pool), fractions, rng)

    test = pd.DataFrame({"prompt_id": prompts.loc[is_test, "prompt_id"], "fold": "test"})
    train_val = pd.DataFrame(
        {"prompt_id": pool["prompt_id"], "fold": pool["dup_group"].map(fold_by_group)}
    )
    out = pd.concat([train_val, test], ignore_index=True)
    out.insert(1, "split", f"lobo_{held_out}")
    return out


def make_splits(prompts: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """All splits in one long table: (prompt_id, split, fold)."""
    seed = cfg["seed"]
    parts = [in_distribution_split(prompts, cfg["in_distribution"]["fractions"], seed)]
    lobo = cfg["leave_one_benchmark_out"]
    for source in lobo["held_out_sources"]:
        parts.append(leave_one_benchmark_out(prompts, source, lobo["val_fraction"], seed))
    out = pd.concat(parts, ignore_index=True)[SPLIT_COLUMNS]
    return out.astype({"prompt_id": "string", "split": "string", "fold": "string"})


def check_no_leakage(splits: pd.DataFrame, groups: pd.DataFrame) -> None:
    """Raise if any prompt or dup_group appears in more than one fold of the same split."""
    merged = splits.merge(groups[["prompt_id", "dup_group"]], on="prompt_id", validate="m:1")
    for name, s in merged.groupby("split"):
        if s["prompt_id"].duplicated().any():
            raise ValueError(f"{name}: a prompt appears in more than one fold")
        folds_per_group = s.groupby("dup_group")["fold"].nunique()
        if (folds_per_group > 1).any():
            raise ValueError(f"{name}: {int((folds_per_group > 1).sum())} groups span folds")


def load_splits(processed_dir: str | Path = "data/processed") -> pd.DataFrame:
    return pd.read_parquet(PROJECT_ROOT / processed_dir / "splits.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/splits.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    prompts = load_prompts()
    prompts = prompts[~prompts["eval_name"].isin(cfg["exclude_eval_names"])].reset_index(drop=True)
    prompts = assign_groups(prompts, cfg)
    splits = make_splits(prompts, cfg)
    check_no_leakage(splits, prompts)

    out_dir = PROJECT_ROOT / cfg["output_dir"]
    prompts[["prompt_id", "source", "dup_group"]].to_parquet(
        out_dir / "dup_groups.parquet", index=False
    )
    splits.to_parquet(out_dir / "splits.parquet", index=False)

    sizes = prompts["dup_group"].value_counts()
    print(
        f"prompts: {len(prompts)} | groups: {len(sizes)} | "
        f"prompts in a multi-prompt group: {int(sizes[sizes > 1].sum())} | "
        f"largest group: {int(sizes.max())}"
    )
    print(splits.groupby(["split", "fold"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
