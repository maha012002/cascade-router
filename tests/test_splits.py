import numpy as np
import pandas as pd
import pytest

from router.config import PROJECT_ROOT
from router.splits import (
    assign_groups,
    check_no_leakage,
    group_ids,
    in_distribution_split,
    leave_one_benchmark_out,
    make_splits,
    near_duplicate_pairs,
)

CFG = {
    "seed": 0,
    "near_duplicates": {"ngram_range": [1, 2], "max_df": 1.0, "threshold": 0.7},
    "in_distribution": {"fractions": {"train": 0.7, "val": 0.15, "test": 0.15}},
    "leave_one_benchmark_out": {"held_out_sources": ["alpha", "beta"], "val_fraction": 0.2},
}


@pytest.fixture
def prompts() -> pd.DataFrame:
    """60 unique prompts over 3 sources, plus planted near-duplicates (one cross-source)."""
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(500)]
    texts = [" ".join(rng.choice(vocab, 12, replace=False)) for _ in range(60)]
    sources = ["alpha"] * 20 + ["beta"] * 20 + ["gamma"] * 20
    texts += [texts[0] + " extra", texts[1], texts[25] + " again"]
    sources += ["alpha", "alpha", "gamma"]  # last one: near-dup of a beta prompt
    return pd.DataFrame(
        {"prompt_id": [f"p{i}" for i in range(len(texts))], "source": sources, "prompt": texts}
    )


def test_near_duplicates_found(prompts):
    pairs = {tuple(p) for p in near_duplicate_pairs(prompts["prompt"].tolist(), 0.7, 1.0)}
    assert {(0, 60), (1, 61), (25, 62)} <= pairs


def test_group_ids_are_transitive():
    labels = group_ids(4, np.array([[0, 1], [1, 2]]))
    assert labels[0] == labels[1] == labels[2] != labels[3]


def test_id_split_has_no_overlap_and_keeps_groups_together(prompts):
    g = assign_groups(prompts, CFG)
    split = in_distribution_split(g, CFG["in_distribution"]["fractions"], seed=0)
    assert split["prompt_id"].is_unique
    assert set(split["prompt_id"]) == set(prompts["prompt_id"])
    check_no_leakage(split, g)
    fold = split.set_index("prompt_id")["fold"]
    assert fold["p0"] == fold["p60"] and fold["p1"] == fold["p61"]


def test_id_split_fractions_roughly_respected(prompts):
    g = assign_groups(prompts, CFG)
    shares = in_distribution_split(g, CFG["in_distribution"]["fractions"], 0)["fold"].value_counts(
        normalize=True
    )
    assert shares["train"] == pytest.approx(0.7, abs=0.1)


def test_id_split_is_deterministic(prompts):
    g = assign_groups(prompts, CFG)
    a = in_distribution_split(g, CFG["in_distribution"]["fractions"], seed=0)
    b = in_distribution_split(g, CFG["in_distribution"]["fractions"], seed=0)
    pd.testing.assert_frame_equal(a, b)


def test_lobo_test_is_exactly_held_out_source_and_train_is_clean(prompts):
    g = assign_groups(prompts, CFG)
    split = leave_one_benchmark_out(g, "beta", val_fraction=0.2, seed=0)
    by_fold = split.merge(g, on="prompt_id")
    test = by_fold[by_fold["fold"] == "test"]
    rest = by_fold[by_fold["fold"] != "test"]

    assert set(test["source"]) == {"beta"}
    assert len(test) == (prompts["source"] == "beta").sum()
    assert "beta" not in set(rest["source"])
    assert set(test["dup_group"]).isdisjoint(rest["dup_group"])
    assert "p62" not in set(rest["prompt_id"])  # cross-source near-dup of a beta prompt
    assert not set(test["prompt_id"]) & set(rest["prompt_id"])


def test_make_splits_passes_leakage_check(prompts):
    g = assign_groups(prompts, CFG)
    splits = make_splits(g, CFG)
    assert set(splits["split"]) == {"id", "lobo_alpha", "lobo_beta"}
    check_no_leakage(splits, g)


def test_check_no_leakage_catches_straddling_group(prompts):
    g = assign_groups(prompts, CFG)
    split = in_distribution_split(g, CFG["in_distribution"]["fractions"], 0)
    split.loc[split["prompt_id"] == "p60", "fold"] = (
        "test" if split.set_index("prompt_id").loc["p0", "fold"] != "test" else "train"
    )
    with pytest.raises(ValueError, match="span folds"):
        check_no_leakage(split, g)


_SPLITS = PROJECT_ROOT / "data/processed/splits.parquet"


@pytest.mark.integration
@pytest.mark.skipif(not _SPLITS.exists(), reason="run `make splits` first")
def test_real_splits_have_zero_overlap():
    splits = pd.read_parquet(_SPLITS)
    groups = pd.read_parquet(PROJECT_ROOT / "data/processed/dup_groups.parquet")
    check_no_leakage(splits, groups)
    for name, s in splits.groupby("split"):
        test_ids = set(s.loc[s["fold"] == "test", "prompt_id"])
        other_ids = set(s.loc[s["fold"] != "test", "prompt_id"])
        assert not test_ids & other_ids, name
