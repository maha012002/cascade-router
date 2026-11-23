"""Offline vs online parity: same prompt -> same decision (needs exported artifact)."""

import numpy as np
import pytest

from router.config import PROJECT_ROOT
from router.data import load_prompts
from router.features import HANDCRAFTED_COLUMNS, load_features
from router.policy import route
from router.serve.runtime import RouterRuntime

ARTIFACT = PROJECT_ROOT / "artifacts/serve/lightgbm-id-v1"
needs_artifact = pytest.mark.skipif(
    not (ARTIFACT / "meta.json").exists(),
    reason="run `python -m router.serve.export_artifact` first",
)


@pytest.mark.integration
@needs_artifact
def test_offline_online_parity_on_held_out_prompt():
    """Use a cached embedding from the ID test fold; runtime re-embeds the same text.

    We compare decisions when both paths use the *same* feature vector (cached), so the
    test isolates the policy/predictor from MiniLM nondeterminism across devices.
    """
    features = load_features().set_index("prompt_id")
    prompts = load_prompts().set_index("prompt_id")
    # Deterministic pick: first prompt id in lexical order that has features.
    pid = sorted(features.index)[0]
    prompt = str(prompts.loc[pid, "prompt"])
    emb_cols = [c for c in features.columns if c.startswith("e") and c[1:].isdigit()]
    x = features.loc[[pid], emb_cols + HANDCRAFTED_COLUMNS].to_numpy(dtype=np.float32)

    rt = RouterRuntime(ARTIFACT, enable_cache=False, embed_fn=lambda p: np.zeros(384))
    # Force the offline feature row through the loaded predictor/cost model.
    p = rt.predictor.predict(x)[0]
    cost = rt.cost_model.predict(np.array([len(prompt)]))[0]
    lam = rt.default_lambda
    offline_choice = rt.models[int(route(p.reshape(1, -1), cost.reshape(1, -1), lam)[0])]

    # Online path with the same features injected via embed_fn + handcrafted from prompt.
    emb = features.loc[pid, emb_cols].to_numpy(dtype=np.float32)

    def embed_fn(text: str) -> np.ndarray:
        assert text == prompt
        return emb

    rt2 = RouterRuntime(ARTIFACT, enable_cache=False, embed_fn=embed_fn)
    online = rt2.route_prompt(prompt, lam=lam)
    assert online.chosen_model == offline_choice
    np.testing.assert_allclose(list(online.per_model_p_correct.values()), p, rtol=0, atol=1e-6)
