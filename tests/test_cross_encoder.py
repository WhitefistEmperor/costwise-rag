"""Cross-encoder strong verifier: free verdicts, app wiring, and the tune-only threshold rule.
No model is loaded: every test injects a scorer."""
from __future__ import annotations

import numpy as np
import pytest

from costwise.adapters.verifiers import CrossEncoderVerifier, LLMVerifier
from costwise.app.container import build_strong_verifier
from costwise.domain.cache import SemanticCache
from costwise.domain.models import CacheEntry
from costwise.eval.verifier_eval import candidate_pairs, confusion, tune_score_threshold
from costwise.settings import Settings


def fake_scorer(pairs):
    return [0.9 if a.split()[0] == b.split()[0] else 0.2 for a, b in pairs]


def test_judge_is_free_and_counts_calls():
    v = CrossEncoderVerifier(threshold=0.5, scorer=fake_scorer)
    assert v.judge("refund rules", "refund policy").same
    verdict = v.judge("refund rules", "billing date")
    assert not verdict.same and verdict.cost_usd == 0.0
    assert v.calls == 2


def unit(x, y):
    v = np.array([x, y], dtype=np.float32)
    return v / np.linalg.norm(v)


def test_cross_encoder_opens_the_band_at_no_cost():
    strong = CrossEncoderVerifier(threshold=0.5, scorer=fake_scorer)
    cache = SemanticCache(clock=lambda: 0.0, threshold=0.95, candidate_threshold=0.80, strong_verifier=strong)
    cache.put(CacheEntry("refund rules", "a", {}, "m", 0.0, 0.01), unit(1, 0))
    res = cache.lookup("refund policy please", unit(1, 0.6))     # similarity ~0.86: in the band
    assert res.hit is not None and res.hit.tier == "verified" and res.cost_usd == 0.0
    res = cache.lookup("billing date", unit(1, 0.6))
    assert res.hit is None and cache.stats.rejected_by_strong_verifier == 1


def test_container_builds_each_strong_verifier():
    s = Settings()
    s.cache_strong_verifier = "none"
    assert build_strong_verifier(s, {}, llm=None) is None
    s.cache_strong_verifier = "llm"
    assert isinstance(build_strong_verifier(s, {}, llm=object()), LLMVerifier)
    s.cache_strong_verifier = "cross_encoder"
    v = build_strong_verifier(s, {"cross_encoder_model": "m", "cross_encoder_threshold": 0.35}, llm=None)
    assert isinstance(v, CrossEncoderVerifier) and v.model_name == "m" and v.threshold == 0.35
    assert v._scorer is None                                      # lazy: no model loaded at build time
    s.cache_strong_verifier = "bogus"
    with pytest.raises(ValueError):
        build_strong_verifier(s, {}, llm=None)


def test_score_threshold_is_lowest_with_no_false_accept_in_either_order():
    cands = [(True, "a", "a2"), (False, "a", "b"), (True, "c", "c2")]
    scores = {("a", "a2"): 0.9, ("a2", "a"): 0.9, ("a", "b"): 0.3, ("b", "a"): 0.42,
              ("c", "c2"): 0.5, ("c2", "c"): 0.6}
    t = tune_score_threshold(cands, scores)
    assert t == 0.45                                              # 0.42 in the reverse order still counts
    c = confusion(cands, scores, t)
    assert c == {"candidates": 3, "same": 2, "true_accepts": 2, "false_accepts": 0,
                 "precision": 1.0, "recall": 1.0}


def test_candidate_pairs_apply_band_and_cheap_checks():
    class No:
        def same_question(self, a, b):
            return b != "blocked"
    pairs = [(0.70, True, "a", "b"), (0.50, True, "a", "c"), (0.95, False, "a", "d"), (0.75, False, "a", "blocked")]
    assert candidate_pairs(pairs, No(), None, 0.6, 0.9) == [(True, "a", "b")]
    ev = {"a": "x", "b": "y"}
    assert candidate_pairs(pairs[:1], None, ev, 0.6, 0.9) == []
