from __future__ import annotations

import numpy as np

from costwise.adapters.storage import SQLiteCacheStore
from costwise.adapters.verifiers import CrossEncoderVerifier, KeyTermGuard, extract_entity_terms
from costwise.domain.cache import SemanticCache, normalise
from costwise.domain.models import CacheEntry
from costwise.domain.ports import Verdict
from costwise.eval.dataset import Query
from costwise.eval.offline import pair_counts, simulate_stream, tune_threshold


ZERO = lambda: 0.0


def entry(query, answer="a", sources=None, created=0.0, tag=None):
    return CacheEntry(query, answer, sources or {}, "m", created, 0.01, tag)


def vec(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_normalise():
    assert normalise("  What IS  the price?? ") == "what is the price"


def test_semantic_hit_above_threshold_and_miss_below():
    cache = SemanticCache(clock=ZERO, threshold=0.9)
    cache.put(entry("refund window"), vec(1, 0, 0))
    hit = cache.lookup("how long to refund", vec(1, 0.1, 0)).hit
    assert hit and hit.tier == "semantic" and hit.similarity > 0.9
    assert cache.lookup("backup schedule", vec(0, 1, 0)).hit is None
    assert cache.stats.semantic_hits == 1 and cache.stats.lookups == 2


def test_exact_tier_wins_even_if_verifier_would_reject():
    class RejectAll:
        name = "reject"
        def same_question(self, a, b): return False

    cache = SemanticCache(clock=ZERO, threshold=0.9, verifier=RejectAll())
    cache.put(entry("What is the price?"), vec(1, 0))
    assert cache.lookup("what is the PRICE", vec(1, 0)).hit.tier == "exact"
    assert cache.lookup("price please", vec(1, 0.01)).hit is None
    assert cache.stats.rejected_by_verifier == 1


def test_verifier_rejection_falls_through_to_next_candidate():
    guard = KeyTermGuard({"team", "business"})
    cache = SemanticCache(clock=ZERO, threshold=0.8, verifier=guard)
    cache.put(entry("rate limit on Team", tag="team"), vec(1, 0.05))
    cache.put(entry("rate limit on Business", tag="business"), vec(1, 0.2))
    hit = cache.lookup("API cap for Business", vec(1, 0.0)).hit     # Team entry is the nearest
    assert hit.entry.tag == "business"


def test_ttl_expiry_removes_entry():
    now = [0.0]
    cache = SemanticCache(0.9, ttl_s=10, clock=lambda: now[0])
    cache.put(entry("q"), vec(1, 0))
    now[0] = 11
    assert cache.lookup("q", vec(1, 0)).hit is None
    assert len(cache) == 0 and cache.stats.expired == 1


def test_changed_source_doc_invalidates_only_that_entry():
    cache = SemanticCache(clock=ZERO, threshold=0.9)
    cache.put(entry("refunds", sources={"refunds.md": "h1"}), vec(1, 0))
    cache.put(entry("pricing", sources={"pricing.md": "p1"}), vec(0, 1))
    docs = {"refunds.md": "h2", "pricing.md": "p1"}
    assert cache.lookup("refunds", vec(1, 0), docs).hit is None
    assert cache.lookup("pricing", vec(0, 1), docs).hit is not None
    assert len(cache) == 1 and cache.stats.invalidated == 1


def test_put_replaces_same_question():
    cache = SemanticCache(clock=ZERO, threshold=0.9)
    cache.put(entry("q", "old"), vec(1, 0))
    cache.put(entry("Q?", "new"), vec(1, 0))
    assert len(cache) == 1 and cache.lookup("q", vec(1, 0)).hit.entry.answer == "new"


def test_sqlite_store_survives_restart(tmp_path):
    path = tmp_path / "cache.db"
    cache = SemanticCache(clock=ZERO, threshold=0.9, store=SQLiteCacheStore(path))
    cache.put(entry("q1", "a1", {"d.md": "h"}), vec(1, 0))
    cache.put(entry("q2", "a2"), vec(0, 1))
    cache.lookup("q2", vec(0, 1), {"x.md": "y"})                     # q2 has no sources: still valid
    reloaded = SemanticCache(clock=ZERO, threshold=0.9, store=SQLiteCacheStore(path))
    assert len(reloaded) == 2
    hit = reloaded.lookup("q1", vec(1, 0)).hit
    assert hit.entry.answer == "a1" and hit.entry.sources == {"d.md": "h"}


def test_entity_terms_come_from_mid_sentence_capitals(chunks):
    terms = extract_entity_terms(chunks)
    assert "team" in terms                                      # "... per seat on Team."
    assert "monthly" not in terms and "extra" not in terms      # sentence-initial words are skipped


def test_guard_rejects_substitutions_and_accepts_additions():
    g = KeyTermGuard({"team", "business", "starter", "gb", "tb"})
    same = g.same_question
    assert not same("rate limit on Team", "rate limit on Business")
    assert not same("refund window for monthly plans", "refund window for annual plans")
    assert not same("cost to migrate 800 GB", "cost to migrate 500 GB")
    assert not same("Team price per month", "Team price with 12 seats and 1.3 TB")   # added number
    assert same("How much storage on Team?", "How many GB or TB do I get on Team?")
    assert same("how long is the free trial", "what is the length of the trial period")
    assert same("price of 2FA on Team", "team 2fa price")


def test_cross_encoder_verifier_uses_scorer_and_threshold():
    v = CrossEncoderVerifier(threshold=0.5, scorer=lambda pairs: [0.9 if a == b else 0.1 for a, b in pairs])
    assert v.same_question("x", "x") and not v.same_question("x", "y")


def test_pair_counts_and_threshold_tuning():
    pairs = [(0.95, True, "a", "a2"), (0.70, True, "b", "b2"), (0.85, False, "a", "c"), (0.40, False, "b", "c")]
    assert pair_counts(pairs, 0.8, None) == {"true_accepts": 1, "same_pairs": 2, "false_accepts": 1,
                                             "different_pairs": 2}
    assert tune_threshold(pairs, None) == 0.88                   # first grid value above 0.85, plus margin


def test_simulate_stream_counts_wrong_hits():
    vecs = {"a": vec(1, 0), "a again": vec(1, 0.05), "b near a": vec(1, 0.1), "c": vec(0, 1)}
    stream = [Query("A", "a", False, False), Query("A", "a again", False, True),
              Query("B", "b near a", False, False), Query("C", "c", False, False)]
    assert simulate_stream(stream, vecs, 0.9, None) == {"queries": 4, "repeats": 1, "hits": 2,
                                                         "correct_hits": 1, "wrong_hits": 1, "strong_verifier_calls": 0}


class FakeStrong:
    """Strong verifier that accepts pairs listed in `same` and bills `cost` per call."""
    name = "fake_strong"

    def __init__(self, same, cost=0.001):
        self.same, self.cost, self.calls = same, cost, 0

    def judge(self, a, b):
        self.calls += 1
        return Verdict((a, b) in self.same or (b, a) in self.same, self.cost)

    def same_question(self, a, b):
        return self.judge(a, b).same


def test_band_candidate_needs_strong_verifier_and_bills_its_cost():
    strong = FakeStrong({("q repeat", "q")})
    cache = SemanticCache(clock=ZERO, threshold=0.95, candidate_threshold=0.80, strong_verifier=strong)
    cache.put(entry("q", tag="Q"), vec(1, 0))
    hit = cache.lookup("q repeat", vec(1, 0.3)).hit            # similarity ~0.96 >= accept: no strong call
    assert hit.tier == "semantic" and strong.calls == 0
    res = cache.lookup("q repeat", vec(1, 0.6))                # ~0.86: in the band
    assert res.hit.tier == "verified" and res.hit.verify_cost_usd == 0.001 and strong.calls == 1
    assert res.cost_usd == 0.001 and res.strong_verifier_calls == 1
    res = cache.lookup("other", vec(1, 0.6))                   # strong verifier says no: a paid miss
    assert res.hit is None and res.cost_usd == 0.001
    assert cache.stats.rejected_by_strong_verifier == 1
    assert cache.stats.strong_verifier_calls == 2 and abs(cache.stats.strong_verifier_cost_usd - 0.002) < 1e-9
    assert cache.lookup("far", vec(1, 2.0)).hit is None and strong.calls == 2   # below the band: no call


def test_at_most_one_strong_verifier_call_per_lookup():
    strong = FakeStrong(set())
    cache = SemanticCache(clock=ZERO, threshold=0.99, candidate_threshold=0.5, strong_verifier=strong)
    cache.put(entry("a"), vec(1, 0.4))
    cache.put(entry("b"), vec(1, 0.5))
    cache.put(entry("c"), vec(1, 0.6))
    assert cache.lookup("x", vec(1, 0)).hit is None and strong.calls == 1


def test_without_strong_verifier_the_band_is_ignored():
    cache = SemanticCache(clock=ZERO, threshold=0.95, candidate_threshold=0.5)
    cache.put(entry("q"), vec(1, 0))
    assert cache.lookup("q2", vec(1, 0.6)).hit is None and cache.low_threshold == 0.95


def test_llm_verifier_parses_yes_no_and_tracks_cost():
    from costwise.adapters.verifiers import LLMVerifier
    from costwise.domain.models import LLMResult

    class FakeLLM:
        def __init__(self): self.calls = []
        def complete(self, model, system, user, max_tokens):
            self.calls.append((model, user, max_tokens))
            text = "YES" if "Team" in user and "Business" not in user else "No."
            return LLMResult(text, model, 40, 1, 0.00001, 5.0)

    llm = FakeLLM()
    v = LLMVerifier(llm, "small-model")
    assert v.same_question("Team price", "cost of Team")
    assert not v.same_question("Team price", "Business price")
    assert v.calls == 2 and abs(v.cost_usd - 0.00002) < 1e-12
    assert v.judge("Team price", "cost of Team") == Verdict(True, 0.00001)
    assert llm.calls[0][0] == "small-model" and "Question A: Team price" in llm.calls[0][1]


def test_candidate_threshold_rule_uses_band_precision():
    from costwise.eval.offline import tune_candidate_threshold
    # band [0.74, 0.90): one same pair, two different -> precision 0.33, too low;
    # band [0.76, 0.90): one same, one different -> 0.5, accepted.
    pairs = [(0.85, True, "a", "a2"), (0.82, False, "a", "c"), (0.74, False, "b", "c"), (0.65, False, "d", "c")]
    assert tune_candidate_threshold(pairs, 0.90, None) == 0.76
    assert tune_candidate_threshold([(0.70, False, "b", "c")], 0.90, None) == 0.90   # no good band: no band


def test_simulate_stream_reports_strong_verifier_calls():
    class Oracle:
        name = "oracle"
        def same_question(self, a, b): return a[0] == b[0]
    vecs = {"a": vec(1, 0), "a again": vec(1, 0.5), "b near a": vec(1, 0.6)}
    stream = [Query("A", "a", False, False), Query("A", "a again", False, True), Query("B", "b near a", False, False)]
    r = simulate_stream(stream, vecs, 0.99, None, candidate_threshold=0.8, strong_verifier=Oracle())
    assert r["hits"] == 1 and r["wrong_hits"] == 0 and r["strong_verifier_calls"] == 2
