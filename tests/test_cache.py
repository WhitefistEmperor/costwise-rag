from __future__ import annotations

import numpy as np

from costwise.adapters.storage import SQLiteCacheStore
from costwise.adapters.verifiers import CrossEncoderVerifier, KeyTermGuard, extract_entity_terms
from costwise.domain.cache import SemanticCache, normalise
from costwise.domain.models import CacheEntry
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
    hit = cache.lookup("how long to refund", vec(1, 0.1, 0))
    assert hit and hit.tier == "semantic" and hit.similarity > 0.9
    assert cache.lookup("backup schedule", vec(0, 1, 0)) is None
    assert cache.stats.semantic_hits == 1 and cache.stats.lookups == 2


def test_exact_tier_wins_even_if_verifier_would_reject():
    class RejectAll:
        name = "reject"
        def same_question(self, a, b): return False

    cache = SemanticCache(clock=ZERO, threshold=0.9, verifier=RejectAll())
    cache.put(entry("What is the price?"), vec(1, 0))
    assert cache.lookup("what is the PRICE", vec(1, 0)).tier == "exact"
    assert cache.lookup("price please", vec(1, 0.01)) is None
    assert cache.stats.rejected_by_verifier == 1


def test_verifier_rejection_falls_through_to_next_candidate():
    guard = KeyTermGuard({"team", "business"})
    cache = SemanticCache(clock=ZERO, threshold=0.8, verifier=guard)
    cache.put(entry("rate limit on Team", tag="team"), vec(1, 0.05))
    cache.put(entry("rate limit on Business", tag="business"), vec(1, 0.2))
    hit = cache.lookup("API cap for Business", vec(1, 0.0))     # Team entry is the nearest
    assert hit.entry.tag == "business"


def test_ttl_expiry_removes_entry():
    now = [0.0]
    cache = SemanticCache(0.9, ttl_s=10, clock=lambda: now[0])
    cache.put(entry("q"), vec(1, 0))
    now[0] = 11
    assert cache.lookup("q", vec(1, 0)) is None
    assert len(cache) == 0 and cache.stats.expired == 1


def test_changed_source_doc_invalidates_only_that_entry():
    cache = SemanticCache(clock=ZERO, threshold=0.9)
    cache.put(entry("refunds", sources={"refunds.md": "h1"}), vec(1, 0))
    cache.put(entry("pricing", sources={"pricing.md": "p1"}), vec(0, 1))
    docs = {"refunds.md": "h2", "pricing.md": "p1"}
    assert cache.lookup("refunds", vec(1, 0), docs) is None
    assert cache.lookup("pricing", vec(0, 1), docs) is not None
    assert len(cache) == 1 and cache.stats.invalidated == 1


def test_put_replaces_same_question():
    cache = SemanticCache(clock=ZERO, threshold=0.9)
    cache.put(entry("q", "old"), vec(1, 0))
    cache.put(entry("Q?", "new"), vec(1, 0))
    assert len(cache) == 1 and cache.lookup("q", vec(1, 0)).entry.answer == "new"


def test_sqlite_store_survives_restart(tmp_path):
    path = tmp_path / "cache.db"
    cache = SemanticCache(clock=ZERO, threshold=0.9, store=SQLiteCacheStore(path))
    cache.put(entry("q1", "a1", {"d.md": "h"}), vec(1, 0))
    cache.put(entry("q2", "a2"), vec(0, 1))
    cache.lookup("q2", vec(0, 1), {"x.md": "y"})                 # q2 has no sources: still valid
    reloaded = SemanticCache(clock=ZERO, threshold=0.9, store=SQLiteCacheStore(path))
    assert len(reloaded) == 2
    hit = reloaded.lookup("q1", vec(1, 0))
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
                                                         "correct_hits": 1, "wrong_hits": 1}
