"""Cache lookups: per-lookup verifier cost, no lock held during the strong verifier, stale
entries do not use up candidate slots, and the pipeline charges the verifier cost."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from costwise.adapters.llm import MockLLM
from costwise.domain.cache import LookupResult, SemanticCache
from costwise.domain.chunking import doc_hashes
from costwise.domain.models import CacheEntry
from costwise.domain.pipeline import Pipeline
from costwise.domain.ports import Verdict


def vec(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def entry(query, created=0.0, tag=None):
    return CacheEntry(query, "a", {}, "m", created, 0.01, tag)


class SlowStrong:
    """Strong verifier that sleeps, bills a cost that depends on the query, and records whether
    the cache lock was held while it ran."""
    name = "slow"

    def __init__(self, cache_ref, costs, delay=0.05, same=True):
        self.cache_ref, self.costs, self.delay, self.same = cache_ref, costs, delay, same
        self.lock_held = []

    def judge(self, query, cached_query):
        self.lock_held.append(self.cache_ref[0]._lock.locked())
        time.sleep(self.delay)
        return Verdict(self.same, self.costs[query])

    def same_question(self, query, cached_query):
        return self.judge(query, cached_query).same


def test_concurrent_lookups_each_get_their_own_verifier_cost():
    ref = []
    strong = SlowStrong(ref, {"qa": 0.001, "qb": 0.002, "qc": 0.004})
    cache = SemanticCache(0.99, candidate_threshold=0.5, clock=lambda: 0.0, strong_verifier=strong)
    ref.append(cache)
    cache.put(entry("q"), vec(1, 0))
    results = {}

    def look(name):
        results[name] = cache.lookup(name, vec(1, 0.5))

    threads = [threading.Thread(target=look, args=(n,)) for n in ("qa", "qb", "qc")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert {n: r.cost_usd for n, r in results.items()} == {"qa": 0.001, "qb": 0.002, "qc": 0.004}
    assert all(isinstance(r, LookupResult) and r.hit.tier == "verified" for r in results.values())
    assert cache.stats.strong_verifier_calls == 3
    assert cache.stats.strong_verifier_cost_usd == pytest.approx(0.007)


def test_lock_is_not_held_during_the_strong_verifier_call():
    ref = []
    strong = SlowStrong(ref, {"x": 0.001}, delay=0.0)
    cache = SemanticCache(0.99, candidate_threshold=0.5, clock=lambda: 0.0, strong_verifier=strong)
    ref.append(cache)
    cache.put(entry("q"), vec(1, 0))
    assert cache.lookup("x", vec(1, 0.5)).hit is not None
    assert strong.lock_held == [False]


def test_cache_changes_during_the_verifier_call_are_handled_by_key():
    ref = []
    cache_box = {}

    class Mutating(SlowStrong):
        def judge(self, query, cached_query):
            c = cache_box["c"]
            c.put(entry("brand new"), vec(0, 1))               # another request stores an entry meanwhile
            return Verdict(False, 0.001)

    now = [20.0]
    cache = SemanticCache(0.99, candidate_threshold=0.5, ttl_s=10, clock=lambda: now[0],
                          strong_verifier=Mutating(ref, {}))
    cache_box["c"] = cache
    ref.append(cache)
    cache.put(entry("old", created=0.0), vec(1, 0.05))         # expired
    cache.put(entry("fresh", created=15.0), vec(1, 0.4))
    res = cache.lookup("x", vec(1, 0))                          # fresh is at ~0.93: in the band
    assert res.hit is None and res.cost_usd == 0.001
    assert sorted(e.query for e in cache.entries) == ["brand new", "fresh"]
    assert cache.vectors.shape[0] == 2


def test_stale_entries_do_not_use_up_candidate_slots():
    now = [20.0]
    cache = SemanticCache(0.9, ttl_s=10, clock=lambda: now[0], max_candidates=2)
    for i in range(3):                                          # expired, and closer to the query
        cache.put(entry(f"stale {i}", created=0.0, tag="stale"), vec(1, 0.01 * i))
    cache.put(entry("fresh", created=15.0, tag="fresh"), vec(1, 0.2))
    res = cache.lookup("query", vec(1, 0))
    assert res.hit is not None and res.hit.entry.tag == "fresh"
    assert cache.stats.expired == 3 and len(cache) == 1


def test_rejected_candidates_still_count_toward_max_candidates():
    class RejectAll:
        name = "reject"
        def same_question(self, a, b): return False

    cache = SemanticCache(0.9, clock=lambda: 0.0, max_candidates=2, verifier=RejectAll())
    cache.put(entry("a"), vec(1, 0.0))
    cache.put(entry("b"), vec(1, 0.01))
    cache.put(entry("c"), vec(1, 0.02))
    assert cache.lookup("q", vec(1, 0)).hit is None and cache.stats.rejected_by_verifier == 2


class RecordingLLM(MockLLM):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.costs = []

    def complete(self, model, system, user, max_tokens):
        r = super().complete(model, system, user, max_tokens)
        self.costs.append(r.cost_usd)
        return r


class FixedStrong:
    name = "fixed"

    def __init__(self, yes, cost):
        self.yes, self.cost, self.calls = yes, cost, 0

    def judge(self, query, cached_query):
        self.calls += 1
        return Verdict(query in self.yes, self.cost)

    def same_question(self, query, cached_query):
        return self.judge(query, cached_query).same


def test_pipeline_charges_the_strong_verifier_cost_on_hit_and_on_miss(chunks, retrievers, embedder):
    llm = RecordingLLM()
    strong = FixedStrong({"What does Starter cost per month?"}, 0.0007)
    cache = SemanticCache(0.999, candidate_threshold=0.05, clock=lambda: 0.0, strong_verifier=strong)
    p = Pipeline(embedder, retrievers["hybrid"], llm, "mock/small", "mock/large", cache=cache,
                 default_route="large", doc_hashes=doc_hashes(chunks))
    first = p.ask("How much does Starter cost per month?")
    assert first.route == "large" and strong.calls == 0 and first.cost_usd == llm.costs[0]
    hit = p.ask("What does Starter cost per month?")
    assert hit.route == "cache" and hit.cache_tier == "verified" and hit.cost_usd == 0.0007
    miss = p.ask("How much does Team cost per month?")
    assert miss.route == "large" and strong.calls == 2
    assert miss.cost_usd == pytest.approx(llm.costs[1] + 0.0007, abs=1e-15)
