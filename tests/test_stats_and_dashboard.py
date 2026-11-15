"""Nearest-rank percentiles, deduplicated sources, verifier call counting, dashboard safety."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from costwise.adapters.llm import MockLLM
from costwise.adapters.storage import SQLiteRequestLog
from costwise.domain.cache import SemanticCache
from costwise.domain.chunking import doc_hashes
from costwise.domain.models import Answer, CacheEntry
from costwise.domain.pipeline import Pipeline
from costwise.domain.stats import percentile
from costwise.eval.live import ConfigResult

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "costwise" / "app" / "dashboard.html"


def test_nearest_rank_percentile():
    values = list(range(1, 101))
    assert percentile(values, 0.5) == 50 and percentile(values, 0.95) == 95
    assert percentile(values, 1.0) == 100 and percentile([7.0], 0.95) == 7.0 and percentile([], 0.5) is None
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.7) == 7     # 0.7 * 10 is 7.000000000000001 in floats


def test_request_log_and_live_summary_use_nearest_rank():
    log = SQLiteRequestLog(None)
    for ms in range(100, 0, -1):
        log.log(Answer("q", "a", "small", "m", latency_ms=float(ms)))
    stats = log.stats()
    assert stats["latency_ms_p50"] == 50 and stats["latency_ms_p95"] == 95
    r = ConfigResult(queries=100, latencies_ms=[float(x) for x in range(1, 101)])
    assert r.summary()["latency_ms_p50"] == 50 and r.summary()["latency_ms_p95"] == 95


def test_miss_sources_are_deduplicated_like_hits(chunks, retrievers, embedder):
    cache = SemanticCache(0.99, clock=lambda: 0.0)
    p = Pipeline(embedder, retrievers["hybrid"], MockLLM(), "mock/small", "mock/large", cache=cache,
                 default_route="large", doc_hashes=doc_hashes(chunks), top_k=4)
    miss = p.ask("How much does Starter cost per month and per extra seat?")
    assert len(miss.sources) == len(set(miss.sources))
    hit = p.ask("How much does Starter cost per month and per extra seat?")
    assert hit.route == "cache" and hit.sources == miss.sources


def test_strong_verifier_calls_count_only_calls_that_returned():
    class Failing:
        name = "failing"
        def judge(self, a, b): raise RuntimeError("budget")
        def same_question(self, a, b): raise RuntimeError("budget")

    cache = SemanticCache(0.99, candidate_threshold=0.5, clock=lambda: 0.0, strong_verifier=Failing())
    cache.put(CacheEntry("q", "a", {}, "m", 0.0, 0.0), np.array([1.0, 0.0], dtype=np.float32))
    with pytest.raises(RuntimeError):
        cache.lookup("x", np.array([0.8, 0.6], dtype=np.float32))
    assert cache.stats.strong_verifier_calls == 0


def test_dashboard_hidden_badge_and_escaping():
    html = DASHBOARD.read_text(encoding="utf-8")
    assert re.search(r"\[hidden\]\s*\{\s*display:\s*none\s*!important", html)
    assert "function esc(" in html
    for raw in ("${x.query", "${x.route}", "${x.model}", "${x.cache_tier}", "${value}", "${label}"):
        assert raw not in html, raw                              # innerHTML templates escape every value
    script = html.split("<script>", 1)[1]
    assert "$('answer').textContent" in script and "$('models').textContent" in script   # plain text, no HTML
    assert script.count(".innerHTML") == 2                      # stats cards and recent rows, both escaped
    assert re.search(r"esc\(\s*s\)\s*\.replace|replace\(/&/g", script)          # & is escaped first
    for ch in ("&amp;", "&lt;", "&gt;", "&quot;", "&#39;"):
        assert ch in script
