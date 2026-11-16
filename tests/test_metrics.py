"""GET /metrics: Prometheus text exposition, parsed back line by line."""
from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from costwise.app.api import create_app
from costwise.app.container import build
from costwise.app.metrics import render
from costwise.settings import Settings

SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(?:[a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\\n]|\\.)*",?)*\})? (\S+)$')


def parse(text: str) -> dict[str, float]:
    """Every line must be a comment or a valid sample; returns {name+labels: value}."""
    out = {}
    assert text.endswith("\n")
    for line in text.splitlines():
        if line.startswith("# HELP ") or line.startswith("# TYPE "):
            continue
        m = SAMPLE.match(line)
        assert m, f"bad exposition line: {line!r}"
        out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return out


@pytest.fixture
def settings(corpus_dir, tmp_path):
    tuned = tmp_path / "tuned.json"
    tuned.write_text(json.dumps({"cache_threshold": 0.95, "cache_candidate_threshold": 0.95,
                                 "cache_require_same_evidence": True, "route_threshold": 0.5}), encoding="utf-8")
    return Settings(corpus_dir=corpus_dir, state_dir=tmp_path / "state", tuned_path=tuned, embed_model="hash",
                    llm_provider="mock", router="heuristic", small_model="mock/small", large_model="mock/large")


def test_metrics_endpoint_counts_requests_and_cache_hits(settings):
    app = create_app(build(settings, persistent=False))
    with TestClient(app) as client:
        empty = client.get("/metrics")
        assert empty.status_code == 200 and empty.headers["content-type"].startswith("text/plain")
        assert "version=0.0.4" in empty.headers["content-type"]
        m = parse(empty.text)
        assert m['costwise_requests_total{route="none"}'] == 0
        for _ in range(2):
            client.post("/ask", json={"query": "How much does Starter cost per month?"})
        m = parse(client.get("/metrics").text)
    assert m['costwise_requests_total{route="cache"}'] == 1
    assert sum(v for k, v in m.items() if k.startswith("costwise_requests_total")) == 2
    assert m['costwise_cache_hits_total{tier="exact"}'] == 1 and m["costwise_cache_entries"] == 1
    assert m["costwise_cache_lookups_total"] == 2 and m["costwise_request_latency_ms_count"] == 2
    assert m["costwise_llm_cost_usd_total"] > 0 and m["costwise_baseline_cost_usd_total"] >= m["costwise_llm_cost_usd_total"]
    info = [k for k in m if k.startswith("costwise_info")][0]
    assert 'mock="true"' in info and 'small_model="mock/small"' in info


def test_render_escapes_labels_and_handles_no_cache():
    text = render({"requests": 0, "by_route": {}}, llm_provider="mock", small_model='we"ird\\name\nx',
                  large_model="l")
    m = parse(text)
    assert r'small_model="we\"ird\\name\nx"' in text
    assert not any(k.startswith("costwise_cache_") for k in m)
    assert text.count('costwise_request_latency_ms{quantile="0.5"} NaN') == 1
