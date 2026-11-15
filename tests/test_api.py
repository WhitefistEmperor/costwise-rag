from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from costwise.app.api import create_app
from costwise.app.container import build
from costwise.settings import Settings


@pytest.fixture
def settings(corpus_dir, tmp_path):
    tuned = tmp_path / "tuned.json"
    tuned.write_text(json.dumps({"cache_threshold": 0.95, "cache_candidate_threshold": 0.95,
                                 "cache_require_same_evidence": True, "route_threshold": 0.5}), encoding="utf-8")
    return Settings(corpus_dir=corpus_dir, state_dir=tmp_path / "state", tuned_path=tuned, embed_model="hash",
                    llm_provider="mock", router="heuristic", small_model="mock/small", large_model="mock/large")


def test_container_defaults_to_mock_and_never_needs_a_key(settings, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    c = build(settings)
    assert c.llm.name == "mock" and c.cache is not None and c.pipeline.router.name == "heuristic"
    assert c.pipeline.route_threshold == 0.5 and c.cache.threshold == 0.95
    assert (settings.state_dir / "cache.db").exists() and (settings.state_dir / "requests.db").exists()


def test_container_refuses_openrouter_without_key(settings, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings.llm_provider = "openrouter"
    with pytest.raises(RuntimeError):
        build(settings)


def test_container_learned_router_needs_weights(settings):
    settings.router = "learned"
    with pytest.raises(RuntimeError):
        build(settings)


def test_api_ask_stats_recent_and_dashboard(settings):
    app = create_app(build(settings, persistent=False))
    with TestClient(app) as client:
        r = client.post("/ask", json={"query": "How much does Starter cost per month?"})
        assert r.status_code == 200
        body = r.json()
        assert body["route"] in ("small", "large") and "9" in body["text"] and body["sources"]
        again = client.post("/ask", json={"query": "How much does Starter cost per month?"}).json()
        assert again["route"] == "cache" and again["cache_tier"] == "exact"

        stats = client.get("/stats").json()
        assert stats["requests"] == 2 and stats["cache_hits"] == 1 and stats["mock"] is True
        assert stats["cache"]["entries"] == 1 and stats["models"]["small"] == "mock/small"
        recent = client.get("/recent?n=5").json()
        assert len(recent) == 2 and recent[0]["route"] == "cache"
        assert client.get("/cache").json()[0]["query"].startswith("How much")
        page = client.get("/")
        assert page.status_code == 200 and "Costwise RAG" in page.text
        assert client.post("/ask", json={"query": ""}).status_code == 422
