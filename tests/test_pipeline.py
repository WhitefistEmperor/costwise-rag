from __future__ import annotations

import json

import httpx
import pytest

from costwise.adapters.llm import MockLLM, OpenRouterLLM
from costwise.adapters.storage import SQLiteRequestLog
from costwise.adapters.verifiers import KeyTermGuard
from costwise.domain.cache import SemanticCache
from costwise.domain.chunking import doc_hashes
from costwise.domain.pipeline import ABSTAIN, Pipeline, abstained, build_user_prompt
from costwise.domain.pricing import cost_usd, estimate_tokens
from costwise.domain.routing import HeuristicRouter


def test_cost_table_and_token_estimate():
    assert cost_usd("anthropic/claude-haiku-5.5", 1_000_000, 0) == 0.10
    assert cost_usd("anthropic/claude-sonnet-5.5", 1000, 100) == pytest.approx(0.002 + 0.001)
    assert cost_usd("unknown/model", 100, 100) == 0.0
    assert estimate_tokens("abcdefgh") == 2 and estimate_tokens("") == 1


def test_mock_llm_answers_with_the_context_and_can_abstain():
    llm = MockLLM()
    r = llm.complete("mock/small", "sys", "Context:\n[a#0] Starter costs $9.\n\nQuestion: price?", 100)
    assert r.text == "[a#0] Starter costs $9." and r.cost_usd > 0 and r.model == "mock/small"
    llm = MockLLM(abstain_if=lambda model, user: "small" in model)
    assert abstained(llm.complete("mock/small", "s", "u", 10).text)
    assert not abstained(llm.complete("mock/large", "s", "u", 10).text)


def test_openrouter_client_parses_usage_and_real_cost():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "anthropic/claude-haiku-5.5",
                                         "choices": [{"message": {"content": " 49 dollars "}}],
                                         "usage": {"prompt_tokens": 120, "completion_tokens": 4, "cost": 0.000014}})

    llm = OpenRouterLLM("test-key-not-real", transport=httpx.MockTransport(handler))
    r = llm.complete("anthropic/claude-haiku-5.5", "sys", "user", 50)
    assert r.text == "49 dollars" and r.prompt_tokens == 120 and r.cost_usd == 0.000014 and r.latency_ms >= 0
    assert seen["auth"] == "Bearer test-key-not-real" and seen["body"]["usage"] == {"include": True}
    assert seen["body"]["messages"][0]["role"] == "system" and seen["body"]["max_tokens"] == 50
    with pytest.raises(ValueError):
        OpenRouterLLM("")


def test_openrouter_falls_back_to_price_table_without_cost():
    handler = lambda req: httpx.Response(200, json={"choices": [{"message": {"content": "x"}}],
                                                    "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}})
    llm = OpenRouterLLM("k", transport=httpx.MockTransport(handler))
    assert llm.complete("anthropic/claude-sonnet-5.5", "s", "u", 5).cost_usd == 2.0


def make_pipeline(chunks, retrievers, embedder, llm, **kw):
    defaults = dict(small_model="mock/small", large_model="mock/large", router=HeuristicRouter(),
                    doc_hashes=doc_hashes(chunks), request_log=SQLiteRequestLog(None), clock=lambda: 0.0)
    defaults.update(kw)
    return Pipeline(embedder, retrievers["hybrid"], llm, **defaults)


def test_pipeline_routes_easy_to_small_and_hard_to_large(chunks, retrievers, embedder):
    p = make_pipeline(chunks, retrievers, embedder, MockLLM())
    easy = p.ask("How much does Starter cost per month?")
    assert easy.route == "small" and easy.model == "mock/small" and "9" in easy.text
    assert easy.cost_usd > 0 and easy.baseline_cost_usd > easy.cost_usd       # baseline = large-model price
    hard = p.ask("Compare the total cost of Team with 3 extra seats and the refund window for annual plans, 30 days?")
    assert hard.route == "large" and hard.route_score > easy.route_score and hard.baseline_cost_usd == hard.cost_usd
    assert p.request_log.stats()["requests"] == 2 and p.request_log.stats()["by_route"] == {"small": 1, "large": 1}


class RecordingLLM(MockLLM):
    """MockLLM that keeps every LLMResult it returned."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.results = []

    def complete(self, model, system, user, max_tokens):
        r = super().complete(model, system, user, max_tokens)
        self.results.append(r)
        return r


def test_small_model_abstention_escalates_to_large(chunks, retrievers, embedder):
    llm = RecordingLLM(abstain_if=lambda model, user: model == "mock/small")
    p = make_pipeline(chunks, retrievers, embedder, llm)
    a = p.ask("How much does Starter cost per month?")
    assert a.escalated and a.route == "large" and a.model == "mock/large" and ABSTAIN not in a.text
    assert [m for m, _ in llm.calls] == ["mock/small", "mock/large"]
    small, large = llm.results
    assert small.cost_usd > 0 and large.cost_usd > 0
    assert a.cost_usd == small.cost_usd + large.cost_usd                 # both calls are paid
    assert a.prompt_tokens == small.prompt_tokens + large.prompt_tokens
    assert a.completion_tokens == small.completion_tokens + large.completion_tokens
    assert a.baseline_cost_usd == large.cost_usd
    assert p.request_log.recent(1)[0]["escalated"] is True


def test_pipeline_without_router_uses_default_route(chunks, retrievers, embedder):
    p = make_pipeline(chunks, retrievers, embedder, MockLLM(), router=None, default_route="small",
                      escalate_on_abstain=False)
    assert p.ask("How much does Starter cost per month?").route == "small"
    p = make_pipeline(chunks, retrievers, embedder, MockLLM(), router=None)
    assert p.ask("How much does Starter cost per month?").route == "large"


def test_cache_hit_skips_the_model_and_charges_only_verification(chunks, retrievers, embedder):
    llm = MockLLM()
    cache = SemanticCache(0.99, verifier=KeyTermGuard.from_corpus(chunks), clock=lambda: 0.0,
                          require_same_evidence=True)
    p = make_pipeline(chunks, retrievers, embedder, llm, cache=cache)
    first = p.ask("How much does Starter cost per month?")
    second = p.ask("how much does starter cost per month")
    assert second.route == "cache" and second.cache_tier == "exact" and second.cost_usd == 0.0
    assert second.text == first.text and second.baseline_cost_usd == first.baseline_cost_usd
    assert len(llm.calls) == 1 and len(cache) == 1
    assert cache.entries[0].evidence_key == "pricing.md#0" and cache.entries[0].sources["pricing.md"]
    stats = p.request_log.stats()
    assert stats["cache_hits"] == 1 and stats["saving_usd"] > 0 and stats["latency_ms_p95"] is not None


def test_abstained_answers_are_not_cached(chunks, retrievers, embedder):
    llm = MockLLM(respond=lambda *_: ABSTAIN)
    cache = SemanticCache(0.99, clock=lambda: 0.0)
    p = make_pipeline(chunks, retrievers, embedder, llm, cache=cache, router=None)
    p.ask("How much does Starter cost per month?")
    assert len(cache) == 0


def test_changed_document_invalidates_cached_answer(chunks, retrievers, embedder):
    llm = MockLLM()
    cache = SemanticCache(0.99, clock=lambda: 0.0)
    p = make_pipeline(chunks, retrievers, embedder, llm, cache=cache, router=None)
    p.ask("How much does Starter cost per month?")
    p.doc_hashes = {**p.doc_hashes, "pricing.md": "changed"}
    a = p.ask("How much does Starter cost per month?")
    assert a.route == "large" and len(llm.calls) == 2 and cache.stats.invalidated == 1


def test_user_prompt_lists_chunks_with_ids(chunks, retrievers, embedder):
    hits = retrievers["hybrid"].search("refunded within days", embedder.encode(["refunded within days"])[0], 2)
    prompt = build_user_prompt("refund?", hits)
    assert prompt.startswith("Context:\n[") and prompt.endswith("Question: refund?") and "refunds.md#" in prompt
