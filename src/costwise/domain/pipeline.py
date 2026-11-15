"""The seven stages: embed, retrieve, cache lookup, route, generate (with escalation), store, log.

Retrieval runs before the cache lookup because the top retrieved chunk is the cache's evidence
key. The strong verifier's cost (if any) is charged to the request whether it hits or misses.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .cache import SemanticCache
from .models import Answer, CacheEntry, Hit, LLMResult
from .ports import Embedder, LLMClient, RequestLog, Retriever, Router
from .pricing import cost_usd
from .routing import decide

ABSTAIN = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = (
    "You answer questions about the Brightline product using ONLY the context provided. Be brief "
    "and state the exact numbers, plan names and conditions from the context. If the context does "
    f"not contain enough information to answer, reply with exactly {ABSTAIN} and nothing else."
)


def build_user_prompt(query: str, hits: Sequence[Hit]) -> str:
    context = "\n\n".join(f"[{h.chunk.id}] {h.chunk.text}" for h in hits)
    return f"Context:\n{context}\n\nQuestion: {query}"


def abstained(text: str) -> bool:
    return ABSTAIN in text.upper() or not text.strip()


@dataclass
class Pipeline:
    embedder: Embedder
    retriever: Retriever
    llm: LLMClient
    small_model: str
    large_model: str
    cache: SemanticCache | None = None           # None: cache off
    router: Router | None = None                 # None: every request goes to `default_route`
    default_route: str = "large"                 # "large" | "small" when there is no router
    route_threshold: float = 0.5
    top_k: int = 4
    max_answer_tokens: int = 300
    escalate_on_abstain: bool = True
    doc_hashes: dict[str, str] = field(default_factory=dict)   # current doc -> hash, for invalidation
    request_log: RequestLog | None = None
    clock: Callable[[], float] = time.time

    def ask(self, query: str, tag: str | None = None) -> Answer:
        """`tag` is eval-only: the id of the question asked, stored on the cache entry so a
        later hit can be graded as right or wrong."""
        t0 = time.perf_counter()
        qvec = self.embedder.encode([query])[0]
        hits = self.retriever.search(query, qvec, self.top_k)
        evidence_key = hits[0].chunk.id if hits else None
        sources = [h.chunk.doc for h in hits]
        verify_cost = 0.0

        if self.cache is not None:
            looked = self.cache.lookup(query, qvec, self.doc_hashes or None, evidence_key)
            hit, verify_cost = looked.hit, looked.cost_usd
            if hit is not None:
                answer = Answer(query, hit.entry.answer, "cache", hit.entry.model, list(hit.entry.sources),
                                cache_tier=hit.tier, cache_similarity=round(hit.similarity, 4),
                                cache_tag=hit.entry.tag, cost_usd=verify_cost,
                                baseline_cost_usd=hit.entry.baseline_cost_usd,
                                latency_ms=(time.perf_counter() - t0) * 1000)
                return self._log(answer)

        score = self.router.score(query, hits) if self.router is not None else None
        route = decide(score, self.route_threshold) if score is not None else self.default_route
        model = self.large_model if route == "large" else self.small_model
        user = build_user_prompt(query, hits)
        result = self.llm.complete(model, SYSTEM_PROMPT, user, self.max_answer_tokens)
        escalated = False
        cost, p_tokens, c_tokens = result.cost_usd, result.prompt_tokens, result.completion_tokens
        if route == "small" and self.escalate_on_abstain and abstained(result.text):
            escalated = True
            second = self.llm.complete(self.large_model, SYSTEM_PROMPT, user, self.max_answer_tokens)
            cost += second.cost_usd
            p_tokens += second.prompt_tokens
            c_tokens += second.completion_tokens
            result, route, model = second, "large", self.large_model
        baseline = cost_usd(self.large_model, result.prompt_tokens, result.completion_tokens) \
            if model != self.large_model else result.cost_usd

        if self.cache is not None and not abstained(result.text):
            entry = CacheEntry(query, result.text, {h.chunk.doc: h.chunk.doc_hash for h in hits}, model,
                               self.clock(), baseline, tag=tag, evidence_key=evidence_key)
            self.cache.put(entry, qvec)

        answer = Answer(query, result.text, route, model, sources, route_score=None if score is None else round(score, 4),
                        escalated=escalated, prompt_tokens=p_tokens, completion_tokens=c_tokens,
                        cost_usd=cost + verify_cost, baseline_cost_usd=baseline,
                        latency_ms=(time.perf_counter() - t0) * 1000)
        return self._log(answer)

    def _log(self, answer: Answer) -> Answer:
        if self.request_log is not None:
            self.request_log.log(answer)
        return answer
