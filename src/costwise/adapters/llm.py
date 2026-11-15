"""LLM clients behind the `LLMClient` port.

MockLLM is the default everywhere: it never touches the network and never costs anything real.
OpenRouterLLM is only built when the settings ask for it explicitly, and the API key is read from
the environment by the caller, never stored or printed.
"""
from __future__ import annotations

import time
from typing import Callable

import httpx

from ..domain.models import LLMResult
from ..domain.pricing import cost_usd, estimate_tokens


class MockLLM:
    """Deterministic stand-in. Its "answer" is the context it was given, so keyword grading of
    retrieved facts works offline. Cost comes from the price table and is NOT a real cost.

    `abstain_if(model, user) -> bool` lets tests and dry runs make the small model abstain.
    `respond(model, system, user) -> str` overrides the answer text entirely.
    """

    name = "mock"

    def __init__(self, latency_ms: float = 0.0, abstain_if: Callable[[str, str], bool] | None = None,
                 respond: Callable[[str, str, str], str] | None = None, abstain_text: str = "INSUFFICIENT_CONTEXT"):
        self.latency_ms, self.abstain_if, self.respond, self.abstain_text = latency_ms, abstain_if, respond, abstain_text
        self.calls: list[tuple[str, str]] = []

    def complete(self, model: str, system: str, user: str, max_tokens: int) -> LLMResult:
        self.calls.append((model, user))
        if self.respond is not None:
            text = self.respond(model, system, user)
        elif self.abstain_if is not None and self.abstain_if(model, user):
            text = self.abstain_text
        else:
            text = user.split("Context:\n", 1)[-1].split("\n\nQuestion:", 1)[0] if "Context:\n" in user else user
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        p, c = estimate_tokens(system + user), min(max_tokens, estimate_tokens(text))
        return LLMResult(text, model, p, c, cost_usd(model, p, c), float(self.latency_ms))


class OpenRouterLLM:
    """Thin httpx client for OpenRouter's chat completions. Asks for `usage.include` so the
    response carries the real cost in credits (USD); falls back to the price table otherwise."""

    name = "openrouter"

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", timeout_s: float = 60.0,
                 transport: httpx.BaseTransport | None = None):
        if not api_key:
            raise ValueError("OpenRouterLLM needs an API key (set OPENROUTER_API_KEY in the environment)")
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport,
                                    headers={"Authorization": f"Bearer {api_key}",
                                             "HTTP-Referer": "https://github.com/costwise-rag",
                                             "X-Title": "costwise-rag"})

    def complete(self, model: str, system: str, user: str, max_tokens: int) -> LLMResult:
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "usage": {"include": True}}
        t0 = time.perf_counter()
        r = self._client.post("/chat/completions", json=body)
        r.raise_for_status()
        latency = (time.perf_counter() - t0) * 1000
        data = r.json()
        usage = data.get("usage") or {}
        p, c = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        cost = usage.get("cost")
        text = (data["choices"][0]["message"].get("content") or "").strip()
        return LLMResult(text, data.get("model", model), p, c,
                         float(cost) if cost is not None else cost_usd(model, p, c), latency)

    def close(self) -> None:
        self._client.close()
