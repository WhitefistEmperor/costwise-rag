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


class ProviderError(RuntimeError):
    """The provider failed to return a usable answer (after any retries). The message never
    contains the API key or request headers."""


class OpenRouterLLM:
    """Thin httpx client for OpenRouter's chat completions. Asks for `usage.include` so the
    response carries the real cost in credits (USD); falls back to the price table otherwise.

    HTTP 429, 5xx and timeouts are retried up to `max_retries` times with exponential backoff
    (`backoff_s`, then twice that, ...). Every retry is treated as a paid call: if
    `before_retry(model, system, user, max_tokens)` is set (the budget wrapper sets it), it runs
    before each retry and may raise to stop it. Other errors, and a 200 response without
    `choices`, raise `ProviderError` at once.
    """

    name = "openrouter"

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", timeout_s: float = 60.0,
                 transport: httpx.BaseTransport | None = None, max_retries: int = 2, backoff_s: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep):
        if not api_key:
            raise ValueError("OpenRouterLLM needs an API key (set OPENROUTER_API_KEY in the environment)")
        self.max_retries, self.backoff_s, self._sleep = max_retries, backoff_s, sleep
        self.before_retry: Callable[[str, str, str, int], None] | None = None
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport,
                                    headers={"Authorization": f"Bearer {api_key}",
                                             "HTTP-Referer": "https://github.com/costwise-rag",
                                             "X-Title": "costwise-rag"})

    def _post(self, body: dict, model: str) -> httpx.Response:
        """One attempt. Returns the response, or raises `_Retryable` / `ProviderError`."""
        try:
            r = self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise _Retryable(f"timeout calling the provider for {model} ({type(e).__name__})") from None
        except httpx.TransportError as e:
            raise _Retryable(f"network error calling the provider for {model} ({type(e).__name__})") from None
        if r.status_code == 429 or r.status_code >= 500:
            raise _Retryable(f"HTTP {r.status_code} from the provider for {model}")
        if r.status_code >= 400:
            raise ProviderError(f"HTTP {r.status_code} from the provider for {model}: {_error_message(r)}")
        return r

    def complete(self, model: str, system: str, user: str, max_tokens: int) -> LLMResult:
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "usage": {"include": True}}
        attempt = 0
        while True:
            t0 = time.perf_counter()
            try:
                r = self._post(body, model)
                break
            except _Retryable as e:
                if attempt >= self.max_retries:
                    raise ProviderError(f"{e} after {attempt + 1} attempts") from None
                self._sleep(self.backoff_s * 2 ** attempt)
                attempt += 1
                if self.before_retry is not None:
                    self.before_retry(model, system, user, max_tokens)
        latency = (time.perf_counter() - t0) * 1000
        try:
            data = r.json()
        except ValueError:
            raise ProviderError(f"provider returned a non-JSON body for {model}") from None
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise ProviderError(f"provider response for {model} has no choices: {_error_message(r)}")
        usage = data.get("usage") or {}
        p, c = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        cost = usage.get("cost")
        text = ((choices[0].get("message") or {}).get("content") or "").strip()
        return LLMResult(text, data.get("model", model), p, c,
                         float(cost) if cost is not None else cost_usd(model, p, c), latency)

    def close(self) -> None:
        self._client.close()


class ZaiLLM:
    """OpenAI-compatible client for Z.ai (GLM Flash, free tier).

    Uses the standard OpenAI chat-completions format; `base_url` defaults to
    `https://api.z.ai/api/paas/v4` (set COSTWISE_ZAI_BASE_URL to override for
    the China platform: `https://open.bigmodel.cn/api/paas/v4`).
    The API key is read from `ZAI_API_KEY` by the caller and passed in; it is
    never printed or logged here.

    Cost is always 0.0 (free tier); prompt_tokens and completion_tokens come
    from the standard OpenAI usage object in the response.
    """

    name = "zai"

    def __init__(self, api_key: str, base_url: str = "https://api.z.ai/api/paas/v4",
                 timeout_s: float = 60.0, transport: httpx.BaseTransport | None = None,
                 max_retries: int = 6, backoff_s: float = 3.0,
                 sleep: Callable[[float], None] = time.sleep, min_interval_s: float = 2.0):
        if not api_key:
            raise ValueError("ZaiLLM needs an API key (set ZAI_API_KEY in the environment)")
        self.max_retries, self.backoff_s, self._sleep = max_retries, backoff_s, sleep
        self.min_interval_s, self._last_call = min_interval_s, 0.0
        self.before_retry: Callable[[str, str, str, int], None] | None = None
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport,
                                    headers={"Authorization": f"Bearer {api_key}"})

    def _post(self, body: dict, model: str) -> httpx.Response:
        try:
            r = self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise _Retryable(f"timeout calling Z.ai for {model} ({type(e).__name__})") from None
        except httpx.TransportError as e:
            raise _Retryable(f"network error calling Z.ai for {model} ({type(e).__name__})") from None
        if r.status_code == 429 or r.status_code >= 500:
            raise _Retryable(f"HTTP {r.status_code} from Z.ai for {model}")
        if r.status_code >= 400:
            raise ProviderError(f"HTTP {r.status_code} from Z.ai for {model}: {_error_message(r)}")
        return r

    def complete(self, model: str, system: str, user: str, max_tokens: int) -> LLMResult:
        # Thinking models spend output tokens before answering and reject tiny limits.
        body = {"model": model, "max_tokens": max(max_tokens, 1024), "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        attempt = 0
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            self._sleep(wait)
        while True:
            t0 = time.perf_counter()
            try:
                r = self._post(body, model)
                break
            except _Retryable as e:
                if attempt >= self.max_retries:
                    raise ProviderError(f"{e} after {attempt + 1} attempts") from None
                self._sleep(self.backoff_s * 2 ** attempt)
                attempt += 1
                if self.before_retry is not None:
                    self.before_retry(model, system, user, max_tokens)
        self._last_call = time.monotonic()
        latency = (time.perf_counter() - t0) * 1000
        try:
            data = r.json()
        except ValueError:
            raise ProviderError(f"Z.ai returned a non-JSON body for {model}") from None
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise ProviderError(f"Z.ai response for {model} has no choices: {_error_message(r)}")
        usage = data.get("usage") or {}
        p, c = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        text = ((choices[0].get("message") or {}).get("content") or "").strip()
        return LLMResult(text, data.get("model", model), p, c, 0.0, latency)

    def close(self) -> None:
        self._client.close()


class _Retryable(Exception):
    pass


def _error_message(r: httpx.Response) -> str:
    """The provider's own error message from the body, if any (bodies never echo the key)."""
    try:
        err = r.json().get("error")
    except (ValueError, AttributeError):
        return "no error message"
    if isinstance(err, dict):
        err = err.get("message")
    return str(err)[:200] if err else "no error message"
