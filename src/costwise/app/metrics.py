"""Prometheus text exposition (format 0.0.4) for GET /metrics, written by hand: no client library.

The request counters come from the SQLite request log, which is append-only, so they only grow
and survive restarts. Cache counters come from the in-process cache stats and reset on restart,
which Prometheus `rate()` handles as a counter reset. When the mock model is in use, cost
metrics are not real money; the `costwise_info` series carries `mock="true"` to say so.
"""
from __future__ import annotations

from typing import Iterable

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(labels: dict | None) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{k}="{_escape(v)}"' for k, v in labels.items()) + "}"


def _num(value: float | int | None) -> str:
    if value is None:
        return "NaN"
    if isinstance(value, bool):
        return "1" if value else "0"
    return repr(float(value)) if isinstance(value, float) else str(value)


class _Writer:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def metric(self, name: str, kind: str, help_text: str,
               samples: Iterable[tuple[dict | None, float | int | None]]) -> None:
        self.lines.append(f"# HELP {name} {help_text}")
        self.lines.append(f"# TYPE {name} {kind}")
        for labels, value in samples:
            self.lines.append(f"{name}{_labels(labels)} {_num(value)}")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def render(request_stats: dict, *, llm_provider: str, small_model: str, large_model: str,
           cache_entries: int | None = None, cache_stats: dict | None = None) -> str:
    """`request_stats` is SQLiteRequestLog.stats(); `cache_stats` is asdict(CacheStats) or None."""
    w = _Writer()
    w.metric("costwise_info", "gauge", "Service info; mock=\"true\" means cost metrics are not real money.",
             [({"llm_provider": llm_provider, "mock": str(llm_provider == "mock").lower(),
                "small_model": small_model, "large_model": large_model}, 1)])
    routes = request_stats.get("by_route") or {}
    w.metric("costwise_requests_total", "counter", "Answered requests by route (small, large, cache).",
             [({"route": r}, n) for r, n in sorted(routes.items())] or [({"route": "none"}, 0)])
    w.metric("costwise_llm_cost_usd_total", "counter", "LLM spend in USD (not real money with the mock model).",
             [(None, request_stats.get("cost_usd", 0.0))])
    w.metric("costwise_baseline_cost_usd_total", "counter",
             "What the same requests would have cost on the large model alone, in USD.",
             [(None, request_stats.get("baseline_cost_usd", 0.0))])
    w.metric("costwise_escalations_total", "counter", "Small-model answers escalated to the large model.",
             [(None, request_stats.get("escalations", 0))])
    w.metric("costwise_request_latency_ms", "summary", "End-to-end request latency in milliseconds.",
             [({"quantile": "0.5"}, request_stats.get("latency_ms_p50")),
              ({"quantile": "0.95"}, request_stats.get("latency_ms_p95"))])
    w.lines.append(f"costwise_request_latency_ms_count {request_stats.get('requests', 0)}")
    if cache_stats is not None:
        w.metric("costwise_cache_entries", "gauge", "Entries in the semantic cache.", [(None, cache_entries or 0)])
        w.metric("costwise_cache_lookups_total", "counter", "Cache lookups since start.",
                 [(None, cache_stats.get("lookups", 0))])
        w.metric("costwise_cache_hits_total", "counter", "Cache hits by tier since start.",
                 [({"tier": t}, cache_stats.get(f"{t}_hits", 0)) for t in ("exact", "semantic", "verified")])
        w.metric("costwise_cache_rejections_total", "counter", "Candidates rejected, by reason, since start.",
                 [({"reason": r}, cache_stats.get(k, 0)) for r, k in (
                     ("verifier", "rejected_by_verifier"), ("evidence", "rejected_by_evidence"),
                     ("strong_verifier", "rejected_by_strong_verifier"), ("expired", "expired"),
                     ("invalidated", "invalidated"))])
        w.metric("costwise_cache_strong_verifier_calls_total", "counter", "Strong-verifier calls since start.",
                 [(None, cache_stats.get("strong_verifier_calls", 0))])
        w.metric("costwise_cache_strong_verifier_cost_usd_total", "counter", "Strong-verifier spend in USD.",
                 [(None, cache_stats.get("strong_verifier_cost_usd", 0.0))])
    return w.text()
