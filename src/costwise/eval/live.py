"""End-to-end A/B eval: the same query stream through five configurations.

    python -m costwise.eval.live                       # MOCK dry run (default): free, numbers not real
    python -m costwise.eval.live --live --max-cost 0.5 # PAID run on OpenRouter, hard budget cap in USD

The paid run builds its client with the app's `build_llm` (app/container.py, the only place that
reads OPENROUTER_API_KEY; the key is never printed or logged). The budget is checked BEFORE every
paid call, retries included, with a price-table estimate of that call; when the next call would
cross the cap, the run stops and the partial results say so. A model with no price-table entry is
refused at startup unless `--price MODEL=IN,OUT` (USD per million tokens) is given, because its
estimate would be 0 and the cap blind. A provider error that survives the retries stops only the
current config: it is marked with the error, its partial results are kept, and the run goes on.
The results file is always written, as strict JSON.

Every percentage is written next to its counts. Thresholds and router weights come from
data/tuned.json (fitted on the tune split); the stream is built from the test split.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Sequence

from ..adapters.corpus import load_corpus
from ..adapters.llm import MockLLM, ProviderError
from ..adapters.retrieval import build_retrievers
from ..adapters.storage import SQLiteRequestLog
from ..app.container import build_cache, build_embedder, build_llm, build_router, route_threshold_for
from ..domain.chunking import doc_hashes
from ..domain.models import LLMResult
from ..domain.pipeline import Pipeline
from ..domain.ports import LLMClient
from ..domain.pricing import PRICES_PER_M, cost_usd, estimate_tokens, has_price
from ..settings import ROOT, Settings
from .dataset import Query, all_texts, build_stream, is_correct, load_questions

# name -> (cache on?, router on?, route when the router is off, escalate on abstain?)
CONFIGS: dict[str, tuple[bool, bool, str, bool]] = {
    "baseline_always_large": (False, False, "large", False),
    "always_small": (False, False, "small", False),
    "router_only": (False, True, "large", True),
    "cache_only": (True, False, "large", False),
    "full": (True, True, "large", True),
}


class BudgetExceeded(RuntimeError):
    pass


class BudgetedLLM:
    """Wraps any LLMClient with a hard spending cap shared across the whole run.

    The cap is checked before every call with a price-table estimate, so the run overshoots by
    at most one call. If the inner client retries (OpenRouterLLM), each retry is checked too,
    counting every attempt of the current request at its estimate. Spend is also kept per config
    (`config`), so calls inside a request that later aborted are still accounted for.
    `max_cost_usd=None` means no cap (mock dry runs).
    """

    def __init__(self, inner: LLMClient, max_cost_usd: float | None,
                 prices: dict[str, tuple[float, float]] | None = None):
        self.inner, self.max_cost_usd = inner, max_cost_usd
        self.prices = {**PRICES_PER_M, **(prices or {})}
        self.spent_usd, self.calls, self.retries = 0.0, 0, 0
        self.spent_by_config: dict[str, float] = {}
        self.config: str | None = None
        self._attempts = 0
        if hasattr(inner, "before_retry"):
            inner.before_retry = self._before_retry

    def _estimate(self, model: str, system: str, user: str, max_tokens: int) -> float:
        return cost_usd(model, estimate_tokens(system + user), max_tokens, self.prices)

    def _check(self, model: str, system: str, user: str, max_tokens: int, attempts: int) -> None:
        if self.max_cost_usd is None:
            return
        estimate = self._estimate(model, system, user, max_tokens)
        if self.spent_usd + attempts * estimate > self.max_cost_usd:
            raise BudgetExceeded(f"next call (about ${estimate:.5f}) would exceed the cap of ${self.max_cost_usd}; "
                                 f"spent ${self.spent_usd:.5f} over {self.calls} calls")

    def _before_retry(self, model: str, system: str, user: str, max_tokens: int) -> None:
        self._attempts += 1
        self._check(model, system, user, max_tokens, self._attempts)
        self.retries += 1

    def complete(self, model: str, system: str, user: str, max_tokens: int) -> LLMResult:
        self._attempts = 1
        self._check(model, system, user, max_tokens, 1)
        result = self.inner.complete(model, system, user, max_tokens)
        if result.cost_usd == 0 and model not in PRICES_PER_M and has_price(model, self.prices):
            # The provider reported no cost and the table has no entry: use the given override.
            result.cost_usd = cost_usd(model, result.prompt_tokens, result.completion_tokens, self.prices)
        self.spent_usd += result.cost_usd
        self.calls += 1
        if self.config is not None:
            self.spent_by_config[self.config] = self.spent_by_config.get(self.config, 0.0) + result.cost_usd
        return result


@dataclass
class ConfigResult:
    queries: int = 0
    correct: int = 0
    cost_usd: float = 0.0
    llm_spent_usd: float = 0.0          # every paid call in this config, finished requests or not
    baseline_cost_usd: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    routes: dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    wrong_cache_hits: int = 0
    correct_on_cache_hits: int = 0
    escalations: int = 0
    abstained: int = 0
    strong_verifier_calls: int = 0
    stopped: str | None = None
    error: str | None = None

    def summary(self) -> dict:
        lat = sorted(self.latencies_ms)
        pct = lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))], 1) if lat else None
        n = self.queries
        share = lambda x, d: round(x / d, 4) if d else None
        return {
            "queries": n, "correct": self.correct, "accuracy": share(self.correct, n),
            "cost_usd": round(self.cost_usd, 6), "cost_per_query_usd": round(self.cost_usd / n, 6) if n else None,
            "llm_spent_usd": round(self.llm_spent_usd, 6),
            "always_large_cost_usd": round(self.baseline_cost_usd, 6),
            "latency_ms_p50": pct(0.5), "latency_ms_p95": pct(0.95),
            "routes": self.routes, "cache_hits": self.cache_hits, "cache_hit_rate": share(self.cache_hits, n),
            "wrong_cache_hits": self.wrong_cache_hits, "correct_on_cache_hits": self.correct_on_cache_hits,
            "escalations": self.escalations, "abstained_final": self.abstained,
            "strong_verifier_calls": self.strong_verifier_calls, "stopped": self.stopped, "error": self.error,
        }


def make_pipeline(settings: Settings, tuned: dict, chunks, retriever, embedder, llm: LLMClient,
                  cache_on: bool, router_on: bool, default_route: str, escalate: bool) -> Pipeline:
    cache = build_cache(settings, tuned, chunks, llm, persistent=False) if cache_on else None
    router = build_router(settings, tuned) if router_on else None
    return Pipeline(embedder, retriever, llm, settings.small_model, settings.large_model, cache=cache,
                    router=router, default_route=default_route,
                    route_threshold=route_threshold_for(settings, tuned),
                    top_k=settings.top_k, max_answer_tokens=settings.max_answer_tokens,
                    escalate_on_abstain=escalate, doc_hashes=doc_hashes(chunks), request_log=SQLiteRequestLog(None))


def run_config(pipeline: Pipeline, stream: Sequence[Query], keywords: dict[str, list],
               qid_of: dict[str, str]) -> ConfigResult:
    r = ConfigResult()
    for q in stream:
        try:
            a = pipeline.ask(q.text, tag=q.qid)
        except BudgetExceeded as e:
            r.stopped = str(e)
            break
        except ProviderError as e:
            r.error = f"provider error: {e}"
            break
        r.queries += 1
        ok = is_correct(a.text, keywords[q.qid])
        r.correct += ok
        r.cost_usd += a.cost_usd
        r.baseline_cost_usd += a.baseline_cost_usd
        r.latencies_ms.append(a.latency_ms)
        r.routes[a.route] = r.routes.get(a.route, 0) + 1
        r.escalations += a.escalated
        r.abstained += "INSUFFICIENT_CONTEXT" in a.text.upper()
        if a.route == "cache":
            r.cache_hits += 1
            r.wrong_cache_hits += (a.cache_tag or qid_of.get(a.query)) != q.qid
            r.correct_on_cache_hits += ok
    if pipeline.cache is not None:
        r.strong_verifier_calls = pipeline.cache.stats.strong_verifier_calls
    return r


def check_prices(models: Sequence[str], prices: dict[str, tuple[float, float]] | None) -> None:
    """Refuses to start a paid run with a model the budget cap cannot price."""
    table = {**PRICES_PER_M, **(prices or {})}
    for model in dict.fromkeys(models):
        if not has_price(model, table):
            raise SystemExit(f"no price for model {model!r}: the budget cap would be blind. Add it to "
                             f"domain/pricing.py or pass --price {model}=IN,OUT (USD per million tokens).")


def run(settings: Settings, llm: LLMClient, mock: bool, max_cost_usd: float | None, split: str = "test",
        repeat_rate: float = 0.5, seeds: Sequence[int] = (0, 1), configs: Sequence[str] = tuple(CONFIGS),
        embedder=None, strong_verifier: str = "none",
        prices: dict[str, tuple[float, float]] | None = None) -> dict:
    if not mock:
        check_prices([settings.small_model, settings.large_model], prices)
    settings.cache_strong_verifier = strong_verifier          # "none" | "llm": opens the cache band
    tuned = settings.load_tuned()
    chunks = load_corpus(settings.corpus_dir)
    embedder = embedder or build_embedder(settings)
    retriever = build_retrievers(chunks, embedder)["hybrid"]
    questions = load_questions(settings.questions_path, split)
    keywords = {q["id"]: q["keywords"] for q in questions}
    qid_of = {t: q["id"] for q in questions for t in all_texts(q)}
    # One long stream: several seeds back to back, so repeats are spread over a longer run.
    stream = [q for seed in seeds for q in build_stream(questions, repeat_rate, seed)]
    budget = BudgetedLLM(llm, max_cost_usd, prices)
    out = {
        "mock": mock,
        "note": ("MOCK MODEL: cost, latency and accuracy are NOT real; they only prove the harness runs."
                 if mock else "Live run on OpenRouter with a hard budget cap."),
        "models": {"small": settings.small_model, "large": settings.large_model},
        "split": split, "questions": len(questions), "stream_queries": len(stream),
        "stream_repeats": sum(q.repeat for q in stream), "repeat_rate": repeat_rate, "seeds": list(seeds),
        "max_cost_usd": max_cost_usd, "strong_verifier": strong_verifier,
        "price_overrides": {m: list(p) for m, p in (prices or {}).items()},
        "tuned": {k: v for k, v in tuned.items() if k != "router_weights"},
        "configs": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name in configs:
        cache_on, router_on, default_route, escalate = CONFIGS[name]
        budget.config = name
        try:
            pipeline = make_pipeline(settings, tuned, chunks, retriever, embedder, budget, cache_on, router_on,
                                     default_route, escalate)
            result = run_config(pipeline, stream, keywords, qid_of)
        except Exception as e:                    # unexpected: keep what we have and stop cleanly
            result = ConfigResult(error=f"{type(e).__name__}: {e}")
        result.llm_spent_usd = budget.spent_by_config.get(name, 0.0)
        out["configs"][name] = result.summary()
        if result.stopped:
            out["stopped"] = f"budget cap reached during config {name!r}"
            break
        if result.error and not result.error.startswith("provider error"):
            out["stopped"] = f"error during config {name!r}"
            break
    out["total_spent_usd"] = round(budget.spent_usd, 6)
    out["total_llm_calls"] = budget.calls
    out["total_retries"] = budget.retries
    return out


def parse_price(text: str) -> tuple[str, tuple[float, float]]:
    model, _, pair = text.partition("=")
    try:
        inp, out = (float(x) for x in pair.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--price wants MODEL=IN,OUT (USD per million tokens), got {text!r}")
    return model, (inp, out)


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="spend OpenRouter credit (default: mock dry run)")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="hard cap in USD (default 0.50 for --live; no cap for the mock dry run)")
    ap.add_argument("--seeds", type=int, default=2, help="number of stream seeds run back to back")
    ap.add_argument("--repeat-rate", type=float, default=0.5)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--strong-verifier", choices=["none", "llm"], default="none",
                    help="llm: small-model YES/NO check opens the cache band [candidate, accept)")
    ap.add_argument("--price", type=parse_price, action="append", default=[],
                    help="MODEL=IN,OUT price override in USD per million tokens (for models not in the table)")
    ap.add_argument("--out", default=None, help="output JSON (default results/live.json or results/live_mock.json)")
    args = ap.parse_args(argv)
    settings = Settings()
    prices = dict(args.price)
    if args.live:
        settings.llm_provider = "openrouter"
        check_prices([settings.small_model, settings.large_model], prices)
        try:
            llm: LLMClient = build_llm(settings)       # the only place that reads OPENROUTER_API_KEY
        except RuntimeError as e:
            raise SystemExit(f"--live: {e}")
    else:
        llm = MockLLM()
    max_cost = args.max_cost if args.max_cost is not None else (0.50 if args.live else None)
    result = run(settings, llm, mock=not args.live, max_cost_usd=max_cost, repeat_rate=args.repeat_rate,
                 seeds=range(args.seeds), configs=args.configs, strong_verifier=args.strong_verifier,
                 prices=prices)
    out = ROOT / "results" / (args.out or ("live.json" if args.live else "live_mock.json"))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "configs"}, indent=2))
    for name, c in result["configs"].items():
        print(f"{name:22} correct {c['correct']}/{c['queries']}  cost ${c['cost_usd']:.5f}  "
              f"p95 {c['latency_ms_p95']} ms  cache {c['cache_hits']} (wrong {c['wrong_cache_hits']})  "
              f"routes {c['routes']}  esc {c['escalations']}" + ("  [MOCK, not real]" if result["mock"] else "")
              + (f"  ERROR: {c['error']}" if c["error"] else ""))
    print(f"\nwritten: {out}")
    return result


if __name__ == "__main__":
    main()
