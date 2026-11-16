# Costwise RAG: architecture and design decisions

A RAG service with a semantic cache and a cost-aware model router in front of the LLM.
The goal is one measurable claim: **lower LLM cost and p95 latency at the same answer quality**,
with every number produced by a reproducible eval.

## 1. What the system does

```
query
  │
  ▼
[1 embed query] ──► [2 retrieve: BM25 + dense, fused]
                              │
                              ▼
                    [3 cache: exact → semantic → guard → (band) strong verifier] ──hit──► answer
                              │ miss                                                    (no LLM call)
                              ▼
                    [4 route: small or large model]
                              │
                              ▼
                    [5 generate] ──small model abstains──► escalate to large
                              │
                              ▼
                    [6 store in cache] ─► [7 log cost, tokens, latency] ─► answer
```

Retrieval runs before the cache lookup because the cache uses the top retrieved chunk as an
evidence key: a candidate answered from a different chunk is never served. Retrieval is local
and costs about a millisecond, so this ordering gives up nothing.

## 2. Code structure: ports and adapters

The core pipeline depends only on small interfaces ("ports"). Every outside thing
(embedding model, LLM provider, storage) is an adapter that can be swapped or faked in tests.

```
src/costwise/
  domain/        pure logic, no I/O
    models.py      Chunk, Hit, CacheEntry, LLMResult, Answer
    ports.py       Embedder, Retriever, Verifier, LLMClient, Router, RequestLog
    chunking.py    markdown → chunks
    pipeline.py    the 7 stages above
    cache.py       SemanticCache (exact tier, semantic tier, verifier, TTL, invalidation)
    routing.py     features, HeuristicRouter, LearnedRouter
    pricing.py     price table and cost maths
  adapters/      everything that touches the outside world
    embedders.py   SentenceTransformerEmbedder, HashEmbedder (tests)
    retrieval.py   DenseIndex, BM25Index, HybridRetriever (rank fusion)
    verifiers.py   KeyTermGuard, CrossEncoderVerifier (optional)
    llm.py         OpenRouterLLM, MockLLM
    corpus.py      reads the markdown corpus from disk (chunking itself stays pure in domain/)
    storage.py     SQLite cache store and request log
  app/           entry points
    container.py   builds a Pipeline from Settings
    api.py         FastAPI: /ask, /stats, /metrics, /recent, /cache, dashboard
    metrics.py     Prometheus text exposition for /metrics
    dashboard.html
  eval/          the numbers
    dataset.py     questions, splits, query stream
    offline.py     retrieval, cache and router evals (no LLM, free)
    live.py        end-to-end A/B with a hard budget cap
```

Why this shape: the cache and router are the product, so they sit in `domain/` as plain Python
that can be unit-tested with fakes in milliseconds. Swapping OpenRouter for another provider, or
SQLite for Redis, touches one adapter file.

## 3. Decisions, with the alternatives

### 3.1 Cache lookup: how to decide two questions are "the same"

This is the hard part of a semantic cache. "Refund window for monthly plans" and "refund window
for annual plans" are near-identical as embeddings and have different answers. A wrong cache hit
returns a confident wrong answer, which is worse than a miss.

| Option | Good | Bad |
|---|---|---|
| A. Exact-match only | Never wrong | Misses every paraphrase; low hit rate |
| B. Embedding similarity over a threshold | Simple; catches paraphrases | Near-miss questions cause wrong hits; one threshold trades hit rate against wrong hits |
| C. B, then a second check on the candidate (verifier) | Keeps paraphrase hits, rejects near-misses | More moving parts; verifier adds a few ms |
| D. B, then ask an LLM "same question?" | Most accurate | Costs money and latency on every candidate, which defeats the cache |

**Chosen: C, in three tiers, with D confined to a narrow band.** (1) Exact match on the
normalised query, free and always right. (2) Embedding similarity picks the best candidates above
a *candidate* threshold. (3) The cheap verifier (KeyTermGuard) must accept the pair. Above the
*accept* threshold that is enough; in the band between the two thresholds a strong verifier
must also agree, and it is called at most once per lookup.

Verifier options, all behind one `Verifier` interface:

- **KeyTermGuard (default, free):** rejects the pair if the numbers differ as a set, or if each
  question has an entity term the other lacks (plan names, regions, "monthly" vs "annual").
  Entity terms are extracted from the corpus. Deterministic, no model download, easy to explain.
- **LLMVerifier (strong, paid):** asks the small model "same information? YES or NO" with a
  three-token answer. About 0.5% of the cost of a large-model answer, so a cache that only
  pays it on real candidates still saves money. Its cost is charged to the request, hit or miss.
- **CrossEncoderVerifier (strong, optional):** a duplicate-question cross-encoder. Free per
  call but needs a model download (about 300 MB) and 10–30 ms on CPU. Not downloaded yet.

**Why two thresholds (finding from the Phase 2 offline eval, tune split):** with MiniLM
embeddings, same-question paraphrases sit mostly at similarity 0.6–0.9 (only 6 of 75 tune
pairs above 0.9), while deliberate near-misses ("refund window for monthly" vs "annual") sit
in the same range. No single threshold separates them: the lowest tune threshold with zero
false accepts under the guard is 0.86 (0.88 with margin), which recovers only 11 of 75 tune
pairs. The guard removes substitutions and added numbers, but true semantic near-misses
remain ("how long can a deleted workspace be restored" vs "how long until it is purged",
similarity 0.86). So the free path keeps the strict accept threshold, and the band below it
is opened only when a strong verifier is configured. The offline eval reports the free
configuration and the *upper bound* of the band with a perfect verifier; the real
strong-verifier numbers can only come from the paid live run.

Both thresholds are fitted on the tune split (accept: lowest grid value with zero false accepts
plus a 0.02 margin; candidate: lowest grid value whose band is at least half true repeats, so
the verifier is called about once per correct hit it adds) and reported on the test split.

### 3.2 Cache freshness

| Option | Good | Bad |
|---|---|---|
| TTL only | Simple | Serves stale answers until expiry after a doc changes |
| Flush everything when the corpus changes | Always fresh | Throws away valid entries |
| Per-entry source tracking | Only entries built from changed docs are dropped | Slightly more bookkeeping |

**Chosen: TTL plus per-entry source tracking.** Each entry stores the content hash of the
documents its answer came from; if any of those documents changed, the entry is invalid.

### 3.3 Routing: which model answers

| Option | Good | Bad |
|---|---|---|
| A. Hand-written rules | Transparent, no data needed | Brittle; hard to tune |
| B. Classifier on hand-made features (length, numbers, multi-part cues, retrieval spread) | Small data is enough; explainable | Features are domain-shaped |
| C. Classifier on the query embedding (RouteLLM style) | Learns its own features | Needs hundreds to thousands of labelled queries |
| D. Cascade: always try small, escalate when it is unsure (FrugalGPT style) | No classifier; safe | Hard questions pay for two calls and wait twice |
| E. Ask an LLM to route | Flexible | Adds a paid call to every request |

**Chosen: B before the call, D as the safety net.** A logistic-regression router on features
decides small or large up front. If the small model replies that the context is insufficient, the
request escalates to the large model. A is kept as the baseline so the eval shows what learning
adds. C is the upgrade path once there is real traffic to label.

The router's threshold is a cost/quality dial: the eval reports the curve, not one point.

### 3.4 Retrieval

| Option | Good | Bad |
|---|---|---|
| Dense only | Handles paraphrase | Weak on exact tokens like "3000" or "Starter" |
| BM25 only | Strong on exact tokens | Weak on paraphrase |
| Hybrid: BM25 + dense, reciprocal rank fusion | Best of both; no tuning of score scales | Two indexes |
| Hybrid + cross-encoder rerank | Highest precision | Extra model, extra latency |

**Chosen: hybrid with rank fusion,** and the eval reports recall for dense, BM25 and hybrid so
the choice is backed by numbers. Reranking is left out: retrieval is not what this project is
about, and the corpus is small.

### 3.5 Storage and providers

| Part | Chosen | Alternative considered | Why |
|---|---|---|---|
| Vector search | NumPy brute force behind `Retriever` | FAISS, hnswlib, Qdrant, Chroma | Exact and instant at this corpus size; an ANN store is a one-file swap when it grows |
| Cache and request log | SQLite | Redis, Postgres | Zero setup, single file, survives restarts; Redis is the swap for multi-instance |
| LLM access | Thin `httpx` client for OpenRouter | LiteLLM, provider SDKs | One key reaches small and large models and returns real cost per call; no heavy dependency |
| Embeddings | `all-MiniLM-L6-v2`, local | API embeddings | Free, fast on CPU, already on this machine |
| Models | small `anthropic/claude-haiku-5.5` ($0.10 / $0.50 per M tokens), large `anthropic/claude-sonnet-5.5` ($2 / $10) | any pair | 20x price gap makes routing matter; both are config values |
| Metrics | SQLite log + one dashboard page + JSON `/stats` + Prometheus text at `/metrics` (hand-written, no client library) | prometheus_client, Grafana, Langfuse | No extra services to run; any Prometheus can scrape `/metrics` |

## 4. How it is evaluated

Two layers, so most numbers cost nothing to produce.

**Offline (free, no LLM):**
- Retrieval recall@k for dense, BM25 and hybrid.
- Cache: hit rate, wrong-hit rate and paraphrase recall across thresholds, for threshold-only
  versus each free verifier, plus the upper bound of the two-threshold band with a perfect
  strong verifier (labelled as such). Reported with the repeat rate of the query stream,
  because hit rate depends on it.
- Router: accuracy, share of hard questions sent to the large model, share of easy ones kept
  on the small model, heuristic versus learned.

**Live (uses OpenRouter credit, hard budget cap):** the same query stream through five
configurations with identical retrieval and prompt:

| Config | Cache | Router |
|---|---|---|
| baseline | off | always large |
| always-small | off | always small |
| + router | off | on |
| + cache | on | always large |
| full | on | on |

Reported per config: answer accuracy, total cost, cost per query, p50 and p95 latency, cache
hits and wrong hits, route split, escalations.

**Honesty rules:** thresholds and router weights are fitted on the tune split and reported on the
test split. The corpus is a 12-document synthetic knowledge base with 60 questions (46 easy,
14 multi-step; 25 tune, 35 test), 2 paraphrases each, and deliberate near-miss pairs; results
say so, and every percentage is reported with its counts. No number
goes on a resume unless a run produced it.

## 5. Known limits

- A small synthetic corpus shows the method works; it does not prove production numbers. The
  stronger follow-up is a public paraphrase set (Quora Question Pairs) for the cache.
- KeyTermGuard depends on entity terms found in the corpus; it will miss near-misses that differ
  only in ordinary words. The strong verifier (LLM or cross-encoder) covers that case, but only
  inside the band; a near-miss above the accept threshold still gets through.
- Single process: the SQLite cache is not shared across instances.
- Answers are graded by required key facts, not by a human or an LLM judge.
