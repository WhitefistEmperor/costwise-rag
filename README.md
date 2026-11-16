# Costwise RAG

![tests](https://github.com/WhitefistEmperor/costwise-rag/actions/workflows/ci.yml/badge.svg)

A retrieval-augmented question-answering service that puts a **semantic cache** and a
**cost-aware model router** in front of the LLM, so cheap questions go to a small model,
repeated questions are answered from the cache, and only the hard ones pay for the large model.

The project's one claim is meant to be measured, not asserted: *lower LLM cost and p95 latency at
the same answer quality.* Every number below comes from a reproducible eval in this repo. The
offline numbers are real, and two small paid live runs (router; cache) are reported in
[Results](#results) with their limits.

## How it works

```mermaid
flowchart TD
    Q[User question] --> E[Embed query<br/>all-MiniLM-L6-v2, local]
    E --> R[Retrieve<br/>BM25 + dense, rank fusion]
    R --> C{Semantic cache}
    C -- exact match --> A[Answer]
    C -- "similarity >= accept<br/>and KeyTermGuard agrees<br/>and same top chunk" --> A
    C -- "in the band [candidate, accept)<br/>guard + strong verifier agree" --> A
    C -- miss --> RT{Router<br/>logistic regression on 7 features}
    RT -- easy --> S[Small model]
    RT -- hard --> L[Large model]
    S -- "INSUFFICIENT_CONTEXT" --> L
    S --> ST[Store in cache, log cost and latency]
    L --> ST
    ST --> A
```

Code layout follows ports and adapters: the cache, router and pipeline are plain Python in
`src/costwise/domain/` with no I/O; embedders, retrieval indexes, verifiers, the OpenRouter client
and SQLite live in `src/costwise/adapters/`; FastAPI and the dashboard in `src/costwise/app/`; the
evals in `src/costwise/eval/`. Full design notes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Design decisions

| Problem | Chosen | Alternatives considered | Why |
|---|---|---|---|
| When are two questions "the same"? | Embedding similarity, then **KeyTermGuard** (numbers must match as a set; units, plan names and contrast words must not be substituted), plus the same top retrieved chunk | Threshold only; always ask an LLM | Near-misses ("refund window for monthly" vs "annual") sit at the same similarity as true paraphrases; no single threshold separates them (see results). An LLM check on every lookup costs what the cache saves. |
| Cheap check vs accurate check | **Two thresholds**: above the accept threshold the free guard decides; in the band below it a strong verifier (small LLM YES/NO, or a cross-encoder) must also agree, at most once per lookup, its cost charged to the request | One threshold | Keeps the free path strict and pays for accuracy only where it is needed. |
| Cache freshness | TTL plus per-entry source-document hashes | Flush all on change; TTL only | Only answers built from a changed document are dropped. |
| Which model answers | Logistic regression on hand-made features (length, numbers, multi-part cues, clauses, documents in top-k, retrieval scores), plus escalation when the small model says the context is insufficient | Hand rules; embedding classifier (RouteLLM style); pure cascade (FrugalGPT style); LLM router | Works with 75 labelled rows, explainable; the cascade is the safety net. The heuristic router is kept as a baseline. |
| Retrieval | Hybrid BM25 + dense with reciprocal rank fusion | Dense only; BM25 only; + reranker | Exact tokens ("3000", "Starter") and paraphrases both matter; recall numbers below back it. |
| Storage, provider | NumPy brute-force search, SQLite, a thin `httpx` OpenRouter client | FAISS/Qdrant, Redis, LiteLLM | Zero setup at this size; each is a one-adapter swap. |

**Honesty rules:** thresholds and router weights are fitted on the **tune** split only and reported
on the **test** split. The corpus is a 12-document synthetic help-centre knowledge base
(39 chunks) with 60 questions (25 tune, 35 test), each with 2 paraphrases. Every percentage is
given with its counts.

## Results

### Offline (real, free to reproduce; from [results/offline.json](results/offline.json))

These need no LLM. They are real measurements, but on a small synthetic corpus, so they show the
method works rather than predict production numbers.

**Retrieval, test split, 105 queries, recall@4:** dense 102/105 (97.1%), BM25 98/105 (93.3%),
**hybrid 103/105 (98.1%)**.

**Cache, test split, 350-query stream with 175 repeats (5 stream seeds, repeat rate 0.5):**

| Variant (threshold from tune) | Hits | Correct | Wrong | Repeats served | Wrong-hit rate |
|---|---|---|---|---|---|
| Similarity threshold only (tune needs 0.99) | 0 | 0 | 0 | 0/175 (0%) | 0/350 (0%) |
| Guard + same evidence, accept 0.88 (the free config) | 22 | 17 | 5 | 17/175 (9.7%) | 5/350 (1.4%) |
| Upper bound: perfect strong verifier in the band [0.60, 0.88) | 139 | 134 | 5 | 134/175 (76.6%) | 5/350 (1.4%) |
| Upper bound: perfect strong verifier on every candidate | 138 | 138 | 0 | 138/175 (78.9%) | 0/350 (0%) |

The upper-bound rows assume a strong verifier that is never wrong; they bound what the paid LLM
verifier can reach. The last row was added after the 5 wrong test hits were seen and is reported
only; the tuned config does not use it.

**Router, test split, 105 query texts (35 questions x 3 phrasings; 21 texts hand-labelled hard, 84 easy):**

| Router (threshold from tune) | Accuracy | Hard sent to large | Easy kept on small | Sent to large |
|---|---|---|---|---|
| Heuristic rules, 0.4 | 93/105 (88.6%) | 14/21 | 79/84 | 19/105 |
| Learned (logistic regression), 0.5 | 97/105 (92.4%) | 18/21 | 79/84 | 23/105 |

"Hard" is a hand label, not an observed small-model failure; the live run decides whether it
matters. The learned router fits all 75 tune rows, so it over-fits a little.

**Charts** (drawn by `scripts/make_charts.py` from results/offline.json):

![Correct and wrong cache hits across accept thresholds, per verifier variant](docs/img/cache_thresholds.png)

The checks are what make low thresholds usable: at 0.60, wrong hits fall from 134 (threshold
only) to 63 with KeyTermGuard and to 40 with the guard plus the same-evidence check, while
correct hits rise from 89 to 121; at the tuned 0.88 the guard plus evidence leaves 5 wrong of 350.

![Router accuracy against share sent to the large model](docs/img/router_tradeoff.png)

### End to end (mock dry run: NOT real numbers)

[results/live_mock.json](results/live_mock.json) runs the same 140-query stream (70 repeats)
through five configurations with a **mock model that echoes the retrieved context**. Its accuracy,
cost and latency are **not real**; they only prove the harness runs end to end.

| Config | Correct | Mock cost | Routes | Cache hits |
|---|---|---|---|---|
| baseline (always large) | 124/140 | $0.3239 (mock) | large 140 | 0 |
| always small | 124/140 | $0.0162 (mock) | small 140 | 0 |
| router only | 124/140 | $0.0816 (mock) | small 112, large 28 | 0 |
| cache only | 124/140 | $0.1955 (mock) | large 84, cache 56 | 56 (2 wrong) |
| full | 124/140 | $0.0502 (mock) | small 67, large 17, cache 56 | 56 (2 wrong) |

### End to end, live (one partial run, 16 Nov 2026)

One paid run on OpenRouter (`results/live.json`): small `anthropic/claude-haiku-5.5`, large
`anthropic/claude-sonnet-5.5`, 70-query stream from the test split (35 questions, 35 repeats),
strong verifier on. It spent $0.1227 over 241 calls; then the OpenRouter account ran out of
credit (HTTP 402), so three of the five configs stopped early. Their partial counts are kept and
marked; costs are compared **per query**, because the configs answered different numbers of
queries.

| Config | Queries answered | Correct | Cost per query | vs baseline | p95 latency |
|---|---|---|---|---|---|
| baseline (always large) | 64 of 70 (stopped) | 60/64 (93.8%) | $0.00113 | n/a | 1558 ms |
| always small | 70 of 70 | 66/70 (94.3%) | $0.000062 | 5.5% of the cost | 1487 ms |
| router only | 70 of 70 | 66/70 (94.3%) | $0.00035 | 31.2% of the cost (68.8% saving) | 1485 ms |
| cache only | 23 of 70 (stopped) | 22/23 | — | too few queries to compare | 2069 ms |
| full (cache + router) | 12 of 70 (stopped) | 12/12 | — | too few queries to compare | 1150 ms |

What this run shows, and what it does not:

- **The router cut cost per query by 68.8% with no loss of accuracy** (66/70 against 60/64 for
  always-large), sending 15 of 70 queries to the large model.
- **On this data the small model alone was just as accurate** (66/70) at 5.5% of the cost. So the
  router's large-model calls bought nothing here; the hand-labelled "hard" questions were not
  hard for the small model. A harder question set is needed to show the router's value.
- **The cache configs did not finish**, so this run measures no cache saving. In the 23 queries
  it reached, the cache served 3 hits, all correct (0 wrong); the full config served 1, correct.
- One seed of 35 synthetic questions: these numbers show the method works, not production
  savings.

### Cache, live, on one model (16 Nov 2026)

Both configs on `glm-5.3-flash` (a "thinking" model, via Bharat Code, small = large), the same
70-query stream, strong verifier on (`results/live_glm.json`). One model means no router test;
this measures only the cache. Costs use an **assumed** $1 / $4 per million tokens (the real rate
is not known), so read them as relative.

| Config | Correct | Answer calls | Verifier calls | Cache hits | Assumed cost | p95 latency |
|---|---|---|---|---|---|---|
| baseline (always ask) | 66/70 (94.3%) | 70 | 0 | n/a | $0.0339 | 9.9 s |
| cache only | 65/70 (92.9%) | 46 | 26 | 24/70 (34.3%), 23 right, 1 wrong | $0.0360 | 10.2 s |

- **The cache removed 24 of 70 model answers** (34%), with 1 wrong hit.
- **It did not save money here:** the 26 verifier calls, each of which makes the thinking model
  reason before answering YES/NO, cost about 6% more than they saved. A cheap non-thinking
  verifier, or a cross-encoder, is the fix; this run shows the trade-off rather than a saving.
- Accuracy 65 vs 66 is within noise at 70 queries; one seed, one synthetic corpus.

## How to run

Requires Python 3.11+. Commands are for Windows Git Bash; on Linux/macOS use `.venv/bin/python`.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
```

The embedder (`all-MiniLM-L6-v2`, about 90 MB) is downloaded by `sentence-transformers` on first
use. Tests never load it and never touch the network.

```bash
.venv/Scripts/python -m pytest -q
```

```bash
PYTHONPATH=src .venv/Scripts/python -m costwise.eval.offline
```

```bash
PYTHONPATH=src .venv/Scripts/python -m costwise.eval.live --seeds 2
```

```bash
.venv/Scripts/python scripts/make_charts.py
```

Load test: N concurrent `/ask` requests, reporting throughput and p50/p95. In-process with the
mock model by default (no server, no download); `--url http://127.0.0.1:8000` hits a running
server. With the mock model it measures the service's own overhead, not LLM latency.

```bash
PYTHONPATH=src .venv/Scripts/python scripts/load_test.py -n 200 -c 8
```

```bash
PYTHONPATH=src .venv/Scripts/python -m uvicorn costwise.app.api:app
```

Then open http://127.0.0.1:8000/ for the dashboard, or `POST /ask` with `{"query": "..."}`.
`GET /stats`, `/recent` and `/cache` return JSON; `GET /metrics` serves Prometheus text format
(request counts by route, LLM and baseline cost, escalations, latency p50/p95, cache hits and
rejections). With the mock model, `costwise_info{mock="true"}` marks the cost metrics as not real. The API uses the mock model unless
`COSTWISE_LLM=openrouter` and `OPENROUTER_API_KEY` are set in the environment.

### Docker

```bash
docker compose up --build
```

The API then serves on http://localhost:8000/ with the **mock model**; the SQLite cache and
request log live in a named volume. The default embedder (MiniLM, about 90 MB) is downloaded once
on first start into a second volume; `COSTWISE_EMBED_MODEL=hash docker compose up` runs with no
download at all (weaker retrieval, demo only). No key is in the image or the compose file: for the
paid model, set `COSTWISE_LLM=openrouter` and `OPENROUTER_API_KEY` in your shell before `up`, and
compose passes them in at runtime.

The paid live eval (`--live`) spends OpenRouter credit and stops at a hard cap; see
[PROGRESS.md](PROGRESS.md) for the exact command.

## Limits

- Small synthetic corpus: the method is shown, production numbers are not.
- KeyTermGuard misses near-misses that differ only in ordinary words; above the accept threshold
  no strong verifier is asked.
- Single process: the SQLite cache is not shared across instances.
- Answers are graded by required key facts, not by a human or an LLM judge.
