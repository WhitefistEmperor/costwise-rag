# Progress

Updated at every phase boundary. Plan: [docs/PLAN.md](docs/PLAN.md). Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Working rule from Sahil (15 Nov 2026): commit each step as its own commit.** One commit per
phase or sub-step, not one big commit, and push each one: `git push origin main`.
Remote: private GitHub repo `costwise-rag` (origin). Never force-push.

## Done
- **Phase 0:** design doc, plan, 12-document synthetic corpus (39 chunks), 60 questions
  (46 easy, 14 hard; 25 tune, 35 test; 2 paraphrases each) built by `scripts/make_questions.py`.
- **Phase 1:** domain models and ports, chunking, embedders, dense + BM25 + hybrid retrieval,
  retrieval eval. Test split, 105 queries, recall@4: dense 102, BM25 98, hybrid 103.
- **Phase 2:** `SemanticCache` with two thresholds (exact tier, semantic tier above the accept
  threshold with `KeyTermGuard`, a band down to the candidate threshold where a strong verifier
  must also agree, at most one strong call per lookup, its cost charged to the request), TTL,
  source-hash invalidation, same-evidence check. Verifiers: `KeyTermGuard` (numbers must match
  as a set; substitution rule for entity terms), `LLMVerifier` (small model, YES/NO, built on
  the `LLMClient` port), `CrossEncoderVerifier` (optional, not downloaded). SQLite store.
  Offline cache eval with tune-only threshold rules. 38 tests pass.

  Offline cache numbers (test split, 5 stream seeds, 350 queries, 175 repeats; thresholds from
  tune; synthetic corpus, so not production numbers):
  - threshold only: tune needs 0.99, so 0 hits.
  - guard (+ same evidence), accept 0.88: 22 hits, 17 correct, 5 wrong; repeat recall 17/175
    (9.7%); wrong-hit rate 5/350 (1.4%). This is the free configuration.
  - UPPER BOUND with a perfect strong verifier in the band [0.60, 0.88): 139 hits, 134 correct,
    5 wrong, 145 verifier calls; repeat recall 134/175 (76.6%). The 5 wrong hits are above the
    accept threshold, where the strong verifier is not asked.
  - UPPER BOUND, strong verifier checks every candidate (accept 0.99, candidate 0.60): 138 hits,
    0 wrong, 164 calls; repeat recall 138/175 (78.9%). This variant was added after the 5 test
    wrong hits were seen; it is reported only and `data/tuned.json` does not use it.
  These bounds are what a real strong verifier could at best reach; real numbers need the paid
  live run.

- **Phase 3:** `domain/routing.py`: 7 features (length, numbers, multi-part cues, clauses,
  distinct docs in top-k, top dense score, score gap), `HeuristicRouter` (rule weights),
  `LearnedRouter` (logistic regression, class-balanced, weights stored as plain numbers in
  `data/tuned.json`), `decide(score, threshold)`. Router eval in `eval/offline.py`; thresholds
  picked on tune by balanced accuracy. 43 tests pass. `scripts/show_router_eval.py` prints it.

  Router numbers (test split, 105 texts: 21 hand-labelled hard, 84 easy; fitted on tune):
  - heuristic, threshold 0.4: accuracy 93/105 (88.6%); 14/21 hard sent to large; 79/84 easy
    kept small; 19/105 sent to large.
  - learned, threshold 0.5: accuracy 97/105 (92.4%); 18/21 hard sent to large; 79/84 easy kept
    small; 23/105 sent to large. (Tune: 75/75, so it is over-fitting a little on 75 rows.)
  - "hard" is a hand label, not an observed small-model failure; the live run decides whether
    the label matters.
  - Disclosure: the first draft of the cue-word list was written after the hard questions of
    both splits had been printed. The list was then filtered mechanically to cues that occur in
    at least one tune text (`scripts/filter_router_cues.py`; 14 cues dropped) and the eval was
    rerun; the numbers above are from that rerun. The learned router's numbers did not change.
    Later, cue matching was switched to whole words ("if a" no longer matches "if any"); under
    that rule "which plan" occurs in no tune text and was dropped. Rerun: the test numbers above
    did not change (heuristic tune accuracy 68/75 -> 70/75). Each router now uses its own
    tune-chosen threshold (`heuristic_route_threshold` in data/tuned.json).

- **Phase 4:** `domain/pricing.py` (price table, cost maths), `domain/pipeline.py` (embed,
  retrieve, cache lookup with the top chunk as evidence key, route, generate, escalate on
  INSUFFICIENT_CONTEXT, store, log; strong-verifier cost charged to the request), `adapters/llm.py`
  (`MockLLM` default, `OpenRouterLLM` with real per-call cost from `usage.cost`, built only when
  `COSTWISE_LLM=openrouter` and the key is in `OPENROUTER_API_KEY`), `SQLiteRequestLog` with
  stats (cost, always-large baseline, saving, p50/p95, route split, escalations),
  `app/container.py`, `app/api.py` (POST /ask, GET /stats, /recent, /cache, / dashboard),
  `app/dashboard.html` (marks mock runs as not real). 58 tests pass, none touch the network.
  `scripts/smoke_api.py` runs /ask end to end with the real embedder and the mock model.

- **Phase 5:** `eval/live.py`: five configs (baseline always-large, always-small, router only,
  cache only, full) over one long stream (test split, several seeds back to back), `BudgetedLLM`
  checks a hard `--max-cost` cap BEFORE every call and stops the run with the partial results
  marked. Mock dry run is the default; the paid run needs `--live` and `OPENROUTER_API_KEY`.
  Every percentage is written next to its counts. `--strong-verifier llm` opens the cache band.
  63 tests pass, none touch the network.

  Mock dry run (`results/live_mock.json`, 140 queries, 70 repeats).
  THESE NUMBERS ARE NOT REAL: the mock model echoes the retrieved context, so accuracy, cost and
  latency only prove the harness runs end to end.

  | config | correct | mock cost | routes | cache hits | escalations |
  |---|---|---|---|---|---|
  | baseline_always_large | 124/140 | $0.3239 | {'large': 140} | 0 (0 wrong) | 0 |
  | always_small | 124/140 | $0.0162 | {'small': 140} | 0 (0 wrong) | 0 |
  | router_only | 124/140 | $0.0816 | {'small': 112, 'large': 28} | 0 (0 wrong) | 0 |
  | cache_only | 124/140 | $0.1955 | {'large': 84, 'cache': 56} | 56 (2 wrong) | 0 |
  | full | 124/140 | $0.0502 | {'small': 67, 'large': 17, 'cache': 56} | 56 (2 wrong) | 0 |

- **Review fixes (15 Nov 2026),** one commit each, every one with a regression test. 153 tests
  pass, none touch the network.
  1. Live run: `OpenRouterLLM` retries 429/5xx/timeouts (2 retries, backoff) and every retry
     passes the budget check; other errors and a 200 without `choices` raise `ProviderError`.
     A provider error stops only the current config (marked `error`, partial results kept) and
     the results file is always written, as strict JSON (`max_cost_usd` is null without a cap).
     Paid runs refuse a model with no price unless `--price MODEL=IN,OUT` is given. Spend is
     kept per config (`llm_spent_usd`, sums to `total_spent_usd`). `--live` builds its client
     with the app's `build_llm`, the only place that reads `OPENROUTER_API_KEY`.
  2. Cache: `lookup` returns a `LookupResult` (hit, verifier cost, calls); strong verifiers
     return a `Verdict` with the cost of that call; the lock is not held during the strong
     verifier call; stale candidates no longer use up `max_candidates`.
  3. `KeyTermGuard`: numbers glued to letters ("800GB", "6h", "v2") are extracted and compared
     exactly ("3,000" == "3000", "1.30" == "1.3"); entity extraction treats the first word of
     each line or bullet as sentence-initial (chunk text keeps line breaks). Cache eval rerun:
     thresholds and all cache numbers unchanged.
  4. Hybrid fusion gives no BM25 rank credit to chunks with BM25 score 0; `load_corpus` moved
     to `adapters/corpus.py` and a test keeps I/O out of `domain/`. Recall@4 unchanged (dense
     102/105, BM25 98/105, hybrid 103/105); mock costs moved slightly (table above is current).
  5. Router: each router uses its own tune-chosen threshold; cues match whole words (see the
     Phase 3 note).
  6. Dashboard: the mock badge respects `hidden`; every value put into `innerHTML` is escaped.
     Percentiles use nearest rank (`domain/stats.py`) in the request log and the live eval.
     A miss lists each source document once, like a hit.
  7. Stronger tests (exact escalation cost, verifier cost inside the config cost, same-evidence
     check, `evidence_key` SQLite round trip, every script and source file compiles); fixed the
     broken f-string in `scripts/show_cache_eval.py`; removed the stale auto-save snapshot
     `PROGRESS-2026-11-15.md` (its open issues are folded in below).

**16 Nov run (backlog A to H done):** the four open defects fixed with regression tests;
README, charts, CI (green on GitHub Actions), `/metrics`, Docker (image built and smoke-tested
with the hash embedder), load test; self-review of the diff since 5d8ba30 found only doc drift
(fixed). 182 tests pass, none touch the network.

**Phases 0 to 5 are done. What remains (Phase 6) spends OpenRouter credit and needs Sahil's
go-ahead.** Command for the paid run (reads the key from the environment, never prints it):
`PYTHONPATH=src .venv/Scripts/python -m costwise.eval.live --live --max-cost 0.50 --seeds 1 --strong-verifier llm`
`--strong-verifier llm` opens the cache band [0.60, 0.88) with the small-model YES/NO check;
without it the run measures the guard-only cache. Estimate for one seed (70 queries x 5 configs,
short answers, plus roughly one extra small-model call per two queries in the two cache configs):
about $0.25 to $0.35; two seeds may touch the $0.50 cap, in which case the run stops and says so.

## Next steps (in order)
1. Phase 6 (needs go-ahead): paid live run, then fill the live table in README.md and the resume bullets.
2. Later (optional): Quora Question Pairs cache benchmark.

## Independent check of the fixes (15 Nov, at 1538b07)
All 12 reviewed defects are fixed; 153 tests pass with no network access; every number above
reproduces from a fresh run. All four were fixed on 16 Nov, each with a regression test:
- FIXED (16 Nov): `--max-cost` must be a finite number >= 0; `inf`, `nan` and negatives are
  refused by the parser (exit 2, nothing run) and by `BudgetedLLM` itself.
- FIXED (16 Nov): Ctrl+C during the live run keeps the partial counts of the current config,
  writes the results file (marked `interrupted`) and then exits with code 130.
- FIXED (16 Nov): when the strong verifier raises (budget cap, provider error),
  `SemanticCache.lookup` still records the checks it made and removes stale entries, then
  re-raises; the failed call is not counted.
- FIXED (16 Nov): `KeyTermGuard` reads the unit written right after a number ("800GB",
  "800 GB", "90-day", "6 hours") as a key term under the substitution rule, so "800GB" vs
  "800MB" conflict and "800GB" vs "800 GB" do not. Offline eval rerun: data/tuned.json and
  results/offline.json unchanged; mock live counts unchanged.

- **README.md (16 Nov):** portfolio README with a Mermaid diagram, design decisions and
  alternatives, offline numbers from results/offline.json, mock numbers labelled not real, and
  a placeholder table for the Phase 6 live numbers.
- **Charts (16 Nov):** `scripts/make_charts.py` draws docs/img/cache_thresholds.png (correct and
  wrong hits per accept threshold and verifier variant) and docs/img/router_tradeoff.png
  (accuracy vs share sent to large) from results/offline.json; linked from the README.
- **CI (16 Nov):** `.github/workflows/ci.yml` runs the suite on every push and pull request
  (Python 3.11 and 3.12, CPU-only torch, `HF_HUB_OFFLINE=1`, mock model; no model download, no
  network in tests, no secret).
- **/metrics (16 Nov):** `app/metrics.py` writes Prometheus text format 0.0.4 by hand (no new
  dependency): requests by route, LLM and always-large cost, escalations, latency p50/p95 summary,
  cache entries, lookups, hits by tier, rejections by reason, strong-verifier calls and cost.
  `costwise_info{mock="true"}` marks mock cost as not real. Tests parse every exposition line.
- **Load test (16 Nov):** `scripts/load_test.py` sends N /ask requests with C in flight, in-process
  (mock LLM, hash embedder, no download) or against `--url`, and reports throughput and p50/p95.
  With the mock model it measures service overhead only, and its report says so.
- **Docker (16 Nov):** `Dockerfile` (python:3.12-slim, CPU-only torch, non-root user, mock LLM
  by default, health check on /stats), `docker-compose.yml` (named volumes for state and the
  embedder cache; key only from the runtime environment), `.dockerignore` (keeps .env, .venv,
  .git and .state out of the build context).

## Resume bullets (DRAFT, 16 Nov)
Offline numbers are real (results/offline.json, synthetic 12-document corpus, test split).
Anything in [brackets] needs the paid Phase 6 run and must not be used until it exists.

- Built a RAG service (FastAPI, hybrid BM25 + dense retrieval, SQLite) with a semantic cache and
  a cost-aware small/large model router, cutting LLM cost by [X% vs always-large] and p95
  latency by [Y ms] at [same / within Z points of] answer accuracy on a [N]-query live A/B.
- Designed a two-threshold semantic cache with a rule-based key-term verifier (numbers, units,
  plan names) that cut wrong cache hits from 134 to 40 of 350 queries at similarity 0.60, and
  served 17/175 repeats with 5/350 wrong hits (1.4%) at the tune-chosen threshold; with a
  perfect strong verifier in the band, up to 134/175 repeats (offline upper bound).
- Trained a logistic-regression router on 7 query and retrieval features from 75 labelled
  rows: 92.4% routing accuracy (97/105) on held-out questions, sending 18 of 21 hard questions
  to the large model while sending only 23/105 there in total.
- Hybrid retrieval with reciprocal rank fusion: recall@4 of 98.1% (103/105) vs 97.1% dense and
  93.3% BM25 on held-out questions.
- Engineered for honest evaluation: tune/test split for every threshold and weight, hard
  budget cap checked before every paid call, mock runs labelled not real, Prometheus /metrics,
  Docker, and CI running 182 offline tests on every push.

## Open issues
- Paid live eval not run; no real cost, latency or answer-quality numbers exist yet. The mock
  numbers in results/live_mock.json are not real.
- Cross-encoder and stronger embedding models need a download; not done without a yes.
- `CrossEncoderVerifier` is only tested with an injected scorer, never with the real model;
  `LLMVerifier` and `OpenRouterLLM` only with fake clients and `httpx.MockTransport`.
- The learned router fits the 75 tune rows perfectly (97/105 on test), so it is a little over-fitted.
- `KeyTermGuard` entity terms include UI labels and languages that the corpus really writes with
  a capital mid-sentence (Settings, Developers, Billing, Export, Request, Restore, Center,
  Drive, Type, English). They are not bullet artefacts (the corpus has no bullets), and no
  stoplist was added; a substitution between two of them is treated as a different question.
- Numbers glued to letters now count, so "2FA" carries the number 2; a paraphrase that drops
  "2FA" no longer passes the guard. Thresholds tuned on tune did not change.
- The repo lives in a folder the app made for this session; it is deleted if the session is
  deleted. The GitHub remote has every commit; move the folder before relying on it.

## How to run
- Tests: `.venv/Scripts/python -m pytest -q`
- Offline eval: `PYTHONPATH=src .venv/Scripts/python -m costwise.eval.offline`
- Cache view: `PYTHONPATH=src .venv/Scripts/python scripts/show_cache_eval.py`
- Router view: `.venv/Scripts/python scripts/show_router_eval.py`
- API (mock model): `PYTHONPATH=src .venv/Scripts/python -m uvicorn costwise.app.api:app --reload`, then open http://127.0.0.1:8000/
- API smoke: `PYTHONPATH=src .venv/Scripts/python scripts/smoke_api.py`
- Live eval dry run (mock, free): `PYTHONPATH=src .venv/Scripts/python -m costwise.eval.live --seeds 2`
- Charts: `.venv/Scripts/python scripts/make_charts.py` (writes docs/img/*.png)
- Live eval paid run: add `--live --max-cost 0.50 --strong-verifier llm` (needs OPENROUTER_API_KEY;
  do not run without a yes). A model missing from `domain/pricing.py` needs
  `--price MODEL=IN,OUT` (USD per million tokens) or the run refuses to start.
- Router cue check: `PYTHONPATH=src .venv/Scripts/python scripts/filter_router_cues.py`
- Tune-split cache diagnosis: `PYTHONPATH=src .venv/Scripts/python scripts/diagnose_cache.py 0.70`
- Load test (mock, in-process): `PYTHONPATH=src .venv/Scripts/python scripts/load_test.py -n 200 -c 8`
- Docker (mock model): `docker compose up --build`, then http://localhost:8000/
- Never paste Python with quotes into a bash heredoc here; write a script file and run it.
