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
    0 wrong, 164 calls; repeat recall 138/175 (78.9%).
  These bounds are what a real strong verifier could at best reach; real numbers need the paid
  live run.

## Next steps (in order)
1. Phase 3: router (`domain/routing.py`: features, HeuristicRouter, LearnedRouter), router eval
   on the test split (fit on tune), tests. Commit "Phase 3: cost-aware router".
2. Phase 4: `domain/pricing.py`, `domain/pipeline.py`, `adapters/llm.py` (OpenRouterLLM,
   MockLLM), request log in `adapters/storage.py`, `app/container.py`, `app/api.py`,
   `app/dashboard.html`, tests. Pipeline order: embed, retrieve, cache lookup (top chunk id as
   evidence key; add `cache.last_lookup_cost_usd` to the request cost), route, generate, store,
   log. Mock model is the default; the paid client needs an explicit setting and the key from
   the environment only. Commit "Phase 4: pipeline, API and dashboard".
3. Phase 5: `eval/live.py` with five configs (always-large baseline, always-small, router only,
   cache only, full), a hard `--max-cost` cap checked before every call, a `--mock` dry run
   whose output file is marked as mock. Longer stream; counts next to every percentage.
   Commit "Phase 5: live eval harness".
4. Phase 6 needs Sahil's go-ahead (spends OpenRouter credit, estimated under $0.50).

## Open issues
- Paid live eval not run; no cost, latency or answer-quality numbers exist yet.
- Cross-encoder and stronger embedding models need a download; not done without a yes.
- `CrossEncoderVerifier` is only tested with an injected scorer, never with the real model.
- The repo lives in a folder the app made for this session; it is deleted if the session is
  deleted. Move it to a permanent folder before relying on it.

## How to run
- Tests: `.venv/Scripts/python -m pytest -q`
- Offline eval: `PYTHONPATH=src .venv/Scripts/python -m costwise.eval.offline`
- Cache view: `PYTHONPATH=src .venv/Scripts/python scripts/show_cache_eval.py`
- Tune-split cache diagnosis: `PYTHONPATH=src .venv/Scripts/python scripts/diagnose_cache.py 0.70`
- Never paste Python with quotes into a bash heredoc here; write a script file and run it.
