# Progress

Updated at every phase boundary. Plan: [docs/PLAN.md](docs/PLAN.md). Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Working rule from Sahil (15 Nov 2026): commit each step as its own commit.** One commit per
phase or sub-step, not one big commit. No remote exists yet, so nothing is pushed.

## Done
- **Phase 0:** design doc, plan, 12-document synthetic corpus (39 chunks), 60 questions
  (46 easy, 14 hard; 25 tune, 35 test; 2 paraphrases each) built by `scripts/make_questions.py`.
- **Phase 1:** domain models and ports, chunking, embedders, dense + BM25 + hybrid retrieval,
  retrieval eval. Test split, 105 queries, recall@4: dense 102, BM25 98, hybrid 103.
- **Phase 2 (code done, design not finished):** `SemanticCache` (exact tier, semantic tier,
  verifier, TTL, source-hash invalidation, same-evidence check), `KeyTermGuard`,
  `CrossEncoderVerifier` (optional, not downloaded), SQLite cache store, cache eval.
  32 tests pass.

## In progress: Phase 2 finding and what to do about it
The cache eval showed that embedding similarity (MiniLM) cannot separate paraphrases from
near-miss questions on this data. Numbers, 5 stream seeds, 350 test queries, 175 repeats:
- threshold only: no threshold below 0.99 is free of wrong pairs on tune, so 0 hits.
- key-term guard, threshold 0.88 from tune: 22 hits, 5 wrong, repeat recall 9.7%.
- adding the same-evidence check changed almost nothing.
- Same-question pairs on tune sit mostly at similarity 0.6 to 0.9 (only 6 of 75 above 0.9).

Diagnosis on the tune split (`scripts/diagnose_cache.py`) found the remaining wrong accepts:
1. A question with extra numbers ("Team with 12 seats and 1.3 TB") matched the plain one
   ("Team price"). Fix decided: numbers must match exactly as a set; keep the substitution rule
   for entity terms only.
2. True semantic near-misses ("how long can a deleted workspace be restored" vs "how long until
   it is purged", similarity 0.86). No free rule separates these.

## Next steps (in order)
1. In `adapters/verifiers.py` `KeyTermGuard.same_question`: require equal number sets; update
   the docstring rule and `tests/test_cache.py`. Re-run `scripts/diagnose_cache.py` (tune only).
2. Make the cache two-threshold: similarity >= `accept_threshold` plus guard accepts directly;
   between `candidate_threshold` and `accept_threshold` a **strong verifier** must also agree.
   Strong verifier is a `Verifier`: add `LLMVerifier` (asks the small model "same question?",
   costs about 0.5% of a large call) next to the existing `CrossEncoderVerifier`.
3. In the offline eval, report the free configuration (guard only) and the upper bound with a
   perfect strong verifier (share of repeats that have a correct candidate above the low
   threshold). The real strong-verifier numbers come from the paid live run.
4. Update `docs/ARCHITECTURE.md` section 3.1 with this finding and the two-threshold design.
   Run `python -m costwise.eval.offline` to write `results/offline.json` and `data/tuned.json`.
   Commit: "Phase 2: two-threshold cache with strong verifier".
5. Phase 3: router (`domain/routing.py`: features, HeuristicRouter, LearnedRouter), router eval
   on the test split, tests. Commit.
6. Phase 4: `domain/pricing.py`, `domain/pipeline.py`, `adapters/llm.py` (OpenRouterLLM,
   MockLLM), request log in `adapters/storage.py`, `app/container.py`, `app/api.py`,
   `app/dashboard.html`, tests. Pipeline order: embed, retrieve, cache lookup (needs the top
   chunk id as evidence key), route, generate, store, log. Commit.
7. Phase 5: `eval/live.py` with five configs (always-large baseline, always-small, router only,
   cache only, full), a hard `--max-cost` cap, a `--mock` dry run. Lengthen the stream and
   report counts with every percentage. Commit.
8. Phase 6 needs Sahil's go-ahead (spends OpenRouter credit, estimated under $0.50).

## Open issues
- Paid live eval not run; no cost, latency or answer-quality numbers exist yet.
- Cross-encoder and stronger embedding models need a download; not done without a yes.
- `CrossEncoderVerifier` is only tested with an injected scorer, never with the real model.
- The repo lives in a folder the app made for this session; it is deleted if the session is
  deleted. Move it to a permanent folder before relying on it.
- No git remote; commits are local only.

## How to run
- Tests: `.venv/Scripts/python -m pytest -q`
- Offline eval: `PYTHONPATH=src .venv/Scripts/python -m costwise.eval.offline`
- Cache view: `PYTHONPATH=src .venv/Scripts/python scripts/show_cache_eval.py`
- Never paste Python with quotes into a bash heredoc here; write a script file and run it.
