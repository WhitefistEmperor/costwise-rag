# Build plan

Each phase ends with passing tests and an updated `PROGRESS.md`, so work can stop and resume at
any phase boundary. Design reasons are in [ARCHITECTURE.md](ARCHITECTURE.md).

| Phase | What gets built | Done when |
|---|---|---|
| 0 | Design doc, repo skeleton, sample corpus, question set | Docs written; corpus and questions in `data/` |
| 1 | Domain models and ports, chunking, embedders, dense + BM25 + hybrid retrieval, retrieval eval | Tests pass; recall@k reported for dense, BM25, hybrid |
| 2 | Semantic cache: exact tier, semantic tier, verifier, TTL, source invalidation; cache eval | Tests pass; hit rate and wrong-hit rate reported across thresholds, with and without verifier |
| 3 | Router: features, heuristic and learned routers, escalation; router eval | Tests pass; router accuracy reported on the test split |
| 4 | Pipeline wiring, pricing, SQLite request log, FastAPI, dashboard | Tests pass; `/ask` works end to end with the mock model; dashboard shows stats |
| 5 | Live eval harness with a hard budget cap, dry-run against the mock model | Dry run passes; the paid run is one command and waits for a go-ahead |
| 6 | Paid live run, README with real numbers, resume bullets | Results table filled from a real run |
| Later | Prometheus `/metrics` endpoint (done), Dockerfile and compose (done), Quora Question Pairs cache benchmark | Optional |
