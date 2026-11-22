"""Retrieval on public BEIR benchmarks (SciFact, NFCorpus): the same retrievers the service uses,
scored with nDCG@10, recall@100 and MRR@10, with bootstrap confidence intervals.

Run with `python -m costwise.eval.beir`. Downloads the datasets from the Hugging Face Hub at a
pinned revision into the HF cache (never into the repo) and, for reranking, the
cross-encoder/ms-marco-MiniLM-L-6-v2 model. No LLM calls.

Honesty rules: the fusion depth and the rerank depth are chosen on a sample of the TRAIN qrels;
the TEST qrels are only scored. Every method is reported with its 95% CI over test queries, and
hybrid and rerank are compared to dense per query with a paired bootstrap.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ..adapters.embedders import SentenceTransformerEmbedder
from ..adapters.retrieval import BM25Index, DenseIndex, HybridRetriever
from ..domain.models import Chunk
from ..settings import ROOT, Settings
from .ir_metrics import bootstrap_ci, mean, mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k

# Pinned dataset revisions (Hugging Face commit shas), so the numbers stay reproducible.
DATASETS = {
    "scifact": {"repo": "BeIR/scifact", "rev": "b3b5335604bf5ee3c4447671af975ea25143d4f5",
                "qrels": "BeIR/scifact-qrels", "qrels_rev": "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029",
                "tune_split": "train"},
    "nfcorpus": {"repo": "BeIR/nfcorpus", "rev": "b5026a0e96e8a7ac4f95f482a596389289d46269",
                 "qrels": "BeIR/nfcorpus-qrels", "qrels_rev": "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65",
                 "tune_split": "dev"},
}
# Published reference points (nDCG@10 on test), used only as a sanity check of this harness.
PUBLISHED = {
    "scifact": {"bm25_elasticsearch_beir_paper": 0.665, "all-MiniLM-L6-v2_mteb": 0.645},
    "nfcorpus": {"bm25_elasticsearch_beir_paper": 0.325, "all-MiniLM-L6-v2_mteb": 0.316},
}
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Fusion grid (rrf_k, depth). depth >= 100 so recall@100 is always over a full top 100.
FUSION_GRID = [(k, d) for k in (10, 30, 60, 100) for d in (100, 300, 1000)]
# Capped at 50: the reranker runs at about 24 query-abstract pairs per second on a laptop CPU.
RERANK_DEPTHS = [10, 20, 50]
TUNE_SAMPLE = 200
RERANK_TUNE_SAMPLE = 100


def load_beir(name: str) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, dict[str, int]]]]:
    """(corpus id -> text, query id -> text, split -> qrels). Reads parquet and TSV only."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    spec = DATASETS[name]
    get = lambda repo, rev, f: hf_hub_download(repo, f, repo_type="dataset", revision=rev)
    corpus_df = pd.read_parquet(get(spec["repo"], spec["rev"], "corpus/corpus-00000-of-00001.parquet"))
    queries_df = pd.read_parquet(get(spec["repo"], spec["rev"], "queries/queries-00000-of-00001.parquet"))
    corpus = {str(i): (f"{t}. {x}" if t else str(x))
              for i, t, x in zip(corpus_df["_id"], corpus_df["title"], corpus_df["text"])}
    queries = {str(i): str(x) for i, x in zip(queries_df["_id"], queries_df["text"])}
    qrels: dict[str, dict[str, dict[str, int]]] = {}
    for split in ("test", spec["tune_split"]):
        rows: dict[str, dict[str, int]] = {}
        with open(get(spec["qrels"], spec["qrels_rev"], f"{split}.tsv"), encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                rows.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
        qrels[split] = {q: r for q, r in rows.items() if q in queries and any(v > 0 for v in r.values())}
    return corpus, queries, qrels


def cached_doc_vectors(name: str, embedder: SentenceTransformerEmbedder, texts: Sequence[str], cache_dir: Path) -> np.ndarray:
    """Document embeddings are the slow part on a CPU; keep them in .state (gitignored)."""
    path = cache_dir / f"{name}-{embedder.name.replace('/', '_')}-{DATASETS[name]['rev'][:8]}-{len(texts)}.npy"
    if path.exists():
        return np.load(path)
    vecs = embedder.encode(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs


def score_run(run: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict:
    qids = sorted(qrels)
    per = {"ndcg@10": [ndcg_at_k(run[q], qrels[q], 10) for q in qids],
           "recall@100": [recall_at_k(run[q], qrels[q], 100) for q in qids],
           "mrr@10": [mrr_at_k(run[q], qrels[q], 10) for q in qids]}
    out = {"queries": len(qids)}
    for metric, xs in per.items():
        lo, hi = bootstrap_ci(xs)
        out[metric] = round(mean(xs), 4)
        out[f"{metric}_ci95"] = [round(lo, 4), round(hi, 4)]
    out["_per_query_ndcg"] = per["ndcg@10"]
    return out


def latency(times_ms: list[float]) -> dict:
    t = sorted(times_ms)
    return {"ms_p50": round(t[len(t) // 2], 2), "ms_p95": round(t[min(len(t) - 1, int(0.95 * len(t)))], 2)}


class Bench:
    """Indexes one corpus once and runs each retriever over any query set."""

    def __init__(self, name: str, settings: Settings, reranker: str | None = RERANKER):
        self.name = name
        self.corpus, self.queries, self.qrels = load_beir(name)
        ids = list(self.corpus)
        self.chunks = [Chunk(i, i, "", self.corpus[i]) for i in ids]
        self.embedder = SentenceTransformerEmbedder(settings.embed_model)
        t = time.perf_counter()
        vecs = cached_doc_vectors(name, self.embedder, [c.text for c in self.chunks], settings.state_dir / "bench")
        self.embed_s = round(time.perf_counter() - t, 1)
        self.dense = DenseIndex(self.chunks, vecs)
        t = time.perf_counter()
        self.bm25 = BM25Index(self.chunks, self.dense)
        self.bm25_build_s = round(time.perf_counter() - t, 1)
        self._qvecs: dict[str, np.ndarray] = {}
        self._reranker_name = reranker
        self._reranker = None

    def qvec(self, qid: str) -> np.ndarray:
        if qid not in self._qvecs:
            missing = [q for q in self.queries if q not in self._qvecs]
            for q, v in zip(missing, self.embedder.encode([self.queries[q] for q in missing])):
                self._qvecs[q] = v
        return self._qvecs[qid]

    def run(self, retriever, qids: Sequence[str], k: int = 100) -> tuple[dict[str, list[str]], list[float]]:
        out, times = {}, []
        for q in qids:
            v = self.qvec(q)                                   # embedding time excluded: same for all
            t = time.perf_counter()
            hits = retriever.search(self.queries[q], v, k)
            times.append((time.perf_counter() - t) * 1000)
            out[q] = [h.chunk.doc for h in hits]
        return out, times

    def rerank(self, first: dict[str, list[str]], depth: int) -> tuple[dict[str, list[str]], list[float]]:
        """Cross-encoder rescoring of the first `depth` candidates; the rest keep their order.
        Pair scores are memoised, so trying several depths scores each pair once. The latency
        reported is per query for the pairs scored in this call (0 when all were memoised)."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self._reranker_name)
            self._ce: dict[tuple[str, str], float] = {}
        out, times = {}, []
        for n, (q, docs) in enumerate(first.items()):
            head = docs[:depth]
            todo = [d for d in head if (q, d) not in self._ce]
            t = time.perf_counter()
            if todo:
                s = self._reranker.predict([(self.queries[q], self.corpus[d]) for d in todo],
                                           batch_size=64, show_progress_bar=False)
                self._ce.update({(q, d): float(v) for d, v in zip(todo, np.asarray(s).reshape(-1))})
                times.append((time.perf_counter() - t) * 1000)
            scores = [self._ce[(q, d)] for d in head]
            order = np.argsort(-np.asarray(scores), kind="stable")
            out[q] = [head[i] for i in order] + docs[depth:]
            if n % 50 == 0:
                print(f"  rerank depth {depth}: {n}/{len(first)} queries", flush=True)
        return out, times


def tune_sample(qrels: dict, n: int = TUNE_SAMPLE, seed: int = 0) -> list[str]:
    qids = sorted(qrels)
    random.Random(seed).shuffle(qids)
    return sorted(qids[:n])


def evaluate(name: str, settings: Settings, rerank: bool = True) -> dict:
    print(f"{name}: loading and indexing", flush=True)
    b = Bench(name, settings)
    print(f"{name}: indexed {len(b.corpus)} docs (embedding {b.embed_s}s)", flush=True)
    tune_split = DATASETS[name]["tune_split"]
    tune_q = tune_sample(b.qrels[tune_split])
    test_qrels = b.qrels["test"]
    test_q = sorted(test_qrels)
    ndcg_on = lambda run, qs, qrels: mean([ndcg_at_k(run[q], qrels[q], 10) for q in qs])

    # Fusion depth: chosen on the tune sample.
    fusion_scores = {kd: ndcg_on(b.run(HybridRetriever(b.dense, b.bm25, rrf_k=kd[0], depth=kd[1]), tune_q)[0],
                                 tune_q, b.qrels[tune_split]) for kd in FUSION_GRID}
    best_k, best_depth = max(FUSION_GRID, key=lambda kd: (round(fusion_scores[kd], 4), kd[0] == 60, -kd[1]))
    hybrid = HybridRetriever(b.dense, b.bm25, rrf_k=best_k, depth=best_depth)

    out: dict = {"dataset": name, "revision": DATASETS[name]["rev"], "documents": len(b.corpus),
                 "test_queries": len(test_q), "tune_split": tune_split, "tune_queries": len(tune_q),
                 "embedder": settings.embed_model, "doc_embedding_s": b.embed_s, "bm25_index_s": b.bm25_build_s,
                 "published_ndcg@10": PUBLISHED[name],
                 "fusion_tune_ndcg@10": {f"rrf_k={k},depth={d}": round(v, 4) for (k, d), v in fusion_scores.items()},
                 "fusion_from_tune": {"rrf_k": best_k, "depth": best_depth}, "methods": {}}
    runs = {}
    for method, retriever in (("bm25", b.bm25), ("dense", b.dense), ("hybrid_rrf", hybrid)):
        run, times = b.run(retriever, test_q)
        runs[method] = run
        out["methods"][method] = {**score_run(run, test_qrels), "search_latency": latency(times)}

    if rerank:
        rr_q = tune_q[:RERANK_TUNE_SAMPLE]
        tune_first = b.run(hybrid, rr_q)[0]
        rr_scores = {d: ndcg_on(b.rerank(tune_first, d)[0], rr_q, b.qrels[tune_split])
                     for d in sorted(RERANK_DEPTHS, reverse=True)}   # deepest first: the rest is memoised
        best_rr = max(RERANK_DEPTHS, key=lambda d: (round(rr_scores[d], 4), -d))
        out["rerank_tune_queries"] = len(rr_q)
        out["rerank_depth_tune_ndcg@10"] = {str(d): round(s, 4) for d, s in rr_scores.items()}
        out["rerank_depth_from_tune"] = best_rr
        out["reranker"] = RERANKER
        run, times = b.rerank(runs["hybrid_rrf"], best_rr)
        runs["hybrid_rrf+rerank"] = run
        out["methods"]["hybrid_rrf+rerank"] = {**score_run(run, test_qrels), "rerank_latency": latency(times),
                                               "rerank_depth": best_rr}

    base = out["methods"]["dense"]["_per_query_ndcg"]
    out["paired_vs_dense_ndcg@10"] = {
        m: paired_bootstrap(out["methods"][m]["_per_query_ndcg"], base) for m in out["methods"] if m != "dense"}
    out["paired_rerank_vs_hybrid_ndcg@10"] = (
        paired_bootstrap(out["methods"]["hybrid_rrf+rerank"]["_per_query_ndcg"],
                         out["methods"]["hybrid_rrf"]["_per_query_ndcg"]) if rerank else None)
    for m in out["methods"].values():
        m.pop("_per_query_ndcg")
    return out


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args(argv)
    settings = Settings()
    result = {"note": "Public BEIR benchmarks, test qrels; depths chosen on train/dev samples. "
                      "Search latency is per query on this CPU, query embedding excluded.",
              "datasets": {}}
    for name in args.datasets:
        r = evaluate(name, settings, rerank=not args.no_rerank)
        result["datasets"][name] = r
        for m, s in r["methods"].items():
            print(f"{name:9} {m:18} nDCG@10 {s['ndcg@10']:.4f} {s['ndcg@10_ci95']}  R@100 {s['recall@100']:.4f}  "
                  f"MRR@10 {s['mrr@10']:.4f}")
        print(f"{name:9} published {r['published_ndcg@10']}  paired vs dense {r['paired_vs_dense_ndcg@10']}")
        path = ROOT / "results" / "beir.json"        # written after each dataset: a later failure keeps it
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"written: {path}")
    return result


if __name__ == "__main__":
    main()
