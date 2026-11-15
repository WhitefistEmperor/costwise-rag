"""Offline evals: no LLM calls, no cost. Run with `python -m costwise.eval.offline`.

Anything fitted here (cache threshold, router weights) is fitted on the tune split and
reported on the test split.
"""
from __future__ import annotations

import json
from itertools import combinations
from typing import Sequence

import numpy as np

from ..adapters.embedders import SentenceTransformerEmbedder
from ..adapters.retrieval import build_retrievers
from ..adapters.verifiers import KeyTermGuard
from ..domain.cache import SemanticCache
from ..domain.chunking import load_corpus
from ..domain.models import CacheEntry
from ..domain.ports import Embedder, Verifier
from ..settings import ROOT, Settings
from .dataset import Query, all_texts, build_stream, load_questions

THRESHOLD_GRID = [round(0.60 + 0.02 * i, 2) for i in range(20)]     # 0.60 .. 0.98
THRESHOLD_MARGIN = 0.02


# ---------------------------------------------------------------- retrieval

def eval_retrieval(retrievers: dict, embedder: Embedder, questions: Sequence[dict], k: int) -> dict:
    """Doc-level recall@k (every gold doc is in the top k) and MRR of the first gold doc,
    over each question and its paraphrases."""
    texts = [(q, t) for q in questions for t in all_texts(q)]
    vecs = embedder.encode([t for _, t in texts])
    out = {}
    for name, retriever in retrievers.items():
        found, rr = 0, 0.0
        for (q, text), vec in zip(texts, vecs):
            docs = [h.chunk.doc for h in retriever.search(text, vec, k)]
            found += all(d in docs for d in q["docs"])
            rr += next((1.0 / (i + 1) for i, d in enumerate(docs) if d in q["docs"]), 0.0)
        out[name] = {"recall_at_k": round(found / len(texts), 4), "mrr": round(rr / len(texts), 4),
                     "found": found, "queries": len(texts), "k": k}
    return out


# -------------------------------------------------------------------- cache

def embed_questions(embedder: Embedder, questions: Sequence[dict]) -> dict[str, np.ndarray]:
    texts = [t for q in questions for t in all_texts(q)]
    return dict(zip(texts, embedder.encode(texts)))


def question_pairs(questions: Sequence[dict], vecs: dict[str, np.ndarray]) -> list[tuple[float, bool, str, str]]:
    """Every pair of question texts: (similarity, same question?, text a, text b)."""
    items = [(q["id"], t) for q in questions for t in all_texts(q)]
    return [(float(vecs[a] @ vecs[b]), qa == qb, a, b) for (qa, a), (qb, b) in combinations(items, 2)]


def top_chunks(retriever, vecs: dict[str, np.ndarray]) -> dict[str, str]:
    """Evidence key per question text: the id of its top retrieved chunk."""
    return {text: retriever.search(text, vec, 1)[0].chunk.id for text, vec in vecs.items()}


def pair_counts(pairs, threshold: float, verifier: Verifier | None, evidence: dict[str, str] | None = None) -> dict:
    accept = lambda s, a, b: (s >= threshold and (evidence is None or evidence[a] == evidence[b])
                              and (verifier is None or verifier.same_question(a, b)))
    same = [p for p in pairs if p[1]]
    diff = [p for p in pairs if not p[1]]
    return {"true_accepts": sum(accept(s, a, b) for s, _, a, b in same), "same_pairs": len(same),
            "false_accepts": sum(accept(s, a, b) for s, _, a, b in diff), "different_pairs": len(diff)}


def tune_threshold(pairs, verifier: Verifier | None, evidence: dict[str, str] | None = None) -> float:
    """Lowest threshold on the grid with no false accept on the tune pairs, plus a safety margin."""
    for t in THRESHOLD_GRID:
        if pair_counts(pairs, t, verifier, evidence)["false_accepts"] == 0:
            return round(min(t + THRESHOLD_MARGIN, 0.99), 2)
    return 0.99


def simulate_stream(stream: Sequence[Query], vecs: dict[str, np.ndarray], threshold: float,
                    verifier: Verifier | None, evidence: dict[str, str] | None = None) -> dict:
    """Runs the query stream through a fresh cache. A hit is wrong if the cached entry came
    from a different question. Misses are stored; wrong hits are not (nothing would be)."""
    cache = SemanticCache(threshold, verifier=verifier, clock=lambda: 0.0,
                          require_same_evidence=evidence is not None)
    key = lambda text: evidence[text] if evidence is not None else None
    hits = wrong = 0
    for q in stream:
        hit = cache.lookup(q.text, vecs[q.text], evidence_key=key(q.text))
        if hit is None:
            cache.put(CacheEntry(q.text, "", {}, "", 0.0, 0.0, tag=q.qid, evidence_key=key(q.text)),
                      vecs[q.text])
        else:
            hits += 1
            wrong += hit.entry.tag != q.qid
    return {"queries": len(stream), "repeats": sum(q.repeat for q in stream), "hits": hits,
            "correct_hits": hits - wrong, "wrong_hits": wrong}


def _rates(c: dict) -> dict:
    return {**c, "hit_rate": round(c["hits"] / c["queries"], 4),
            "wrong_hit_rate": round(c["wrong_hits"] / c["queries"], 4),
            "repeat_recall": round(c["correct_hits"] / c["repeats"], 4) if c["repeats"] else None}


def eval_cache(embedder: Embedder, retriever, tune: Sequence[dict], test: Sequence[dict],
               variants: dict[str, tuple[Verifier | None, bool]],
               repeat_rate: float = 0.5, seeds: Sequence[int] = range(5)) -> dict:
    """`variants` maps a name to (verifier or None, require same evidence?)."""
    tune_vecs, test_vecs = embed_questions(embedder, tune), embed_questions(embedder, test)
    tune_ev, test_ev = top_chunks(retriever, tune_vecs), top_chunks(retriever, test_vecs)
    tune_pairs, test_pairs = question_pairs(tune, tune_vecs), question_pairs(test, test_vecs)
    streams = [build_stream(test, repeat_rate, seed) for seed in seeds]

    def stream_totals(threshold: float, verifier, ev) -> dict:
        total: dict[str, int] = {}
        for stream in streams:
            for key, value in simulate_stream(stream, test_vecs, threshold, verifier, ev).items():
                total[key] = total.get(key, 0) + value
        return _rates(total)

    out = {"repeat_rate": repeat_rate, "stream_seeds": len(streams), "variants": {}}
    for name, (verifier, same_evidence) in variants.items():
        tev, sev = (tune_ev, test_ev) if same_evidence else (None, None)
        threshold = tune_threshold(tune_pairs, verifier, tev)
        out["variants"][name] = {
            "threshold_from_tune": threshold,
            "tune_pairs": pair_counts(tune_pairs, threshold, verifier, tev),
            "test_pairs": pair_counts(test_pairs, threshold, verifier, sev),
            "test_stream": stream_totals(threshold, verifier, sev),
            "test_stream_by_threshold": {str(t): stream_totals(t, verifier, sev) for t in THRESHOLD_GRID[::2]},
        }
    return out


# --------------------------------------------------------------------- main

def run(settings: Settings | None = None) -> tuple[dict, dict]:
    """Returns (results, tuned values)."""
    settings = settings or Settings()
    embedder = SentenceTransformerEmbedder(settings.embed_model)
    chunks = load_corpus(settings.corpus_dir)
    retrievers = build_retrievers(chunks, embedder)
    tune = load_questions(settings.questions_path, "tune")
    test = load_questions(settings.questions_path, "test")
    guard = KeyTermGuard.from_corpus(chunks)

    cache = eval_cache(embedder, retrievers["hybrid"], tune, test, {
        "threshold_only": (None, False),
        "key_term_guard": (guard, False),
        "same_evidence": (None, True),
        "guard_and_evidence": (guard, True),
    })
    results = {
        "corpus": {"documents": len({c.doc for c in chunks}), "chunks": len(chunks), "synthetic": True},
        "questions": {"tune": len(tune), "test": len(test), "texts_per_question": 3},
        "retrieval_test": eval_retrieval(retrievers, embedder, test, settings.top_k),
        "cache": cache,
    }
    tuned = {"cache_threshold": cache["variants"]["guard_and_evidence"]["threshold_from_tune"],
             "cache_verifier": guard.name, "cache_require_same_evidence": True}
    return results, tuned


def main() -> None:
    settings = Settings()
    results, tuned = run(settings)
    out = ROOT / "results" / "offline.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    settings.tuned_path.write_text(json.dumps(tuned, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nresults: {out}\ntuned values: {settings.tuned_path}")


if __name__ == "__main__":
    main()
