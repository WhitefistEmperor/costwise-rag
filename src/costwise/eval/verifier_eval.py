"""Real strong verifier, offline: a local cross-encoder in the cache band, at no API cost.

Run with `python -m costwise.eval.verifier_eval`. Needs the cross-encoder models (downloaded by
sentence-transformers on first use); no LLM calls.

The offline cache eval (`eval.offline`) can only bound the two-threshold design with a perfect
("oracle") strong verifier. This eval puts real cross-encoders in that slot and measures what
they reach on the same test streams. Everything that is fitted (the cross-encoder's score
threshold) is fitted on the tune split; the accept and candidate thresholds are the ones the
offline eval already chose on tune (data/tuned.json). Results go to results/verifier.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ..adapters.corpus import load_corpus
from ..adapters.embedders import SentenceTransformerEmbedder
from ..adapters.retrieval import build_retrievers
from ..adapters.verifiers import CrossEncoderVerifier, KeyTermGuard
from ..settings import ROOT, Settings
from .dataset import build_stream, load_questions
from .offline import _rates, embed_questions, question_pairs, simulate_stream, top_chunks

MODELS = ["cross-encoder/quora-distilroberta-base", "cross-encoder/stsb-distilroberta-base"]
SCORE_GRID = [round(0.05 * i, 2) for i in range(1, 20)]       # 0.05 .. 0.95
LOW = 0.60                                                     # lowest similarity ever checked


def score_pairs(model_name: str, pairs: Sequence[tuple[str, str]], batch_size: int = 64) -> dict[tuple[str, str], float]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    scores = model.predict(list(pairs), batch_size=batch_size, show_progress_bar=False)
    return {p: float(s) for p, s in zip(pairs, np.asarray(scores).reshape(-1))}


def time_single_pair(model_name: str, pairs: Sequence[tuple[str, str]], n: int = 100) -> dict:
    """Latency of one pair per call, the way the cache calls it (batch of 1), on this CPU."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    model.predict([pairs[0]])                                  # warm-up
    times = []
    for p in list(pairs)[:n]:
        t = time.perf_counter()
        model.predict([p], show_progress_bar=False)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"pairs": len(times), "ms_p50": round(times[len(times) // 2], 1),
            "ms_p95": round(times[min(len(times) - 1, int(0.95 * len(times)))], 1)}


def candidate_pairs(pairs, guard: KeyTermGuard | None, evidence: dict[str, str] | None, low: float, high: float):
    """Pairs a strong verifier would be asked about: similarity in [low, high), cheap checks passed."""
    return [(same, a, b) for s, same, a, b in pairs
            if low <= s < high and (evidence is None or evidence[a] == evidence[b])
            and (guard is None or guard.same_question(a, b))]


def tune_score_threshold(cands, scores: dict) -> float:
    """Lowest score threshold with no false accept among the tune candidates (both pair orders)."""
    for t in SCORE_GRID:
        if not any(not same and (scores[(a, b)] >= t or scores[(b, a)] >= t) for same, a, b in cands):
            return t
    return 1.0


def confusion(cands, scores: dict, t: float) -> dict:
    """Pair-level counts at score threshold t, scoring each pair in the (a, b) order."""
    tp = sum(same and scores[(a, b)] >= t for same, a, b in cands)
    fp = sum((not same) and scores[(a, b)] >= t for same, a, b in cands)
    pos = sum(same for same, _, _ in cands)
    return {"candidates": len(cands), "same": pos, "true_accepts": tp, "false_accepts": fp,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / pos, 4) if pos else None}


def run(settings: Settings | None = None, models: Sequence[str] = MODELS, seeds: Sequence[int] = range(5),
        repeat_rate: float = 0.5) -> dict:
    settings = settings or Settings()
    tuned = settings.load_tuned()
    accept, candidate = tuned["cache_threshold"], tuned["cache_candidate_threshold"]
    embedder = SentenceTransformerEmbedder(settings.embed_model)
    chunks = load_corpus(settings.corpus_dir)
    retriever = build_retrievers(chunks, embedder)["hybrid"]
    tune = load_questions(settings.questions_path, "tune")
    test = load_questions(settings.questions_path, "test")
    guard = KeyTermGuard.from_corpus(chunks)
    tune_vecs, test_vecs = embed_questions(embedder, tune), embed_questions(embedder, test)
    tune_ev, test_ev = top_chunks(retriever, tune_vecs), top_chunks(retriever, test_vecs)
    tune_pairs, test_pairs = question_pairs(tune, tune_vecs), question_pairs(test, test_vecs)
    streams = [build_stream(test, repeat_rate, seed) for seed in seeds]

    # Every pair a verifier could ever see (similarity >= LOW), in both orders.
    to_score = sorted({p for s, _, a, b in tune_pairs + test_pairs if s >= LOW for p in ((a, b), (b, a))})

    def totals(threshold, verifier, ev, cand, strong) -> dict:
        total: dict[str, int] = {}
        for stream in streams:
            for k, v in simulate_stream(stream, test_vecs, threshold, verifier, ev, cand, strong).items():
                total[k] = total.get(k, 0) + v
        return _rates(total)

    out = {"note": "Real local cross-encoders as the strong verifier, offline, no API cost. The "
                   "score threshold is fitted on tune; accept and candidate thresholds come from "
                   "data/tuned.json (also fitted on tune). Counts are totals over all stream seeds.",
           "accept_threshold": accept, "candidate_threshold": candidate, "stream_seeds": len(streams),
           "repeat_rate": repeat_rate, "scored_pairs": len(to_score), "models": {}}
    for model in models:
        name = f"fine-tuned/{Path(model).name}" if Path(model).exists() else model
        t0 = time.perf_counter()
        scores = score_pairs(model, to_score)
        scoring_s = time.perf_counter() - t0
        lookup = lambda pairs: [scores[p] for p in pairs]
        res: dict = {"batch_scoring_s": round(scoring_s, 1),
                     "single_pair_latency": time_single_pair(model, to_score)}
        # Three ways to use it; each gets its own score threshold, fitted on tune.
        setups = {
            "band": (accept, guard, True),                  # deployed design: guard + evidence, band below accept
            "verify_all": (0.99, guard, True),              # strong verifier on every semantic candidate
            "cross_encoder_alone": (0.99, None, False),     # no guard, no evidence check
        }
        for setup, (acc, cheap, same_ev) in setups.items():
            tev, sev = (tune_ev, test_ev) if same_ev else (None, None)
            t = tune_score_threshold(candidate_pairs(tune_pairs, cheap, tev, candidate, acc), scores)
            strong = CrossEncoderVerifier(model, t, scorer=lookup)
            res[setup] = {
                "accept_threshold": acc, "score_threshold_from_tune": t,
                "tune_band_pairs": confusion(candidate_pairs(tune_pairs, cheap, tev, candidate, acc), scores, t),
                "test_band_pairs": confusion(candidate_pairs(test_pairs, cheap, sev, candidate, acc), scores, t),
                "test_stream": totals(acc, cheap, sev, candidate, strong),
            }
        out["models"][name] = res
    return out


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="default: the two public cross-encoders, plus the fine-tuned verifier if trained")
    ap.add_argument("--write-tuned", action="store_true",
                    help="store the best band model and its threshold in data/tuned.json for the app")
    args = ap.parse_args(argv)
    settings = Settings()
    tuned_model = settings.state_dir / "models" / "verifier-minilm-paws-qqp"
    models = args.models or (MODELS + ([str(tuned_model)] if tuned_model.exists() else []))
    result = run(settings, models)
    path = ROOT / "results" / "verifier.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    for name, r in result["models"].items():
        for setup in ("band", "verify_all", "cross_encoder_alone"):
            s = r[setup]["test_stream"]
            print(f"{name:42} {setup:20} t={r[setup]['score_threshold_from_tune']:<5} hits {s['hits']:3} "
                  f"correct {s['correct_hits']:3}/{s['repeats']} wrong {s['wrong_hits']:3}/{s['queries']} "
                  f"calls {s['strong_verifier_calls']}")
        print(f"{'':42} single-pair latency {r['single_pair_latency']}")
    if args.write_tuned:
        # Pick on TUNE: most true accepts in the band with no false accept on tune.
        # Hub models only: a locally fine-tuned model is not in a fresh clone or the Docker image.
        hub = [m for m in result["models"] if not m.startswith("fine-tuned/")]
        best = max(hub, key=lambda m: result["models"][m]["band"]["tune_band_pairs"]["true_accepts"])
        tuned = settings.load_tuned()
        tuned["cross_encoder_model"] = best
        tuned["cross_encoder_threshold"] = result["models"][best]["band"]["score_threshold_from_tune"]
        settings.tuned_path.write_text(json.dumps(tuned, indent=2), encoding="utf-8")
        print(f"tuned.json: cross_encoder_model={best}")
    print(f"written: {path}")
    return result


if __name__ == "__main__":
    main()
