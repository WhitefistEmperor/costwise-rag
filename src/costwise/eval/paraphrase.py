"""Cache decisions on public paraphrase benchmarks: QQP (real duplicate questions) and PAWS
(adversarial pairs built by swapping words, so most non-paraphrases share almost every word).

Run with `python -m costwise.eval.paraphrase`. Downloads GLUE QQP and PAWS labeled_final as
parquet at pinned revisions; uses the MiniLM embedder and local cross-encoders. No LLM calls.

Each labelled pair is one cache decision: a new query meets a cached query; serving the cached
answer is right only if the pair is a duplicate. So precision is the share of served hits that
are right (1 - precision = wrong answers served) and recall is the share of duplicates served.
These are pairwise decisions; QQP and PAWS have no document corpus, so the service's
same-evidence check cannot run here.

Policies, each with its thresholds fitted on a TUNE sample (QQP train, PAWS train) for a target
precision and then scored on a TEST sample (QQP validation, PAWS test):
  cosine            embedding similarity >= t
  cosine+guard      the same, and KeyTermGuard agrees (numbers, units, contrast words and
                    capitalised terms, the terms taken from the tune texts)
  cascade           the service design: guard, then cosine >= accept serves at once, cosine in
                    [candidate, accept) serves only if the cross-encoder scores >= s
  cross_encoder     cross-encoder score >= s on every pair (most expensive)

Leakage, disclosed: all-MiniLM-L6-v2 was trained on a large paraphrase mix that includes Quora
duplicate questions, and cross-encoder/quora-distilroberta-base was trained on QQP itself, so
their QQP numbers are optimistic. Neither was trained on PAWS. The STS-B cross-encoder was
trained on neither.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ..adapters.embedders import SentenceTransformerEmbedder
from ..adapters.verifiers import KeyTermGuard, extract_entity_terms
from ..domain.models import Chunk
from ..settings import ROOT, Settings

DATASETS = {
    "qqp": {"repo": "nyu-mll/glue", "rev": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
            "tune": "qqp/train-00000-of-00001.parquet", "test": "qqp/validation-00000-of-00001.parquet",
            "cols": ("question1", "question2")},
    "paws": {"repo": "google-research-datasets/paws", "rev": "161ece9501cf0a11f3e48bd356eaa82de46d6a09",
             "tune": "labeled_final/train-00000-of-00001.parquet", "test": "labeled_final/test-00000-of-00001.parquet",
             "cols": ("sentence1", "sentence2")},
}
CROSS_ENCODERS = ["cross-encoder/stsb-distilroberta-base", "cross-encoder/quora-distilroberta-base"]
TARGETS = [0.80, 0.90, 0.95]
COS_GRID = [round(0.50 + 0.02 * i, 2) for i in range(25)]          # 0.50 .. 0.98
SCORE_GRID = [round(0.05 * i, 2) for i in range(1, 20)]            # 0.05 .. 0.95
TUNE_N, TEST_N = 3000, 5000
# Entity-term rule for open-domain text: keep words capitalised in at least this share of their
# mid-sentence occurrences (0 = the service's closed-corpus rule). Chosen on tune by average precision.
GUARD_RATIOS = [0.0, 0.5, 0.8, 0.95]


def fine_tuned_verifier(settings: Settings) -> str | None:
    """Path of the verifier trained by eval/train_verifier.py, if it has been trained."""
    path = settings.state_dir / "models" / "verifier-minilm-paws-qqp"
    return str(path) if path.exists() else None


def model_key(m: str) -> str:
    return Path(m).name if Path(m).exists() else m.replace("/", "_")


def load_pairs(name: str, split: str, n: int, seed: int = 0) -> list[tuple[str, str, int]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    spec = DATASETS[name]
    df = pd.read_parquet(hf_hub_download(spec["repo"], spec[split], repo_type="dataset", revision=spec["rev"]))
    a, b = spec["cols"]
    df = df[df["label"].isin([0, 1])].sample(n=min(n, len(df)), random_state=seed)
    return [(str(x), str(y), int(z)) for x, y, z in zip(df[a], df[b], df["label"])]


def counts(accept: np.ndarray, y: np.ndarray) -> dict:
    tp = int((accept & (y == 1)).sum())
    fp = int((accept & (y == 0)).sum())
    pos = int((y == 1).sum())
    return {"served": tp + fp, "right": tp, "wrong": fp, "duplicates": pos, "pairs": int(len(y)),
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / pos, 4) if pos else None}


def bootstrap(accept: np.ndarray, y: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    prec, rec = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        a, t = accept[i], y[i]
        tp, fp, pos = (a & (t == 1)).sum(), (a & (t == 0)).sum(), (t == 1).sum()
        if tp + fp:
            prec.append(tp / (tp + fp))
        rec.append(tp / pos if pos else 0.0)
    q = lambda xs: [round(float(np.quantile(xs, 0.025)), 4), round(float(np.quantile(xs, 0.975)), 4)] if xs else None
    return {"precision_ci95": q(prec), "recall_ci95": q(rec)}


def best_on_tune(candidates: list[tuple[dict, np.ndarray]], y: np.ndarray, target: float) -> dict | None:
    """Among parameter settings, the one with the highest tune recall whose tune precision reaches
    the target; None if none does."""
    ok = []
    for params, accept in candidates:
        c = counts(accept, y)
        if c["precision"] is not None and c["precision"] >= target:
            ok.append((c["recall"], c["precision"], params))
    if not ok:
        return None
    return max(ok, key=lambda r: (r[0], r[1]))[2]


class Scored:
    """Everything a policy needs for one split: cosine, guard verdict, cross-encoder scores."""

    def __init__(self, pairs, embedder, guard: KeyTermGuard, ce_scores: dict[str, np.ndarray]):
        a = embedder.encode([p[0] for p in pairs])
        b = embedder.encode([p[1] for p in pairs])
        self.cos = (a * b).sum(axis=1)
        self.guard = np.array([guard.same_question(x, y) for x, y, _ in pairs])
        self.y = np.array([p[2] for p in pairs])
        self.ce = ce_scores


def policies(s: Scored, ce: str | None):
    """(policy name, list of (params, accept mask)) for one split."""
    yield "cosine", [({"cos": t}, s.cos >= t) for t in COS_GRID]
    yield "cosine+guard", [({"cos": t}, (s.cos >= t) & s.guard) for t in COS_GRID]
    if ce is None:
        return
    score = s.ce[ce]
    cascade = []
    for acc in COS_GRID + [1.01]:                                       # 1.01: nothing served without the CE
        for cand in [c for c in COS_GRID if c < acc][::2]:
            for t in SCORE_GRID:
                mask = s.guard & ((s.cos >= acc) | ((s.cos >= cand) & (s.cos < acc) & (score >= t)))
                cascade.append(({"accept": acc, "candidate": cand, "score": t}, mask))
    yield f"cascade[{ce}]", cascade
    yield f"cross_encoder[{ce}]", [({"score": t}, score >= t) for t in SCORE_GRID]


def apply(policy: str, params: dict, s: Scored) -> tuple[np.ndarray, int]:
    """Accept mask on a split and how many cross-encoder calls the policy makes there."""
    if policy == "cosine":
        return s.cos >= params["cos"], 0
    if policy == "cosine+guard":
        return (s.cos >= params["cos"]) & s.guard, 0
    ce = policy[policy.index("[") + 1:-1]
    score = s.ce[ce]
    if policy.startswith("cross_encoder"):
        return score >= params["score"], len(score)
    band = s.guard & (s.cos >= params["candidate"]) & (s.cos < params["accept"])
    mask = s.guard & ((s.cos >= params["accept"]) | (band & (score >= params["score"])))
    return mask, int(band.sum())


def ce_scores(model_name: str, pairs, cache_path=None) -> tuple[np.ndarray, float | None]:
    """Cross-encoder score per pair and batched ms per pair (None when read from the cache)."""
    if cache_path is not None and cache_path.exists():
        ms_path = cache_path.with_suffix(".ms")
        return np.load(cache_path), (float(ms_path.read_text()) if ms_path.exists() else None)
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    t = time.perf_counter()
    out = np.asarray(model.predict([(a, b) for a, b, _ in pairs], batch_size=64, show_progress_bar=False)).reshape(-1)
    ms = (time.perf_counter() - t) * 1000 / len(pairs)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, out)
        cache_path.with_suffix(".ms").write_text(str(ms))
    return out, ms


def average_precision(score: np.ndarray, y: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise, as scikit-learn computes it)."""
    from sklearn.metrics import average_precision_score

    return round(float(average_precision_score(y, score)), 4)


def evaluate(name: str, settings: Settings, cross_encoders: Sequence[str] = CROSS_ENCODERS) -> dict:
    tune, test = load_pairs(name, "tune", TUNE_N), load_pairs(name, "test", TEST_N)
    embedder = SentenceTransformerEmbedder(settings.embed_model)
    tune_chunks = [Chunk(str(i), "", "", f"x\n{a}\n{b}") for i, (a, b, _) in enumerate(tune)]
    ta, tb = embedder.encode([p[0] for p in tune]), embedder.encode([p[1] for p in tune])
    tune_cos, tune_y = (ta * tb).sum(axis=1), np.array([p[2] for p in tune])
    guard_ap = {}
    for r in GUARD_RATIOS:                              # entity-term rule chosen on tune
        g = KeyTermGuard(extract_entity_terms(tune_chunks, r))
        mask = np.array([g.same_question(a, b) for a, b, _ in tune])
        guard_ap[r] = average_precision(np.where(mask, tune_cos, -1.0), tune_y)
    ratio = max(GUARD_RATIOS, key=lambda r: (guard_ap[r], -r))
    guard = KeyTermGuard(extract_entity_terms(tune_chunks, ratio))
    closed_rule_guard = KeyTermGuard(extract_entity_terms(tune_chunks, 0.0))
    ce_tune, ce_test, ce_ms = {}, {}, {}
    cache = settings.state_dir / "bench"
    key = lambda m, split, n: cache / f"{name}-{split}-{n}-{DATASETS[name]['rev'][:8]}-{model_key(m)}.npy"
    labels = []
    for m in cross_encoders:
        label = f"fine-tuned/{Path(m).name}" if Path(m).exists() else m
        labels.append(label)
        print(f"{name}: scoring with {label}", flush=True)
        ce_tune[label], _ = ce_scores(m, tune, key(m, "tune", len(tune)))
        ce_test[label], ce_ms[label] = ce_scores(m, test, key(m, "test", len(test)))
    st, ss = Scored(tune, embedder, guard, ce_tune), Scored(test, embedder, guard, ce_test)
    out = {"dataset": name, "revision": DATASETS[name]["rev"], "tune_pairs": len(tune), "test_pairs": len(test),
           "tune_duplicate_share": round(float(st.y.mean()), 4), "test_duplicate_share": round(float(ss.y.mean()), 4),
           "guard_entity_terms": len(guard.entity_terms),
           "guard_cap_ratio_tune_ap": {str(r): v for r, v in guard_ap.items()},
           "guard_cap_ratio_from_tune": ratio,
           "guard_closed_rule_test": {
               "duplicates_blocked": int(sum(not closed_rule_guard.same_question(a, b) for a, b, y in test if y == 1)),
               "non_duplicates_blocked": int(sum(not closed_rule_guard.same_question(a, b) for a, b, y in test if y == 0))},
           "cross_encoder_ms_per_pair_batched": {m: (round(v, 2) if v else None) for m, v in ce_ms.items()},
           # Threshold-free: average precision of each score on test (guard-blocked pairs score -1).
           "test_average_precision": {
               "cosine": average_precision(ss.cos, ss.y),
               "cosine+guard": average_precision(np.where(ss.guard, ss.cos, -1.0), ss.y),
               **{f"cross_encoder[{m}]": average_precision(ss.ce[m], ss.y) for m in labels},
               **{f"cross_encoder+guard[{m}]": average_precision(np.where(ss.guard, ss.ce[m], -1.0), ss.y)
                  for m in labels}},
           "guard_blocks": {"test_duplicates_blocked": int((~ss.guard & (ss.y == 1)).sum()),
                            "test_non_duplicates_blocked": int((~ss.guard & (ss.y == 0)).sum())},
           "targets": {}}
    for target in TARGETS:
        rows = {}
        seen = set()
        for ce in [None, *labels]:
            for policy, cands in policies(st, ce):
                if policy in seen:
                    continue
                seen.add(policy)
                params = best_on_tune(cands, st.y, target)
                if params is None:
                    rows[policy] = {"params_from_tune": None, "note": "no setting reaches the target on tune"}
                    continue
                mask, calls = apply(policy, params, ss)
                rows[policy] = {"params_from_tune": params, "test": counts(mask, ss.y), **bootstrap(mask, ss.y),
                                "cross_encoder_calls_share": round(calls / len(ss.y), 4)}
        out["targets"][str(target)] = rows
    return out


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    args = ap.parse_args(argv)
    settings = Settings()
    result = {"note": "Pairwise cache decisions; thresholds fitted on tune for a target precision, "
                      "scored on test. See the module docstring for leakage.", "datasets": {}}
    models = list(CROSS_ENCODERS) + ([fine_tuned_verifier(settings)] if fine_tuned_verifier(settings) else [])
    for name in args.datasets:
        r = evaluate(name, settings, models)
        result["datasets"][name] = r
        print(f"{name:5} average precision {r['test_average_precision']}  guard {r['guard_blocks']}", flush=True)
        for target, rows in r["targets"].items():
            for policy, row in rows.items():
                if row.get("test"):
                    t = row["test"]
                    print(f"{name:5} target {target}  {policy:52} P {t['precision']}  R {t['recall']}  "
                          f"wrong {t['wrong']}/{t['pairs']}  CE calls {row['cross_encoder_calls_share']}")
                else:
                    print(f"{name:5} target {target}  {policy:52} -- {row['note']}")
    path = ROOT / "results" / "paraphrase.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"written: {path}")
    return result


if __name__ == "__main__":
    main()
