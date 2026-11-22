"""Router on public data with observed small-model failures: RouteLLM's gpt4_dataset.

Run with `python -m costwise.eval.routerbench`. Downloads `valid.jsonl` (10,000 real chat
prompts from LMSYS/ShareGPT etc., each with a GPT-4 response, a Mixtral-8x7B response, and a
GPT-4 judge score of the Mixtral answer from 1 to 5) at a pinned revision. No LLM calls.

What this measures: given only the prompt, can a router tell which prompts the small model
(Mixtral) fails on, so that only those go to the large model (GPT-4)? A prompt "needs the large
model" when Mixtral's judged score is 3 or lower. Quality of a routing decision: 1 if the
answer used is acceptable (GPT-4's answer is treated as acceptable; Mixtral's when it scored 4
or 5), else 0. Costs are SIMULATED from list prices and response lengths (about 4 characters
per token); no call was made.

Routers compared, all fitted on the fit half and scored on the held-out half:
  random                 the diagonal any router must beat (computed exactly);
  costwise_text          the service's learned router restricted to its 4 text features
                         (words, numbers, multi-part cues, clauses). Its 3 retrieval features
                         do not exist for open chat prompts, so this is a text-only variant,
                         not the deployed router;
  embedding_lr           logistic regression on the MiniLM query embedding the service
                         already computes for retrieval (so no extra latency at serving time);
  oracle                 routes exactly the failing prompts (the ceiling).

Metrics (RouteLLM's): PGR(f) = share of the quality gap between always-small and always-large
recovered when a fraction f of prompts goes to the large model; APGR = mean PGR over f in
[0, 1]; CPT(x) = the smallest share of large-model calls that recovers x of the gap.
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Sequence

import numpy as np

from ..adapters.embedders import SentenceTransformerEmbedder
from ..domain.routing import features
from ..settings import ROOT, Settings

REPO, REVISION, FILE = "routellm/gpt4_dataset", "7ef62d690ff71ce9c21f4226cfc7bbc2ef9e6227", "valid.jsonl"
# List prices in USD per million tokens (input, output), used only for the simulated cost.
PRICES = {"large": ("gpt-4-1106-preview", 10.0, 30.0), "small": ("mixtral-8x7b-instruct", 0.6, 0.6)}
TEXT_FEATURES = [0, 1, 2, 3]          # words, numbers, cues, clauses (see domain/routing.py)
FRACTIONS = np.linspace(0.0, 1.0, 101)
C_GRID = [0.01, 0.1, 1.0, 10.0]


def load_rows() -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            try:
                score = int(r["mixtral_score"])
            except (TypeError, ValueError):
                continue
            if r.get("prompt"):
                rows.append({"prompt": r["prompt"], "score": score,
                             "large_chars": len(r.get("gpt4_response") or ""),
                             "small_chars": len(r.get("mixtral_response") or "")})
    return rows


def simulated_cost(row: dict, tier: str) -> float:
    _, pin, pout = PRICES[tier]
    out_chars = row["large_chars"] if tier == "large" else row["small_chars"]
    return (len(row["prompt"]) / 4 * pin + out_chars / 4 * pout) / 1e6


def curve(scores: np.ndarray, needs_large: np.ndarray) -> np.ndarray:
    """Quality at each fraction f of prompts sent to the large model, highest scores first."""
    order = np.argsort(-scores, kind="stable")
    rescued = np.concatenate([[0], np.cumsum(needs_large[order])])     # failures fixed after n calls
    n = len(scores)
    base = 1.0 - needs_large.mean()                                    # quality of always-small
    return np.array([base + rescued[int(round(f * n))] / n for f in FRACTIONS])


def summarise(q: np.ndarray, weak: float, strong: float) -> dict:
    pgr = (q - weak) / (strong - weak)
    cpt = lambda x: float(FRACTIONS[np.argmax(pgr >= x - 1e-12)]) if (pgr >= x - 1e-12).any() else None
    return {"apgr": round(float(pgr.mean()), 4), "cpt50": cpt(0.5), "cpt80": cpt(0.8),
            "quality_at_20pct_large": round(float(q[20]), 4), "quality_at_50pct_large": round(float(q[50]), 4)}


def bootstrap_apgr(scores: np.ndarray, y: np.ndarray, n: int = 300, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        s, t = scores[idx], y[idx]
        weak = 1.0 - t.mean()
        vals.append(float(((curve(s, t) - weak) / (1.0 - weak)).mean()))
    vals.sort()
    return [round(vals[int(0.025 * n)], 4), round(vals[int(0.975 * n) - 1], 4)]


def fit_lr(x_fit: np.ndarray, y_fit: np.ndarray, seed: int = 0):
    """Logistic regression with C chosen on an inner 20% hold-out of the fit half (by APGR)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    idx = np.random.default_rng(seed).permutation(len(y_fit))
    cut = int(0.8 * len(idx))
    tr, va = idx[:cut], idx[cut:]
    def model(c):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000))
    weak_va = 1.0 - y_fit[va].mean()
    def apgr(c):
        m = model(c).fit(x_fit[tr], y_fit[tr])
        return float(((curve(m.predict_proba(x_fit[va])[:, 1], y_fit[va]) - weak_va) / (1 - weak_va)).mean())
    picks = {c: round(apgr(c), 4) for c in C_GRID}
    best = max(C_GRID, key=lambda c: (picks[c], -c))
    return model(best).fit(x_fit, y_fit), best, picks


def run(settings: Settings | None = None, seed: int = 0, bootstrap: int = 300) -> dict:
    settings = settings or Settings()
    rows = load_rows()
    random.Random(seed).shuffle(rows)
    half = len(rows) // 2
    fit_rows, test_rows = rows[:half], rows[half:]
    y = lambda rs: np.array([r["score"] <= 3 for r in rs], dtype=float)
    y_fit, y_test = y(fit_rows), y(test_rows)

    text = lambda rs: np.array([[features(r["prompt"], [])[i] for i in TEXT_FEATURES] for r in rs])
    embedder = SentenceTransformerEmbedder(settings.embed_model)
    emb_fit, emb_test = embedder.encode([r["prompt"] for r in fit_rows]), embedder.encode([r["prompt"] for r in test_rows])

    routers: dict[str, np.ndarray] = {}
    chosen: dict[str, dict] = {}
    for name, (xf, xt) in {"costwise_text": (text(fit_rows), text(test_rows)),
                           "embedding_lr": (emb_fit, emb_test)}.items():
        model, c, picks = fit_lr(xf, y_fit, seed)
        routers[name] = model.predict_proba(xt)[:, 1]
        chosen[name] = {"C_from_fit_holdout": c, "fit_holdout_apgr_by_C": {str(k): v for k, v in picks.items()}}
    routers["oracle"] = y_test.copy()

    weak, strong = 1.0 - y_test.mean(), 1.0
    random_curve = weak + FRACTIONS * (strong - weak)
    cost_large = sum(simulated_cost(r, "large") for r in test_rows)
    cost_small = sum(simulated_cost(r, "small") for r in test_rows)

    def cost_at(scores: np.ndarray, f: float) -> float:
        order = np.argsort(-scores, kind="stable")
        k = int(round(f * len(order)))
        big = set(order[:k].tolist())
        return sum(simulated_cost(r, "large" if i in big else "small") for i, r in enumerate(test_rows))

    out = {
        "note": "Public data with observed small-model failures (GPT-4 judge scores of Mixtral answers). "
                "Fitted on one half, reported on the other. Costs are SIMULATED from list prices and "
                "response lengths; no model was called. costwise_text is a text-only variant of the "
                "service router (its retrieval features do not exist for open prompts).",
        "dataset": {"repo": REPO, "revision": REVISION, "file": FILE, "rows": len(rows),
                    "fit": len(fit_rows), "test": len(test_rows), "split_seed": seed},
        "label": "needs large = Mixtral judged score <= 3",
        "test_needs_large": int(y_test.sum()), "always_small_quality": round(weak, 4),
        "prices_per_million": {k: {"model": m, "in": i, "out": o} for k, (m, i, o) in PRICES.items()},
        "simulated_cost_usd": {"always_large": round(cost_large, 4), "always_small": round(cost_small, 4)},
        "random": summarise(random_curve, weak, strong),
        "routers": {},
    }
    for name, scores in routers.items():
        q = curve(scores, y_test)
        s = summarise(q, weak, strong)
        # Cost to reach 95% of always-large quality, against always-large.
        f95 = next((float(f) for f, v in zip(FRACTIONS, q) if v >= 0.95 * strong - 1e-12), 1.0)
        c95 = cost_at(scores, f95)
        out["routers"][name] = {
            **s, **chosen.get(name, {}),
            "apgr_ci95": bootstrap_apgr(scores, y_test, bootstrap, seed) if name != "oracle" else None,
            "large_share_for_95pct_quality": f95,
            "simulated_cost_for_95pct_quality": round(c95, 4),
            "simulated_saving_vs_always_large": round(1 - c95 / cost_large, 4),
            "curve": [round(float(v), 4) for v in q[::5]],
        }
    rf95 = next(float(f) for f, v in zip(FRACTIONS, random_curve) if v >= 0.95 * strong - 1e-12)
    out["random"]["large_share_for_95pct_quality"] = rf95
    out["random"]["curve"] = [round(float(v), 4) for v in random_curve[::5]]
    out["curve_fractions"] = [round(float(f), 2) for f in FRACTIONS[::5]]
    return out


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    result = run(seed=args.seed)
    path = ROOT / "results" / "routerbench.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"always-small quality {result['always_small_quality']}  random APGR {result['random']['apgr']}")
    for name, r in result["routers"].items():
        print(f"{name:14} APGR {r['apgr']} {r['apgr_ci95']}  CPT50 {r['cpt50']}  CPT80 {r['cpt80']}  "
              f"large share @95% {r['large_share_for_95pct_quality']}  saving {r['simulated_saving_vs_always_large']}")
    print(f"written: {path}")
    return result


if __name__ == "__main__":
    main()
