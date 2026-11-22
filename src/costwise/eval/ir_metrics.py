"""Ranking metrics and bootstrap confidence intervals. Pure functions, no I/O.

nDCG@k uses graded relevance (BEIR qrels give 0, 1 or 2) with the standard log2 discount and
the ideal ranking built from all judged documents of the query, as pytrec_eval / BEIR do.
"""
from __future__ import annotations

import math
import random
from typing import Mapping, Sequence


def ndcg_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    dcg = sum((2 ** qrels.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((r for r in qrels.values() if r > 0), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    relevant = {d for d, r in qrels.items() if r > 0}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def mrr_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    return next((1.0 / (i + 1) for i, d in enumerate(ranked[:k]) if qrels.get(d, 0) > 0), 0.0)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(xs: Sequence[float], n: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling queries."""
    rng = random.Random(seed)
    m = len(xs)
    means = sorted(mean([xs[rng.randrange(m)] for _ in range(m)]) for _ in range(n))
    return means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n) - 1]


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n: int = 2000, seed: int = 0) -> dict:
    """Mean of a - b per query, its 95% CI, and the share of resamples where a does not beat b
    (a one-sided bootstrap p-value)."""
    assert len(a) == len(b)
    d = [x - y for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(d, n, seed)
    rng = random.Random(seed + 1)
    m = len(d)
    not_better = sum(mean([d[rng.randrange(m)] for _ in range(m)]) <= 0 for _ in range(n))
    return {"mean_diff": round(mean(d), 4), "ci95": [round(lo, 4), round(hi, 4)], "p_not_better": round(not_better / n, 4)}
