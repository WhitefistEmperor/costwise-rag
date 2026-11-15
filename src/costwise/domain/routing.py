"""Cost-aware routing: decide before the call whether the small model is likely enough.

Both routers return a score in [0, 1] that the query needs the large model; the pipeline sends
it to the large model when score >= route_threshold. The threshold is the cost/quality dial.

- HeuristicRouter: hand-written rules over the same features (the baseline, option A in the
  design doc).
- LearnedRouter: logistic regression over the features (option B). Fitted on the tune split by
  the offline eval; weights are stored as plain numbers so scoring needs no scikit-learn.

"Hard" is a hand label on the question set (multi-step questions), not "the small model
failed"; the live eval is what shows whether the label predicts small-model failures.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Sequence

from .models import Hit

_WORD = re.compile(r"[a-z0-9.]+")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")

# Cues that a question combines facts, compares, or computes something. The list was filtered
# to cues that occur in at least one TUNE-split text (scripts/filter_router_cues.py), because the
# first draft was written after the hard questions of both splits had been looked at. Cues match
# whole words only; under that rule "which plan" occurs in no tune text and was dropped.
MULTI_PART_CUES = (
    "between", "both", "and what", "and how", "total", "cheapest", "lowest", "if i",
    "if a", "after upgrading", "per hour", "in one hour", "maximum number", "at most", "longest",
    "and its", "and how long", "what does it cost and",
)

_CUE_PATTERNS = {cue: re.compile(rf"\b{re.escape(cue)}\b") for cue in MULTI_PART_CUES}


def matched_cues(text: str) -> list[str]:
    """Cues found as whole words ("if a" does not match "if any", "and its" not "and itself")."""
    q = text.lower()
    return [cue for cue, pattern in _CUE_PATTERNS.items() if pattern.search(q)]


def cue_count(text: str) -> int:
    return len(matched_cues(text))


FEATURE_NAMES = (
    "words",            # question length in words / 10
    "numbers",          # numbers written in the question
    "cues",             # multi-part cues matched
    "clauses",          # commas, semicolons and "and" / "or" joins
    "docs_in_top_k",    # distinct source documents among the retrieved hits
    "top_score",        # dense similarity of the best hit
    "score_gap",        # best minus second-best dense similarity (small gap = spread evidence)
)


def features(query: str, hits: Sequence[Hit]) -> list[float]:
    q = query.lower()
    words = _WORD.findall(q)
    cues = cue_count(q)
    clauses = q.count(",") + q.count(";") + sum(w in ("and", "or") for w in words)
    docs = len({h.chunk.doc for h in hits})
    scores = sorted((h.dense_score for h in hits), reverse=True)
    top = scores[0] if scores else 0.0
    gap = (scores[0] - scores[1]) if len(scores) > 1 else top
    return [len(words) / 10.0, float(len(_NUMBER.findall(q))), float(cues), float(clauses),
            float(docs), float(top), float(gap)]


def decide(score: float, threshold: float) -> str:
    return "large" if score >= threshold else "small"


class HeuristicRouter:
    """Each rule that fires adds its weight; the score is the capped sum."""

    name = "heuristic"

    def score(self, query: str, hits: Sequence[Hit]) -> float:
        words, numbers, cues, clauses, docs, top, gap = features(query, hits)
        s = 0.0
        s += 0.45 if cues >= 1 else 0.0
        s += 0.25 if cues >= 2 else 0.0
        s += 0.30 if docs >= 3 else 0.0
        s += 0.20 if numbers >= 1 else 0.0
        s += 0.15 if clauses >= 2 else 0.0
        s += 0.15 if words >= 1.6 else 0.0          # 16+ words
        s += 0.10 if gap < 0.03 else 0.0
        return min(1.0, s)


@dataclass
class LearnedRouter:
    """Logistic regression on standardised features. `weights` holds everything needed to score."""

    name: str = "learned"
    coef: list[float] = field(default_factory=lambda: [0.0] * len(FEATURE_NAMES))
    intercept: float = 0.0
    mean: list[float] = field(default_factory=lambda: [0.0] * len(FEATURE_NAMES))
    scale: list[float] = field(default_factory=lambda: [1.0] * len(FEATURE_NAMES))

    def score(self, query: str, hits: Sequence[Hit]) -> float:
        x = features(query, hits)
        z = self.intercept + sum(c * (v - m) / s for c, v, m, s in zip(self.coef, x, self.mean, self.scale))
        return 1.0 / (1.0 + math.exp(-z))

    def to_weights(self) -> dict:
        return {"features": list(FEATURE_NAMES), "coef": [round(c, 6) for c in self.coef],
                "intercept": round(self.intercept, 6), "mean": [round(m, 6) for m in self.mean],
                "scale": [round(s, 6) for s in self.scale]}

    @classmethod
    def from_weights(cls, w: dict) -> "LearnedRouter":
        return cls(coef=list(w["coef"]), intercept=float(w["intercept"]), mean=list(w["mean"]),
                   scale=list(w["scale"]))

    @classmethod
    def fit(cls, rows: Sequence[Sequence[float]], labels: Sequence[int], c: float = 1.0) -> "LearnedRouter":
        """Fits on feature rows (from `features`) and 0/1 labels (1 = needs the large model).
        Classes are re-weighted to balance, because hard questions are the minority."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        x = np.asarray(rows, dtype=float)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0] = 1.0
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=1000)
        model.fit((x - mean) / scale, np.asarray(labels))
        return cls(coef=[float(v) for v in model.coef_[0]], intercept=float(model.intercept_[0]),
                   mean=[float(v) for v in mean], scale=[float(v) for v in scale])
