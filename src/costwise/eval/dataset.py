"""Eval question set: loading, answer grading and the query stream used for cache tests."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Query:
    qid: str          # id of the base question this text asks
    text: str
    hard: bool
    repeat: bool      # True if the same question appeared earlier in the stream


def load_questions(path: Path, split: str | None = None) -> list[dict]:
    qs = json.loads(Path(path).read_text(encoding="utf-8"))
    return [q for q in qs if split is None or q["split"] == split]


def all_texts(q: dict) -> list[str]:
    return [q["q"], *q["paraphrases"]]


def _has(answer: str, kw: str) -> bool:
    pat = re.escape(kw)
    if kw[0].isdigit():
        pat = r"(?<![\d.])" + pat      # "60" must not match inside "600" or "1.60"
    if kw[-1].isdigit():
        pat += r"(?!\d)"
    return re.search(pat, answer) is not None


def is_correct(answer: str, keywords: list) -> bool:
    """True if every keyword group appears in the answer. A group may list alternatives."""
    a = answer.lower().replace(",", "")
    return all(any(_has(a, alt.lower()) for alt in (g if isinstance(g, list) else [g])) for g in keywords)


def build_stream(questions: list[dict], repeat_rate: float, seed: int = 7) -> list[Query]:
    """Each base question once, in random order, plus paraphrases inserted after their base
    question until `repeat_rate` of the stream is repeats (capped by the paraphrases available)."""
    rng = random.Random(seed)
    base = list(questions)
    rng.shuffle(base)
    stream = [Query(q["id"], q["q"], q["difficulty"] == "hard", False) for q in base]
    pool = [(q, p) for q in base for p in q["paraphrases"]]
    rng.shuffle(pool)
    want = len(pool) if repeat_rate >= 1 else min(len(pool), round(repeat_rate * len(base) / (1 - repeat_rate)))
    for q, p in pool[:want]:
        first = next(i for i, s in enumerate(stream) if s.qid == q["id"])
        stream.insert(rng.randint(first + 1, len(stream)), Query(q["id"], p, q["difficulty"] == "hard", True))
    return stream
