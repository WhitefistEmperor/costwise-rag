"""Cache verifiers: the second check that two similar-looking questions really are the same.

KeyTermGuard is the default. Its rule, fixed before any test-split result was looked at:

  Two questions conflict when EACH has a key term the other lacks (a substitution, such as
  "Team" for "Business" or "800" for "500"). One side merely adding a term is not a conflict,
  because paraphrases often add or drop words.

Key terms are:
  1. entity terms taken from the corpus alone: words written with a capital letter in the
     middle of a sentence (plan names, regions, products, acronyms);
  2. numbers in the question;
  3. a small configured list of contrast words that are not capitalised in the corpus
     (CONTRAST_GROUPS below). This list is hand-written domain config, not learned.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable, Sequence

from ..domain.models import Chunk

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:\.[0-9]+)?")
_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_SENTENCE = re.compile(r"(?<=[.!?:;])\s+")

# word -> canonical contrast label. Words with different labels in the same group conflict.
CONTRAST_GROUPS: list[dict[str, str]] = [
    {"monthly": "billing:monthly", "annual": "billing:annual", "annually": "billing:annual",
     "yearly": "billing:annual"},
    {"daily": "freq:daily", "hourly": "freq:hourly", "weekly": "freq:weekly"},
    {"desktop": "client:desktop", "mobile": "client:mobile"},
    {"rest": "state:rest", "transit": "state:transit"},
    {"minimum": "bound:min", "maximum": "bound:max"},
]
_CONTRAST = {word: label for group in CONTRAST_GROUPS for word, label in group.items()}


def extract_entity_terms(chunks: Iterable[Chunk]) -> set[str]:
    """Lower-cased words that appear with a capital letter away from the start of a sentence."""
    terms: set[str] = set()
    for chunk in chunks:
        body = chunk.text.split("\n", 1)[-1]          # drop the "Title - Heading" line
        for sentence in _SENTENCE.split(body):
            for token in _TOKEN.findall(sentence)[1:]:
                if any(c.isupper() for c in token) and not token[0].isdigit():
                    terms.add(token.lower())
    return terms


class KeyTermGuard:
    name = "key_term_guard"

    def __init__(self, entity_terms: Iterable[str]):
        self.entity_terms = {t.lower() for t in entity_terms}

    @classmethod
    def from_corpus(cls, chunks: Iterable[Chunk]) -> "KeyTermGuard":
        return cls(extract_entity_terms(chunks))

    def key_terms(self, question: str) -> set[str]:
        out: set[str] = set()
        for token in _TOKEN.findall(question.lower()):
            if _NUMBER.fullmatch(token):
                out.add(f"num:{float(token):g}")
            elif token in self.entity_terms:
                out.add(token)
            elif token in _CONTRAST:
                out.add(_CONTRAST[token])
        return out

    def same_question(self, query: str, cached_query: str) -> bool:
        a, b = self.key_terms(query), self.key_terms(cached_query)
        return not (a - b and b - a)


class CrossEncoderVerifier:
    """Optional: scores the pair with a duplicate-question cross-encoder.

    Needs a model download on first use (about 300 MB), so it is off by default. Pass `scorer`
    to supply the pair-scoring function directly (used in tests).
    """

    name = "cross_encoder"

    def __init__(self, model_name: str = "cross-encoder/quora-distilroberta-base", threshold: float = 0.5,
                 scorer: Callable[[Sequence[tuple[str, str]]], Sequence[float]] | None = None):
        self.model_name, self.threshold, self._scorer = model_name, threshold, scorer

    def same_question(self, query: str, cached_query: str) -> bool:
        if self._scorer is None:
            from sentence_transformers import CrossEncoder

            self._scorer = CrossEncoder(self.model_name).predict
        return float(self._scorer([(query, cached_query)])[0]) >= self.threshold
