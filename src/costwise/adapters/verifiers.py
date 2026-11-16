"""Cache verifiers: the second check that two similar-looking questions really are the same.

KeyTermGuard is the default. Its rule, written before any test-split result was looked at and
revised once from a diagnosis of the TUNE split only:

  1. The numbers in the two questions must be the same set. A question that adds a number
     ("Team with 12 seats and 1.3 TB") asks something different from the one without it.
     Numbers are found even when glued to letters ("800GB", "6h", "v2") and compared exactly
     ("3,000" == "3000", "1.30" == "1.3", no rounding).
  2. For all other key terms, the questions conflict when EACH has a term the other lacks (a
     substitution, such as "Team" for "Business"). One side merely adding a term is not a
     conflict, because paraphrases often add or drop words.

Key terms are:
  1. entity terms taken from the corpus alone: words written with a capital letter in the
     middle of a sentence (plan names, regions, products, acronyms);
  2. numbers in the question;
  3. a small configured list of contrast words that are not capitalised in the corpus
     (CONTRAST_GROUPS below). This list is hand-written domain config, not learned;
  4. the unit written right after a number, glued, spaced or hyphenated ("800GB", "800 GB", "90-day"), mapped
     to a canonical unit (UNITS below). Units follow the substitution rule, so "800GB" vs
     "800MB" conflict while "800GB" vs "800 GB" and "800GB" vs "800 of storage" do not.
"""
from __future__ import annotations

import re
import threading
from decimal import Decimal
from typing import Callable, Iterable, Sequence

from ..domain.models import Chunk
from ..domain.ports import LLMClient, Verdict

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:\.[0-9]+)?")
# A number anywhere, also glued to letters ("800GB", "6h", "v2"); "3,000" only in the thousands shape.
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_SENTENCE = re.compile(r"(?<=[.!?:;])\s+")
_LIST_MARK = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def number_key(text: str) -> str:
    """Exact canonical form of a number: "3,000" -> "3000", "1.30" -> "1.3". No rounding."""
    return format(Decimal(text.replace(",", "")).normalize(), "f")

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

# Unit word -> canonical unit. Only counted directly after a number.
UNITS: dict[str, str] = {
    **{w: "kb" for w in ("kb", "kilobyte", "kilobytes")},
    **{w: "mb" for w in ("mb", "megabyte", "megabytes")},
    **{w: "gb" for w in ("gb", "gigabyte", "gigabytes")},
    **{w: "tb" for w in ("tb", "terabyte", "terabytes")},
    **{w: "ms" for w in ("ms", "millisecond", "milliseconds")},
    **{w: "s" for w in ("s", "sec", "secs", "second", "seconds")},
    **{w: "min" for w in ("min", "mins", "minute", "minutes")},
    **{w: "h" for w in ("h", "hr", "hrs", "hour", "hours")},
    **{w: "day" for w in ("d", "day", "days")},
    **{w: "week" for w in ("wk", "wks", "week", "weeks")},
    **{w: "month" for w in ("mo", "month", "months")},
    **{w: "year" for w in ("yr", "yrs", "year", "years")},
}
_NUMBER_UNIT = re.compile(rf"(?:{_NUMBER.pattern})[\s-]?([a-z]+)")


def extract_entity_terms(chunks: Iterable[Chunk]) -> set[str]:
    """Lower-cased words that appear with a capital letter away from the start of a sentence.
    Each source line (chunk text keeps the document's line breaks) and each list item starts a
    sentence, so the first word of a bullet is not mistaken for a mid-sentence capital."""
    terms: set[str] = set()
    for chunk in chunks:
        body = chunk.text.split("\n", 1)[-1]          # drop the "Title - Heading" line
        for line in body.splitlines():
            for sentence in _SENTENCE.split(_LIST_MARK.sub("", line)):
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
        q = question.lower()
        out: set[str] = {f"num:{number_key(m.group())}" for m in _NUMBER.finditer(q)}
        out |= {f"unit:{UNITS[m.group(1)]}" for m in _NUMBER_UNIT.finditer(q) if m.group(1) in UNITS}
        for token in _TOKEN.findall(q):
            if token[0].isdigit():
                continue                                # numbers are handled above
            if token in self.entity_terms:
                out.add(token)
            elif token in _CONTRAST:
                out.add(_CONTRAST[token])
        return out

    @staticmethod
    def _split(terms: set[str]) -> tuple[set[str], set[str]]:
        nums = {t for t in terms if t.startswith("num:")}
        return nums, terms - nums

    def same_question(self, query: str, cached_query: str) -> bool:
        a_num, a = self._split(self.key_terms(query))
        b_num, b = self._split(self.key_terms(cached_query))
        if a_num != b_num:
            return False
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


class LLMVerifier:
    """Strong verifier: asks a small LLM whether two questions ask for the same information.

    Used only inside the cache's similarity band, at most once per lookup. One call is a few
    dozen tokens on the small model, roughly 0.5% of a large-model answer. `judge` returns the
    verdict together with the cost of that call, so the cache can charge it to the request
    without a shared "last cost" field that concurrent requests could overwrite.
    """

    name = "llm_verifier"
    SYSTEM = ("You judge whether two user questions ask for exactly the same information from the "
              "same product documentation. Questions about different plans, periods, regions, "
              "quantities or features are NOT the same. Answer with one word: YES or NO.")

    def __init__(self, llm: LLMClient, model: str, max_tokens: int = 3):
        self.llm, self.model, self.max_tokens = llm, model, max_tokens
        self.calls = 0
        self.cost_usd = 0.0
        self._lock = threading.Lock()

    def judge(self, query: str, cached_query: str) -> Verdict:
        user = f"Question A: {query}\nQuestion B: {cached_query}\nSame information? YES or NO."
        result = self.llm.complete(self.model, self.SYSTEM, user, self.max_tokens)
        with self._lock:
            self.calls += 1
            self.cost_usd += result.cost_usd
        return Verdict(result.text.strip().upper().startswith("YES"), result.cost_usd)

    def same_question(self, query: str, cached_query: str) -> bool:
        return self.judge(query, cached_query).same
