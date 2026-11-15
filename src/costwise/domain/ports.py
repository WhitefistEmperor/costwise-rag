"""Interfaces the core pipeline depends on. Adapters implement these."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .models import Answer, Hit, LLMResult


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Returns one L2-normalised float32 row per text."""


class Retriever(Protocol):
    def search(self, query: str, qvec: np.ndarray, k: int) -> list[Hit]: ...


class Verifier(Protocol):
    """Second check on a cache candidate: are these two questions asking the same thing?"""

    name: str

    def same_question(self, query: str, cached_query: str) -> bool: ...


@dataclass(frozen=True)
class Verdict:
    same: bool
    cost_usd: float = 0.0


class PaidVerifier(Verifier, Protocol):
    """A verifier whose check can cost money (an LLM). `judge` returns the verdict together with
    the cost of that one call, so concurrent requests never read each other's cost."""

    def judge(self, query: str, cached_query: str) -> Verdict: ...


class LLMClient(Protocol):
    def complete(self, model: str, system: str, user: str, max_tokens: int) -> LLMResult: ...


class Router(Protocol):
    name: str

    def score(self, query: str, hits: Sequence[Hit]) -> float:
        """Score in [0, 1] that the query needs the large model."""


class RequestLog(Protocol):
    def log(self, answer: Answer) -> None: ...
