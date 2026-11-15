"""Retrievers: dense (cosine), BM25 (lexical) and a hybrid that fuses both by rank."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

from ..domain.models import Chunk, Hit
from ..domain.ports import Embedder

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class DenseIndex:
    """Exact cosine search over normalised vectors. Fine up to roughly 100k chunks."""

    def __init__(self, chunks: Sequence[Chunk], vectors: np.ndarray):
        self.chunks, self.vectors = list(chunks), vectors

    @classmethod
    def build(cls, chunks: Sequence[Chunk], embedder: Embedder) -> "DenseIndex":
        return cls(chunks, embedder.encode([c.text for c in chunks]))

    def scores(self, qvec: np.ndarray) -> np.ndarray:
        return self.vectors @ qvec

    def search(self, query: str, qvec: np.ndarray, k: int) -> list[Hit]:
        s = self.scores(qvec)
        return [Hit(self.chunks[i], float(s[i]), float(s[i])) for i in np.argsort(-s, kind="stable")[:k]]


class BM25Index:
    """Okapi BM25."""

    def __init__(self, chunks: Sequence[Chunk], dense: DenseIndex | None = None, k1: float = 1.5, b: float = 0.75):
        self.chunks, self.dense, self.k1, self.b = list(chunks), dense, k1, b
        self.tf = [Counter(tokenize(c.text)) for c in self.chunks]
        self.lengths = np.array([sum(tf.values()) for tf in self.tf], dtype=np.float32)
        self.avg_len = float(self.lengths.mean())
        n = len(self.chunks)
        df = Counter(term for tf in self.tf for term in tf)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.chunks), dtype=np.float32)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                    out[i] += idf * f * (self.k1 + 1) / (f + norm)
        return out

    def search(self, query: str, qvec: np.ndarray, k: int) -> list[Hit]:
        s = self.scores(query)
        d = self.dense.scores(qvec) if self.dense else np.zeros_like(s)
        return [Hit(self.chunks[i], float(s[i]), float(d[i])) for i in np.argsort(-s, kind="stable")[:k]]


class HybridRetriever:
    """Reciprocal rank fusion of the dense and BM25 rankings: score = sum(1 / (rrf_k + rank))."""

    def __init__(self, dense: DenseIndex, bm25: BM25Index, rrf_k: int = 60, depth: int = 20):
        self.dense, self.bm25, self.rrf_k, self.depth = dense, bm25, rrf_k, depth

    def search(self, query: str, qvec: np.ndarray, k: int) -> list[Hit]:
        d, b = self.dense.scores(qvec), self.bm25.scores(query)
        fused: dict[int, float] = {}
        for ranking in (np.argsort(-d, kind="stable")[: self.depth], np.argsort(-b, kind="stable")[: self.depth]):
            for rank, i in enumerate(ranking):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (self.rrf_k + rank + 1)
        top = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        return [Hit(self.dense.chunks[i], fused[i], float(d[i])) for i in top]


def build_retrievers(chunks: Sequence[Chunk], embedder: Embedder) -> dict[str, object]:
    dense = DenseIndex.build(chunks, embedder)
    bm25 = BM25Index(chunks, dense)
    return {"dense": dense, "bm25": bm25, "hybrid": HybridRetriever(dense, bm25)}
