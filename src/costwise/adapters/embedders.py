"""Embedding adapters."""
from __future__ import annotations

import re
import zlib
from typing import Sequence

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1.0, norms)).astype(np.float32)


class SentenceTransformerEmbedder:
    """Local sentence-transformers model, loaded on first use."""

    def __init__(self, model_name: str):
        self.name = model_name
        self._model = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name)
        vecs = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class HashEmbedder:
    """Deterministic bag-of-words embedding. For tests only: no model, no download."""

    def __init__(self, dim: int = 512):
        self.name, self.dim = f"hash-{dim}", dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in _WORD.findall(text.lower()):
                m[i, zlib.crc32(word.encode()) % self.dim] += 1.0
        return _normalise(m)
