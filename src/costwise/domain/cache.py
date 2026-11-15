"""Semantic cache with three tiers: exact match, embedding similarity, then verifiers.

Two similarity thresholds:

  similarity >= threshold (the accept threshold)
      the cheap verifier (KeyTermGuard) alone decides;
  candidate_threshold <= similarity < threshold (the band)
      the cheap verifier must accept AND a strong verifier (an LLM or cross-encoder asked
      "same question?") must agree. At most one strong-verifier call per lookup, and its cost
      is reported so the pipeline can charge it to the request even on a miss.

Without a strong verifier the band is empty and only the accept threshold applies.

A semantic candidate only counts as a hit if it is fresh (TTL), its source documents are
unchanged and, when required, it was answered from the same evidence (same top retrieved chunk).
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from .models import CacheEntry
from .ports import Verifier

_SPACE = re.compile(r"\s+")


def normalise(query: str) -> str:
    return _SPACE.sub(" ", query.strip().lower()).rstrip("?!. ")


class CacheStore(Protocol):
    """Optional persistence for cache entries."""

    def load(self) -> list[tuple[CacheEntry, np.ndarray]]: ...
    def add(self, entry: CacheEntry, vec: np.ndarray) -> None: ...
    def remove(self, query: str) -> None: ...


@dataclass
class CacheHit:
    entry: CacheEntry
    tier: str            # "exact" | "semantic" | "verified" (band, strong verifier agreed)
    similarity: float
    verify_cost_usd: float = 0.0


@dataclass
class CacheStats:
    lookups: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    verified_hits: int = 0
    rejected_by_verifier: int = 0
    rejected_by_evidence: int = 0
    rejected_by_strong_verifier: int = 0
    strong_verifier_calls: int = 0
    strong_verifier_cost_usd: float = 0.0
    expired: int = 0
    invalidated: int = 0
    extra: dict = field(default_factory=dict)


class SemanticCache:
    def __init__(self, threshold: float, ttl_s: float = 24 * 3600, verifier: Verifier | None = None,
                 store: CacheStore | None = None, clock: Callable[[], float] = time.time,
                 max_candidates: int = 5, require_same_evidence: bool = False,
                 candidate_threshold: float | None = None, strong_verifier: Verifier | None = None):
        self.threshold, self.ttl_s, self.verifier = threshold, ttl_s, verifier
        self.store, self.clock, self.max_candidates = store, clock, max_candidates
        self.require_same_evidence = require_same_evidence
        self.strong_verifier = strong_verifier
        if candidate_threshold is not None and candidate_threshold > threshold:
            raise ValueError("candidate_threshold must not exceed threshold")
        self.candidate_threshold = candidate_threshold
        self.entries: list[CacheEntry] = []
        self.vectors: np.ndarray | None = None
        self.stats = CacheStats()
        self.last_lookup_cost_usd = 0.0
        self._lock = threading.Lock()
        for entry, vec in (store.load() if store else []):
            self._append(entry, vec)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def low_threshold(self) -> float:
        """Lowest similarity that is still considered; the band only exists with a strong verifier."""
        if self.strong_verifier is not None and self.candidate_threshold is not None:
            return self.candidate_threshold
        return self.threshold

    def _append(self, entry: CacheEntry, vec: np.ndarray) -> None:
        row = np.asarray(vec, dtype=np.float32)[None, :]
        self.vectors = row if self.vectors is None else np.vstack([self.vectors, row])
        self.entries.append(entry)

    def _remove(self, i: int) -> None:
        entry = self.entries.pop(i)
        self.vectors = np.delete(self.vectors, i, axis=0)
        if self.store:
            self.store.remove(entry.query)

    def _stale(self, entry: CacheEntry, current_docs: dict[str, str] | None) -> str | None:
        if self.clock() - entry.created > self.ttl_s:
            return "expired"
        if current_docs is not None and any(current_docs.get(d) != h for d, h in entry.sources.items()):
            return "invalidated"
        return None

    def lookup(self, query: str, qvec: np.ndarray, current_docs: dict[str, str] | None = None,
               evidence_key: str | None = None) -> CacheHit | None:
        """`current_docs` maps doc name to its current content hash; entries built from a doc
        that has since changed are dropped. `evidence_key` is the id of the top chunk retrieved
        for this query; with `require_same_evidence`, a semantic candidate must share it.
        After the call `last_lookup_cost_usd` holds what the strong verifier spent, hit or miss."""
        with self._lock:
            self.stats.lookups += 1
            self.last_lookup_cost_usd = 0.0
            if not self.entries:
                return None
            key = normalise(query)
            sims = self.vectors @ qvec
            order = [int(i) for i in np.argsort(-sims, kind="stable")[: self.max_candidates]]
            # An exact match is tried first whatever its rank.
            exact = [i for i, e in enumerate(self.entries) if normalise(e.query) == key]
            low = self.low_threshold
            candidates = exact + [i for i in order if i not in exact and sims[i] >= low]
            stale: list[int] = []
            hit: CacheHit | None = None
            strong_calls = 0
            for i in candidates:
                entry = self.entries[i]
                reason = self._stale(entry, current_docs)
                if reason:
                    setattr(self.stats, reason, getattr(self.stats, reason) + 1)
                    stale.append(i)
                    continue
                if i in exact:
                    self.stats.exact_hits += 1
                    hit = CacheHit(entry, "exact", float(sims[i]))
                    break
                if self.require_same_evidence and evidence_key != entry.evidence_key:
                    self.stats.rejected_by_evidence += 1
                    continue
                if self.verifier and not self.verifier.same_question(query, entry.query):
                    self.stats.rejected_by_verifier += 1
                    continue
                if sims[i] >= self.threshold:
                    self.stats.semantic_hits += 1
                    hit = CacheHit(entry, "semantic", float(sims[i]))
                    break
                # In the band: the strong verifier decides, once per lookup.
                if strong_calls >= 1:
                    continue
                strong_calls += 1
                self.stats.strong_verifier_calls += 1
                agreed = self.strong_verifier.same_question(query, entry.query)
                cost = float(getattr(self.strong_verifier, "last_cost_usd", 0.0))
                self.stats.strong_verifier_cost_usd += cost
                self.last_lookup_cost_usd += cost
                if not agreed:
                    self.stats.rejected_by_strong_verifier += 1
                    continue
                self.stats.verified_hits += 1
                hit = CacheHit(entry, "verified", float(sims[i]), verify_cost_usd=cost)
                break
            for i in sorted(stale, reverse=True):
                self._remove(i)
            return hit

    def put(self, entry: CacheEntry, qvec: np.ndarray) -> None:
        with self._lock:
            key = normalise(entry.query)
            for i, e in enumerate(self.entries):
                if normalise(e.query) == key:
                    self._remove(i)
                    break
            self._append(entry, qvec)
            if self.store:
                self.store.add(entry, qvec)
