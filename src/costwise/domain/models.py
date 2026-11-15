"""Plain data types shared across the system."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    id: str          # "<doc>#<n>"
    doc: str         # source file name
    doc_hash: str    # content hash of the source file, used for cache invalidation
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float         # ranking score of the retriever that produced the hit
    dense_score: float   # cosine similarity to the query, comparable across retrievers


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float


@dataclass
class CacheEntry:
    query: str
    answer: str
    sources: dict[str, str]      # doc name -> doc hash at the time the answer was made
    model: str
    created: float
    baseline_cost_usd: float     # what the large model would have cost for this answer
    tag: str | None = None       # eval only: id of the question that produced the entry
    evidence_key: str | None = None   # id of the top retrieved chunk when the answer was made


@dataclass
class Answer:
    query: str
    text: str
    route: str                   # "cache" | "small" | "large"
    model: str
    sources: list[str] = field(default_factory=list)
    cache_tier: str | None = None    # "exact" | "semantic" when route == "cache"
    cache_similarity: float | None = None
    cache_tag: str | None = None
    route_score: float | None = None
    escalated: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0
    latency_ms: float = 0.0
