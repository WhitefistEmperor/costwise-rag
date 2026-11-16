"""Builds a Pipeline and its adapters from Settings. The only place that reads the API key."""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..adapters.embedders import HashEmbedder, SentenceTransformerEmbedder
from ..adapters.llm import MockLLM, OpenRouterLLM, ZaiLLM
from ..adapters.retrieval import build_retrievers
from ..adapters.storage import SQLiteCacheStore, SQLiteRequestLog
from ..adapters.verifiers import KeyTermGuard, LLMVerifier
from ..domain.cache import SemanticCache
from ..adapters.corpus import load_corpus
from ..domain.chunking import doc_hashes
from ..domain.models import Chunk
from ..domain.pipeline import Pipeline
from ..domain.ports import Embedder, LLMClient, Router
from ..domain.routing import HeuristicRouter, LearnedRouter
from ..settings import Settings


@dataclass
class Container:
    settings: Settings
    pipeline: Pipeline
    cache: SemanticCache | None
    request_log: SQLiteRequestLog
    llm: LLMClient
    chunks: list[Chunk]


def build_embedder(settings: Settings) -> Embedder:
    return HashEmbedder() if settings.embed_model == "hash" else SentenceTransformerEmbedder(settings.embed_model)


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise RuntimeError("COSTWISE_LLM=openrouter but OPENROUTER_API_KEY is not set in the environment")
        return OpenRouterLLM(key, settings.openrouter_base_url)
    if settings.llm_provider == "zai":
        key = os.environ.get("ZAI_API_KEY", "")
        if not key:
            raise RuntimeError("COSTWISE_LLM=zai but ZAI_API_KEY is not set in the environment")
        return ZaiLLM(key, settings.zai_base_url)
    if settings.llm_provider == "mock":
        return MockLLM()
    raise ValueError(f"unknown llm provider {settings.llm_provider!r} (mock | openrouter | zai)")


def build_router(settings: Settings, tuned: dict) -> Router | None:
    if settings.router == "none":
        return None
    if settings.router == "heuristic":
        return HeuristicRouter()
    if settings.router == "learned":
        if "router_weights" not in tuned:
            raise RuntimeError("no router weights in data/tuned.json; run `python -m costwise.eval.offline` "
                               "or set COSTWISE_ROUTER=heuristic")
        return LearnedRouter.from_weights(tuned["router_weights"])
    raise ValueError(f"unknown router {settings.router!r} (learned | heuristic | none)")


def route_threshold_for(settings: Settings, tuned: dict) -> float:
    """The tune-chosen threshold of the selected router. data/tuned.json keeps the learned
    router's under `route_threshold` and the heuristic's under `heuristic_route_threshold`;
    an older file with only `route_threshold` is used for both."""
    learned = tuned.get("route_threshold", settings.route_threshold)
    if settings.router == "heuristic":
        return tuned.get("heuristic_route_threshold", learned)
    return learned


def build_cache(settings: Settings, tuned: dict, chunks: list[Chunk], llm: LLMClient,
                persistent: bool = True) -> SemanticCache | None:
    if not settings.cache_enabled:
        return None
    strong = LLMVerifier(llm, settings.small_model) if settings.cache_strong_verifier == "llm" else None
    return SemanticCache(
        threshold=tuned.get("cache_threshold", settings.cache_threshold),
        candidate_threshold=tuned.get("cache_candidate_threshold", settings.cache_candidate_threshold),
        ttl_s=settings.cache_ttl_s, verifier=KeyTermGuard.from_corpus(chunks), strong_verifier=strong,
        require_same_evidence=tuned.get("cache_require_same_evidence", True),
        store=SQLiteCacheStore(settings.cache_db) if persistent else None)


def build(settings: Settings | None = None, embedder: Embedder | None = None, llm: LLMClient | None = None,
          persistent: bool = True) -> Container:
    settings = settings or Settings()
    tuned = settings.load_tuned()
    embedder = embedder or build_embedder(settings)
    llm = llm or build_llm(settings)
    chunks = load_corpus(settings.corpus_dir)
    retriever = build_retrievers(chunks, embedder)["hybrid"]
    cache = build_cache(settings, tuned, chunks, llm, persistent)
    log = SQLiteRequestLog(settings.request_log_db if persistent else None)
    pipeline = Pipeline(embedder, retriever, llm, settings.small_model, settings.large_model, cache=cache,
                        router=build_router(settings, tuned), default_route=settings.default_route,
                        route_threshold=route_threshold_for(settings, tuned),
                        top_k=settings.top_k, max_answer_tokens=settings.max_answer_tokens,
                        doc_hashes=doc_hashes(chunks), request_log=log)
    return Container(settings, pipeline, cache, log, llm, chunks)
