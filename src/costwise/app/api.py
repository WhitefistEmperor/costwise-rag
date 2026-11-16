"""FastAPI entry point. Run: `uvicorn costwise.app.api:app --reload` (PYTHONPATH=src).

Endpoints: POST /ask {query}, GET /stats, GET /metrics (Prometheus text), GET /recent?n=20,
GET /cache, GET / (dashboard).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from ..settings import Settings
from .container import Container, build
from .metrics import CONTENT_TYPE as METRICS_CONTENT_TYPE
from .metrics import render as render_metrics

DASHBOARD = Path(__file__).with_name("dashboard.html")


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


def create_app(container: Container | None = None, settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = container or build(settings)
        yield

    app = FastAPI(title="Costwise RAG", lifespan=lifespan)

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        c: Container = app.state.container
        try:
            return asdict(c.pipeline.ask(req.query))
        except Exception as e:                       # provider errors surface as 502, not 500
            raise HTTPException(status_code=502, detail=str(e)) from e

    @app.get("/stats")
    def stats() -> dict:
        c: Container = app.state.container
        out = c.request_log.stats()
        out["llm_provider"] = c.settings.llm_provider
        out["mock"] = c.settings.llm_provider == "mock"
        out["models"] = {"small": c.settings.small_model, "large": c.settings.large_model}
        if c.cache is not None:
            out["cache"] = {"entries": len(c.cache), **asdict(c.cache.stats)}
        return out

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        c: Container = app.state.container
        body = render_metrics(c.request_log.stats(), llm_provider=c.settings.llm_provider,
                              small_model=c.settings.small_model, large_model=c.settings.large_model,
                              cache_entries=len(c.cache) if c.cache is not None else None,
                              cache_stats=asdict(c.cache.stats) if c.cache is not None else None)
        return PlainTextResponse(body, media_type=METRICS_CONTENT_TYPE)

    @app.get("/recent")
    def recent(n: int = 20) -> list[dict]:
        c: Container = app.state.container
        return c.request_log.recent(max(1, min(n, 200)))

    @app.get("/cache")
    def cache_entries() -> list[dict]:
        c: Container = app.state.container
        if c.cache is None:
            return []
        return [{"query": e.query, "model": e.model, "created": e.created, "sources": list(e.sources)}
                for e in c.cache.entries]

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return DASHBOARD.read_text(encoding="utf-8")

    return app


app = create_app()
