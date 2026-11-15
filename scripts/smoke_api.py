"""End-to-end smoke test of the API with the real embedder and the MOCK model (free, offline).

Usage: PYTHONPATH=src .venv/Scripts/python scripts/smoke_api.py
"""
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from costwise.app.api import create_app
from costwise.app.container import build
from costwise.settings import Settings

settings = Settings(state_dir=Path(tempfile.mkdtemp()) / "state", llm_provider="mock")
app = create_app(build(settings, persistent=False))
with TestClient(app) as client:
    for q in ["How much does the Team plan cost per month?", "What is the monthly price of the Team plan?",
              "What is the total monthly cost of a Team plan with 12 seats and 1.3 TB of storage?"]:
        a = client.post("/ask", json={"query": q}).json()
        print(f"{a['route']:6} {a['model']:12} tier={a['cache_tier']} score={a['route_score']} "
              f"cost={a['cost_usd']:.6f} (mock, not real) | {q}")
    s = client.get("/stats").json()
    print("stats:", {k: s[k] for k in ("requests", "cache_hits", "by_route", "escalations")}, "mock =", s["mock"])
