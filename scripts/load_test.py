"""Load test: N /ask requests with C in flight at once; reports throughput and p50/p95 latency.

    # in-process app with the mock LLM (default; free, no server, no network)
    PYTHONPATH=src .venv/Scripts/python scripts/load_test.py -n 200 -c 8
    # against a running server (for example `docker compose up`)
    PYTHONPATH=src .venv/Scripts/python scripts/load_test.py -n 200 -c 8 --url http://127.0.0.1:8000

Queries cycle through every question text in data/questions.json, so later requests repeat
earlier ones and exercise the cache. With the mock model the numbers measure this service's own
overhead (embedding, retrieval, cache, routing, logging), not LLM latency, and the report says so.
In-process mode uses `--embed-model` (default "hash": no model download).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from costwise.domain.stats import percentile  # noqa: E402

Send = Callable[[str], tuple[int, dict]]


def load_queries(path: Path) -> list[str]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    return [t for q in questions for t in [q["q"], *q.get("paraphrases", [])]]


def run_load(send: Send, queries: Sequence[str], n: int, concurrency: int) -> dict:
    """Sends n requests (queries cycled) with `concurrency` workers; `send` returns (status, body)."""
    if n < 1 or concurrency < 1 or not queries:
        raise ValueError("n, concurrency and queries must be positive")

    def one(i: int) -> tuple[int, float, str | None]:
        t0 = time.perf_counter()
        try:
            status, body = send(queries[i % len(queries)])
            route = body.get("route") if status == 200 else None
        except Exception:                         # a failed request counts as an error, run goes on
            status, route = 0, None
        return status, (time.perf_counter() - t0) * 1000, route

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, range(n)))
    wall_s = time.perf_counter() - t0
    ok = [r for r in results if r[0] == 200]
    lat = [r[1] for r in ok]
    routes: dict[str, int] = {}
    for r in ok:
        routes[r[2]] = routes.get(r[2], 0) + 1
    pct = lambda p: None if not lat else round(percentile(lat, p), 1)
    return {"requests": n, "concurrency": concurrency, "ok": len(ok), "errors": n - len(ok),
            "wall_s": round(wall_s, 3), "throughput_rps": round(len(ok) / wall_s, 1) if wall_s else None,
            "latency_ms_p50": pct(0.5), "latency_ms_p95": pct(0.95), "routes": routes}


def in_process_sender(embed_model: str = "hash") -> tuple[Send, Callable[[], None], dict]:
    from fastapi.testclient import TestClient

    from costwise.app.api import create_app
    from costwise.app.container import build
    from costwise.settings import Settings

    settings = Settings(llm_provider="mock", embed_model=embed_model)
    app = create_app(build(settings, persistent=False))
    client = TestClient(app)
    client.__enter__()

    def send(query: str) -> tuple[int, dict]:
        r = client.post("/ask", json={"query": query})
        return r.status_code, r.json() if r.status_code == 200 else {}

    info = {"mode": "in-process", "llm_provider": "mock", "embed_model": embed_model}
    return send, lambda: client.__exit__(None, None, None), info


def http_sender(url: str, timeout_s: float = 60.0) -> tuple[Send, Callable[[], None], dict]:
    import httpx

    client = httpx.Client(base_url=url.rstrip("/"), timeout=timeout_s)
    try:
        provider = client.get("/stats").json().get("llm_provider", "unknown")
    except Exception:
        provider = "unknown"

    def send(query: str) -> tuple[int, dict]:
        r = client.post("/ask", json={"query": query})
        return r.status_code, r.json() if r.status_code == 200 else {}

    return send, client.close, {"mode": "http", "url": url, "llm_provider": provider}


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--requests", type=int, default=200)
    ap.add_argument("-c", "--concurrency", type=int, default=8)
    ap.add_argument("--url", default=None, help="hit a running server instead of the in-process app")
    ap.add_argument("--embed-model", default="hash", help="in-process only; 'hash' needs no download")
    ap.add_argument("--questions", default=str(ROOT / "data" / "questions.json"))
    args = ap.parse_args(argv)
    send, close, info = http_sender(args.url) if args.url else in_process_sender(args.embed_model)
    try:
        report = run_load(send, load_queries(Path(args.questions)), args.requests, args.concurrency)
    finally:
        close()
    report = {**info, **report}
    if info.get("llm_provider") == "mock":
        report["note"] = "MOCK LLM: latency is service overhead only, not real model latency."
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
