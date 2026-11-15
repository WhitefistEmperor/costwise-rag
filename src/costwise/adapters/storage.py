"""SQLite persistence: cache entries and the per-request log."""
from __future__ import annotations

import json
import time
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..domain.models import Answer, CacheEntry


def _connect(path: Path | str | None) -> sqlite3.Connection:
    if path is None:
        return sqlite3.connect(":memory:", check_same_thread=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path), check_same_thread=False)


class SQLiteCacheStore:
    def __init__(self, path: Path | str | None = None):
        self.db, self._lock = _connect(path), threading.Lock()
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (query TEXT PRIMARY KEY, entry TEXT NOT NULL, "
                        "vec BLOB NOT NULL)")
        self.db.commit()

    def load(self) -> list[tuple[CacheEntry, np.ndarray]]:
        with self._lock:
            rows = self.db.execute("SELECT entry, vec FROM cache ORDER BY rowid").fetchall()
        return [(CacheEntry(**json.loads(e)), np.frombuffer(v, dtype=np.float32)) for e, v in rows]

    def add(self, entry: CacheEntry, vec: np.ndarray) -> None:
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                            (entry.query, json.dumps(asdict(entry)), np.asarray(vec, dtype=np.float32).tobytes()))
            self.db.commit()

    def remove(self, query: str) -> None:
        with self._lock:
            self.db.execute("DELETE FROM cache WHERE query = ?", (query,))
            self.db.commit()


class SQLiteRequestLog:
    """One row per answered request, plus the aggregates the dashboard shows."""

    COLUMNS = ("ts", "query", "route", "model", "cache_tier", "cache_similarity", "route_score", "escalated",
               "prompt_tokens", "completion_tokens", "cost_usd", "baseline_cost_usd", "latency_ms", "sources")

    def __init__(self, path: Path | str | None = None):
        self.db, self._lock = _connect(path), threading.Lock()
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, query TEXT, "
            "route TEXT, model TEXT, cache_tier TEXT, cache_similarity REAL, route_score REAL, escalated INTEGER, "
            "prompt_tokens INTEGER, completion_tokens INTEGER, cost_usd REAL, baseline_cost_usd REAL, "
            "latency_ms REAL, sources TEXT)")
        self.db.commit()

    def log(self, answer: Answer) -> None:
        row = (time.time(), answer.query, answer.route, answer.model, answer.cache_tier, answer.cache_similarity,
               answer.route_score, int(answer.escalated), answer.prompt_tokens, answer.completion_tokens,
               answer.cost_usd, answer.baseline_cost_usd, answer.latency_ms, json.dumps(answer.sources))
        with self._lock:
            self.db.execute(f"INSERT INTO requests ({', '.join(self.COLUMNS)}) VALUES ({', '.join('?' * len(row))})", row)
            self.db.commit()

    def recent(self, n: int = 20) -> list[dict]:
        with self._lock:
            rows = self.db.execute(f"SELECT {', '.join(self.COLUMNS)} FROM requests ORDER BY id DESC LIMIT ?",
                                   (n,)).fetchall()
        out = []
        for r in rows:
            d = dict(zip(self.COLUMNS, r))
            d["sources"] = json.loads(d["sources"] or "[]")
            d["escalated"] = bool(d["escalated"])
            out.append(d)
        return out

    def stats(self) -> dict:
        with self._lock:
            rows = self.db.execute("SELECT route, cost_usd, baseline_cost_usd, latency_ms, escalated, cache_tier "
                                   "FROM requests").fetchall()
        n = len(rows)
        lat = sorted(r[3] for r in rows)
        pct = lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))], 1) if lat else None
        cost = sum(r[1] for r in rows)
        baseline = sum(r[2] for r in rows)
        by_route = {}
        for r in rows:
            by_route[r[0]] = by_route.get(r[0], 0) + 1
        return {"requests": n, "cost_usd": round(cost, 6), "baseline_cost_usd": round(baseline, 6),
                "saving_usd": round(baseline - cost, 6),
                "saving_share": round((baseline - cost) / baseline, 4) if baseline else None,
                "by_route": by_route, "cache_hits": by_route.get("cache", 0),
                "escalations": sum(r[4] for r in rows), "latency_ms_p50": pct(0.5), "latency_ms_p95": pct(0.95)}
