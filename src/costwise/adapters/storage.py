"""SQLite persistence: cache entries and the per-request log."""
from __future__ import annotations

import json
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
