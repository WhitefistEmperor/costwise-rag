"""scripts/load_test.py: counts, errors and percentiles, in-process with the mock model."""
from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("load_test", ROOT / "scripts" / "load_test.py")
load_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_test)


def test_run_load_counts_errors_and_concurrency():
    lock, state = threading.Lock(), {"now": 0, "peak": 0, "calls": 0}

    def send(query):
        with lock:
            state["now"] += 1
            state["calls"] += 1
            state["peak"] = max(state["peak"], state["now"])
            n = state["calls"]
        threading.Event().wait(0.01)
        with lock:
            state["now"] -= 1
        if n % 5 == 0:
            raise ConnectionError("boom")
        return (200, {"route": "small"}) if query != "bad" else (500, {})

    r = load_test.run_load(send, ["a", "b", "bad", "c"], n=20, concurrency=4)
    assert r["requests"] == 20 and r["ok"] + r["errors"] == 20 and r["errors"] >= 4
    assert sum(r["routes"].values()) == r["ok"] and set(r["routes"]) == {"small"}
    assert 1 < state["peak"] <= 4
    assert r["latency_ms_p50"] <= r["latency_ms_p95"] and r["throughput_rps"] > 0


def test_main_in_process_with_mock_model(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    r = load_test.main(["-n", "24", "-c", "4", "--embed-model", "hash"])
    assert r["mode"] == "in-process" and r["ok"] == 24 and r["errors"] == 0
    assert "MOCK" in r["note"] and r["latency_ms_p95"] is not None
    assert sum(r["routes"].values()) == 24
