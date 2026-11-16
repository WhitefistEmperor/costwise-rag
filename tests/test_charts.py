"""The chart script draws both PNGs from results/offline.json, offline."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
ROOT = Path(__file__).resolve().parents[1]


def test_make_charts_writes_both_pngs(tmp_path):
    spec = importlib.util.spec_from_file_location("make_charts", ROOT / "scripts" / "make_charts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    paths = mod.main([str(ROOT / "results" / "offline.json"), str(tmp_path)])
    assert [p.name for p in paths] == ["cache_thresholds.png", "router_tradeoff.png"]
    for p in paths:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and p.stat().st_size > 10_000
