"""Pure parts of the public-benchmark evals (no downloads, no models)."""
from __future__ import annotations

import numpy as np
import pytest

from costwise.eval import paraphrase as pp
from costwise.eval import routerbench as rb


def test_counts_precision_recall():
    accept = np.array([True, True, False, True])
    y = np.array([1, 0, 1, 1])
    assert pp.counts(accept, y) == {"served": 3, "right": 2, "wrong": 1, "duplicates": 3, "pairs": 4,
                                    "precision": round(2 / 3, 4), "recall": round(2 / 3, 4)}


def test_best_on_tune_respects_the_precision_target():
    y = np.array([1, 1, 0, 1])
    loose = ({"t": 0.1}, np.array([True, True, True, True]))     # P 0.75, R 1.0
    strict = ({"t": 0.9}, np.array([True, False, False, False]))  # P 1.0,  R 0.33
    assert pp.best_on_tune([loose, strict], y, 0.9) == {"t": 0.9}
    assert pp.best_on_tune([loose, strict], y, 0.7) == {"t": 0.1}
    assert pp.best_on_tune([loose], y, 0.9) is None


class FakeScored:
    def __init__(self):
        self.cos = np.array([0.95, 0.80, 0.80, 0.50])
        self.guard = np.array([True, True, False, True])
        self.y = np.array([1, 1, 1, 0])
        self.ce = {"m": np.array([0.1, 0.9, 0.9, 0.9])}


def test_cascade_serves_above_accept_and_asks_the_ce_only_in_the_band():
    mask, calls = pp.apply("cascade[m]", {"accept": 0.9, "candidate": 0.7, "score": 0.5}, FakeScored())
    assert mask.tolist() == [True, True, False, False]           # row 3 blocked by the guard
    assert calls == 1                                            # only row 2 is in the band and passes the guard
    mask, calls = pp.apply("cross_encoder[m]", {"score": 0.5}, FakeScored())
    assert mask.tolist() == [False, True, True, True] and calls == 4


def test_router_curve_random_and_oracle():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    q_oracle = rb.curve(y.copy(), y)
    weak = 0.8
    assert q_oracle[0] == pytest.approx(weak) and q_oracle[-1] == pytest.approx(1.0)
    assert q_oracle[20] == pytest.approx(1.0)                    # 20% to large fixes both failures
    s = rb.summarise(q_oracle, weak, 1.0)
    assert 0.1 < s["cpt80"] <= 0.2 and s["apgr"] > 0.85          # 10 rows: 15% rounds to 2 calls
    worst = rb.curve(-y, y)                                      # failures routed last
    assert rb.summarise(worst, weak, 1.0)["apgr"] < 0.2


def test_simulated_cost_uses_list_prices():
    row = {"prompt": "x" * 400, "large_chars": 4000, "small_chars": 4000}
    assert rb.simulated_cost(row, "large") == pytest.approx((100 * 10 + 1000 * 30) / 1e6)
    assert rb.simulated_cost(row, "small") == pytest.approx((100 * 0.6 + 1000 * 0.6) / 1e6)
