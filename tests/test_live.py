from __future__ import annotations

import json

import pytest

from costwise.adapters.embedders import HashEmbedder
from costwise.adapters.llm import MockLLM
from costwise.eval import live
from costwise.eval.live import CONFIGS, BudgetedLLM, BudgetExceeded, run
from costwise.settings import Settings


def test_budget_is_checked_before_each_call():
    b = BudgetedLLM(MockLLM(), max_cost_usd=0.00005)
    b.complete("mock/small", "s", "short", 10)                 # tiny call fits
    with pytest.raises(BudgetExceeded):
        b.complete("mock/large", "s", "x" * 4000, 300)          # estimate alone exceeds the cap
    assert b.calls == 1 and 0 < b.spent_usd <= 0.00005


def test_mock_dry_run_covers_all_configs_and_is_marked_mock(live_settings):
    out = run(live_settings, MockLLM(), mock=True, max_cost_usd=1.0, seeds=(0, 1), embedder=HashEmbedder())
    assert out["mock"] is True and "NOT real" in out["note"] and set(out["configs"]) == set(CONFIGS)
    assert out["stream_queries"] > out["questions"] and out["stream_repeats"] >= 1
    for name, c in out["configs"].items():
        assert c["queries"] == out["stream_queries"] and c["stopped"] is None
        assert 0 <= c["correct"] <= c["queries"] and c["cost_usd"] > 0
    assert out["configs"]["baseline_always_large"]["routes"] == {"large": out["stream_queries"]}
    assert out["configs"]["always_small"]["routes"] == {"small": out["stream_queries"]}
    assert set(out["configs"]["router_only"]["routes"]) <= {"small", "large"}
    assert out["configs"]["cache_only"]["cache_hits"] >= out["questions"]   # second seed repeats every base text exactly
    assert out["configs"]["cache_only"]["cost_usd"] < out["configs"]["baseline_always_large"]["cost_usd"]
    assert out["total_llm_calls"] > 0 and "stopped" not in out


def test_run_stops_at_the_budget_cap_and_records_it(live_settings):
    out = run(live_settings, MockLLM(), mock=True, max_cost_usd=0.004, seeds=(0,), embedder=HashEmbedder())
    assert "stopped" in out and out["total_spent_usd"] <= 0.004
    first = out["configs"]["baseline_always_large"]
    assert first["stopped"] and first["queries"] < out["stream_queries"]


def test_cli_defaults_to_mock_and_writes_marked_file(live_settings, monkeypatch, tmp_path):
    monkeypatch.setattr(live, "Settings", lambda: live_settings)
    monkeypatch.setattr(live, "ROOT", tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = live.main(["--seeds", "1", "--configs", "full"])
    written = json.loads((tmp_path / "results" / "live_mock.json").read_text(encoding="utf-8"))
    assert written["mock"] is True and list(written["configs"]) == ["full"] and result["mock"] is True
    with pytest.raises(SystemExit):
        live.main(["--live"])                                   # no key: refuses before any call


def test_strong_verifier_flag_opens_the_band_and_bills_its_calls(live_settings):
    def respond(model, system, user):
        if "Question A:" in user:                               # verifier prompt
            return "YES"
        return user.split("Context:\n", 1)[-1].split("\n\nQuestion:", 1)[0]

    live_settings.tuned_path.write_text(json.dumps({"cache_threshold": 0.99, "cache_candidate_threshold": 0.3,
                                                    "cache_require_same_evidence": False, "route_threshold": 0.4}),
                                        encoding="utf-8")
    class Recording(MockLLM):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.billed = []                                    # (is verifier call, cost)

        def complete(self, model, system, user, max_tokens):
            r = super().complete(model, system, user, max_tokens)
            self.billed.append(("Question A:" in user, r.cost_usd))
            return r

    llm = Recording(respond=respond)
    out = run(live_settings, llm, mock=True, max_cost_usd=1.0, seeds=(0,),
              configs=["full"], embedder=HashEmbedder(), strong_verifier="llm")
    full = out["configs"]["full"]
    verifier_costs = [c for is_verifier, c in llm.billed if is_verifier]
    assert out["strong_verifier"] == "llm" and full["strong_verifier_calls"] == len(verifier_costs) > 0
    assert full["cache_hits"] > 0 and sum(verifier_costs) > 0
    # The config cost is every call, verifier calls included, and equals what the budget saw.
    assert full["cost_usd"] == pytest.approx(round(sum(c for _, c in llm.billed), 6), abs=1e-9)
    assert full["cost_usd"] == full["llm_spent_usd"] == out["total_spent_usd"]
    answers_only = sum(c for is_verifier, c in llm.billed if not is_verifier)
    assert full["cost_usd"] > round(answers_only, 6)
    off = run(live_settings, MockLLM(respond=respond), mock=True, max_cost_usd=1.0, seeds=(0,),
              configs=["full"], embedder=HashEmbedder())
    assert off["configs"]["full"]["strong_verifier_calls"] == 0
