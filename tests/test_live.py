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


@pytest.fixture
def live_settings(corpus_dir, tmp_path):
    questions = [
        {"id": "price:starter", "q": "How much does Starter cost per month?",
         "paraphrases": ["What is the monthly price of Starter?", "Starter monthly cost?"],
         "keywords": ["9"], "docs": ["pricing.md"], "difficulty": "easy", "split": "test"},
        {"id": "refund:monthly", "q": "What is the refund window for monthly plans?",
         "paraphrases": ["Within how many days can monthly plans be refunded?"],
         "keywords": ["7"], "docs": ["refunds.md"], "difficulty": "easy", "split": "test"},
        {"id": "hard:0", "q": "Compare the total of Team with 2 extra seats and the backup frequency, 6 hours?",
         "paraphrases": ["Total for Team plus 2 seats, and how often is it backed up?"],
         "keywords": ["49"], "docs": ["pricing.md", "backups.md"], "difficulty": "hard", "split": "test"},
    ]
    qpath = tmp_path / "questions.json"
    qpath.write_text(json.dumps(questions), encoding="utf-8")
    tuned = tmp_path / "tuned.json"
    tuned.write_text(json.dumps({"cache_threshold": 0.95, "cache_candidate_threshold": 0.95,
                                 "cache_require_same_evidence": True, "route_threshold": 0.4}), encoding="utf-8")
    return Settings(corpus_dir=corpus_dir, questions_path=qpath, tuned_path=tuned, state_dir=tmp_path / "state",
                    embed_model="hash", router="heuristic", small_model="mock/small", large_model="mock/large")


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
