"""Live-run robustness: provider errors, unknown prices, budget on every paid call, strict JSON."""
from __future__ import annotations

import json

import httpx
import pytest

from costwise.adapters.embedders import HashEmbedder
from costwise.adapters.llm import MockLLM, OpenRouterLLM, ProviderError
from costwise.adapters.verifiers import LLMVerifier
from costwise.domain.cache import SemanticCache
from costwise.domain.chunking import doc_hashes
from costwise.domain.pipeline import Pipeline
from costwise.eval import live
from costwise.eval.live import BudgetedLLM, BudgetExceeded, run


def strict_loads(text: str):
    def bad(name):
        raise ValueError(f"non-standard JSON constant {name}")
    return json.loads(text, parse_constant=bad)


def ok_response(text="x"):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}],
                                     "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.00001}})


def test_openrouter_retries_429_and_5xx_then_succeeds():
    replies = [httpx.Response(429), httpx.Response(503), ok_response("fine")]
    seen = []

    def handler(request):
        seen.append(1)
        return replies.pop(0)

    sleeps = []
    llm = OpenRouterLLM("k", transport=httpx.MockTransport(handler), max_retries=2, backoff_s=0.5,
                        sleep=sleeps.append)
    assert llm.complete("mock/small", "s", "u", 5).text == "fine"
    assert len(seen) == 3 and sleeps == [0.5, 1.0]


@pytest.mark.parametrize("status", [500, 501, 505, 529])
def test_openrouter_retries_every_5xx(status):
    replies = [httpx.Response(status), ok_response("fine")]
    llm = OpenRouterLLM("k", transport=httpx.MockTransport(lambda r: replies.pop(0)), sleep=lambda s: None)
    assert llm.complete("mock/small", "s", "u", 5).text == "fine" and replies == []


def test_unexpected_error_mid_config_keeps_partial_counts(live_settings):
    state = {"n": 0}

    def respond(model, system, user):
        state["n"] += 1
        if state["n"] > 2:
            raise KeyError("odd response shape")
        return user.split("Context:\n", 1)[-1].split("\n\nQuestion:", 1)[0]

    out = run(live_settings, MockLLM(respond=respond), mock=True, max_cost_usd=1.0, seeds=(0,),
              configs=["baseline_always_large", "always_small"], embedder=HashEmbedder())
    first = out["configs"]["baseline_always_large"]
    assert first["queries"] == 2 and first["correct"] >= 1 and "KeyError" in first["error"]
    assert out["stopped"] == "error during config 'baseline_always_large'"
    assert list(out["configs"]) == ["baseline_always_large"]


def test_openrouter_retries_timeouts_and_gives_up_with_provider_error():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    llm = OpenRouterLLM("secret-key", transport=httpx.MockTransport(handler), max_retries=2, sleep=lambda s: None)
    with pytest.raises(ProviderError) as e:
        llm.complete("mock/small", "s", "u", 5)
    assert len(calls) == 3 and "secret-key" not in str(e.value)


def test_openrouter_does_not_retry_other_4xx():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    llm = OpenRouterLLM("k", transport=httpx.MockTransport(handler), sleep=lambda s: None)
    with pytest.raises(ProviderError):
        llm.complete("mock/small", "s", "u", 5)
    assert len(calls) == 1


def test_openrouter_200_without_choices_is_a_clear_error():
    llm = OpenRouterLLM("k", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"error": {"message": "upstream overloaded"}})), sleep=lambda s: None)
    with pytest.raises(ProviderError, match="no choices"):
        llm.complete("mock/small", "s", "u", 5)


def test_each_retry_passes_the_budget_check():
    replies = [httpx.Response(500), ok_response()]
    llm = OpenRouterLLM("k", transport=httpx.MockTransport(lambda r: replies.pop(0)), sleep=lambda s: None)
    # Cap fits one estimated large call but not two: the retry must be refused.
    one = BudgetedLLM(MockLLM(), None)._estimate("mock/large", "s", "u" * 400, 300)
    budget = BudgetedLLM(llm, max_cost_usd=one * 1.5)
    with pytest.raises(BudgetExceeded):
        budget.complete("mock/large", "s", "u" * 400, 300)
    assert len(replies) == 1                                    # the retry was never sent


def test_provider_error_mid_run_keeps_partial_results_and_writes_file(live_settings, monkeypatch, tmp_path):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        return ok_response("Starter costs $9") if state["n"] <= 3 else httpx.Response(500)

    llm = OpenRouterLLM("k", transport=httpx.MockTransport(handler), max_retries=1, sleep=lambda s: None)
    monkeypatch.setattr(live, "Settings", lambda: live_settings)
    monkeypatch.setattr(live, "ROOT", tmp_path)
    monkeypatch.setattr(live, "build_llm", lambda settings: llm)
    monkeypatch.setattr(live, "build_embedder", lambda settings: HashEmbedder())
    live.main(["--live", "--seeds", "1", "--configs", "baseline_always_large", "always_small"])
    written = strict_loads((tmp_path / "results" / "live.json").read_text(encoding="utf-8"))
    first, second = written["configs"]["baseline_always_large"], written["configs"]["always_small"]
    assert first["queries"] == 3 and "HTTP 500" in first["error"]
    assert second["queries"] == 0 and second["error"]          # the run went on to the next config
    assert written["mock"] is False


def test_unknown_model_is_refused_before_any_paid_call(live_settings):
    live_settings.large_model = "someone/unpriced-model"
    llm = MockLLM()
    with pytest.raises(SystemExit, match="unpriced-model"):
        run(live_settings, llm, mock=False, max_cost_usd=1.0, seeds=(0,), embedder=HashEmbedder())
    assert llm.calls == []
    out = run(live_settings, llm, mock=False, max_cost_usd=1.0, seeds=(0,), embedder=HashEmbedder(),
              configs=["baseline_always_large"], prices={"someone/unpriced-model": (2.0, 10.0)})
    assert out["configs"]["baseline_always_large"]["stopped"] is None and out["total_spent_usd"] > 0


def test_cap_trips_on_an_escalation_and_records_the_small_call(live_settings):
    llm = MockLLM(abstain_if=lambda model, user: model == "mock/small")
    out = run(live_settings, llm, mock=True, max_cost_usd=0.001, seeds=(0,), configs=["router_only"],
              embedder=HashEmbedder())
    c = out["configs"]["router_only"]
    assert c["stopped"] and "stopped" in out
    assert [m for m, _ in llm.calls][-1] == "mock/small"       # the large call was refused, not made
    assert c["llm_spent_usd"] > 0 and c["llm_spent_usd"] == out["total_spent_usd"]


def test_cap_trips_on_a_strong_verifier_call(chunks, retrievers, embedder):
    inner = MockLLM(respond=lambda m, s, u: "YES" if "Question A:" in u else "Starter costs $9")
    budget = BudgetedLLM(inner, max_cost_usd=1.0)
    cache = SemanticCache(0.999, candidate_threshold=0.1, clock=lambda: 0.0,
                          strong_verifier=LLMVerifier(budget, "mock/small"))
    p = Pipeline(embedder, retrievers["hybrid"], budget, "mock/small", "mock/large", cache=cache,
                 default_route="large", doc_hashes=doc_hashes(chunks))
    p.ask("How much does Starter cost per month?")
    budget.max_cost_usd = budget.spent_usd + 1e-9               # nothing more fits
    with pytest.raises(BudgetExceeded):
        p.ask("What does Starter cost each month?")
    assert len(inner.calls) == 1                                # the verifier call was refused


def test_config_spend_reconciles_with_total(live_settings):
    out = run(live_settings, MockLLM(), mock=True, max_cost_usd=0.004, seeds=(0,), embedder=HashEmbedder())
    spent = sum(c["llm_spent_usd"] for c in out["configs"].values())
    assert abs(spent - out["total_spent_usd"]) < 3e-6 and len(out["configs"]) >= 1


def test_mock_cli_writes_strict_json_with_null_cap(live_settings, monkeypatch, tmp_path):
    monkeypatch.setattr(live, "Settings", lambda: live_settings)
    monkeypatch.setattr(live, "ROOT", tmp_path)
    live.main(["--seeds", "1", "--configs", "full"])
    written = strict_loads((tmp_path / "results" / "live_mock.json").read_text(encoding="utf-8"))
    assert written["max_cost_usd"] is None


@pytest.mark.parametrize("value", ["inf", "-inf", "nan", "-1"])
def test_max_cost_must_be_finite_and_not_negative(value, live_settings, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(live, "Settings", lambda: live_settings)
    monkeypatch.setattr(live, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as exc:
        live.main(["--seeds", "1", "--configs", "full", f"--max-cost={value}"])
    assert exc.value.code == 2
    assert "--max-cost" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()                  # refused before anything ran


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -0.01])
def test_budgeted_llm_refuses_a_blind_cap(value):
    with pytest.raises(ValueError):
        BudgetedLLM(MockLLM(), value)


def test_ctrl_c_mid_run_writes_partial_results(live_settings, monkeypatch, tmp_path):
    state = {"n": 0}

    def respond(model, system, user):
        state["n"] += 1
        if state["n"] > 2:
            raise KeyboardInterrupt
        return user.split("Context:\n", 1)[-1].split("\n\nQuestion:", 1)[0]

    monkeypatch.setattr(live, "Settings", lambda: live_settings)
    monkeypatch.setattr(live, "ROOT", tmp_path)
    monkeypatch.setattr(live, "MockLLM", lambda: MockLLM(respond=respond))
    with pytest.raises(SystemExit) as exc:
        live.main(["--seeds", "1", "--configs", "baseline_always_large", "always_small"])
    assert exc.value.code == 130
    written = strict_loads((tmp_path / "results" / "live_mock.json").read_text(encoding="utf-8"))
    first = written["configs"]["baseline_always_large"]
    assert first["queries"] == 2 and first["interrupted"] is True
    assert written["interrupted"] is True and "interrupted" in written["stopped"]
    assert list(written["configs"]) == ["baseline_always_large"]
