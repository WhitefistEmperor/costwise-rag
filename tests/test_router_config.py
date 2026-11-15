"""Router threshold per router type, word-boundary cue matching, and tune-only fitting."""
from __future__ import annotations

import json

import pytest

import costwise.eval.offline as offline
from costwise.app.container import build, route_threshold_for
from costwise.domain.routing import cue_count, matched_cues
from costwise.eval.dataset import all_texts
from costwise.settings import Settings


@pytest.mark.parametrize("router,expected", [("heuristic", 0.3), ("learned", 0.6)])
def test_container_uses_the_threshold_of_the_selected_router(router, expected, corpus_dir, tmp_path):
    from costwise.domain.routing import LearnedRouter
    tuned = tmp_path / "tuned.json"
    tuned.write_text(json.dumps({"route_threshold": 0.6, "heuristic_route_threshold": 0.3,
                                 "router_weights": LearnedRouter().to_weights()}), encoding="utf-8")
    s = Settings(corpus_dir=corpus_dir, state_dir=tmp_path / "state", tuned_path=tuned, embed_model="hash",
                 llm_provider="mock", router=router)
    assert route_threshold_for(s, s.load_tuned()) == expected
    assert build(s, persistent=False).pipeline.route_threshold == expected


def test_route_threshold_falls_back_for_old_tuned_files():
    s = Settings(router="heuristic")
    assert route_threshold_for(s, {"route_threshold": 0.4}) == 0.4
    assert route_threshold_for(s, {}) == s.route_threshold


def test_cues_match_whole_words_only():
    assert "if a" not in matched_cues("is there a fee if any data moves")
    assert "and its" not in matched_cues("the plan itself and itself again")
    assert "if a" in matched_cues("what happens if a payment fails")
    assert "and its" in matched_cues("the cheapest plan and its price")
    assert "total" not in matched_cues("totally fine")
    assert cue_count("what is the total, and what does it cost") == 2      # "total", "and what"


def test_router_fit_and_thresholds_use_only_tune_rows(retrievers, embedder, monkeypatch):
    tune = [{"q": "How much does Starter cost?", "paraphrases": ["Starter price?"], "difficulty": "easy"},
            {"q": "Compare Starter and Team prices including 3 extra seats, and the refund window?",
             "paraphrases": ["Total for Team with 3 seats and the refund window, compared with Starter?"],
             "difficulty": "hard"}]
    test = [{"q": "How often is Team backed up?", "paraphrases": [], "difficulty": "easy"},
            {"q": "If I add 5 seats to Team and pay annually, what is the total and the refund window?",
             "paraphrases": [], "difficulty": "hard"}]
    fitted, tuned_on = [], []
    real_features, real_tune = offline.features, offline.tune_route_threshold
    monkeypatch.setattr(offline, "features", lambda text, hits: fitted.append(text) or real_features(text, hits))
    monkeypatch.setattr(offline, "tune_route_threshold",
                        lambda router, rows: tuned_on.append([t for t, _, _ in rows]) or real_tune(router, rows))
    out, tuned = offline.eval_router(retrievers["hybrid"], embedder, tune, test, 3)
    tune_texts = [t for q in tune for t in all_texts(q)]
    assert fitted == tune_texts
    assert tuned_on == [tune_texts, tune_texts]                # heuristic and learned thresholds
    assert tuned["heuristic_route_threshold"] == out["routers"]["heuristic"]["threshold_from_tune"]
    assert tuned["route_threshold"] == out["routers"]["learned"]["threshold_from_tune"]
