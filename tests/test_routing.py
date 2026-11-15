from __future__ import annotations

import costwise.domain.routing as routing
from costwise.domain.routing import FEATURE_NAMES, HeuristicRouter, LearnedRouter, decide, features
from costwise.eval.offline import eval_router, router_counts, tune_route_threshold


def search(retrievers, embedder, q, k=4):
    return retrievers["hybrid"].search(q, embedder.encode([q])[0], k)


def test_features_shape_and_values(retrievers, embedder):
    q = "Compare refund windows for monthly and annual plans, 7 or 30 days?"
    f = dict(zip(FEATURE_NAMES, features(q, search(retrievers, embedder, q))))
    assert f["numbers"] == 2 and f["cues"] >= 1 and f["clauses"] >= 2
    assert 1 <= f["docs_in_top_k"] <= 3 and -1 <= f["top_score"] <= 1 and f["score_gap"] >= 0
    assert len(f) == len(FEATURE_NAMES)


def test_heuristic_scores_multi_step_higher_than_lookup(retrievers, embedder):
    r = HeuristicRouter()
    easy = "How much does Starter cost per month?"
    hard = "What is the total monthly cost of Team with 12 seats, including backups, compared with Starter?"
    assert r.score(hard, search(retrievers, embedder, hard)) > r.score(easy, search(retrievers, embedder, easy))
    assert decide(0.7, 0.5) == "large" and decide(0.2, 0.5) == "small"


def test_learned_router_fits_separable_data_and_roundtrips_weights(monkeypatch):
    rows = [[0.5, 0, 0, 0, 1, 0.9, 0.2], [0.6, 0, 0, 1, 1, 0.8, 0.1], [0.7, 0, 0, 0, 2, 0.7, 0.2],
            [1.8, 2, 2, 3, 3, 0.5, 0.01], [2.0, 1, 1, 2, 2, 0.6, 0.02], [1.5, 3, 1, 2, 3, 0.4, 0.0]]
    labels = [0, 0, 0, 1, 1, 1]
    again = LearnedRouter.from_weights(LearnedRouter.fit(rows, labels).to_weights())
    monkeypatch.setattr(routing, "features", lambda q, hits: hits)    # feed feature rows straight in
    probs = [again.score("q", row) for row in rows]
    assert all(p < 0.5 for p in probs[:3]) and all(p > 0.5 for p in probs[3:])
    assert again.to_weights()["features"] == list(FEATURE_NAMES)


def test_router_counts_and_threshold_tuning():
    class Fixed:
        name = "fixed"

        def score(self, q, hits):
            return float(q)

    rows = [("0.9", [], 1), ("0.8", [], 1), ("0.3", [], 0), ("0.6", [], 0)]
    c = router_counts(Fixed(), rows, 0.7)
    assert c == {"threshold": 0.7, "queries": 4, "hard": 2, "easy": 2, "hard_to_large": 2, "easy_to_small": 2,
                 "sent_to_large": 2, "accuracy": 1.0, "hard_recall": 1.0, "easy_kept_small": 1.0}
    assert tune_route_threshold(Fixed(), rows) == 0.8    # highest threshold with perfect balanced accuracy


def test_eval_router_fits_on_tune_and_reports_test(retrievers, embedder):
    tune = [{"q": "How much does Starter cost?", "paraphrases": ["Starter price?"], "difficulty": "easy"},
            {"q": "Compare Starter and Team prices including 3 extra seats, and the refund window?",
             "paraphrases": ["Total for Team with 3 seats and the refund window, compared with Starter?"],
             "difficulty": "hard"}]
    test = [{"q": "How often is Team backed up?", "paraphrases": [], "difficulty": "easy"},
            {"q": "If I add 5 seats to Team and pay annually, what is the total and the refund window?",
             "paraphrases": [], "difficulty": "hard"}]
    out, tuned = eval_router(retrievers["hybrid"], embedder, tune, test, 3)
    assert set(out["routers"]) == {"heuristic", "learned"}
    assert out["routers"]["learned"]["test"]["queries"] == 2
    assert tuned["router_weights"]["features"] == list(FEATURE_NAMES) and 0 < tuned["route_threshold"] < 1
