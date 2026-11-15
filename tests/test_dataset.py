from __future__ import annotations

from costwise.eval.dataset import build_stream, is_correct, load_questions
from costwise.settings import Settings


def test_is_correct_number_boundaries():
    assert is_correct("The limit is 60 requests per minute.", ["60"])
    assert not is_correct("The limit is 600 requests per minute.", ["60"])
    assert not is_correct("It costs $1.60.", ["60"])
    assert is_correct("That comes to $1,910.40 per year.", [["1910.4", "1910.40"]])
    assert is_correct("180,000 requests", ["180000"])


def test_is_correct_needs_every_group_and_accepts_alternatives():
    assert is_correct("Starter is daily, Business every hour.", [["daily"], ["hourly", "every hour"]])
    assert not is_correct("Starter is daily.", [["daily"], ["hourly", "every hour"]])
    assert is_correct("TEAM or BUSINESS", ["team", "business"])


def test_question_set_shape():
    qs = load_questions(Settings().questions_path)
    assert len({q["id"] for q in qs}) == len(qs) == 60
    assert all(len(q["paraphrases"]) == 2 and q["keywords"] and q["docs"] for q in qs)
    assert {q["split"] for q in qs} == {"tune", "test"}


def test_stream_repeat_rate_and_order():
    qs = load_questions(Settings().questions_path, "test")
    stream = build_stream(qs, repeat_rate=0.5, seed=1)
    repeats = [s for s in stream if s.repeat]
    assert len(stream) == 2 * len(qs) and len(repeats) == len(qs)
    seen = set()
    for s in stream:
        assert (s.qid in seen) == s.repeat      # a repeat always follows its base question
        seen.add(s.qid)
    assert build_stream(qs, 0.5, seed=1) == stream
