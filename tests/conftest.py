from __future__ import annotations

import json

import pytest

from costwise.adapters.embedders import HashEmbedder
from costwise.adapters.retrieval import build_retrievers
from costwise.adapters.corpus import load_corpus
from costwise.settings import Settings

DOCS = {
    "pricing.md": "# Acme Pricing\n\n## Plans\nStarter costs $9 per month. Team costs $49 per month.\n\n"
                  "## Add-ons\nExtra seats cost $6 per seat on Team.\n",
    "refunds.md": "# Acme Refunds\n\n## Windows\nMonthly plans can be refunded within 7 days. "
                  "Annual plans can be refunded within 30 days.\n",
    "backups.md": "# Acme Backups\n\n## Frequency\nStarter is backed up daily. Team is backed up every 6 hours.\n",
}


@pytest.fixture
def corpus_dir(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    for name, text in DOCS.items():
        (d / name).write_text(text, encoding="utf-8")
    return d


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def chunks(corpus_dir):
    return load_corpus(corpus_dir)


@pytest.fixture
def retrievers(chunks, embedder):
    return build_retrievers(chunks, embedder)


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
