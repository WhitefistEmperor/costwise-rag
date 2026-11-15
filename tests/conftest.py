from __future__ import annotations

import pytest

from costwise.adapters.embedders import HashEmbedder
from costwise.adapters.retrieval import build_retrievers
from costwise.domain.chunking import load_corpus

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
