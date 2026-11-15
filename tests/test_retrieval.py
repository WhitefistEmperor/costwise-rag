from __future__ import annotations

import numpy as np
import pytest

from costwise.adapters.corpus import load_corpus
from costwise.domain.chunking import chunk_markdown, doc_hashes


def test_chunking_one_chunk_per_section_with_title():
    chunks = chunk_markdown("a.md", "# Title\n\n## One\nfirst body\n\n## Two\nsecond body\n", "h")
    assert [c.id for c in chunks] == ["a.md#0", "a.md#1"]
    assert chunks[0].text == "Title - One\nfirst body"
    assert chunks[1].doc_hash == "h"


def test_load_corpus_hashes_change_with_content(corpus_dir):
    before = doc_hashes(load_corpus(corpus_dir))
    (corpus_dir / "refunds.md").write_text("# Acme Refunds\n\n## Windows\nNo refunds.\n", encoding="utf-8")
    after = doc_hashes(load_corpus(corpus_dir))
    assert before["refunds.md"] != after["refunds.md"]
    assert before["pricing.md"] == after["pricing.md"]


def test_empty_corpus_raises(tmp_path):
    with pytest.raises(ValueError):
        load_corpus(tmp_path)


def test_embedder_rows_are_normalised(embedder):
    vecs = embedder.encode(["refund window", "backup frequency", ""])
    assert np.allclose(np.linalg.norm(vecs[:2], axis=1), 1.0)
    assert np.linalg.norm(vecs[2]) == 0.0


@pytest.mark.parametrize("name", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize("query,doc", [
    ("within how many days can monthly plans be refunded", "refunds.md"),
    ("how often is Starter backed up", "backups.md"),
    ("extra seats cost on Team", "pricing.md"),
])
def test_each_retriever_finds_the_right_doc(retrievers, embedder, name, query, doc):
    hits = retrievers[name].search(query, embedder.encode([query])[0], 2)
    assert hits[0].chunk.doc == doc
    assert len(hits) == 2


def test_hybrid_scores_are_rank_fusion_and_sorted(retrievers, embedder):
    q = "refunded within 30 days"
    hits = retrievers["hybrid"].search(q, embedder.encode([q])[0], 3)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] <= 2 / 61 + 1e-9          # best possible: rank 1 in both lists
    assert all(-1.0001 <= h.dense_score <= 1.0001 for h in hits)


def test_bm25_ignores_unknown_terms(retrievers):
    assert float(retrievers["bm25"].scores("zzzz qqqq").sum()) == 0.0
