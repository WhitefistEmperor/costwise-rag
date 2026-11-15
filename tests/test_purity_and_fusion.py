"""Hybrid fusion gives no rank credit to BM25 non-matches; domain/ has no I/O."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import costwise.domain

DOMAIN = Path(costwise.domain.__file__).parent
FORBIDDEN_MODULES = {"pathlib", "os", "io", "shutil", "sqlite3", "httpx", "requests", "socket", "glob", "urllib"}


def test_hybrid_with_no_bm25_match_ranks_by_dense_only(retrievers, embedder):
    q = "zzzz qqqq"                                   # no term in the corpus: every BM25 score is 0
    assert float(retrievers["bm25"].scores(q).sum()) == 0.0
    qvec = embedder.encode(["refund window"])[0]      # a real vector, so dense still ranks
    hybrid = retrievers["hybrid"].search(q, qvec, 4)
    dense = retrievers["dense"].search(q, qvec, 4)
    assert [h.chunk.id for h in hybrid] == [h.chunk.id for h in dense]
    assert [h.score for h in hybrid] == pytest.approx([1 / (60 + r) for r in (1, 2, 3, 4)])


def test_hybrid_gives_rank_credit_only_to_bm25_matches(retrievers, embedder):
    q = "Starter"                                     # BM25 matches only chunks containing "starter"
    hits = retrievers["hybrid"].search(q, embedder.encode([q])[0], 10)
    matched = {retrievers["bm25"].chunks[i].id for i, s in enumerate(retrievers["bm25"].scores(q)) if s > 0}
    for h in hits:
        if h.chunk.id not in matched:
            assert h.score <= 1 / 61 + 1e-12          # dense credit only


@pytest.mark.parametrize("path", sorted(DOMAIN.glob("*.py")), ids=lambda p: p.name)
def test_domain_modules_do_no_io_and_import_no_adapters(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
            assert not (node.level and (node.module or "").startswith(("adapters", "app", "eval"))), \
                f"{path.name} imports {node.module}"
            assert "adapters" not in (node.module or "").split("."), f"{path.name} imports {node.module}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            raise AssertionError(f"{path.name} calls open()")
        else:
            continue
        assert not set(names) & FORBIDDEN_MODULES, f"{path.name} imports {names}"


def test_corpus_loader_lives_in_adapters(corpus_dir):
    from costwise.adapters.corpus import load_corpus
    import costwise.domain.chunking as chunking
    assert not hasattr(chunking, "load_corpus")
    assert {c.doc for c in load_corpus(corpus_dir)} == {"pricing.md", "refunds.md", "backups.md"}
