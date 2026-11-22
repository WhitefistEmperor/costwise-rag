"""Ranking metrics against hand-computed values, and the inverted-index BM25 against the plain
formula. No downloads."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from costwise.adapters.retrieval import BM25Index, tokenize
from costwise.domain.models import Chunk
from costwise.eval.ir_metrics import bootstrap_ci, mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k


def test_ndcg_binary_and_graded():
    assert ndcg_at_k(["a", "b"], {"a": 1}) == 1.0
    assert ndcg_at_k(["x", "a"], {"a": 1}) == pytest.approx(1 / math.log2(3))
    # graded: relevance 2 at rank 2, 1 at rank 1; ideal puts the 2 first
    dcg = 1 / 1 + 3 / math.log2(3)
    idcg = 3 / 1 + 1 / math.log2(3)
    assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["x"], {"a": 0}) == 0.0                     # no relevant judged doc
    assert ndcg_at_k(["a"] + ["x"] * 20, {"a": 1}, k=10) == 1.0


def test_recall_and_mrr():
    assert recall_at_k(["a", "x", "b"], {"a": 1, "b": 1, "c": 1}, 2) == pytest.approx(1 / 3)
    assert recall_at_k(["a"], {"a": 0}, 5) == 0.0
    assert mrr_at_k(["x", "y", "a"], {"a": 1}) == pytest.approx(1 / 3)
    assert mrr_at_k(["x"] * 10 + ["a"], {"a": 1}, k=10) == 0.0


def test_bootstrap_ci_contains_mean_and_paired_detects_a_clear_win():
    xs = [0.0, 1.0] * 50
    lo, hi = bootstrap_ci(xs)
    assert lo < 0.5 < hi
    better = paired_bootstrap([1.0] * 40, [0.0] * 40)
    assert better["mean_diff"] == 1.0 and better["p_not_better"] == 0.0
    tie = paired_bootstrap([0.5] * 40, [0.5] * 40)
    assert tie["p_not_better"] == 1.0


def naive_bm25(chunks, query, k1=1.5, b=0.75):
    tfs = [Counter(tokenize(c.text)) for c in chunks]
    lengths = [sum(t.values()) for t in tfs]
    avg = sum(lengths) / len(lengths)
    df = Counter(t for tf in tfs for t in tf)
    n = len(chunks)
    out = []
    for tf, length in zip(tfs, lengths):
        s = 0.0
        for term in set(tokenize(query)):
            if term in df and tf.get(term):
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                f = tf[term]
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * length / avg))
        out.append(s)
    return np.array(out)


def test_inverted_index_bm25_matches_the_formula():
    texts = ["refund window for monthly plans", "annual plan refund refund policy", "export data to csv",
             "the team plan has 12 seats", "monthly monthly monthly billing", "nothing here"]
    chunks = [Chunk(str(i), str(i), "", t) for i, t in enumerate(texts)]
    index = BM25Index(chunks)
    for q in ["monthly refund", "team seats csv", "unknown words only", "refund refund"]:
        np.testing.assert_allclose(index.scores(q), naive_bm25(chunks, q), rtol=1e-5, atol=1e-6)


def test_bm25_empty_corpus_does_not_crash():
    assert BM25Index([]).scores("anything").shape == (0,)
