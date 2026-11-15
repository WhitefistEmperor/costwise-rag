"""Lists cache mistakes on the TUNE split only, so design choices never look at test data.

Usage: python scripts/diagnose_cache.py [threshold]
"""
import sys

from costwise.adapters.embedders import SentenceTransformerEmbedder
from costwise.adapters.retrieval import build_retrievers
from costwise.adapters.verifiers import KeyTermGuard
from costwise.adapters.corpus import load_corpus
from costwise.eval.dataset import load_questions
from costwise.eval.offline import embed_questions, question_pairs, top_chunks
from costwise.settings import Settings

threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.70
s = Settings()
embedder = SentenceTransformerEmbedder(s.embed_model)
chunks = load_corpus(s.corpus_dir)
guard = KeyTermGuard.from_corpus(chunks)
tune = load_questions(s.questions_path, "tune")
vecs = embed_questions(embedder, tune)
ev = top_chunks(build_retrievers(chunks, embedder)["hybrid"], vecs)
pairs = sorted(question_pairs(tune, vecs), reverse=True)

print(f"entity terms ({len(guard.entity_terms)}):", sorted(guard.entity_terms))
print(f"\n-- false accepts at {threshold} (different questions that pass guard + evidence)")
for sim, same, a, b in pairs:
    if not same and sim >= threshold and guard.same_question(a, b) and ev[a] == ev[b]:
        print(f"  {sim:.2f} | {a} | {b} | {ev[a]}")
same = [p for p in pairs if p[1]]
print(f"\n-- same-question pairs: {len(same)}")
for lo, hi in [(0.9, 1.01), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.0, 0.6)]:
    print(f"  similarity {lo:.1f}-{hi:.1f}: {sum(lo <= p[0] < hi for p in same)}")
print("\n-- same-question pairs rejected by the guard")
for sim, _, a, b in same:
    if not guard.same_question(a, b):
        print(f"  {sim:.2f} | {a} | {b} | {guard.key_terms(a) ^ guard.key_terms(b)}")
print("\n-- same-question pairs with different top chunk:", sum(ev[a] != ev[b] for _, _, a, b in same))
