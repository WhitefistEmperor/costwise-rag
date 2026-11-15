"""Prints which MULTI_PART_CUES occur in TUNE-split texts. The cue list in domain/routing.py must
only contain cues that pass this check, so the router is never shaped by test-split questions.
Uses the router's own whole-word matcher."""
import json
from pathlib import Path

from costwise.domain.routing import MULTI_PART_CUES, matched_cues

qs = json.loads((Path(__file__).resolve().parents[1] / "data" / "questions.json").read_text(encoding="utf-8"))
tune = [t.lower() for q in qs if q["split"] == "tune" for t in [q["q"], *q["paraphrases"]]]
for cue in MULTI_PART_CUES:
    n = sum(cue in matched_cues(t) for t in tune)
    print(f"{'ok ' if n else 'NOT IN TUNE'} {cue!r}: {n} tune texts")
