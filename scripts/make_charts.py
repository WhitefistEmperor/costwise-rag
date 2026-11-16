"""Draws the offline-eval charts from results/offline.json (no model, no network).

    .venv/Scripts/python scripts/make_charts.py [results/offline.json] [docs/img]

Writes:
  cache_thresholds.png  correct and wrong cache hits on the test stream, per accept threshold
                        and cheap-verifier variant
  router_tradeoff.png   router accuracy against the share of questions sent to the large
                        model, across router thresholds; the tune-chosen threshold is marked
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"threshold_only": "threshold only", "key_term_guard": "KeyTermGuard",
          "same_evidence": "same evidence", "guard_and_evidence": "guard + evidence (used)"}


def cache_chart(data: dict, out: Path) -> None:
    variants = data["cache"]["variants"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    queries = repeats = None
    for name, v in variants.items():
        rows = sorted((float(t), r) for t, r in v["test_stream_by_threshold"].items())
        xs = [t for t, _ in rows]
        left.plot(xs, [r["correct_hits"] for _, r in rows], marker="o", ms=3, label=LABELS.get(name, name))
        right.plot(xs, [r["wrong_hits"] for _, r in rows], marker="o", ms=3, label=LABELS.get(name, name))
        queries, repeats = rows[0][1]["queries"], rows[0][1]["repeats"]
    accept = variants["guard_and_evidence"]["threshold_from_tune"]
    for ax, title in ((left, f"Correct hits (of {repeats} repeats)"), (right, f"Wrong hits (of {queries} queries)")):
        ax.axvline(accept, color="grey", ls=":", lw=1)
        ax.set_title(title)
        ax.set_xlabel("accept threshold (cosine similarity)")
        ax.grid(alpha=0.3)
    left.text(accept, left.get_ylim()[1] * 0.95, f" tuned {accept}", color="grey", va="top", fontsize=8)
    right.legend(fontsize=8)
    fig.suptitle("Semantic cache on the test stream (synthetic corpus, offline, no strong verifier)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def router_chart(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, r in data["router"]["routers"].items():
        rows = r["test_by_threshold"]
        share = [row["sent_to_large"] / row["queries"] for row in rows]
        ax.plot(share, [row["accuracy"] for row in rows], marker="o", ms=3, label=name)
        chosen = r["test"]
        ax.scatter([chosen["sent_to_large"] / chosen["queries"]], [chosen["accuracy"]], s=60, zorder=3,
                   edgecolor="black", label=f"{name}, tuned threshold {r['threshold_from_tune']}")
        hard = chosen["hard"] / chosen["queries"]
    ax.axvline(hard, color="grey", ls=":", lw=1)
    ax.text(hard, 0.02, " share labelled hard", color="grey", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xlabel("share of test questions sent to the large model")
    ax.set_ylabel("routing accuracy vs hand labels")
    ax.set_title("Router trade-off across thresholds (test split, 105 questions)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> list[Path]:
    argv = sys.argv[1:] if argv is None else argv
    src = Path(argv[0]) if argv else ROOT / "results" / "offline.json"
    out_dir = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(src.read_text(encoding="utf-8"))
    paths = [out_dir / "cache_thresholds.png", out_dir / "router_tradeoff.png"]
    cache_chart(data, paths[0])
    router_chart(data, paths[1])
    for p in paths:
        print(f"written: {p}")
    return paths


if __name__ == "__main__":
    main()
