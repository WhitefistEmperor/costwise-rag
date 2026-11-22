"""Draws the public-benchmark charts from results/*.json (no model, no network).

    .venv/Scripts/python scripts/make_bench_charts.py [results] [docs/img]

Writes, for each results file present:
  beir_ndcg.png          nDCG@10 with 95% CIs per retriever on SciFact and NFCorpus (beir.json)
  router_curve.png       quality vs share sent to the large model on RouteLLM data (routerbench.json)
  paraphrase_ap.png      average precision of each cache-decision score on QQP and PAWS (paraphrase.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def beir_chart(data: dict, out: Path) -> None:
    sets = list(data["datasets"])
    fig, axes = plt.subplots(1, len(sets), figsize=(5 * len(sets), 3.8), squeeze=False)
    for ax, name in zip(axes[0], sets):
        d = data["datasets"][name]
        methods = list(d["methods"])
        vals = [d["methods"][m]["ndcg@10"] for m in methods]
        err = [[v - d["methods"][m]["ndcg@10_ci95"][0] for m, v in zip(methods, vals)],
               [d["methods"][m]["ndcg@10_ci95"][1] - v for m, v in zip(methods, vals)]]
        ax.bar(methods, vals, yerr=err, capsize=4, color=["#8c8c8c", "#5b8bd0", "#3c6db0", "#1f3f7a"][: len(methods)])
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_title(f"{name} ({d['documents']} docs, {d['test_queries']} test queries)", fontsize=9)
        ax.set_ylabel("nDCG@10 (95% CI)")
        ax.tick_params(axis="x", labelsize=7, rotation=15)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Retrieval on BEIR test qrels (depths chosen on train/dev)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def router_chart(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = data["curve_fractions"]
    ax.plot(xs, data["random"]["curve"], ls="--", color="grey", label=f"random (APGR {data['random']['apgr']})")
    for name, r in data["routers"].items():
        ax.plot(xs, r["curve"], marker="o", ms=2, label=f"{name} (APGR {r['apgr']})")
    ax.axhline(0.95, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("share of prompts sent to the large model (GPT-4)")
    ax.set_ylabel("share of acceptable answers")
    ax.set_title(f"Routing on RouteLLM gpt4_dataset ({data['dataset']['test']} held-out prompts)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def paraphrase_chart(data: dict, out: Path) -> None:
    """Average precision of each cache-decision score on the test pairs (no threshold needed);
    the duplicate share is what a random score would get."""
    sets = list(data["datasets"])
    names = [k for k in data["datasets"][sets[0]]["test_average_precision"] if "+guard[" not in k]
    short = lambda k: (k.replace("cross_encoder[", "").replace("]", "").replace("cross-encoder/", "")
                       .replace("-distilroberta-base", " CE").replace("fine-tuned/verifier-minilm-paws-qqp",
                                                                       "fine-tuned MiniLM (ours)"))
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(sets)
    for j, name in enumerate(sets):
        d = data["datasets"][name]
        vals = [d["test_average_precision"][k] for k in names]
        xs = [i + j * width for i in range(len(names))]
        ax.bar(xs, vals, width, label=f"{name.upper()} (base rate {d['test_duplicate_share']:.2f})")
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks([i + width * (len(sets) - 1) / 2 for i in range(len(names))])
    ax.set_xticklabels([short(k) for k in names], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("average precision on test pairs")
    ax.set_title("Is this cached question the same question? (QQP validation, PAWS test)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> list[Path]:
    argv = sys.argv[1:] if argv is None else argv
    src = Path(argv[0]) if argv else ROOT / "results"
    out_dir = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [("beir.json", "beir_ndcg.png", beir_chart), ("routerbench.json", "router_curve.png", router_chart),
            ("paraphrase.json", "paraphrase_ap.png", paraphrase_chart)]
    written = []
    for name, png, draw in jobs:
        path = src / name
        if path.exists():
            draw(json.loads(path.read_text(encoding="utf-8")), out_dir / png)
            written.append(out_dir / png)
            print(f"written: {out_dir / png}")
    return written


if __name__ == "__main__":
    main()
