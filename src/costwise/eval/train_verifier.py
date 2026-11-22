"""Fine-tunes a small cross-encoder as the cache's strong verifier, on CPU.

Run with `python -m costwise.eval.train_verifier`. Starts from
cross-encoder/ms-marco-MiniLM-L-6-v2 (22M parameters) and trains it with binary cross-entropy
on duplicate / not-duplicate pairs from the TRAIN splits of PAWS (word-swap near-misses) and
QQP (real duplicate questions). Pairs that appear in the paraphrase eval's tune samples are
excluded, so that eval can still fit thresholds on unseen pairs; the test splits (PAWS test,
QQP validation) and the synthetic help-centre questions are never used for training.

Writes the model to .state/models/<name> (gitignored, about 90 MB) and a training report to
results/train_verifier.json.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from typing import Sequence

from ..settings import ROOT, Settings
from .paraphrase import TUNE_N, load_pairs

BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NAME = "verifier-minilm-paws-qqp"


def training_pairs(sizes: dict[str, int], seed: int = 1) -> list[tuple[str, str, int]]:
    out = []
    for name, per_dataset in sizes.items():
        held = {(a, b) for a, b, _ in load_pairs(name, "tune", TUNE_N)}          # the eval's tune sample
        pool = [p for p in load_pairs(name, "tune", per_dataset + TUNE_N + 500, seed=seed) if (p[0], p[1]) not in held]
        out += pool[:per_dataset]
    random.Random(seed).shuffle(out)
    return out


def train(pairs: Sequence[tuple[str, str, int]], epochs: int = 1, batch_size: int = 32, lr: float = 3e-5,
          max_length: int = 128, seed: int = 0) -> tuple[object, dict]:
    import torch
    from sentence_transformers import CrossEncoder

    torch.manual_seed(seed)
    ce = CrossEncoder(BASE, num_labels=1, max_length=max_length)
    model, tok = ce.model, ce.tokenizer
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(pairs) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = random.Random(seed)
    t0, step, losses = time.perf_counter(), 0, []
    for epoch in range(epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = [pairs[j] for j in order[i:i + batch_size]]
            enc = tok([a for a, _, _ in batch], [b for _, b, _ in batch], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            logits = model(**enc).logits.view(-1)
            loss = loss_fn(logits, torch.tensor([float(y) for _, _, y in batch]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(float(loss))
            step += 1
            if step % 50 == 0:
                print(f"  step {step}/{steps}  loss {sum(losses[-50:]) / 50:.4f}  "
                      f"{(time.perf_counter() - t0) / step:.2f}s/step", flush=True)
    model.eval()
    ce.activation_fn = torch.nn.Sigmoid()          # scores in [0, 1], like the other verifiers
    report = {"base": BASE, "pairs": len(pairs), "duplicates": sum(y for _, _, y in pairs), "epochs": epochs,
              "batch_size": batch_size, "lr": lr, "max_length": max_length, "steps": steps,
              "train_seconds": round(time.perf_counter() - t0, 1),
              "loss_first_50": round(sum(losses[:50]) / min(50, len(losses)), 4),
              "loss_last_50": round(sum(losses[-50:]) / min(50, len(losses)), 4)}
    return ce, report


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paws", type=int, default=16000, help="PAWS train pairs")
    ap.add_argument("--qqp", type=int, default=8000, help="QQP train pairs")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args(argv)
    settings = Settings()
    pairs = training_pairs({"paws": args.paws, "qqp": args.qqp})
    print(f"training on {len(pairs)} pairs", flush=True)
    ce, report = train(pairs, epochs=args.epochs, lr=args.lr)
    path = settings.state_dir / "models" / NAME
    ce.save(str(path))
    report["saved_to"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    (ROOT / "results" / "train_verifier.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
