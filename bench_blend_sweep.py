#!/usr/bin/env python3
"""Zero-risk baseline for the retrieval-rebuild question: sweep the
static bm25_weight over a range on SQuAD dev and see if the shipped
default of 0.85 is already optimal.

If a different static weight hits 55%+ recall@1 on dev, a per-query-
adaptive reranker has a higher bar to beat and the 'shipped default
is suboptimal' answer is a zero-evolution fix.

If 0.85 really is the static optimum on dev, then per-query adaptation
is the only path — build qadapt v2.
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.model import GenregLM


def main():
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "checkpoints")
    m = GenregLM(ckpt, device="cuda")

    dev_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "../LLM/data_raw/squad_dev.json")
    with open(dev_path) as f:
        dev = json.load(f)
    cases = []
    for art in dev["data"]:
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                if qa.get("answers"):
                    cases.append({"q": qa["question"], "ctx": para["context"]})
    rng = np.random.default_rng(7)
    if len(cases) > 300:
        cases = list(rng.choice(cases, size=300, replace=False))

    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.80,
                0.85, 0.90, 0.95, 1.0]
    results = {}
    for w in weights:
        r1 = 0; r3 = 0
        t0 = time.time()
        for c in cases:
            hits = m.retrieve(c["q"], k=3, bm25_weight=w)
            if hits and hits[0]["text"] == c["ctx"]:
                r1 += 1
            if any(h["text"] == c["ctx"] for h in hits):
                r3 += 1
        n = len(cases)
        print(f"  bm25_w={w:.2f}  r@1={r1/n:.3f} ({r1}/{n})  "
              f"r@3={r3/n:.3f} ({r3}/{n})  ({time.time()-t0:.0f}s)",
              flush=True)
        results[w] = {"r1": r1/n, "r3": r3/n, "r1_count": r1, "r3_count": r3}

    best_r1 = max(results.items(), key=lambda kv: kv[1]["r1"])
    best_r3 = max(results.items(), key=lambda kv: kv[1]["r3"])
    print(f"\nBest r@1: bm25_w={best_r1[0]} -> {best_r1[1]['r1']:.3f}",
          flush=True)
    print(f"Best r@3: bm25_w={best_r3[0]} -> {best_r3[1]['r3']:.3f}",
          flush=True)
    print(f"Shipped default (0.85): r@1={results[0.85]['r1']:.3f} "
          f"r@3={results[0.85]['r3']:.3f}", flush=True)

    with open("blend_sweep_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
