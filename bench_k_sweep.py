#!/usr/bin/env python3
"""Sweep k_retrieve over {3, 5, 7, 10} on extractive QA. Each larger k
gives extract_answer more chunks to consider. Expect containment to
rise toward the recall@k ceiling.
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
                    cases.append({"q": qa["question"], "ctx": para["context"],
                                   "answers": [a["text"] for a in qa["answers"]]})
    rng = np.random.default_rng(7)
    if len(cases) > 300:
        cases = list(rng.choice(cases, size=300, replace=False))

    ks = [3, 5, 7, 10]
    hits_per_k = {k: 0 for k in ks}
    rk_per_k = {k: 0 for k in ks}

    t0 = time.time()
    for i, c in enumerate(cases):
        # recall@k per k (measure once with the biggest k then subset)
        hits10 = m.retrieve(c["q"], k=10)
        for k in ks:
            sub = hits10[:k]
            if any(h["text"] == c["ctx"] for h in sub):
                rk_per_k[k] += 1
        # Extract answer per k
        for k in ks:
            text, _ = m.generate_qa(c["q"], k_retrieve=k)
            text_l = text.lower()
            if any(a.lower() in text_l for a in c["answers"]):
                hits_per_k[k] += 1
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            line = f"  {i+1}/{len(cases)} ({el:.0f}s)  "
            for k in ks:
                line += f"k{k}: r@k={rk_per_k[k]/(i+1):.3f} ans={hits_per_k[k]/(i+1):.3f}  "
            print(line, flush=True)

    n = len(cases)
    print(f"\n{'='*60}", flush=True)
    print(f"k-sweep on {n} SQuAD dev Q", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'k':>4} | {'recall@k':>10} | {'answer-cont':>12} | conversion")
    for k in ks:
        rk = rk_per_k[k] / n
        h = hits_per_k[k] / n
        conv = h / max(rk, 1e-6)
        print(f"  {k:2d} | {rk:>10.3f} | {h:>12.3f} | {conv:>10.3f}",
              flush=True)


if __name__ == "__main__":
    main()
