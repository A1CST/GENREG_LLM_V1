#!/usr/bin/env python3
"""Bypass the MLP span scorer. Test whether the fallback heuristic
alone beats MLP+fallback on dev.

Disables _span_mlp by setting it to None after load, then runs same
300 dev questions. Compares to v13.3's 27.3% baseline.
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.model import GenregLM


def main():
    m = GenregLM("./checkpoints", device="cuda")
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
    cases = list(rng.choice(cases, size=300, replace=False))

    print(f"MLP loaded: {m._span_mlp is not None}", flush=True)
    # Disable MLP, force fallback heuristic path
    m._span_mlp = None
    m._span_qa = None
    # Ensure v2 scorer is also disabled to hit pure fallback
    m._span_scorer = None

    hits = 0
    t0 = time.time()
    for i, c in enumerate(cases):
        text, _ = m.generate_qa(c["q"], k_retrieve=3)
        text_l = text.lower()
        if any(a.lower() in text_l for a in c["answers"]):
            hits += 1
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(cases)} ({el:.0f}s)  "
                  f"fallback-only ans={hits/(i+1):.3f}", flush=True)
    n = len(cases)
    print(f"\n=== No-MLP fallback heuristic: {hits}/{n} = {hits/n:.3f} ===",
          flush=True)
    print(f"  (v13.3 with MLP: 82/300 = 0.273)", flush=True)


if __name__ == "__main__":
    main()
