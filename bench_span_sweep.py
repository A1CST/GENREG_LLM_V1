#!/usr/bin/env python3
"""Sweep max_span ∈ {8, 20, 40, 60, 100} on same 300 dev Q to find
where span-length gains plateau. Also report conversion rate of
retrieval → answer per span length."""
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
                    cases.append({"q": qa["question"],
                                   "ctx": para["context"],
                                   "answers": [a["text"] for a in qa["answers"]]})
    rng = np.random.default_rng(7)
    if len(cases) > 300:
        cases = list(rng.choice(cases, size=300, replace=False))

    spans = [8, 20, 40, 60, 100]
    hits = {s: 0 for s in spans}
    t0 = time.time()

    for i, c in enumerate(cases):
        for s in spans:
            text, _ = m.generate_qa(c["q"], k_retrieve=3, max_span=s)
            text_l = text.lower()
            if any(a.lower() in text_l for a in c["answers"]):
                hits[s] += 1
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            line = f"  {i+1}/{len(cases)}  ({el:.0f}s)  "
            line += "  ".join(f"s{s}={hits[s]/(i+1):.3f}" for s in spans)
            print(line, flush=True)

    n = len(cases)
    print(f"\nFINAL:", flush=True)
    print(f"  span | answer-containment | conversion vs r@3=0.68")
    for s in spans:
        p = hits[s] / n
        conv = p / 0.68
        print(f"   {s:3d} | {p:6.3f}  ({hits[s]}/{n}) | {conv:6.3f}",
              flush=True)


if __name__ == "__main__":
    main()
