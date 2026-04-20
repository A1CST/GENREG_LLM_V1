#!/usr/bin/env python3
"""retr_weight=3.0 + k=5 test — does it beat k=3 @ w=1.0 (27.3% baseline)?"""
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

    counts = {3: 0, 5: 0, 7: 0}
    t0 = time.time()
    for i, c in enumerate(cases):
        for k in [3, 5, 7]:
            text, _ = m.generate_qa(c["q"], k_retrieve=k)
            text_l = text.lower()
            if any(a.lower() in text_l for a in c["answers"]):
                counts[k] += 1
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(cases)} ({el:.0f}s)  "
                  f"k3={counts[3]/(i+1):.3f}  "
                  f"k5={counts[5]/(i+1):.3f}  "
                  f"k7={counts[7]/(i+1):.3f}", flush=True)
    n = len(cases)
    print(f"\nFINAL: k3={counts[3]/n:.3f}  k5={counts[5]/n:.3f}  "
          f"k7={counts[7]/n:.3f}", flush=True)


if __name__ == "__main__":
    main()
