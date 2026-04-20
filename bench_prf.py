#!/usr/bin/env python3
"""Test pseudo-relevance feedback on retrieval.

retrieve() already implements PRF via the `prf=True` flag: take top-N
retrieved chunks, extract their rarest content tokens, add them to the
BM25 query with reduced IDF weight, and rescore. Classical IR trick —
entity-like tokens near the query match get amplified.

Measures r@1, r@3 on SQuAD dev with and without PRF. Also runs
extractive QA on top of each to see if any retrieval lift converts.
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

    configs = [
        ("baseline (prf=False)", {"prf": False}),
        ("prf top=3 terms=3", {"prf": True, "prf_top": 3, "prf_terms": 3}),
        ("prf top=5 terms=5", {"prf": True, "prf_top": 5, "prf_terms": 5}),
    ]

    results = {}
    for label, kwargs in configs:
        r1 = 0; r3 = 0
        t0 = time.time()
        for c in cases:
            hits = m.retrieve(c["q"], k=3, **kwargs)
            if hits and hits[0]["text"] == c["ctx"]:
                r1 += 1
            if any(h["text"] == c["ctx"] for h in hits):
                r3 += 1
        n = len(cases)
        results[label] = {"r1": r1/n, "r3": r3/n, "time": time.time()-t0}
        print(f"  [{label}] r@1={r1/n:.3f} ({r1}/{n})  "
              f"r@3={r3/n:.3f} ({r3}/{n})  "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"PRF comparison on {len(cases)} SQuAD dev Q", flush=True)
    print(f"{'='*60}", flush=True)
    for label, v in results.items():
        print(f"  {label:<30s} r@1={v['r1']:.3f}  r@3={v['r3']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
