#!/usr/bin/env python3
"""Head-to-head retrieval bench: default vs Phase A vs Phase B vs A+B.

Measures r@1 / r@3 / r@10 on 300 SQuAD dev Q for each configuration.
Prints a comparison table. Deploys whichever configuration wins r@1
on completion (honest — pick by dev metric, not train).
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.model import GenregLM


def load_cases(dev_path, n):
    with open(dev_path) as f:
        dev = json.load(f)
    cases = []
    for art in dev["data"]:
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                if qa.get("answers"):
                    cases.append({"q": qa["question"],
                                   "ctx": para["context"]})
    rng = np.random.default_rng(7)
    if 0 < n < len(cases):
        cases = list(rng.choice(cases, size=n, replace=False))
    return cases


def classify(q):
    ql = q.lower().strip()
    if ql.startswith("when") or "what year" in ql or "what date" in ql:
        return "when"
    if ql.startswith("who") or ql.startswith("whom"):
        return "who"
    if ql.startswith("where"):
        return "where"
    if ql.startswith("how"):
        return "how"
    if ql.startswith("what") or ql.startswith("which"):
        return "what"
    return "other"


def bench(m, cases):
    tp1 = tp3 = tp10 = 0
    for c in cases:
        hits = m.retrieve(c["q"], k=10)
        texts = [h["text"] for h in hits]
        if c["ctx"] in texts[:1]: tp1 += 1
        if c["ctx"] in texts[:3]: tp3 += 1
        if c["ctx"] in texts[:10]: tp10 += 1
    n = len(cases)
    return tp1/n, tp3/n, tp10/n


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cases = load_cases(os.path.join(here, "../LLM/data_raw/squad_dev.json"),
                        args.n)
    print(f"bench on {len(cases)} Q", flush=True)

    m = GenregLM(os.path.join(here, "checkpoints"), device=args.device)
    print(f"  head={'yes' if getattr(m, '_retrieval_head', None) else 'no'}"
          f"  qtype={'yes' if getattr(m, '_retrieval_qtype', None) else 'no'}",
          flush=True)

    rows = []
    # Default (both off)
    m._disable_retrieval_head = True
    m._disable_retrieval_qtype = True
    t0 = time.time()
    r1, r3, r10 = bench(m, cases)
    rows.append(("default", r1, r3, r10, time.time()-t0))

    if getattr(m, "_retrieval_head", None) is not None:
        m._disable_retrieval_head = False
        m._disable_retrieval_qtype = True
        t0 = time.time()
        r1, r3, r10 = bench(m, cases)
        rows.append(("A head", r1, r3, r10, time.time()-t0))

    if getattr(m, "_retrieval_qtype", None) is not None:
        m._disable_retrieval_head = True
        m._disable_retrieval_qtype = False
        t0 = time.time()
        r1, r3, r10 = bench(m, cases)
        rows.append(("B qtype", r1, r3, r10, time.time()-t0))

    if (getattr(m, "_retrieval_head", None) is not None
            and getattr(m, "_retrieval_qtype", None) is not None):
        m._disable_retrieval_head = False
        m._disable_retrieval_qtype = False
        t0 = time.time()
        r1, r3, r10 = bench(m, cases)
        rows.append(("A + B", r1, r3, r10, time.time()-t0))

    print(f"\n{'config':20s}  r@1    r@3    r@10  t(s)")
    for label, r1, r3, r10, el in rows:
        print(f"{label:20s}  {r1:.3f}  {r3:.3f}  {r10:.3f}  {el:.0f}")

    best = max(rows, key=lambda x: x[1])
    print(f"\nBest by r@1: {best[0]}  ({best[1]:.3f})")


if __name__ == "__main__":
    run()
