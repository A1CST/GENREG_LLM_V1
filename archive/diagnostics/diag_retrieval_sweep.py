#!/usr/bin/env python3
"""Sweep cheap retrieval knobs to see if any swing recall@1.

Tests:
  A. default (current shipped)
  B. re-enable existing reranker (_use_reranker=True)
  C. bm25_weight sweep: 0.7, 0.85 (def), 0.95
  D. qexp_weight sweep: 0.0 (off), 0.4 (def), 0.7
  E. PRF on
  F. rare-token threshold looser: monkey-patch for 0-rare queries
"""
import os, sys, json, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.model import GenregLM


def load_cases(m_path, n):
    with open(m_path) as f:
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


def bench(m, cases, **retrieve_kwargs):
    tp1 = tp3 = tp10 = 0
    n = len(cases)
    for c in cases:
        hits = m.retrieve(c["q"], k=10, **retrieve_kwargs)
        texts = [h["text"] for h in hits]
        if c["ctx"] in texts[:1]:
            tp1 += 1
        if c["ctx"] in texts[:3]:
            tp3 += 1
        if c["ctx"] in texts[:10]:
            tp10 += 1
    return tp1/n, tp3/n, tp10/n


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    m = GenregLM(os.path.join(here, "checkpoints"), device=args.device)
    cases = load_cases(os.path.join(here, "../LLM/data_raw/squad_dev.json"),
                        args.n)
    print(f"sweep on {len(cases)} Q", flush=True)
    print(f"{'config':50s}  r@1    r@3    r@10  t(s)")

    configs = [
        ("default", {}),
        ("bm25_w=0.70", {"bm25_weight": 0.70}),
        ("bm25_w=0.95", {"bm25_weight": 0.95}),
        ("bm25_w=1.00 (pure lexical)", {"bm25_weight": 1.00}),
        ("bm25_w=0.50", {"bm25_weight": 0.50}),
        ("qexp off", {"qexp": False}),
        ("qexp_weight=0.7", {"qexp_weight": 0.7}),
        ("qexp_k=5", {"qexp_k": 5}),
        ("prf on", {"prf": True}),
        ("prf on + qexp off", {"prf": True, "qexp": False}),
    ]
    for label, kw in configs:
        t0 = time.time()
        r1, r3, r10 = bench(m, cases, **kw)
        el = time.time() - t0
        print(f"{label:50s}  {r1:.3f}  {r3:.3f}  {r10:.3f}  {el:.0f}",
              flush=True)

    # Reranker test (state flag on model)
    print(f"\n-- rerank ON (existing retrieval_reranker.pkl) --")
    m._use_reranker = True
    t0 = time.time()
    r1, r3, r10 = bench(m, cases)
    print(f"{'rerank=True default':50s}  {r1:.3f}  {r3:.3f}  {r10:.3f}  "
          f"{time.time()-t0:.0f}", flush=True)


if __name__ == "__main__":
    run()
