#!/usr/bin/env python3
"""Apples-to-apples bench: generate_qa (extractive) on same 300 SQuAD dev
questions used in bench_rag.py baseline. Compare answer containment.

Hypothesis: since retrieval gets the right passage in 68% of cases (recall@3)
but generative only answers 6%, extractive span-picking should convert a
much larger fraction of retrievals into correct answers.
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
    print(f"  RAG index: {len(m._rag['texts']):,} paragraphs", flush=True)

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

    retrieve_top1 = 0
    retrieve_top3 = 0
    qa_ans = 0
    samples = []

    t0 = time.time()
    for i, c in enumerate(cases):
        hits = m.retrieve(c["q"], k=3)
        if hits and hits[0]["text"] == c["ctx"]:
            retrieve_top1 += 1
        if any(h["text"] == c["ctx"] for h in hits):
            retrieve_top3 += 1

        # Extractive: generate_qa
        text, ids = m.generate_qa(c["q"], k_retrieve=3, max_span=20)
        text_l = text.lower()
        ans_lower = [a.lower() for a in c["answers"]]
        hit = any(a in text_l for a in ans_lower)
        if hit:
            qa_ans += 1
        if i < 12:
            samples.append({"q": c["q"], "gold": c["answers"][0],
                             "extracted": text, "hit": hit})

        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(cases)}  ({el:.0f}s)  "
                  f"r@1={retrieve_top1/(i+1):.3f}  "
                  f"r@3={retrieve_top3/(i+1):.3f}  "
                  f"qa-ans={qa_ans/(i+1):.3f}", flush=True)

    n = len(cases)
    print(f"\n{'='*60}", flush=True)
    print(f"EXTRACTIVE QA on {n} SQuAD dev", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  recall@1:         {retrieve_top1/n:.3f} ({retrieve_top1}/{n})")
    print(f"  recall@3:         {retrieve_top3/n:.3f} ({retrieve_top3}/{n})")
    print(f"  answer containment (extractive): {qa_ans/n:.3f} ({qa_ans}/{n})")
    print(f"  conversion (ans / retrieval@3):  "
          f"{qa_ans/max(retrieve_top3,1):.3f}")

    print(f"\nSample extractions:", flush=True)
    for s in samples:
        mark = "✓" if s["hit"] else "✗"
        print(f"  {mark} Q: {s['q']!r}", flush=True)
        print(f"    gold: {s['gold']!r}", flush=True)
        print(f"    extr: {s['extracted']!r}", flush=True)


if __name__ == "__main__":
    main()
