#!/usr/bin/env python3
"""Properly categorize misses: wrong-passage vs wrong-span-in-right-passage.

Distinguishes:
  HIT                    — extr contains gold
  MISS_NO_PASSAGE        — gold passage NOT in top-3 retrievals
  MISS_WRONG_SPAN_RIGHT  — gold passage IS in top-3, but extracted span
                            doesn't contain gold (the big lever)
  MISS_RIGHT_CHUNK_WRONG — the specific matched chunk we extract from
                            does contain gold, but we still miss (pure
                            scorer failure)
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
                    cases.append({"q": qa["question"],
                                   "ctx": para["context"],
                                   "answers": [a["text"] for a in qa["answers"]]})
    rng = np.random.default_rng(7)
    if len(cases) > 300:
        cases = list(rng.choice(cases, size=300, replace=False))

    counts = {"HIT": 0, "MISS_NO_PASSAGE": 0,
               "MISS_WRONG_SPAN_RIGHT_PASSAGE": 0,
               "MISS_SCORER_FAILURE": 0}
    wrong_span_but_chunk_has_answer = []
    t0 = time.time()
    for i, c in enumerate(cases):
        hits = m.retrieve(c["q"], k=3)
        text, _, meta = m.extract_answer(c["q"], k_retrieve=3)
        text_l = text.lower()
        gold_l = [a.lower() for a in c["answers"]]
        is_hit = any(a in text_l for a in gold_l)

        if is_hit:
            counts["HIT"] += 1
            continue

        # Find gold_passage_in_hits
        gold_in_hits = any(h["text"] == c["ctx"] for h in hits)
        if not gold_in_hits:
            counts["MISS_NO_PASSAGE"] += 1
            continue

        # Gold passage IS retrieved. Now check the CHUNK we actually
        # extracted from (meta["passage_idx"]): does it contain the
        # answer? If yes, pure scorer failure. If no, we picked the
        # wrong chunk from within the right parent.
        pidx = meta.get("passage_idx", 0) if meta else 0
        chunk_text = hits[pidx].get("text") if pidx < len(hits) else ""
        chunk_l = chunk_text.lower()
        if any(a in chunk_l for a in gold_l):
            counts["MISS_SCORER_FAILURE"] += 1
            if len(wrong_span_but_chunk_has_answer) < 5:
                wrong_span_but_chunk_has_answer.append({
                    "q": c["q"], "gold": c["answers"][0],
                    "extr": text[:200],
                    "chunk_excerpt": chunk_text[:200],
                })
        else:
            counts["MISS_WRONG_SPAN_RIGHT_PASSAGE"] += 1

        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(cases)} ({el:.0f}s)  "
                  f"HIT={counts['HIT']}  "
                  f"no_passage={counts['MISS_NO_PASSAGE']}  "
                  f"wrong_chunk={counts['MISS_WRONG_SPAN_RIGHT_PASSAGE']}  "
                  f"scorer={counts['MISS_SCORER_FAILURE']}", flush=True)

    n = len(cases)
    print(f"\n{'='*60}", flush=True)
    print(f"CATEGORIZATION on {n} SQuAD dev Q", flush=True)
    print(f"{'='*60}", flush=True)
    for k, v in counts.items():
        print(f"  {k:<35s} {v:4d}  ({v/n:.3f})", flush=True)

    print(f"\nSamples of PURE SCORER FAILURE (right chunk, wrong span):",
          flush=True)
    for s in wrong_span_but_chunk_has_answer:
        print(f"\n  Q: {s['q']}", flush=True)
        print(f"  gold: {s['gold']!r}", flush=True)
        print(f"  chunk (truncated): {s['chunk_excerpt']!r}", flush=True)
        print(f"  extr (truncated):  {s['extr']!r}", flush=True)

    with open("miss_diagnosis.json", "w") as f:
        json.dump({"counts": counts, "n": n,
                    "samples": wrong_span_but_chunk_has_answer}, f, indent=2)


if __name__ == "__main__":
    main()
