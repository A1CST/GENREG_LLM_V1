#!/usr/bin/env python3
"""Collect annotated hit/miss samples from extractive QA for README.

Runs generate_qa on same 300 SQuAD dev questions as bench_extractive,
flags each as HIT (gold appears in EXTR) or MISS, and writes two
balanced panels of samples: 6 clean hits + 6 instructive misses.
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.model import GenregLM


def truncate(text, n=180):
    if len(text) <= n:
        return text
    return text[:n].rstrip() + " …"


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

    hits = []; misses = []
    empties = 0
    t0 = time.time()
    for i, c in enumerate(cases):
        text, _ = m.generate_qa(c["q"], k_retrieve=3)
        text_l = text.lower()
        gold_l = [a.lower() for a in c["answers"]]
        is_hit = any(a in text_l for a in gold_l)
        is_empty = not text.strip()
        rec = {"q": c["q"], "gold": c["answers"][0], "extr": text,
                "is_empty": is_empty}
        if is_hit:
            hits.append(rec)
        else:
            misses.append(rec)
            if is_empty:
                empties += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(cases)}  hit={len(hits)}  miss={len(misses)}  "
                  f"empty={empties}  ({time.time()-t0:.0f}s)", flush=True)

    n = len(cases)
    print(f"\nFinal: hits={len(hits)}/{n} ({len(hits)/n:.3f})  "
          f"misses={len(misses)}/{n}  empty_extracts={empties}", flush=True)

    # Pick 6 hits + 6 misses (mix of empty and non-empty misses)
    rng2 = np.random.default_rng(11)
    hit_idx = rng2.choice(len(hits), size=min(6, len(hits)), replace=False)
    non_empty_miss = [m for m in misses if not m["is_empty"]]
    empty_miss = [m for m in misses if m["is_empty"]]
    # 4 non-empty misses (scorer picked wrong span), 2 empty (scorer gave up)
    pick_non_empty = list(rng2.choice(len(non_empty_miss),
                                       size=min(4, len(non_empty_miss)),
                                       replace=False))
    pick_empty = list(rng2.choice(len(empty_miss),
                                    size=min(2, len(empty_miss)),
                                    replace=False))
    sel_hits = [hits[int(j)] for j in hit_idx]
    sel_miss_ne = [non_empty_miss[int(j)] for j in pick_non_empty]
    sel_miss_e = [empty_miss[int(j)] for j in pick_empty]

    print(f"\n\n{'=' * 70}", flush=True)
    print("HITS (6 samples)", flush=True)
    print('=' * 70, flush=True)
    for r in sel_hits:
        print(f"\n  Q: {r['q']}", flush=True)
        print(f"  gold: {r['gold']!r}", flush=True)
        print(f"  extr: {truncate(r['extr'], 220)!r}", flush=True)

    print(f"\n\n{'=' * 70}", flush=True)
    print("MISSES — wrong span picked (4 samples)", flush=True)
    print('=' * 70, flush=True)
    for r in sel_miss_ne:
        print(f"\n  Q: {r['q']}", flush=True)
        print(f"  gold: {r['gold']!r}", flush=True)
        print(f"  extr: {truncate(r['extr'], 220)!r}", flush=True)

    print(f"\n\n{'=' * 70}", flush=True)
    print("MISSES — empty extraction (2 samples)", flush=True)
    print('=' * 70, flush=True)
    for r in sel_miss_e:
        print(f"\n  Q: {r['q']}", flush=True)
        print(f"  gold: {r['gold']!r}", flush=True)
        print(f"  extr: {r['extr']!r}", flush=True)

    # Save raw for reference
    out = {
        "stats": {"n": n, "hits": len(hits), "misses": len(misses),
                   "empty": empties},
        "hits": sel_hits, "miss_wrong_span": sel_miss_ne,
        "miss_empty": sel_miss_e,
    }
    with open("qa_samples_annotated.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> qa_samples_annotated.json", flush=True)


if __name__ == "__main__":
    main()
