#!/usr/bin/env python3
"""Retrieval miss-distribution diagnostic.

For each SQuAD dev question, retrieve a large top-K and record the
rank of the gold chunk (or None if not in top-K). Tells us whether
recall@1 is bottlenecked by candidate generation (gold not in top-50)
or by reranking (gold in top-10 but not top-1).

Also breaks down by question type and rare-token count to see if
specific slices fail systematically.
"""
import os, sys, json, argparse, time, re
from collections import Counter, defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.model import GenregLM


def classify(q):
    ql = q.lower().strip()
    for key in ("when", "who", "whom", "where", "why", "how many",
                "how much", "what year", "what date"):
        if ql.startswith(key) or f" {key} " in ql[:30]:
            return key
    if ql.startswith("how"):
        return "how"
    if ql.startswith("what") or ql.startswith("which"):
        return "what"
    return "other"


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad-dev", default="../LLM/data_raw/squad_dev.json")
    ap.add_argument("--n-questions", type=int, default=300)
    ap.add_argument("--max-k", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    m = GenregLM(os.path.join(here, "checkpoints"), device=args.device)
    if m._rag is None:
        raise SystemExit("rag_index.pkl missing")

    dev_path = os.path.join(here, args.squad_dev)
    with open(dev_path) as f:
        dev = json.load(f)
    cases = []
    for art in dev["data"]:
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                if not qa.get("answers"):
                    continue
                cases.append({
                    "q": qa["question"],
                    "ctx": para["context"],
                    "answers": [a["text"] for a in qa["answers"]],
                })
    rng = np.random.default_rng(7)
    if 0 < args.n_questions < len(cases):
        cases = list(rng.choice(cases, size=args.n_questions, replace=False))
    print(f"diagnosing {len(cases)} questions up to top-{args.max_k}",
          flush=True)

    # Buckets of the rank of the gold parent paragraph in retrieval output
    ranks = []   # list of int rank (0-indexed), or -1 if not in top-K
    type_buckets = defaultdict(list)
    rare_buckets = defaultdict(list)

    t0 = time.time()
    for i, c in enumerate(cases):
        hits = m.retrieve(c["q"], k=args.max_k)
        rank = -1
        for r, h in enumerate(hits):
            if h["text"] == c["ctx"]:
                rank = r
                break
        ranks.append(rank)
        type_buckets[classify(c["q"])].append(rank)

        # Count rare content tokens in the query
        q_ids = m.tokenize(c["q"]).cpu().tolist()
        rare_n = 0
        for t in q_ids:
            if t >= 96 and float(m._tok_weight[t].item()) > 0.3:
                rare_n += 1
        rare_buckets[min(rare_n, 5)].append(rank)

        if (i + 1) % 50 == 0:
            r1 = sum(1 for r in ranks if r == 0) / (i + 1)
            r10 = sum(1 for r in ranks if 0 <= r < 10) / (i + 1)
            rk = sum(1 for r in ranks if r != -1) / (i + 1)
            el = time.time() - t0
            print(f"  {i+1}/{len(cases)} ({el:.0f}s)  "
                  f"r@1={r1:.3f}  r@10={r10:.3f}  r@{args.max_k}={rk:.3f}",
                  flush=True)

    n = len(cases)
    ranks_arr = np.array(ranks)
    found = ranks_arr[ranks_arr != -1]
    missed = int((ranks_arr == -1).sum())

    print("\n" + "=" * 60)
    print(f"RETRIEVAL MISS-DISTRIBUTION ({n} Q, max-k={args.max_k})")
    print("=" * 60)
    for k in (1, 2, 3, 5, 10, 20, 50, args.max_k):
        cnt = int(((ranks_arr >= 0) & (ranks_arr < k)).sum())
        print(f"  recall@{k:3d}:  {cnt/n:.3f}  ({cnt}/{n})")
    print(f"  missed (>= {args.max_k}): {missed/n:.3f}  ({missed}/{n})")

    if len(found):
        print(f"\n  of {len(found)} found: median rank {int(np.median(found))}"
              f"  mean {found.mean():.1f}  p90 {int(np.percentile(found, 90))}")

    print("\n  by question type (recall@1 / recall@10 / miss%):")
    for qt, rs in sorted(type_buckets.items(), key=lambda x: -len(x[1])):
        rs_a = np.array(rs); nn = len(rs_a)
        r1 = ((rs_a >= 0) & (rs_a < 1)).sum() / nn
        r10 = ((rs_a >= 0) & (rs_a < 10)).sum() / nn
        miss = (rs_a == -1).sum() / nn
        print(f"    {qt:12s} ({nn:3d})  r@1={r1:.3f}  r@10={r10:.3f}  "
              f"miss={miss:.3f}")

    print("\n  by rare-content-token count in query:")
    for rc in sorted(rare_buckets.keys()):
        rs = np.array(rare_buckets[rc]); nn = len(rs)
        r1 = ((rs >= 0) & (rs < 1)).sum() / nn
        r10 = ((rs >= 0) & (rs < 10)).sum() / nn
        miss = (rs == -1).sum() / nn
        label = f"{rc}" if rc < 5 else "5+"
        print(f"    {label} rare-toks ({nn:3d})  r@1={r1:.3f}  "
              f"r@10={r10:.3f}  miss={miss:.3f}")

    # Interpretive takeaway
    r1 = int((ranks_arr == 0).sum()) / n
    r10 = int(((ranks_arr >= 0) & (ranks_arr < 10)).sum()) / n
    r50 = int(((ranks_arr >= 0) & (ranks_arr < 50)).sum()) / n
    print("\n  DIAGNOSIS:")
    print(f"  - r@1={r1:.3f}  r@10={r10:.3f}  r@50={r50:.3f}")
    recoverable = r10 - r1
    unreachable = 1 - r50
    print(f"  - {recoverable:.3f} of queries have gold in top-10 but miss "
          f"at top-1 → reranker headroom")
    print(f"  - {unreachable:.3f} of queries don't have gold even in top-50 "
          f"→ candidate-generation problem")
    print(f"  - {(r50 - r10):.3f} in top-50 but not top-10 → deep rerank")


if __name__ == "__main__":
    run()
