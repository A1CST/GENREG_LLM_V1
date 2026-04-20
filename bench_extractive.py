#!/usr/bin/env python3
"""Extractive QA benchmark on SQuAD v1.1 dev.

Compares three modes:
  1. No retrieval (just rerank generation from query)
  2. RAG generation (retrieve + generate with copy pool)
  3. EXTRACTIVE (retrieve + return best-scoring span)

Metrics:
  - answer containment (case-insensitive substring of any gold)
  - exact-token-overlap F1 (standard SQuAD eval proxy)
"""
import os, sys, json, argparse, re, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.model import GenregLM


def normalize_text(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_f1(pred, gold):
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()
    if not pred_toks or not gold_toks:
        return 0.0
    common = {}
    for w in pred_toks:
        common[w] = min(pred_toks.count(w), gold_toks.count(w))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)


def best_f1(pred, golds):
    return max(token_f1(pred, g) for g in golds)


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad-dev", default="../LLM/data_raw/squad_dev.json")
    ap.add_argument("--n-questions", type=int, default=300)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="run_extractive_bench.log")
    args = ap.parse_args()

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "checkpoints")
    print("loading model...", flush=True)
    m = GenregLM(ckpt_dir, device=args.device)
    if m._rag is None:
        raise RuntimeError("rag_index.pkl missing")

    dev_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             args.squad_dev)
    with open(dev_path) as f:
        dev = json.load(f)
    cases = []
    for article in dev["data"]:
        for para in article["paragraphs"]:
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
    print(f"evaluating on {len(cases):,} questions, k={args.k}", flush=True)

    counts = {"norag_contain": 0, "rag_contain": 0, "extr_contain": 0,
              "retrieve_top1": 0, "retrieve_topk": 0}
    f1s = {"norag": [], "rag": [], "extr": []}
    log = []

    t0 = time.time()
    for i, c in enumerate(cases):
        hits = m.retrieve(c["q"], k=args.k)
        if hits and hits[0]["text"] == c["ctx"]:
            counts["retrieve_top1"] += 1
        if any(h["text"] == c["ctx"] for h in hits):
            counts["retrieve_topk"] += 1

        # No-RAG generation
        torch.manual_seed(42 + i)
        gen_txt, _ = m.generate_rerank(c["q"], max_tokens=30,
                                        alpha=5.0, temperature=0.7, top_k=30)
        # RAG generation
        torch.manual_seed(42 + i)
        rag_txt, _, _ = m.generate_rag(c["q"], max_tokens=30, k=1,
                                         alpha=5.0, temperature=0.7, top_k=30)
        # Extractive
        extr_txt, extr_ids = m.generate_qa(c["q"], k_retrieve=args.k,
                                            max_span=100)

        gold_low = [a.lower() for a in c["answers"]]
        if any(a in gen_txt.lower() for a in gold_low):
            counts["norag_contain"] += 1
        if any(a in rag_txt.lower() for a in gold_low):
            counts["rag_contain"] += 1
        if extr_txt and any(a in extr_txt.lower() for a in gold_low):
            counts["extr_contain"] += 1

        f1s["norag"].append(best_f1(gen_txt, c["answers"]))
        f1s["rag"].append(best_f1(rag_txt, c["answers"]))
        f1s["extr"].append(best_f1(extr_txt, c["answers"]))

        if i < 15:
            line = (f"\nQ: {c['q']}\n"
                    f"  gold: {c['answers'][0]!r}\n"
                    f"  no-RAG: {gen_txt!r}\n"
                    f"  RAG:    {rag_txt!r}\n"
                    f"  EXTR:   {extr_txt!r}")
            print(line, flush=True)
            log.append(line)

        if (i + 1) % 50 == 0:
            n_ = i + 1
            el = time.time() - t0
            print(f"  {n_}/{len(cases)}  ({el:.0f}s)  "
                  f"recall@1={counts['retrieve_top1']/n_:.3f}  "
                  f"no-RAG={counts['norag_contain']/n_:.3f}  "
                  f"RAG={counts['rag_contain']/n_:.3f}  "
                  f"EXTR={counts['extr_contain']/n_:.3f}",
                  flush=True)

    n = len(cases)
    summary = (
        "\n" + "=" * 60 + "\n"
        f"EXTRACTIVE BENCHMARK ({n} questions, k={args.k})\n"
        + "=" * 60 + "\n"
        f"  retrieval recall@1:   {counts['retrieve_top1']/n:.3f}\n"
        f"  retrieval recall@{args.k}:   {counts['retrieve_topk']/n:.3f}\n"
        f"\n"
        f"  answer containment:\n"
        f"    no-RAG generation:   {counts['norag_contain']/n:.3f}\n"
        f"    RAG generation:      {counts['rag_contain']/n:.3f}\n"
        f"    EXTRACTIVE:          {counts['extr_contain']/n:.3f}\n"
        f"\n"
        f"  token-F1 (mean):\n"
        f"    no-RAG generation:   {np.mean(f1s['norag']):.3f}\n"
        f"    RAG generation:      {np.mean(f1s['rag']):.3f}\n"
        f"    EXTRACTIVE:          {np.mean(f1s['extr']):.3f}\n"
    )
    print(summary, flush=True)
    log.append(summary)
    with open(args.out, "w") as f:
        f.write("\n".join(log))
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    run()
