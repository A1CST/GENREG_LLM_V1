#!/usr/bin/env python3
"""RAG benchmark on SQuAD v1.1 dev.

For each dev question:
  - retrieve top-K paragraphs by cosine
  - check retrieval recall: is the gold paragraph among top-K?
  - generate a response with RAG context
  - check answer containment: is any gold answer substring in the
    lowercased response?

Reports retrieval-recall and answer-containment.
"""
import os, sys, json, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.model import GenregLM


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad-dev",
                     default="../LLM/data_raw/squad_dev.json")
    ap.add_argument("--n-questions", type=int, default=200)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="run_rag_bench.log")
    args = ap.parse_args()

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "checkpoints")
    print(f"loading model...", flush=True)
    m = GenregLM(ckpt_dir, device=args.device)
    if m._rag is None:
        raise RuntimeError("rag_index.pkl missing from checkpoints/")
    print(f"  RAG index: {len(m._rag['texts']):,} paragraphs", flush=True)

    # Load SQuAD dev
    dev_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             args.squad_dev)
    with open(dev_path) as f:
        dev = json.load(f)

    # Flatten (question, gold_context, answers) tuples
    cases = []
    for article in dev["data"]:
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                if not qa.get("answers"):
                    continue
                cases.append({
                    "q": qa["question"],
                    "ctx": ctx,
                    "answers": [a["text"] for a in qa["answers"]],
                })
    print(f"  total dev QA: {len(cases):,}", flush=True)

    # Sample
    rng = np.random.default_rng(7)
    if args.n_questions > 0 and len(cases) > args.n_questions:
        cases = list(rng.choice(cases, size=args.n_questions, replace=False))
    print(f"  evaluating on {len(cases):,} questions, k={args.k}", flush=True)

    # Metrics
    retrieve_top1 = 0
    retrieve_topk = 0
    answer_in_response = 0
    answer_in_response_rerank = 0
    log_lines = []

    torch.manual_seed(42)
    t0 = time.time()
    for i, c in enumerate(cases):
        hits = m.retrieve(c["q"], k=args.k)
        if hits and hits[0]["text"] == c["ctx"]:
            retrieve_top1 += 1
        if any(h["text"] == c["ctx"] for h in hits):
            retrieve_topk += 1

        # Generate with RAG: prepend top-1 retrieved passage
        torch.manual_seed(42 + i)
        text, ids, _ = m.generate_rag(
            c["q"], max_tokens=args.max_tokens, k=1,
            alpha=5.0, temperature=0.7, top_k=30)
        text_l = text.lower()
        answers = [a.lower() for a in c["answers"]]
        if any(a in text_l for a in answers):
            answer_in_response += 1

        # Baseline: rerank without RAG (shows gain from retrieval)
        torch.manual_seed(42 + i)
        text_r, ids_r = m.generate_rerank(
            c["q"], max_tokens=args.max_tokens,
            alpha=5.0, temperature=0.7, top_k=30)
        text_rl = text_r.lower()
        if any(a in text_rl for a in answers):
            answer_in_response_rerank += 1

        if i < 10:
            line = (f"Q: {c['q']}\n"
                    f"  gold: {c['answers'][0]!r}\n"
                    f"  rag-top1-title: {hits[0]['title'] if hits else '?'!r}  "
                    f"score={hits[0]['score']:.3f}\n" if hits else f"Q: {c['q']}\n"
                    f"  gold: {c['answers'][0]!r}\n"
                    f"  no hits\n")
            line += f"  RAG reply: {text}\n  no-RAG reply: {text_r}\n"
            print(line, flush=True)
            log_lines.append(line)

        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(cases)}  ({el:.0f}s)  "
                  f"recall@1={retrieve_top1/(i+1):.3f}  "
                  f"recall@{args.k}={retrieve_topk/(i+1):.3f}  "
                  f"RAG-ans={answer_in_response/(i+1):.3f}  "
                  f"no-RAG-ans={answer_in_response_rerank/(i+1):.3f}",
                  flush=True)

    n = len(cases)
    r1 = retrieve_top1 / n
    rk = retrieve_topk / n
    a_rag = answer_in_response / n
    a_no = answer_in_response_rerank / n
    summary = (
        "\n" + "=" * 60 + "\n"
        f"RAG BENCHMARK ({n} questions, k={args.k}, max_tokens={args.max_tokens})\n"
        + "=" * 60 + "\n"
        f"  retrieval recall@1:   {r1:.3f} ({retrieve_top1}/{n})\n"
        f"  retrieval recall@{args.k}:   {rk:.3f} ({retrieve_topk}/{n})\n"
        f"  answer containment  (RAG):    {a_rag:.3f} ({answer_in_response}/{n})\n"
        f"  answer containment  (no-RAG): {a_no:.3f} ({answer_in_response_rerank}/{n})\n"
        f"  lift from retrieval:  {a_rag - a_no:+.3f}\n"
    )
    print(summary, flush=True)
    log_lines.append(summary)
    with open(args.out, "w") as f:
        f.write("\n".join(log_lines))
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    run()
