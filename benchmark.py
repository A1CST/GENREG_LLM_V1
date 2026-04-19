#!/usr/bin/env python3
"""Benchmark script for the GENREG LM.

Measures:
  1. Load time
  2. Model details (parameter counts, n-gram sizes)
  3. Memory footprint
  4. Generation throughput (tokens/sec)
  5. Top-1 / top-5 next-token accuracy on a held-out stream (rerank path)
  6. Generation diversity (rerank vs pure n-gram)
  7. Sample generations

Everything in this benchmark exercises the gradient-free rerank path.

Usage:
  python benchmark.py              # full suite
  python benchmark.py --quick      # speed + small sample
  python benchmark.py --device cpu # force CPU
  python benchmark.py --both       # CPU and GPU side by side
"""
import os, sys, time, math, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from lib.model import GenregLM, MAX_LEN, frozen_forward


EVAL_PROMPTS = [
    "the king sat on the",
    "during the second world war",
    "she was born in",
    "in the year",
    "the film was directed by",
    "it was the first",
    "he played for the",
    "the battle of",
    "the president of the",
    "during this time",
    "the first world war was",
    "the city of london",
    "he was born in",
    "the new york times",
    "the team won the championship",
    "the album was released",
]


def _count_params(*tensors):
    total = 0
    for t in tensors:
        if t is None:
            continue
        if isinstance(t, torch.Tensor):
            total += t.numel()
        else:
            total += int(np.prod(np.asarray(t).shape))
    return total


def bench_load(ckpt_dir, device):
    t0 = time.time()
    model = GenregLM(ckpt_dir, device=device)
    load_s = time.time() - t0
    print(f"  load time: {load_s:.2f} s")
    return model, load_s


def bench_model_details(model, ckpt_dir):
    """Architecture and parameter counts. All of these were evolved."""
    emb = model.embed
    emb_params = _count_params(
        emb.hash_in, emb.W_skip, emb.W_enc, emb.enc_b,
        emb.W_out, emb.out_b,
        emb.act_ids, emb.act_p1, emb.act_p2, emb.act_p3, emb.act_p4)
    pos = model.posenc
    pos_params = _count_params(
        pos.P, pos.dim_gain,
        pos.act_ids, pos.act_p1, pos.act_p2, pos.act_p3, pos.act_p4)

    def _layer_params(l):
        return _count_params(
            l.W_Q, l.W_K, l.W_V, l.W_O, l.head_gain,
            l.logit_act_ids, l.logit_act_p1, l.logit_act_p2,
            l.logit_act_p3, l.logit_act_p4)
    ce_params = sum(_layer_params(l) for l in model.attn.layers)
    rerank_layers = model.rerank_stack.layers[len(model.attn.layers):]
    rerank_params = sum(_layer_params(l) for l in rerank_layers)

    ngram_entries = {
        "bigram": len(model.bigram),
        "trigram": len(model.trigram),
        "fourgram": len(model.fourgram),
        "fivegram": len(model.fivegram),
    }
    disk_mb = 0.0
    for root, _, files in os.walk(ckpt_dir):
        for name in files:
            disk_mb += os.path.getsize(os.path.join(root, name))
    disk_mb /= 1024 ** 2

    evolved = emb_params + pos_params + ce_params + rerank_params

    def _fmt(n):
        return f"{n:>12,}"

    print(f"  vocabulary size V:            {model.V:>12,}")
    print(f"  context window MAX_LEN:       {MAX_LEN:>12}")
    print(f"  embedding dim D:              {model.embed.D:>12}")
    print()
    print(f"  embedding parameters:         {_fmt(emb_params)}")
    print(f"  positional encoding:          {_fmt(pos_params)}")
    print(f"  attention CE layers:          {_fmt(ce_params)}  "
          f"({len(model.attn.layers)} layers)")
    print(f"  attention rerank layers:      {_fmt(rerank_params)}  "
          f"({len(rerank_layers)} layers)")
    print(f"  -------------------------------------------")
    print(f"  ALL LEARNED PARAMETERS:       {_fmt(evolved)}")
    print(f"  (every parameter produced by gradient-free evolution)")
    print()
    print(f"  n-gram table entries (counted, not learned):")
    for k, v in ngram_entries.items():
        print(f"    {k:10s}   {v:>12,}")
    print()
    print(f"  checkpoint disk footprint:    {disk_mb:.1f} MB")


def bench_memory(device):
    if device == "cuda" and torch.cuda.is_available():
        mb = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"  GPU memory: {mb:.1f} MB")
        return mb
    import resource
    mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  process RSS: {mb:.1f} MB")
    return mb


def bench_throughput(model, n_tokens=100, n_runs=3):
    times = []
    for i in range(n_runs):
        prompt = EVAL_PROMPTS[i % len(EVAL_PROMPTS)]
        t0 = time.time()
        model.generate_rerank(prompt, max_tokens=n_tokens, alpha=5.0,
                               temperature=0.7, top_k=30)
        times.append(time.time() - t0)
    avg = sum(times) / len(times)
    tps = n_tokens / avg
    print(f"  avg: {avg:.2f} s for {n_tokens} tokens  -->  {tps:.1f} tok/s")
    return tps


def bench_next_token_accuracy(model, ckpt_dir):
    """Rerank-path top-1 / top-5 on a held-out token stream.

    For each position we construct the n-gram candidate set, score it
    with `alpha * cos(attn_feat, cand_emb) + log ngram_prob`, and check
    whether the true next token is ranked first or in the top 5. Char
    targets and positions where the n-gram cascade never proposes the
    true token are excluded (same policy the rerank fitness was trained
    under).
    """
    held_path = os.path.join(ckpt_dir, "heldout_sample.pkl")
    if not os.path.exists(held_path):
        print("  (heldout_sample.pkl not found, skipping)")
        return
    with open(held_path, "rb") as f:
        held = pickle.load(f)
    windows = np.asarray(held["windows"])
    EVAL_UP_TO = 256
    emb_table_n = model._emb_table_normalized()
    alpha, topk = 5.0, 30
    n_seen = n_hit1 = n_hit5 = n_inset = 0

    with torch.no_grad():
        for w in windows:
            ids = torch.tensor(w, device=model.device, dtype=torch.long)
            pos = torch.arange(len(ids), device=model.device)
            x = frozen_forward(model.embed, model.posenc, ids, pos)
            x = model.rerank_stack.forward(x, causal=True)
            x_n = F.normalize(x, dim=1)
            for t in range(min(EVAL_UP_TO, len(ids) - 1)):
                true_tok = int(ids[t + 1].item())
                if true_tok < 96:
                    continue
                context = w[: t + 1].tolist()
                cands = model._ngram_candidates(context, K=topk)
                if not cands:
                    continue
                cand_ids = [c[0] for c in cands]
                cand_probs = [c[1] for c in cands]
                if true_tok not in cand_ids:
                    n_seen += 1
                    continue
                cand_t = torch.tensor(cand_ids, device=model.device)
                cand_embs = emb_table_n[cand_t]
                cos = (cand_embs @ x_n[t:t + 1].t()).squeeze(1)
                ng_logp = torch.tensor(
                    [math.log(p + 1e-10) for p in cand_probs],
                    device=model.device)
                score = alpha * cos + ng_logp
                order = score.argsort(descending=True).cpu().tolist()
                ranked = [cand_ids[i] for i in order]
                if ranked[0] == true_tok:
                    n_hit1 += 1
                if true_tok in ranked[:5]:
                    n_hit5 += 1
                n_inset += 1
                n_seen += 1

    if n_seen == 0:
        print("  (no evaluable positions)")
        return
    top1 = n_hit1 / n_seen
    top5 = n_hit5 / n_seen
    in_set = n_inset / n_seen
    print(f"  held-out positions scored: {n_seen}")
    print(f"  true token proposed by n-gram cascade: {in_set:.3f}")
    print(f"  top-1 (rerank path): {top1:.3f}")
    print(f"  top-5 (rerank path): {top5:.3f}")


def bench_diversity(model, seeds=(42, 100), n_tokens=25):
    """Rerank vs pure n-gram on the same prompts, same seeds."""
    def _summarize(ids):
        if not ids:
            return 0.0, 0
        uniq = len(set(ids)) / len(ids)
        cur = run = 1
        for i in range(1, len(ids)):
            if ids[i] == ids[i - 1]:
                cur += 1; run = max(run, cur)
            else:
                cur = 1
        return uniq, run

    configs = [
        ("rerank 4-layer (alpha=5)",
         lambda p: model.generate_rerank(p, max_tokens=n_tokens, alpha=5.0,
                                          temperature=0.7, top_k=30)),
        ("pure n-gram (alpha=0)",
         lambda p: model.generate_rerank(p, max_tokens=n_tokens, alpha=0.0,
                                          temperature=0.7, top_k=30)),
    ]
    for name, fn in configs:
        divs, runs = [], []
        for s in seeds:
            torch.manual_seed(s)
            for p in EVAL_PROMPTS[:8]:
                _, ids = fn(p)
                d, r = _summarize(ids)
                divs.append(d); runs.append(r)
        print(f"  {name:28s}  div={np.mean(divs):.3f}  "
              f"max_repeat={np.mean(runs):.2f}")


def bench_samples(model, n_tokens=25):
    """Print a few sample generations for eyeball inspection."""
    print("\n  Sample generations — rerank (alpha=5, temp=0.7):\n")
    for p in EVAL_PROMPTS[:5]:
        text, _ = model.generate_rerank(p, max_tokens=n_tokens, alpha=5.0,
                                         temperature=0.7, top_k=30)
        print(f"  > {p}")
        print(f"  < {p} {text}")
        print()


CHATBOT_PROMPTS = [
    "what is the capital of france",
    "who wrote romeo and juliet",
    "a telescope is a device that",
    "albert einstein was a",
    "napoleon bonaparte was born in",
    "the reason the sky is blue is",
    "democracy is a system of",
    "once upon a time there was a king who",
    "music is an art form that",
    "history is the study of",
]


def bench_chatbot_shape(model, seeds=(42, 100), n_tokens=40):
    """Score chatbot-shape metrics on question/definition prompts.

    - natural_stop: fraction of responses that end on . / ? / ! / <eos>
    - answer_shape: first token looks like an answer-start
    - mean length: avg tokens per response
    - topic_rel: cosine of prompt-mean-embedding to response-mean-embedding
    """
    emb_n = model._emb_table_normalized()
    eos = model.token_to_id.get("<eos>", -1)
    stops = {model.token_to_id.get(".", -1),
             model.token_to_id.get("?", -1),
             model.token_to_id.get("!", -1),
             eos}
    answer_starts = {"is", "was", "were", "are", "a", "an", "the",
                     "in", "on", "at", "of", "by", "from", "for",
                     "he", "she", "it", "they", "we"}

    tot = n_stop = n_ans = 0
    tot_len = 0
    rels = []
    for seed in seeds:
        torch.manual_seed(seed)
        for p in CHATBOT_PROMPTS:
            text, ids = model.generate_rerank(p, max_tokens=n_tokens,
                                               alpha=5.0, temperature=0.7,
                                               top_k=30)
            tot += 1
            tot_len += len(ids)
            # natural stop
            if ids and ids[-1] in stops:
                n_stop += 1
            # answer shape
            first_word = text.split()[0] if text.split() else ""
            if first_word in answer_starts or (first_word and first_word[:1].isupper()):
                n_ans += 1
            # topic relevance
            p_ids = model.tokenize(p).cpu().tolist()
            if p_ids and ids:
                pv = emb_n[torch.tensor(p_ids, device=model.device)].mean(dim=0)
                gv = emb_n[torch.tensor(ids, device=model.device)].mean(dim=0)
                pv = F.normalize(pv.unsqueeze(0), dim=1)
                gv = F.normalize(gv.unsqueeze(0), dim=1)
                rels.append((pv @ gv.t()).item())

    print(f"  prompts x seeds: {len(CHATBOT_PROMPTS)} x {len(seeds)} = {tot}")
    print(f"  natural stop rate: {n_stop}/{tot}  ({100*n_stop/tot:.0f}%)")
    print(f"  answer-shaped output: {n_ans}/{tot}  ({100*n_ans/tot:.0f}%)")
    print(f"  mean response length: {tot_len/tot:.1f} tokens")
    if rels:
        print(f"  mean prompt->response topic cosine: {np.mean(rels):.3f}")


def bench_chatbot_samples(model, seeds=(42,), n_tokens=40):
    """Print a handful of chatbot-shape completions."""
    print("\n  Chatbot sample generations (alpha=5, temp=0.7):\n")
    for seed in seeds:
        torch.manual_seed(seed)
        for p in CHATBOT_PROMPTS[:5]:
            text, _ = model.generate_rerank(p, max_tokens=n_tokens,
                                             alpha=5.0, temperature=0.7,
                                             top_k=30)
            print(f"  Q: {p}")
            print(f"  A: {text}\n")


def bench_rag_and_extractive(model, n_questions=150, k=3,
                              squad_dev_path=None):
    """Retrieval + RAG generation + extractive QA, measured on SQuAD dev."""
    if model._rag is None:
        print("  (rag_index.pkl missing — skipping RAG metrics)")
        return

    if squad_dev_path is None:
        squad_dev_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "LLM", "data_raw", "squad_dev.json")
    if not os.path.exists(squad_dev_path):
        print(f"  (SQuAD dev not found at {squad_dev_path} — skipping)")
        return

    import json
    with open(squad_dev_path) as f:
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
    if 0 < n_questions < len(cases):
        cases = list(rng.choice(cases, size=n_questions, replace=False))

    rec1 = reck = 0
    n_norag = n_rag = n_extr = 0
    n = len(cases)

    t0 = time.time()
    for i, c in enumerate(cases):
        hits = model.retrieve(c["q"], k=k)
        if hits and hits[0]["text"] == c["ctx"]:
            rec1 += 1
        if any(h["text"] == c["ctx"] for h in hits):
            reck += 1

        torch.manual_seed(42 + i)
        gen_txt, _ = model.generate_rerank(
            c["q"], max_tokens=30, alpha=5.0, temperature=0.7, top_k=30)
        torch.manual_seed(42 + i)
        rag_txt, _, _ = model.generate_rag(
            c["q"], max_tokens=30, k=1, alpha=5.0,
            temperature=0.7, top_k=30)
        extr_txt, _ = model.generate_qa(c["q"], k_retrieve=k, max_span=8)

        golds_l = [a.lower() for a in c["answers"]]
        if any(a in gen_txt.lower() for a in golds_l):
            n_norag += 1
        if any(a in rag_txt.lower() for a in golds_l):
            n_rag += 1
        if extr_txt and any(a in extr_txt.lower() for a in golds_l):
            n_extr += 1

    elapsed = time.time() - t0
    print(f"  evaluated {n} SQuAD dev questions in {elapsed:.0f}s, k={k}")
    print()
    print(f"  retrieval recall@1:    {rec1/n:.3f}  ({rec1}/{n})")
    print(f"  retrieval recall@{k}:    {reck/n:.3f}  ({reck}/{n})")
    print()
    print(f"  answer containment (any gold span is substring of reply):")
    print(f"    no-RAG generation:   {n_norag/n:.3f}  ({n_norag}/{n})")
    print(f"    RAG generation:      {n_rag/n:.3f}  ({n_rag}/{n})")
    print(f"    extractive:          {n_extr/n:.3f}  ({n_extr}/{n})")


def main():
    ap = argparse.ArgumentParser(description="GENREG LM benchmark")
    ap.add_argument("--quick", action="store_true",
                     help="Skip slow tests (accuracy + sample gen)")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--both", action="store_true",
                     help="Run on both CPU and GPU, compare")
    ap.add_argument("--rag-n", type=int, default=150,
                     help="SQuAD dev questions for RAG/extractive bench")
    ap.add_argument("--rag-k", type=int, default=3,
                     help="retrieval top-k for RAG bench")
    args = ap.parse_args()

    if args.both:
        for dev in ["cpu", "cuda"]:
            if dev == "cuda" and not torch.cuda.is_available():
                continue
            print(f"\n{'#'*60}\n# {dev.upper()}\n{'#'*60}")
            args.device = dev
            _run_single(args, dev)
        return

    _run_single(args, args.device)


def _run_single(args, device):
    print("=" * 60)
    print(f"GENREG LM — Benchmark ({device})")
    print("=" * 60)

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "checkpoints")

    print("\n[1] LOAD")
    model, _ = bench_load(ckpt_dir, device)

    print("\n[2] MODEL DETAILS")
    bench_model_details(model, ckpt_dir)

    print("\n[3] MEMORY")
    bench_memory(device)

    print("\n[4] THROUGHPUT")
    n_tokens = args.tokens if not args.quick else 30
    bench_throughput(model, n_tokens=n_tokens, n_runs=3)

    if not args.quick:
        print("\n[5] NEXT-TOKEN ACCURACY (rerank path, held-out stream)")
        bench_next_token_accuracy(model, ckpt_dir)

    print("\n[6] GENERATION DIVERSITY")
    bench_diversity(model)

    print("\n[7] CHATBOT SHAPE METRICS")
    bench_chatbot_shape(model)

    if not args.quick:
        print("\n[8] RAG + EXTRACTIVE ON SQUAD DEV")
        bench_rag_and_extractive(model, n_questions=args.rag_n,
                                  k=args.rag_k)
        print("\n[9] SAMPLE GENERATIONS (wiki-continuation)")
        bench_samples(model)
        print("\n[10] CHATBOT SAMPLE GENERATIONS")
        bench_chatbot_samples(model)

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
