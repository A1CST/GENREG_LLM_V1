#!/usr/bin/env python3
"""Chatbot-shape diagnostic on the current clean stack.

Runs 30 chatbot-style prompts at 2 seeds and measures:
- answer-shapedness: does output start like an answer (start with a verb
  like 'is' / 'was' / 'are', a preposition, or a named entity)?
- length control: does a natural stopping point emerge, or does it run
  on to max_tokens?
- topic relevance: cosine between prompt topic embedding and
  response topic embedding (higher = more on-topic)
- self-consistency: does the response stay on its own opening topic?

Nothing here changes the model; it just scores what the current stack
already produces, so we can see which failure mode is dominant before
building any chatbot-specific training.
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from lib.model import GenregLM


CHATBOT_PROMPTS = [
    # Factual Q
    "what is the capital of france",
    "who wrote romeo and juliet",
    "when was the battle of waterloo",
    "where is the eiffel tower located",
    "what language is spoken in brazil",
    # Definitional
    "a telescope is a device that",
    "photosynthesis is the process by which",
    "a president is a person who",
    "democracy is a system of",
    "an atom is composed of",
    # Biographical
    "albert einstein was a",
    "marie curie is known for",
    "napoleon bonaparte was born in",
    "leonardo da vinci was an",
    "shakespeare wrote plays about",
    # Explanation
    "the reason the sky is blue is",
    "water boils at a temperature of",
    "the human heart pumps",
    "gravity is a force that",
    "the first world war began in",
    # Story-start
    "once upon a time there was a king who",
    "in a distant galaxy a ship was",
    "the old man walked into the forest and",
    "she opened the letter and read",
    "the scientist looked at the microscope and",
    # Topic completion
    "music is an art form that",
    "the internet is a global",
    "a computer processes",
    "history is the study of",
    "mathematics is the science of",
]


def topic_emb(ids, emb_n):
    if not ids:
        return None
    v = emb_n[torch.tensor(ids, device=emb_n.device)]
    m = v.mean(dim=0, keepdim=True)
    return F.normalize(m, dim=1)


def looks_like_answer(text):
    """Crude heuristic. Does the first token start an answer-shaped
    continuation? This catches 'is / was / are / in / of / the' and
    common proper-noun starts."""
    if not text:
        return False
    first = text.split()[0] if text.split() else ""
    answer_starts = {"is", "was", "were", "are", "a", "an", "the",
                     "in", "on", "at", "of", "by", "from", "for",
                     "he", "she", "it", "they", "we"}
    return first in answer_starts or (first and first[0].isupper())


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-tokens", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 100])
    ap.add_argument("--out-log", default="diagnostic_chatbot.log")
    args = ap.parse_args()

    device = args.device
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "checkpoints")
    print(f"Loading model on {device}...", flush=True)
    m = GenregLM(ckpt_dir, device=device)
    emb_n = m._emb_table_normalized()
    print(f"Loaded. {len(m.attn)} CE + {len(m.rerank_stack)-len(m.attn)} rerank layers",
          flush=True)

    ce_punct = m.token_to_id.get(".", None)
    eos_tok = m.token_to_id.get("<eos>", -1)

    log_lines = []
    totals = {
        "n": 0, "answer_shape": 0,
        "natural_stop": 0, "ran_to_max": 0,
        "topic_rel": [], "self_consist": [],
    }

    for seed in args.seeds:
        torch.manual_seed(seed)
        for p in CHATBOT_PROMPTS:
            text, gen_ids = m.generate_rerank(
                p, max_tokens=args.max_tokens, alpha=args.alpha,
                temperature=args.temp, top_k=args.topk)
            prompt_ids = m.tokenize(p).cpu().tolist()
            pt = topic_emb(prompt_ids, emb_n)
            gt = topic_emb(gen_ids, emb_n)
            if pt is not None and gt is not None:
                rel = (pt @ gt.t()).item()
            else:
                rel = 0.0
            if len(gen_ids) >= 6:
                half = len(gen_ids) // 2
                ot = topic_emb(gen_ids[:half], emb_n)
                ct = topic_emb(gen_ids[half:], emb_n)
                self_c = (ot @ ct.t()).item() if ot is not None and ct is not None else 0.0
            else:
                self_c = 0.0

            ans_shape = looks_like_answer(text)
            ran_to_max = len(gen_ids) >= args.max_tokens - 1
            natural_stop = eos_tok in gen_ids or (
                ce_punct is not None and ce_punct in gen_ids[-5:])

            totals["n"] += 1
            totals["answer_shape"] += int(ans_shape)
            totals["natural_stop"] += int(natural_stop)
            totals["ran_to_max"] += int(ran_to_max)
            totals["topic_rel"].append(rel)
            totals["self_consist"].append(self_c)

            line = (f"[seed={seed}] Q: {p}\n"
                    f"  A: {text}\n"
                    f"  ans_shape={ans_shape}  natural_stop={natural_stop}  "
                    f"topic_rel={rel:.3f}  self_consist={self_c:.3f}  "
                    f"len={len(gen_ids)}")
            print(line, flush=True)
            log_lines.append(line)

    n = totals["n"]
    avg_rel = sum(totals["topic_rel"]) / n
    avg_sc = sum(totals["self_consist"]) / n
    summary = (
        "\n" + "=" * 60 + "\n"
        f"DIAGNOSTIC SUMMARY (n={n})\n"
        + "=" * 60 + "\n"
        f"  answer-shaped output:    {totals['answer_shape']}/{n}  "
        f"({100*totals['answer_shape']/n:.0f}%)\n"
        f"  natural stop (. or EOS): {totals['natural_stop']}/{n}  "
        f"({100*totals['natural_stop']/n:.0f}%)\n"
        f"  ran to max_tokens:       {totals['ran_to_max']}/{n}  "
        f"({100*totals['ran_to_max']/n:.0f}%)\n"
        f"  mean prompt->answer topic cosine: {avg_rel:.3f}\n"
        f"  mean answer self-consistency:     {avg_sc:.3f}\n"
    )
    print(summary, flush=True)
    log_lines.append(summary)

    with open(args.out_log, "w") as f:
        f.write("\n".join(log_lines))
    print(f"saved -> {args.out_log}", flush=True)


if __name__ == "__main__":
    run()
