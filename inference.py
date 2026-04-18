#!/usr/bin/env python3
"""Interactive inference for the GENREG LM.

N-gram cascade proposes K grammatical candidates. The evolved 4-layer
attention stack picks among them by cosine similarity to its last-token
feature. No ridge regression. No gradients. Anywhere.

Usage:
  python inference.py                            # interactive REPL
  python inference.py --prompt "hello world"    # one-shot
  python inference.py --alpha 0                 # pure n-gram (attention off)
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from lib.model import GenregLM


def one_shot(model, prompt, max_tokens, temperature, top_k, rep_penalty,
             alpha, top_p):
    t0 = time.time()
    text, _ = model.generate_rerank(
        prompt, max_tokens=max_tokens, alpha=alpha,
        temperature=temperature, top_k=top_k,
        rep_penalty=rep_penalty, top_p=top_p)
    dt = time.time() - t0
    print(f"\n> {prompt}")
    print(f"< {text}")
    tps = max_tokens / max(dt, 1e-6)
    print(f"\n({max_tokens} tokens in {dt:.2f}s, {tps:.1f} tok/s)")


def repl(model, args):
    print()
    print("=" * 60)
    print("  GENREG LM — gradient-free rerank")
    print("=" * 60)
    print("  Type a prompt. The 4-layer attention stack reranks n-gram")
    print("  candidates by semantic cosine.")
    print("  Commands:")
    print("    /temp <float>   sampling temperature (default 0.7)")
    print("    /topk  <int>    n-gram candidates to consider (default 30)")
    print("    /len   <int>    max generation length (default 30)")
    print("    /alpha <float>  attention weight: 0=pure n-gram, 5=default")
    print("    /rep   <float>  repetition penalty (default 1.5)")
    print("    /topp  <float>  nucleus sampling cutoff (0<p<1, or 0 to disable)")
    print("    /quit           exit")
    print()

    temp = args.temperature
    topk = args.top_k
    maxt = args.max_tokens
    alpha = args.alpha
    rep = args.rep_penalty
    topp = args.top_p

    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0][1:]
            if cmd in ("quit", "exit", "q"):
                break
            try:
                if cmd == "temp" and len(parts) == 2:
                    temp = float(parts[1]); print(f"  temperature = {temp}")
                elif cmd == "topk" and len(parts) == 2:
                    topk = int(parts[1]); print(f"  top_k = {topk}")
                elif cmd == "len" and len(parts) == 2:
                    maxt = int(parts[1]); print(f"  max_tokens = {maxt}")
                elif cmd == "alpha" and len(parts) == 2:
                    alpha = float(parts[1]); print(f"  alpha = {alpha}")
                elif cmd == "rep" and len(parts) == 2:
                    rep = float(parts[1]); print(f"  rep_penalty = {rep}")
                elif cmd == "topp" and len(parts) == 2:
                    v = float(parts[1])
                    topp = None if v <= 0 or v >= 1 else v
                    print(f"  top_p = {topp}")
                else:
                    print(f"  unknown: /{cmd}")
            except ValueError:
                print("  bad arg")
            continue
        one_shot(model, prompt, maxt, temp, topk, rep, alpha, topp)
        print()


def main():
    ap = argparse.ArgumentParser(description="GENREG LM inference")
    ap.add_argument("--prompt", type=str, default=None,
                     help="One-shot prompt. Omit for interactive mode.")
    ap.add_argument("--max-tokens", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=5.0,
                     help="attention cosine weight: 0=pure n-gram, 5=default")
    ap.add_argument("--top-p", type=float, default=None,
                     help="nucleus sampling cutoff (0<p<1)")
    ap.add_argument("--rep-penalty", type=float, default=1.5)
    ap.add_argument("--device", type=str, default="cpu",
                     help="'cpu' (default) or 'cuda'")
    args = ap.parse_args()

    device = args.device
    print(f"device: {device}")

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "checkpoints")
    t0 = time.time()
    print("loading...")
    model = GenregLM(ckpt_dir, device=device)
    n_ce = len(model.attn)
    n_total = len(model.rerank_stack)
    n_rerank = n_total - n_ce
    print(f"loaded in {time.time()-t0:.1f}s  (V={model.V}, "
          f"attn layers: {n_ce} CE + {n_rerank} rerank)")

    if args.prompt:
        one_shot(model, args.prompt, args.max_tokens, args.temperature,
                 args.top_k, args.rep_penalty, args.alpha, args.top_p)
    else:
        repl(model, args)


if __name__ == "__main__":
    main()
