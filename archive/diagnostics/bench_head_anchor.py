#!/usr/bin/env python3
"""Head-anchor vs shipped cosine-anchor generation A/B.

Drop-in replacement for the score inside generate_rerank: keep
everything identical (n-gram candidate set, α, rep penalty, temperature,
top-k, seed) EXCEPT swap `cos(last_attn_output, cand_embedding)` for
`γ-scaled W_head logit at that candidate`.

If the head-anchor produces cleaner text than shipped cosine at the
same α, it's a real generation win — and we should wire it into the
repo. If not, γ-scale is a calibration fix only and doesn't belong in
the generation path.
"""
import os, sys, math, pickle, torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from lib.model import GenregLM, MAX_LEN, CHAR_CUTOFF, frozen_forward

GSCALE = os.path.join(os.path.dirname(_HERE), "LLM", "components",
                       "predhead", "predhead_wiki_causal_refit_gscale.pkl")


@torch.no_grad()
def generate_head_anchor(m, prompt, W_head, max_tokens=25, alpha=5.0,
                           temperature=0.8, top_k=30, rep_penalty=1.5,
                           min_tokens=5, seed=42):
    """Same control flow as generate_rerank EXCEPT score = α*head_logit
    + ng_logp instead of α*cosine + ng_logp."""
    torch.manual_seed(seed)
    tok = m.tokenize(prompt)
    gen = []
    for _ in range(max_tokens):
        if tok.shape[0] > MAX_LEN:
            tok = tok[-MAX_LEN:]
        ids_list = tok.cpu().tolist()
        cands = m._ngram_candidates(ids_list, K=top_k)
        if not cands:
            if ids_list[-1] in m.bigram:
                cands = [(next(iter(m.bigram[ids_list[-1]])), 0.5)]
            else:
                break
        positions = torch.arange(tok.shape[0], device=m.device)
        x = frozen_forward(m.embed, m.posenc, tok, positions)
        x = m.rerank_stack.forward(x, causal=True)
        last = x[-1]                               # (D,)
        cand_ids = [c[0] for c in cands]
        # Head logit restricted to valid ids (cap at W_head width)
        V = W_head.shape[1]
        safe_ids = [c if c < V else 0 for c in cand_ids]
        cand_logits = last @ W_head[:, safe_ids]   # (n_cands,)
        ng_logp = torch.tensor([math.log(p + 1e-10) for _, p in cands],
                                device=m.device)
        score = alpha * cand_logits + ng_logp
        if rep_penalty > 1.0 and gen:
            seen = set(gen[-15:])
            adj = torch.tensor(
                [-math.log(rep_penalty) if c in seen else 0.0
                 for c in cand_ids], device=m.device)
            score = score + adj
        score = score / max(temperature, 1e-6)
        probs = F.softmax(score, dim=0)
        pick = torch.multinomial(probs, 1).item()
        nxt = cand_ids[pick]
        gen.append(nxt)
        tok = torch.cat([tok, torch.tensor([nxt], device=m.device,
                                             dtype=torch.long)])
        if len(gen) >= min_tokens and nxt in (
                m.token_to_id.get(".", -1),
                m.token_to_id.get("?", -1),
                m.token_to_id.get("!", -1)):
            break
    return m.detokenize(gen), gen


def main():
    print("Loading GenregLM...", flush=True)
    m = GenregLM("./checkpoints", device="cuda")
    with open(GSCALE, "rb") as f:
        W_head = torch.from_numpy(pickle.load(f)["W_head"]).to(m.device)
    print(f"γ-scaled head: shape={tuple(W_head.shape)} "
          f"std={W_head.std().item():.4f}", flush=True)

    prompts = [
        "the king sat on the",
        "during the second world war",
        "the film was directed by",
        "she was born in",
    ]

    # Try multiple alphas for head-anchor (different scale than cosine!)
    # head_logit std ~1-5 per candidate, cos is [-1,1], so alpha
    # scales differently. Sweep a few.
    for alpha in (5.0, 25.0, 100.0, 500.0):
        print(f"\n{'=' * 70}\nα = {alpha}\n{'=' * 70}", flush=True)
        for p in prompts:
            torch.manual_seed(42)
            shipped_text, _ = m.generate_rerank(
                p, max_tokens=25, alpha=5.0, temperature=0.8,
                top_k=30, rep_penalty=1.5)
            head_text, _ = generate_head_anchor(
                m, p, W_head, max_tokens=25, alpha=alpha,
                temperature=0.8, top_k=30, rep_penalty=1.5, seed=42)
            print(f"\n[{p!r}]", flush=True)
            print(f"  SHIPPED  (α=5 cos):  {shipped_text}", flush=True)
            print(f"  HEAD-γ   (α={alpha}):    {head_text}", flush=True)


if __name__ == "__main__":
    main()
