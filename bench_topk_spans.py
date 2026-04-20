#!/usr/bin/env python3
"""Concatenate top-K best-scoring spans (across all retrieval hits)
instead of returning only the single best. Strictly non-decreasing
in answer containment — if gold was in top-1 before, it's still in
top-K; and sometimes gold is in the #2 or #3 scoring span.

Measures containment on 300 dev for K ∈ {1, 2, 3, 5}.
"""
import os, sys, json, time, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.model import GenregLM, CHAR_CUTOFF


def collect_topk_spans(m, query, k_retrieve=3, K=3, max_span=100):
    """Return the top-K spans (detokenized) across the retrieval pool,
    ranked by the same scoring path as extract_answer but keeping K
    instead of just 1."""
    import math as _math
    if m._rag is None:
        return []
    hits = m.retrieve(query, k=k_retrieve)
    if not hits:
        return []
    emb_n = m._emb_table_normalized()
    q_ids_t = m.tokenize(query)
    q_tok_ids = set(int(t) for t in q_ids_t.cpu().tolist())
    STRUCTURAL = {66, 67}
    q_content = {t for t in q_tok_ids
                  if t >= CHAR_CUTOFF or t in m._ALLOWED_PUNCT_IDS}
    q_content.difference_update(STRUCTURAL)
    q_content_rare = {t for t in q_content
                       if float(m._tok_weight[t].item()) > 0.3}
    q_content_tokens_list = list(q_content_rare) or list(q_content) or list(q_tok_ids)
    q_vec = None
    if q_content_tokens_list:
        q_ct_tens = torch.tensor(q_content_tokens_list, device=m.device)
        q_w = m._tok_weight[q_ct_tens].unsqueeze(1)
        q_vec = (emb_n[q_ct_tens] * q_w).sum(dim=0) / (q_w.sum() + 1e-8)
        q_vec = F.normalize(q_vec.unsqueeze(0), dim=1).squeeze(0)
    qtype = m._classify_question(query)
    tok_weight_np = m._tok_weight.cpu().numpy()
    token_df_dict = m._rag["token_df"] if m._rag else {}
    N_docs_val = m._rag["N_docs"] if m._rag else 1

    # Normalize retrieval scores to [0,1]
    scores_arr = np.array([h.get("score", 1.0) for h in hits],
                           dtype=np.float32)
    if scores_arr.size and scores_arr.max() > scores_arr.min():
        retr_score_by_h = (scores_arr - scores_arr.min()) / (
            scores_arr.max() - scores_arr.min() + 1e-8)
    else:
        retr_score_by_h = np.ones(scores_arr.size, dtype=np.float32)

    all_candidates = []  # (score, text, meta)
    for h_idx, h in enumerate(hits):
        passage = h.get("chunk_token_ids") or h["token_ids"]
        hit_positions = [i for i, t in enumerate(passage)
                          if t in q_content_rare]
        if not hit_positions:
            hit_positions = [i for i, t in enumerate(passage)
                              if t in q_content]
        content_idx = [i for i, t in enumerate(passage)
                        if t not in STRUCTURAL
                        and (t >= CHAR_CUTOFF or t in m._ALLOWED_PUNCT_IDS)]
        if not hit_positions:
            hit_positions = list(content_idx) or [0]
        for L in range(1, max_span + 1):
            for ci_start in range(len(content_idx) - L + 1):
                start = content_idx[ci_start]
                span = [passage[i] for i in content_idx[ci_start:ci_start + L]]
                if all(t in q_content for t in span):
                    continue
                if any(t == 1 for t in span):
                    continue
                rarity_sum = float(sum(tok_weight_np[t] for t in span))
                semantic = 0.0
                if q_vec is not None:
                    s_tens = torch.tensor(span, device=m.device)
                    s_w = m._tok_weight[s_tens].unsqueeze(1)
                    s_vec = (emb_n[s_tens] * s_w).sum(dim=0) / (s_w.sum() + 1e-8)
                    s_vec = F.normalize(s_vec.unsqueeze(0), dim=1).squeeze(0)
                    semantic = float((s_vec * q_vec).sum().item())
                inv_u = -(abs(semantic - 0.5) ** 2) + 0.25
                numeric = 1.0 if m._span_is_numeric(span) else 0.0
                is_when = 1.0 if qtype in ("when", "year", "num") else 0.0
                qt = 0.0
                if is_when and numeric > 0.5: qt = 3.0
                elif is_when: qt = -0.8
                # v13.3 heuristic
                score = (rarity_sum + qt + inv_u + 0.5
                          - 0.05 * max(0, L - 5)
                          + 1.0 * float(retr_score_by_h[h_idx]))
                text = m.detokenize(span)
                all_candidates.append((score, text, h_idx, start, L))
    # Dedupe identical text; keep highest score
    all_candidates.sort(key=lambda x: -x[0])
    seen = set(); top = []
    for sc, text, hi, s, L in all_candidates:
        if text in seen:
            continue
        seen.add(text)
        top.append(text)
        if len(top) >= K:
            break
    return top


def main():
    m = GenregLM("./checkpoints", device="cuda")
    dev_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "../LLM/data_raw/squad_dev.json")
    with open(dev_path) as f:
        dev = json.load(f)
    cases = []
    for art in dev["data"]:
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                if qa.get("answers"):
                    cases.append({"q": qa["question"], "ctx": para["context"],
                                   "answers": [a["text"] for a in qa["answers"]]})
    rng = np.random.default_rng(7)
    cases = list(rng.choice(cases, size=300, replace=False))

    Ks = [1, 2, 3, 5]
    hits_k = {K: 0 for K in Ks}
    t0 = time.time()
    for i, c in enumerate(cases):
        top = collect_topk_spans(m, c["q"], k_retrieve=3, K=5)
        gold_l = [a.lower() for a in c["answers"]]
        for K in Ks:
            joined = " ".join(top[:K]).lower()
            if any(a in joined for a in gold_l):
                hits_k[K] += 1
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            line = f"  {i+1}/{len(cases)} ({el:.0f}s)  "
            for K in Ks:
                line += f"K{K}={hits_k[K]/(i+1):.3f}  "
            print(line, flush=True)
    n = len(cases)
    print(f"\n{'='*60}", flush=True)
    print(f"Top-K span concatenation on {n} SQuAD dev Q", flush=True)
    print(f"{'='*60}", flush=True)
    for K in Ks:
        print(f"  K={K}: {hits_k[K]/n:.3f} ({hits_k[K]}/{n})", flush=True)
    print(f"\n  (v13.3 single-span baseline: 82/300 = 0.273)", flush=True)


if __name__ == "__main__":
    main()
