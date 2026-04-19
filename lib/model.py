"""Inference-only model classes for the GENREG LM.

Loads frozen checkpoints and runs forward pass. No training code.
"""
import os, math, pickle, glob, re
import numpy as np
import torch
import torch.nn.functional as F
from .encoder import apply_evolved_activations

MAX_LEN = 512
EMBED_DIM = 768
N_HEADS = 6
HEAD_DIM = 128
CHAR_CUTOFF = 96   # ids < 96 are specials + chars + punctuation


class FrozenEmbed:
    """Loads the evolved embedding: PPMI-SVD hash + evolved skip+encoder."""
    def __init__(self, ckpt_path, device):
        with open(ckpt_path, "rb") as f:
            st = pickle.load(f)
        d = torch.device(device)
        # hash_in is stored as float16 on disk to halve file size;
        # upcast to float32 for computation.
        self.hash_in = torch.from_numpy(
            st["hash_in"].astype(np.float32)).to(d)
        self.W_skip = torch.from_numpy(st["W_skip"]).to(d)
        self.skip_gain = float(st["skip_gain"])
        self.W_enc = torch.from_numpy(st["W_enc"]).to(d)
        self.enc_b = torch.from_numpy(st["enc_b"]).to(d)
        self.act_ids = torch.from_numpy(st["act_ids"]).to(d)
        self.act_p1 = torch.from_numpy(st["act_p1"]).to(d)
        self.act_p2 = torch.from_numpy(st["act_p2"]).to(d)
        self.act_p3 = torch.from_numpy(st["act_p3"]).to(d)
        self.act_p4 = torch.from_numpy(st["act_p4"]).to(d)
        self.W_out = torch.from_numpy(st["W_out"]).to(d)
        self.out_b = torch.from_numpy(st["out_b"]).to(d)
        self.V, self.K = self.hash_in.shape
        self.H = self.W_enc.shape[0]
        self.D = self.W_out.shape[0]
        self.dev = d

    @torch.no_grad()
    def embed(self, ids, chunk=4096):
        ids = ids.to(self.dev)
        N = ids.shape[0]
        out = torch.empty(N, self.D, device=self.dev)
        enc_T = self.W_enc.t(); out_T = self.W_out.t(); skip_T = self.W_skip.t()
        H = self.H
        for s in range(0, N, chunk):
            e = min(s + chunk, N); n = e - s
            h_in = self.hash_in[ids[s:e]]
            skip = (h_in @ skip_T) * self.skip_gain
            hidden = h_in @ enc_T + self.enc_b
            h_arr = hidden.t().reshape(H, n)
            p1 = self.act_p1.reshape(H, 1).expand(H, n).contiguous()
            p2 = self.act_p2.reshape(H, 1).expand(H, n).contiguous()
            p3 = self.act_p3.reshape(H, 1).expand(H, n).contiguous()
            p4 = self.act_p4.reshape(H, 1).expand(H, n).contiguous()
            a = apply_evolved_activations(h_arr, self.act_ids, p1, p2, p3, p4)
            out[s:e] = skip + a.reshape(H, n).t() @ out_T + self.out_b
        return out


class FrozenPosEnc:
    """Evolved positional encoder."""
    def __init__(self, ckpt_path, device):
        with open(ckpt_path, "rb") as f:
            st = pickle.load(f)
        d = torch.device(device)
        self.P = torch.from_numpy(st["P"]).to(d)
        self.dim_gain = torch.from_numpy(st["dim_gain"]).to(d)
        self.global_gain = float(st["global_gain"])
        self.act_ids = torch.from_numpy(st["act_ids"]).to(d)
        self.act_p1 = torch.from_numpy(st["act_p1"]).to(d)
        self.act_p2 = torch.from_numpy(st["act_p2"]).to(d)
        self.act_p3 = torch.from_numpy(st["act_p3"]).to(d)
        self.act_p4 = torch.from_numpy(st["act_p4"]).to(d)
        self.D = self.P.shape[1]
        self.dev = d

    @torch.no_grad()
    def encode(self, positions):
        positions = positions.to(self.dev)
        N = positions.shape[0]
        x = self.P[positions].t().contiguous()
        D = self.D
        p1 = self.act_p1.reshape(D, 1).expand(D, N).contiguous()
        p2 = self.act_p2.reshape(D, 1).expand(D, N).contiguous()
        p3 = self.act_p3.reshape(D, 1).expand(D, N).contiguous()
        p4 = self.act_p4.reshape(D, 1).expand(D, N).contiguous()
        a = apply_evolved_activations(x, self.act_ids, p1, p2, p3, p4)
        return a.t() * self.dim_gain.unsqueeze(0) * self.global_gain


def frozen_forward(embed, posenc, token_ids, positions):
    return embed.embed(token_ids) + posenc.encode(positions)


class FrozenAttnLayer:
    """Loads a frozen causal attention layer."""
    def __init__(self, ckpt_path, device):
        with open(ckpt_path, "rb") as f:
            st = pickle.load(f)
        d = torch.device(device)
        self.W_Q = torch.from_numpy(st["W_Q"]).to(d)
        self.W_K = torch.from_numpy(st["W_K"]).to(d)
        self.W_V = torch.from_numpy(st["W_V"]).to(d)
        self.W_O = torch.from_numpy(st["W_O"]).to(d)
        self.head_gain = torch.from_numpy(st["head_gain"]).to(d)
        self.logit_act_ids = torch.from_numpy(st["logit_act_ids"]).to(d)
        self.logit_act_p1 = torch.from_numpy(st["logit_act_p1"]).to(d)
        self.logit_act_p2 = torch.from_numpy(st["logit_act_p2"]).to(d)
        self.logit_act_p3 = torch.from_numpy(st["logit_act_p3"]).to(d)
        self.logit_act_p4 = torch.from_numpy(st["logit_act_p4"]).to(d)
        cfg = st["config"]
        self.D = cfg["D"]; self.n_heads = cfg["N_HEADS"]
        self.head_dim = cfg["HEAD_DIM"]
        self.dev = d

    @torch.no_grad()
    def forward(self, x, causal=True):
        S, D = x.shape
        Nh, Hd = self.n_heads, self.head_dim
        Q = (x @ self.W_Q.t()).view(S, Nh, Hd).permute(1, 0, 2)
        K = (x @ self.W_K.t()).view(S, Nh, Hd).permute(1, 0, 2)
        V = (x @ self.W_V.t()).view(S, Nh, Hd).permute(1, 0, 2)
        scale = 1.0 / (Hd ** 0.5)
        logits = torch.bmm(Q, K.transpose(1, 2)) * scale
        pos_idx = torch.arange(S, device=x.device)
        pos_bias = -0.1 * (pos_idx.unsqueeze(0) - pos_idx.unsqueeze(1)).abs().float()
        logits = logits + pos_bias.unsqueeze(0)
        if causal:
            m = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            logits = logits.masked_fill(m.unsqueeze(0), -1e9)
        for h in range(Nh):
            lf = logits[h:h+1].reshape(1, S * S)
            aid = self.logit_act_ids[h:h+1]
            p1 = self.logit_act_p1[h:h+1].unsqueeze(1).expand(1, S*S).contiguous()
            p2 = self.logit_act_p2[h:h+1].unsqueeze(1).expand(1, S*S).contiguous()
            p3 = self.logit_act_p3[h:h+1].unsqueeze(1).expand(1, S*S).contiguous()
            p4 = self.logit_act_p4[h:h+1].unsqueeze(1).expand(1, S*S).contiguous()
            logits[h] = apply_evolved_activations(lf, aid, p1, p2, p3, p4).view(S, S)
        attn = F.softmax(logits, dim=-1)
        head_out = torch.bmm(attn, V) * self.head_gain.view(Nh, 1, 1)
        concat = head_out.permute(1, 0, 2).contiguous().view(S, D)
        out = concat @ self.W_O.t()
        return x + out


class FrozenAttnStack:
    def __init__(self):
        self.layers = []

    def add_layer(self, ckpt_path, device):
        self.layers.append(FrozenAttnLayer(ckpt_path, device))

    @torch.no_grad()
    def forward(self, x, causal=True):
        for layer in self.layers:
            x = layer.forward(x, causal=causal)
        return x

    def __len__(self):
        return len(self.layers)


class GenregLM:
    """Full inference pipeline: embed + posenc + evolved attention stack
    + n-gram cascade reranked by cosine similarity.

    Every learned weight was produced by gradient-free evolution. The
    n-gram tables are counted corpus statistics. No ridge regression,
    no backprop, no SGD — anywhere.
    """
    def __init__(self, ckpt_dir, device="cuda"):
        self.device = torch.device(device)
        # Vocab
        with open(os.path.join(ckpt_dir, "vocab.pkl"), "rb") as f:
            v = pickle.load(f)
        self.token_to_id = v["token_to_id"]
        self.id_to_token = v["id_to_token"]
        self.V = v["V"]

        # Frozen components (all evolved)
        self.embed = FrozenEmbed(os.path.join(ckpt_dir, "embed.pkl"), device)
        self.posenc = FrozenPosEnc(os.path.join(ckpt_dir, "posenc.pkl"), device)
        self.attn = FrozenAttnStack()
        for layer_file in sorted(glob.glob(os.path.join(ckpt_dir, "attn_L*.pkl"))):
            self.attn.add_layer(layer_file, device)

        # Rerank stack = CE stack + evolved rerank layers on top
        self.rerank_stack = FrozenAttnStack()
        for l in self.attn.layers:
            self.rerank_stack.layers.append(l)
        rerank_files = sorted(glob.glob(
            os.path.join(ckpt_dir, "attn_rerank_L*.pkl")))
        for layer_file in rerank_files:
            self.rerank_stack.add_layer(layer_file, device)
        self._emb_table_n = None   # lazy; normalized embedding table for rerank

        # N-gram cascade — counted corpus statistics, stored per-order
        def _load_ngram(name):
            path = os.path.join(ckpt_dir, f"ngrams_{name}.pkl")
            if not os.path.exists(path):
                return {}
            with open(path, "rb") as f:
                return pickle.load(f)
        self.bigram = _load_ngram("bigram")
        self.trigram = _load_ngram("trigram")
        self.fourgram = _load_ngram("fourgram")
        self.fivegram = _load_ngram("fivegram")

        # RAG paragraph index (optional — only loaded if present)
        self._rag = None
        rag_path = os.path.join(ckpt_dir, "rag_index.pkl")
        if os.path.exists(rag_path):
            with open(rag_path, "rb") as f:
                self._rag = pickle.load(f)
            self._rag_emb = torch.from_numpy(
                self._rag["embeddings"].astype(np.float32)).to(self.device)
            self._sif_mean = torch.from_numpy(
                self._rag["sif_mean"]).to(self.device)
            self._sif_top_pc = torch.from_numpy(
                self._rag["sif_top_pc"]).to(self.device)
            self._tok_weight = torch.from_numpy(
                self._rag["tok_weight"]).to(self.device)

        # Evolved span-scorer for extractive QA
        self._span_scorer = None
        sp_path = os.path.join(ckpt_dir, "span_scorer.pkl")
        if os.path.exists(sp_path):
            with open(sp_path, "rb") as f:
                sp = pickle.load(f)
            self._span_scorer_w = torch.from_numpy(
                sp["w"].astype(np.float32)).to(self.device)
            self._span_scorer_b = float(sp["b"])
            self._span_scorer = sp

    def tokenize(self, text):
        words = text.lower().split()
        ids = [self.token_to_id.get(w, self.token_to_id["<unk>"]) for w in words]
        return torch.tensor(ids, device=self.device, dtype=torch.long)

    def detokenize(self, ids):
        return " ".join(self.id_to_token.get(int(i), "<?>") for i in ids)

    # ----- Rerank generation ---------------------------------------------
    # N-gram proposes K grammatical candidates; the evolved attention
    # stack picks the semantically best one by cosine to its last-token
    # feature.

    def _emb_table_normalized(self):
        if self._emb_table_n is None:
            full = self.embed.embed(torch.arange(self.V, device=self.device))
            self._emb_table_n = F.normalize(full, dim=1)
        return self._emb_table_n

    # Punctuation tokens ALLOWED as generation candidates (sentence-structure
    # markers). Period is EOS-adjacent; others shape flow.
    _ALLOWED_PUNCT_IDS = {69, 70, 73, 74}   # . , ! ?

    def _ngram_candidates(self, ids_list, K=30):
        """Top-K next-token candidates drawn from the longest n-gram that
        matches the current context tail. Returns [(token_id, prob), ...].

        N-gram tables built from the punctuated stream include
        punctuation tokens as valid successors — we allow those through
        the CHAR_CUTOFF filter. Words plus the allowed punctuation set
        are the generation vocabulary.

        Lookup keys are built from 'content' tokens (words + allowed
        punctuation), skipping structural space/newline tokens.
        """
        STRUCTURAL = {66, 67}   # space, newline
        content_before = []
        j = len(ids_list) - 1
        while j >= 0 and len(content_before) < 4:
            t = int(ids_list[j])
            if t in STRUCTURAL:
                j -= 1; continue
            if t >= CHAR_CUTOFF or t in self._ALLOWED_PUNCT_IDS:
                content_before.append(t)
            j -= 1
        if not content_before:
            return []
        content_before.reverse()
        prev1 = content_before[-1]
        prev2 = tuple(content_before[-2:]) if len(content_before) >= 2 else None
        prev3 = tuple(content_before[-3:]) if len(content_before) >= 3 else None
        prev4 = tuple(content_before[-4:]) if len(content_before) >= 4 else None
        cands = {}
        for tbl, key in [(self.fivegram, prev4), (self.fourgram, prev3),
                         (self.trigram, prev2)]:
            if key is not None and key in tbl:
                d = tbl[key]
                total = sum(d.values()) + 1e-6
                for tok, cnt in d.items():
                    if tok < self.V and tok not in cands and (
                            tok >= CHAR_CUTOFF
                            or tok in self._ALLOWED_PUNCT_IDS):
                        cands[tok] = cnt / total
                if len(cands) >= K:
                    break
        if prev1 in self.bigram and len(cands) < K:
            d = self.bigram[prev1]
            total = sum(d.values()) + 1e-6
            for tok, cnt in d.items():
                if CHAR_CUTOFF <= tok < self.V and tok not in cands:
                    cands[tok] = cnt / total * 0.5
                if len(cands) >= K:
                    break
        return sorted(cands.items(), key=lambda x: -x[1])[:K]

    # ----- Retrieval (RAG) --------------------------------------------

    @torch.no_grad()
    def retrieve(self, query, k=3, bm25_weight=0.7):
        """Return top-K paragraphs for a query, ranked by a hybrid of
        BM25 lexical match and SIF-weighted cosine similarity on the
        evolved embeddings.

        bm25_weight in [0,1]: mixing weight between BM25 (lexical) and
        dense (semantic) scores, after each is normalized to zero-mean
        unit-var. BM25 is classical lexical retrieval and dominates on
        factual queries where rare content words match. SIF cosine
        adds semantic generalization.
        """
        if self._rag is None:
            raise RuntimeError("RAG index not loaded (rag_index.pkl missing)")
        emb_n = self._emb_table_normalized()
        q_ids = self.tokenize(query)
        if q_ids.numel() == 0:
            return []

        # ---- Dense (SIF) score ----
        w = self._tok_weight[q_ids].unsqueeze(1)
        vecs = emb_n[q_ids] * w
        q_emb = vecs.sum(dim=0) / (w.sum() + 1e-8)
        q_emb = q_emb - self._sif_mean.squeeze(0)
        q_emb = q_emb - (q_emb @ self._sif_top_pc) * self._sif_top_pc
        q_emb = F.normalize(q_emb.unsqueeze(0), dim=1).squeeze(0)
        dense_scores = self._rag_emb @ q_emb
        # Normalize to zero-mean unit-var for blending
        dense_scores = (dense_scores - dense_scores.mean()) / (
            dense_scores.std() + 1e-8)

        # ---- BM25 lexical score ----
        import math as _math
        k1 = 1.2; b = 0.75
        STRUCTURAL = {66, 67}
        ALLOWED_PUNCT = self._ALLOWED_PUNCT_IDS
        q_content = set()
        for t in q_ids.cpu().tolist():
            if t in STRUCTURAL:
                continue
            if t >= CHAR_CUTOFF or t in ALLOWED_PUNCT:
                q_content.add(t)

        N_docs = self._rag["N_docs"]
        avgdl = self._rag["avgdl"]
        token_df = self._rag["token_df"]
        para_tf = self._rag["para_tf"]
        para_len = self._rag["para_len"]  # np.int32 array

        # Per-token idf
        idf = {}
        for t in q_content:
            df = token_df.get(t, 0)
            if df == 0:
                continue
            idf[t] = _math.log((N_docs - df + 0.5) / (df + 0.5) + 1.0)

        bm25 = np.zeros(len(para_tf), dtype=np.float32)
        for i, tf_dict in enumerate(para_tf):
            dl = float(para_len[i])
            norm = 1 - b + b * dl / max(avgdl, 1e-6)
            s = 0.0
            for t, id_val in idf.items():
                f = tf_dict.get(t, 0)
                if f == 0:
                    continue
                s += id_val * (f * (k1 + 1)) / (f + k1 * norm)
            bm25[i] = s
        bm25_t = torch.from_numpy(bm25).to(self.device)
        if bm25_t.std() > 0:
            bm25_t = (bm25_t - bm25_t.mean()) / (bm25_t.std() + 1e-8)

        scores = bm25_weight * bm25_t + (1 - bm25_weight) * dense_scores
        top = scores.topk(min(k, scores.shape[0]))
        out = []
        for idx, s in zip(top.indices.cpu().tolist(),
                           top.values.cpu().tolist()):
            out.append({
                "text": self._rag["texts"][idx],
                "title": self._rag["titles"][idx],
                "token_ids": self._rag["token_lists"][idx],
                "score": s,
            })
        return out

    # ----- Extractive QA on retrieved passages ------------------------
    #
    # Given a question and a passage, score every span of length 1..10
    # in the passage and return the best. No new training — uses the
    # stored tok_weight (SIF-style IDF) and question-word heuristics.

    _QTYPE_KEYWORDS = {
        "when": "when",
        "where": "where",
        "who": "who",
        "whom": "who",
        "how many": "num",
        "how much": "num",
        "what year": "year",
        "what date": "year",
        "what percent": "num",
    }

    def _classify_question(self, query):
        """Return a question type tag for span-scoring bias."""
        ql = query.lower()
        for key, tag in self._QTYPE_KEYWORDS.items():
            if key in ql:
                return tag
        # fallback on first content word
        for w in ql.split():
            if w in ("what", "which", "why", "how"):
                return w
        return "other"

    def _span_is_numeric(self, span_ids):
        """True if the span contains a digit-dominant token run."""
        digit_tok_ids = {self.token_to_id.get(c) for c in "0123456789"}
        digit_tok_ids.discard(None)
        if any(t in digit_tok_ids for t in span_ids):
            return True
        # Also check token strings for digit-start (covers tokens
        # like '1994' if present in vocab as a single token)
        for t in span_ids:
            s = self.id_to_token.get(int(t), "")
            if s and s[0].isdigit():
                return True
        return False

    @torch.no_grad()
    def extract_answer(self, query, k_retrieve=3, max_span=8, min_span=1,
                        return_passage=False):
        """Retrieve top-k passages, score every span, return best.

        Returns (answer_text, answer_token_ids, metadata).
        """
        if self._rag is None:
            raise RuntimeError("RAG index not loaded")
        hits = self.retrieve(query, k=k_retrieve)
        if not hits:
            return "", [], {}

        emb_n = self._emb_table_normalized()
        q_ids_t = self.tokenize(query)
        q_tok_ids = set(int(t) for t in q_ids_t.cpu().tolist())
        STRUCTURAL = {66, 67}
        q_content = {t for t in q_tok_ids
                      if t >= CHAR_CUTOFF or t in self._ALLOWED_PUNCT_IDS}
        q_content.difference_update(STRUCTURAL)
        q_content_rare = {t for t in q_content
                           if float(self._tok_weight[t].item()) > 0.3}

        # Question mean embedding (SIF-weighted, content only)
        q_content_tokens_list = list(q_content_rare) or list(q_content) or list(q_tok_ids)
        if q_content_tokens_list:
            q_ct_tens = torch.tensor(q_content_tokens_list, device=self.device)
            q_w = self._tok_weight[q_ct_tens].unsqueeze(1)
            q_vec = (emb_n[q_ct_tens] * q_w).sum(dim=0) / (q_w.sum() + 1e-8)
            q_vec = F.normalize(q_vec.unsqueeze(0), dim=1).squeeze(0)
        else:
            q_vec = None

        qtype = self._classify_question(query)
        tok_weight_np = self._tok_weight.cpu().numpy()

        best_score = -1e9
        best = ("", [], {})

        for h_idx, h in enumerate(hits):
            passage = h["token_ids"]
            hit_positions = [i for i, t in enumerate(passage)
                              if t in q_content_rare]
            if not hit_positions:
                continue
            dists = []
            for i in range(len(passage)):
                d = min(abs(i - p) for p in hit_positions)
                dists.append(d)

            content_idx = [i for i, t in enumerate(passage)
                            if t not in STRUCTURAL
                            and (t >= CHAR_CUTOFF or t in self._ALLOWED_PUNCT_IDS)]

            for L in range(min_span, max_span + 1):
                for ci_start in range(len(content_idx) - L + 1):
                    start = content_idx[ci_start]
                    span = [passage[i] for i in content_idx[ci_start:ci_start + L]]
                    if all(t in q_content for t in span):
                        continue
                    if any(t == 1 for t in span):
                        continue

                    # Compute features matching the span-scorer trainer
                    L_ = L
                    prox = 1.0 / (1.0 + dists[start])
                    rarity = float(sum(tok_weight_np[t] for t in span))
                    length = float(L_)
                    semantic = 0.0
                    inv_u = 0.0
                    if q_vec is not None:
                        s_tens = torch.tensor(span, device=self.device)
                        s_w = self._tok_weight[s_tens].unsqueeze(1)
                        s_vec = (emb_n[s_tens] * s_w).sum(dim=0) / (s_w.sum() + 1e-8)
                        s_vec = F.normalize(s_vec.unsqueeze(0), dim=1).squeeze(0)
                        sim = float((s_vec * q_vec).sum().item())
                        semantic = sim
                        inv_u = -(abs(sim - 0.5) ** 2) + 0.25
                    numeric = 1.0 if self._span_is_numeric(span) else 0.0
                    is_when = 1.0 if qtype in ("when", "year", "num") else 0.0
                    is_who = 1.0 if qtype in ("who", "whom") else 0.0
                    is_where = 1.0 if qtype == "where" else 0.0
                    is_what = 1.0 if qtype in ("what", "which") else 0.0
                    first_rare = float(tok_weight_np[span[0]]) if span else 0.0
                    # mean/max distance to query hits across span positions
                    if hit_positions:
                        mean_d = sum(
                            min(abs(start + i - p) for p in hit_positions)
                            for i in range(L_)) / L_
                        max_d = max(
                            min(abs(start + i - p) for p in hit_positions)
                            for i in range(L_))
                    else:
                        mean_d = 999.0; max_d = 999.0
                    mean_dist_score = 1.0 / (1.0 + mean_d)
                    max_dist_score = 1.0 / (1.0 + max_d)
                    overlap = float(sum(1 for t in span if t in q_content))
                    retr_conf = 1.0 / (1.0 + h_idx)

                    feat_vec = [prox, rarity, length, semantic, inv_u,
                                 numeric, is_when, is_who, is_where,
                                 is_what, first_rare, mean_dist_score,
                                 max_dist_score, overlap, retr_conf]

                    if self._span_scorer is not None:
                        ft = torch.tensor(feat_vec, device=self.device,
                                           dtype=torch.float32)
                        score = float((ft * self._span_scorer_w).sum().item()
                                       + self._span_scorer_b)
                    else:
                        # Fallback heuristic
                        qtype_bonus = 0.0
                        if is_when and numeric:
                            qtype_bonus = 3.0
                        elif is_when:
                            qtype_bonus = -0.8
                        score = (prox * 2.0 + rarity + qtype_bonus
                                  + inv_u + retr_conf * 0.5
                                  - 0.05 * max(0, L_ - 5))
                    if score > best_score:
                        best_score = score
                        meta = {
                            "passage_title": h.get("title"),
                            "passage_idx": h_idx,
                            "span_start": start,
                            "span_len": L,
                            "score": score,
                            "qtype": qtype,
                            "prox": prox,
                            "rarity": rarity,
                            "semantic": semantic,
                        }
                        text = self.detokenize(span)
                        best = (text, list(span), meta)
        if return_passage:
            return best + (hits,)
        return best

    @torch.no_grad()
    def generate_qa(self, query, k_retrieve=3, max_span=8):
        """Extractive QA: retrieve, extract, return span wrapped as sentence."""
        text, ids, _ = self.extract_answer(query, k_retrieve=k_retrieve,
                                             max_span=max_span)
        if not text:
            return "", []
        # Wrap as a minimal sentence: "[answer] ."
        period_id = self.token_to_id.get(".", -1)
        wrapped = ids + ([period_id] if period_id > 0 else [])
        return self.detokenize(wrapped), wrapped

    @torch.no_grad()
    def generate_rag(self, query, max_tokens=30, k=1, alpha=5.0,
                      temperature=0.7, top_k=30, rep_penalty=1.5,
                      top_p=None, min_tokens=5, max_context=380):
        """RAG generation. Retrieves top-k paragraphs, prepends their
        tokens to the query, then calls rerank generation.

        max_context caps the prepended passage length (in tokens) so
        there's room for the question + generation within MAX_LEN.
        """
        hits = self.retrieve(query, k=k)
        if not hits:
            return self.generate_rerank(
                query, max_tokens=max_tokens, alpha=alpha,
                temperature=temperature, top_k=top_k,
                rep_penalty=rep_penalty, top_p=top_p, min_tokens=min_tokens), []
        # Concatenate retrieved paragraphs (truncated), then a newline,
        # then the query.
        ctx_ids = []
        for h in hits:
            ctx_ids.extend(h["token_ids"])
            ctx_ids.append(self.token_to_id.get("\n", 67))
        # Cap context length to leave room for query + generation
        if len(ctx_ids) > max_context:
            ctx_ids = ctx_ids[-max_context:]
        query_ids = self.tokenize(query).cpu().tolist()
        full_ids = ctx_ids + query_ids
        # Run generate_rerank but seed tok with full context instead
        # of just the query.
        tok = torch.tensor(full_ids, device=self.device, dtype=torch.long)
        if tok.shape[0] > MAX_LEN:
            tok = tok[-MAX_LEN:]

        emb_table_n = self._emb_table_normalized()
        gen = []
        # Passage-copy pool: rare/content tokens from retrieved passages,
        # weighted by inverse document frequency so entity-like tokens
        # dominate over generic fillers. The tok_weight from the SIF
        # index is already `a / (a + df/N)` which strongly downweights
        # common words.
        passage_weighted = {}
        tok_weight_np = self._tok_weight.cpu().numpy() if hasattr(
            self, "_tok_weight") else None
        for h in hits:
            for t in h["token_ids"]:
                if (t >= CHAR_CUTOFF or t in self._ALLOWED_PUNCT_IDS) \
                        and t not in (66, 67):
                    w = float(tok_weight_np[t]) if tok_weight_np is not None else 1.0
                    # Cap very common passage tokens (df-weight < 0.5)
                    # at zero to keep the copy pool focused
                    if w < 0.2:
                        continue
                    passage_weighted[t] = max(passage_weighted.get(t, 0.0), w)
        for _ in range(max_tokens):
            if tok.shape[0] > MAX_LEN:
                tok = tok[-MAX_LEN:]
            ids_list = tok.cpu().tolist()
            cands = self._ngram_candidates(ids_list, K=top_k)
            if not cands:
                if ids_list[-1] in self.bigram:
                    cands = [(next(iter(self.bigram[ids_list[-1]])), 0.5)]
                else:
                    break
            # Augment with rare passage tokens, weighted by inverse
            # document frequency so entity-like tokens dominate.
            existing = set(c[0] for c in cands)
            if cands:
                passage_base = max(c[1] for c in cands)
            else:
                passage_base = 0.2
            for t, w in passage_weighted.items():
                if t not in existing:
                    # Rarer tokens (higher SIF weight) get boosted
                    cands.append((t, passage_base * w))
            positions = torch.arange(tok.shape[0], device=self.device)
            x = frozen_forward(self.embed, self.posenc, tok, positions)
            x = self.rerank_stack.forward(x, causal=True)
            last_n = F.normalize(x[-1:], dim=1)
            cand_ids = [c[0] for c in cands]
            cand_embs = emb_table_n[torch.tensor(cand_ids, device=self.device)]
            cos = (cand_embs @ last_n.t()).squeeze(1)
            ng_logp = torch.tensor([math.log(p + 1e-10) for _, p in cands],
                                    device=self.device)
            score = alpha * cos + ng_logp
            if rep_penalty > 1.0 and gen:
                seen = set(gen[-15:])
                adj = torch.tensor(
                    [-math.log(rep_penalty) if c in seen else 0.0
                     for c in cand_ids], device=self.device)
                score = score + adj
            score = score / max(temperature, 1e-6)
            if top_p is not None and top_p < 1.0:
                sv, si = score.sort(descending=True)
                p_cum = F.softmax(sv, dim=0).cumsum(dim=0)
                keep = p_cum <= top_p
                keep[0] = True
                m = torch.zeros_like(score, dtype=torch.bool)
                m[si[keep]] = True
                score = score.masked_fill(~m, -1e9)
            probs = F.softmax(score, dim=0)
            pick = torch.multinomial(probs, 1).item()
            nxt = cand_ids[pick]
            gen.append(nxt)
            tok = torch.cat([tok, torch.tensor([nxt], device=self.device,
                                                 dtype=torch.long)])
            if len(gen) >= min_tokens and nxt in (
                    self.token_to_id.get(".", -1),
                    self.token_to_id.get("?", -1),
                    self.token_to_id.get("!", -1),
                    self.token_to_id.get("<eos>", -1)):
                break
        return self.detokenize(gen), gen, hits

    @torch.no_grad()
    def generate_rerank(self, prompt, max_tokens=30, alpha=5.0, temperature=0.7,
                        top_k=30, rep_penalty=1.5, top_p=None, min_tokens=5):
        """Rerank generation over the 4-layer attention stack.

        alpha > 0 blends attention cosine into n-gram log-probs. alpha=0
        degenerates to pure n-gram sampling from the candidate set.
        """
        if len(self.rerank_stack) == 0:
            raise RuntimeError("No attention layers loaded")
        emb_table_n = self._emb_table_normalized()
        tok = self.tokenize(prompt)
        if tok.numel() == 0:
            return "", []
        gen = []
        for _ in range(max_tokens):
            if tok.shape[0] > MAX_LEN:
                tok = tok[-MAX_LEN:]
            ids_list = tok.cpu().tolist()
            cands = self._ngram_candidates(ids_list, K=top_k)
            if not cands:
                if ids_list[-1] in self.bigram:
                    cands = [(next(iter(self.bigram[ids_list[-1]])), 0.5)]
                else:
                    break
            positions = torch.arange(tok.shape[0], device=self.device)
            x = frozen_forward(self.embed, self.posenc, tok, positions)
            x = self.rerank_stack.forward(x, causal=True)
            last_n = F.normalize(x[-1:], dim=1)
            cand_ids = [c[0] for c in cands]
            cand_embs = emb_table_n[torch.tensor(cand_ids, device=self.device)]
            cos = (cand_embs @ last_n.t()).squeeze(1)
            ng_logp = torch.tensor([math.log(p + 1e-10) for _, p in cands],
                                    device=self.device)
            score = alpha * cos + ng_logp
            if rep_penalty > 1.0 and gen:
                seen = set(gen[-15:])
                adj = torch.tensor(
                    [-math.log(rep_penalty) if c in seen else 0.0
                     for c in cand_ids], device=self.device)
                score = score + adj
            score = score / max(temperature, 1e-6)
            if top_p is not None and top_p < 1.0:
                sv, si = score.sort(descending=True)
                p = F.softmax(sv, dim=0).cumsum(dim=0)
                keep = p <= top_p
                keep[0] = True
                m = torch.zeros_like(score, dtype=torch.bool)
                m[si[keep]] = True
                score = score.masked_fill(~m, -1e9)
            probs = F.softmax(score, dim=0)
            pick = torch.multinomial(probs, 1).item()
            nxt = cand_ids[pick]
            gen.append(nxt)
            tok = torch.cat([tok, torch.tensor([nxt], device=self.device,
                                                 dtype=torch.long)])
            # Natural-stop triggers: period / ? / ! / <eos>. Require at
            # least min_tokens of generation first so we don't collapse
            # to empty ".".
            if len(gen) >= min_tokens and nxt in (
                    self.token_to_id.get(".", -1),
                    self.token_to_id.get("?", -1),
                    self.token_to_id.get("!", -1),
                    self.token_to_id.get("<eos>", -1)):
                break
            if nxt == self.token_to_id.get("<eos>", -1):
                break
        return self.detokenize(gen), gen
