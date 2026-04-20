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

        # RAG index (optional — supports both chunked_v1 and legacy
        # paragraph-level formats)
        self._rag = None
        self._rag_chunked = False
        rag_path = os.path.join(ckpt_dir, "rag_index.pkl")
        if os.path.exists(rag_path):
            with open(rag_path, "rb") as f:
                self._rag = pickle.load(f)
            self._rag_chunked = self._rag.get("format") == "chunked_v1"
            emb_key = "chunk_embeddings" if self._rag_chunked else "embeddings"
            raw_emb = self._rag[emb_key]
            # int8-quantized? dequantize on load
            if raw_emb.dtype == np.int8:
                scale = self._rag.get("chunk_embeddings_scale", 127.0)
                emb_f32 = raw_emb.astype(np.float32) / float(scale)
                # re-normalize to unit length after dequant
                norms = np.linalg.norm(emb_f32, axis=1, keepdims=True)
                emb_f32 = emb_f32 / np.maximum(norms, 1e-8)
            else:
                emb_f32 = raw_emb.astype(np.float32)
            self._rag_emb = torch.from_numpy(emb_f32).to(self.device)
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

        # Evolved query-adaptive retrieval reranker (v1)
        self._reranker = None
        rr_path = os.path.join(ckpt_dir, "retrieval_reranker.pkl")
        if os.path.exists(rr_path):
            with open(rr_path, "rb") as f:
                rr = pickle.load(f)
            self._reranker_W = torch.from_numpy(
                rr["W"].astype(np.float32)).to(self.device)
            self._reranker = rr

        # Evolved query-adaptive span scorer (v3)
        self._span_qa = None
        sv3_path = os.path.join(ckpt_dir, "span_scorer_qa.pkl")
        if os.path.exists(sv3_path):
            with open(sv3_path, "rb") as f:
                sv3 = pickle.load(f)
            if sv3.get("version") == "span_scorer_v3_query_adaptive":
                self._span_qa_W = torch.from_numpy(
                    sv3["W"].astype(np.float32)).to(self.device)
                self._span_qa = sv3

        # Evolved MLP span scorer (ensemble with heuristic)
        self._span_mlp = None
        mlp_path = os.path.join(ckpt_dir, "span_mlp.pkl")
        if os.path.exists(mlp_path):
            with open(mlp_path, "rb") as f:
                mlp = pickle.load(f)
            if mlp.get("version") == "span_mlp_v1_ensemble":
                self._span_mlp_W1 = torch.from_numpy(
                    mlp["W1"].astype(np.float32)).to(self.device)
                self._span_mlp_b1 = torch.from_numpy(
                    mlp["b1"].astype(np.float32)).to(self.device)
                self._span_mlp_W2 = torch.from_numpy(
                    mlp["W2"].astype(np.float32)).to(self.device)
                self._span_mlp_b2 = torch.from_numpy(
                    mlp["b2"].astype(np.float32)).to(self.device)
                self._span_mlp_beta = float(mlp["beta"])
                self._span_mlp = mlp

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

    def _compute_qfeats(self, q_ids_tensor):
        """10-dim query features for the retrieval reranker.
        Matches genreg_retrieval_reranker.py order."""
        if q_ids_tensor.numel() == 0:
            return np.zeros(10, dtype=np.float32)
        q_ids = q_ids_tensor.cpu().tolist()
        ql = self.detokenize(q_ids).lower().strip()
        # Classify
        is_when = is_who = is_where = is_what = is_how = 0.0
        if ql.startswith("when") or "what year" in ql or "what date" in ql:
            is_when = 1.0
        elif ql.startswith("who") or ql.startswith("whom"):
            is_who = 1.0
        elif ql.startswith("where"):
            is_where = 1.0
        elif ql.startswith("how"):
            is_how = 1.0
        elif ql.startswith("what") or ql.startswith("which"):
            is_what = 1.0
        # has_digit
        digit_tok_ids = {self.token_to_id.get(c) for c in "0123456789"} - {None}
        has_digit = 0.0
        for t in q_ids:
            if t in digit_tok_ids:
                has_digit = 1.0; break
            s = self.id_to_token.get(int(t), "")
            if s and s[0].isdigit():
                has_digit = 1.0; break
        STRUCTURAL = {66, 67}
        content = [t for t in q_ids if t >= CHAR_CUTOFF and t not in STRUCTURAL]
        tok_w = self._tok_weight if hasattr(self, "_tok_weight") else None
        if tok_w is None or not content:
            return np.array([is_when, is_who, is_where, is_what, is_how,
                              has_digit, 0.0, 0.0,
                              min(len(q_ids) / 20.0, 1.0), 1.0],
                             dtype=np.float32)
        tw_np = tok_w.cpu().numpy() if tok_w.is_cuda else tok_w.numpy()
        rare_count = sum(1 for t in content if tw_np[t] > 0.3)
        max_sif = max((tw_np[t] for t in content), default=0.0)
        return np.array([is_when, is_who, is_where, is_what, is_how,
                          has_digit,
                          min(rare_count / 5.0, 1.0),
                          float(max_sif),
                          min(len(q_ids) / 20.0, 1.0),
                          1.0], dtype=np.float32)

    @torch.no_grad()
    def retrieve(self, query, k=3, bm25_weight=0.85, bm25_k1=1.2, bm25_b=0.5,
                  prf=False, prf_top=3, prf_terms=6,
                  qexp=True, qexp_k=3, qexp_weight=0.4):
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
        k1 = bm25_k1; b = bm25_b
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
        if self._rag_chunked:
            tf_list = self._rag["chunk_tf"]
            len_arr = self._rag["chunk_len"]
        else:
            tf_list = self._rag.get("para_tf", [])
            len_arr = self._rag.get("para_len", [])

        # Per-token idf
        idf = {}
        for t in q_content:
            df = token_df.get(t, 0)
            if df == 0:
                continue
            idf[t] = _math.log((N_docs - df + 0.5) / (df + 0.5) + 1.0)

        # ---- Query expansion via embedding neighbors ----
        # For rare content tokens in the query, find the k nearest
        # neighbors in the evolved embedding space and add them to the
        # BM25 query with reduced idf weight. Handles paraphrase
        # (automobile ↔ car, film ↔ movie) without any training.
        if qexp and q_content:
            rare_q = [t for t in q_content
                       if float(self._tok_weight[t].item()) > 0.3]
            if rare_q:
                q_vecs = emb_n[torch.tensor(rare_q, device=self.device)]
                # Cosine to all embeddings
                sims = q_vecs @ emb_n.t()     # (n_rare, V)
                # Mask already-in-query tokens + <unk>/specials
                in_q = torch.zeros(self.V, dtype=torch.bool,
                                    device=self.device)
                for t in q_content:
                    in_q[t] = True
                in_q[1] = True   # <unk>
                sims[:, in_q] = -1.0
                # Top-k per query token
                vals, top_ids = sims.topk(qexp_k, dim=1)
                for row in range(vals.shape[0]):
                    for j in range(qexp_k):
                        nb = int(top_ids[row, j].item())
                        sim_val = float(vals[row, j].item())
                        if sim_val < 0.3:
                            continue
                        if nb in idf:
                            continue
                        # Only expand with content tokens
                        if nb < CHAR_CUTOFF and nb not in self._ALLOWED_PUNCT_IDS:
                            continue
                        df = token_df.get(nb, 0)
                        if df == 0:
                            continue
                        base_idf = _math.log(
                            (N_docs - df + 0.5) / (df + 0.5) + 1.0)
                        idf[nb] = base_idf * qexp_weight * sim_val

        bm25 = np.zeros(len(tf_list), dtype=np.float32)
        for i, tf_dict in enumerate(tf_list):
            dl = float(len_arr[i])
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

        # ---- Query-adaptive rerank (evolved 10×5 head, OPT-IN) ----
        # Small improvement at recall@1 but slight regression at @3,
        # so off by default. Pass rerank=True to enable.
        if (self._reranker is not None and self._rag_chunked
                and getattr(self, "_use_reranker", False)):
            pre_k = min(20, scores.shape[0])
            pre_top = scores.topk(pre_k)
            top_indices = pre_top.indices.cpu().tolist()
            # Query features
            qf = self._compute_qfeats(q_ids)               # (10,)
            qf_t = torch.tensor(qf, device=self.device,
                                 dtype=torch.float32)
            # Softmax weights over the 5 signals
            logits_per_sig = torch.einsum('ig,i->g',
                                           self._reranker_W, qf_t)
            weights = F.softmax(logits_per_sig, dim=0)     # (5,)
            # Per-candidate signal vectors
            q_content_list = list(q_content)
            q_has_digit = qf[5]
            # Build query SIF vector once for signals
            emb_n_q = emb_n
            if q_content_list:
                q_t = torch.tensor(q_content_list, device=self.device)
                qw_sig = self._tok_weight[q_t].unsqueeze(1)
                q_vec_sig = (emb_n_q[q_t] * qw_sig).sum(dim=0) / (qw_sig.sum() + 1e-8)
                q_vec_sig = q_vec_sig - self._sif_mean.squeeze(0)
                q_vec_sig = q_vec_sig - (q_vec_sig @ self._sif_top_pc) * self._sif_top_pc
                q_vec_sig = F.normalize(q_vec_sig.unsqueeze(0), dim=1).squeeze(0)
            else:
                q_vec_sig = None
            # Query bigrams for bm25_bigram signal
            q_bigrams = set()
            for i in range(len(q_content_list) - 1):
                q_bigrams.add((q_content_list[i], q_content_list[i + 1]))
            digit_tok_ids = {self.token_to_id.get(c) for c in "0123456789"} - {None}
            id2tok = self.id_to_token
            chunk_tokens_list = self._rag.get("chunk_token_lists", [])

            new_scores = torch.full_like(scores, -1e9)
            for ci in top_indices:
                # Use RAW (not z-scored) BM25 and SIF values to match
                # training distribution.
                s1 = float(bm25[ci])
                # Raw dense cosine requires recomputing (pre-z-score).
                # Approximate from the embedding directly.
                s2 = float((self._rag_emb[ci] * q_emb).sum().item())
                # s3: BM25 over bigrams
                chunk_toks = chunk_tokens_list[ci] if chunk_tokens_list else []
                c_bigrams = {}
                for i in range(len(chunk_toks) - 1):
                    pair = (chunk_toks[i], chunk_toks[i + 1])
                    c_bigrams[pair] = c_bigrams.get(pair, 0) + 1
                s3 = 0.0
                for bg in q_bigrams:
                    if bg in c_bigrams:
                        s3 += _math.log(1.0 + c_bigrams[bg])
                # s4: length match
                cl = float(len_arr[ci])
                s4 = 1.0 - min(abs(cl - avgdl) / max(avgdl, 1e-6), 1.0)
                # s5: numeric coincidence
                chunk_has_digit = 0.0
                for t in chunk_toks[:200]:
                    if t in digit_tok_ids:
                        chunk_has_digit = 1.0; break
                    s = id2tok.get(int(t), "")
                    if s and s[0].isdigit():
                        chunk_has_digit = 1.0; break
                s5 = float(q_has_digit) * chunk_has_digit
                sig_vec = torch.tensor([s1, s2, s3, s4, s5],
                                         device=self.device,
                                         dtype=torch.float32)
                new_scores[ci] = float((weights * sig_vec).sum().item())
            scores = new_scores

        # Pseudo-relevance feedback: take top-prf_top chunks, extract
        # the top prf_terms rarest content tokens (excluding query
        # tokens), add them to the BM25 query with half-weight idf, and
        # rescore. Classical IR trick — concentrates on entity-like
        # tokens that appear near the query match.
        if prf:
            top_ci = scores.topk(prf_top).indices.cpu().tolist()
            exp_counts = {}
            for ci in top_ci:
                tf = tf_list[ci]
                for t, cnt in tf.items():
                    if t in q_content:
                        continue
                    # content token with meaningful SIF weight
                    if float(self._tok_weight[t].item()) < 0.3:
                        continue
                    exp_counts[t] = exp_counts.get(t, 0) + cnt
            # Rank by inverse DF × total count (rare + salient)
            ranked = []
            for t, c in exp_counts.items():
                df = token_df.get(t, 1)
                idf_t = _math.log((N_docs - df + 0.5) / (df + 0.5) + 1.0)
                ranked.append((t, idf_t * c, idf_t))
            ranked.sort(key=lambda x: -x[1])
            exp_terms = ranked[:prf_terms]
            if exp_terms:
                bm25_exp = np.zeros(len(tf_list), dtype=np.float32)
                for i, tf_dict in enumerate(tf_list):
                    dl = float(len_arr[i])
                    norm = 1 - b + b * dl / max(avgdl, 1e-6)
                    s = 0.0
                    for t, _score, id_val in exp_terms:
                        f = tf_dict.get(t, 0)
                        if f == 0: continue
                        s += id_val * (f * (k1 + 1)) / (f + k1 * norm)
                    bm25_exp[i] = s
                bm25_exp_t = torch.from_numpy(bm25_exp).to(self.device)
                if bm25_exp_t.std() > 0:
                    bm25_exp_t = (bm25_exp_t - bm25_exp_t.mean()) / (
                        bm25_exp_t.std() + 1e-8)
                # Blend expansion score at 0.4 weight
                scores = scores + 0.4 * bm25_exp_t

        # Chunked retrieval: dedupe chunks by parent paragraph, keep
        # the best chunk per parent, then return the PARENT as the hit.
        if self._rag_chunked:
            parent_ids = self._rag["chunk_parent"]
            # For each parent, keep max-score chunk index
            best_per_parent = {}
            # Get top 4k chunks first to save compute
            topk_raw = scores.topk(min(4 * k + 20, scores.shape[0]))
            for ci, s in zip(topk_raw.indices.cpu().tolist(),
                              topk_raw.values.cpu().tolist()):
                p = int(parent_ids[ci])
                if p not in best_per_parent or s > best_per_parent[p][1]:
                    best_per_parent[p] = (ci, s)
            ranked = sorted(best_per_parent.values(),
                             key=lambda x: -x[1])[:k]
            out = []
            for ci, s in ranked:
                p = int(parent_ids[ci])
                out.append({
                    "text": self._rag["texts"][p],
                    "title": self._rag["titles"][p],
                    "token_ids": self._rag["token_lists"][p],
                    "chunk_token_ids": self._rag["chunk_token_lists"][ci],
                    "chunk_idx": ci,
                    "score": s,
                })
            return out
        # Legacy paragraph-level
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
    def extract_answer(self, query, k_retrieve=3, max_span=100, min_span=1,
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

        # Precompute globals for v2/v3 span scorer features
        token_df_dict = self._rag["token_df"] if self._rag else {}
        N_docs_val = self._rag["N_docs"] if self._rag else 1

        # For the v3 query-adaptive scorer: precompute the query
        # feature vector + softmax weights over span features.
        v3_weights = None
        if self._span_qa is not None and self._rag is not None:
            qf_np = self._compute_qfeats(q_ids_t)
            qf_t = torch.tensor(qf_np, device=self.device,
                                 dtype=torch.float32)
            logits_per_sig = torch.einsum('ig,i->g',
                                           self._span_qa_W, qf_t)
            v3_weights = F.softmax(logits_per_sig, dim=0)   # (8,)

        # MLP ensemble span scorer: precompute query feature tensor
        mlp_qf_t = None
        if self._span_mlp is not None and self._rag is not None:
            mlp_qf_np = self._compute_qfeats(q_ids_t)
            mlp_qf_t = torch.tensor(mlp_qf_np, device=self.device,
                                     dtype=torch.float32)

        # Normalize retrieval scores across hits to [0,1]
        import math as _math
        scores_arr = np.array([h.get("score", 1.0) for h in hits],
                               dtype=np.float32)
        if scores_arr.size and scores_arr.max() > scores_arr.min():
            retr_score_by_h = (scores_arr - scores_arr.min()) / (
                scores_arr.max() - scores_arr.min() + 1e-8)
        else:
            retr_score_by_h = np.ones(scores_arr.size, dtype=np.float32)

        best_score = -1e9
        best = ("", [], {})

        # TWO-STAGE MLP PATH: when the v4 MLP ensemble scorer is loaded,
        # collect ALL candidate spans with their heuristic score first,
        # then filter to the heuristic's top-K (matching training
        # FILTER_K=50), then let the MLP rerank those. This matches
        # the training distribution exactly — training only saw top-50
        # heuristic candidates, so inference must too.
        mlp_stage_spans = []   # list of (h_idx, span, feat_dict, heur_sc)

        for h_idx, h in enumerate(hits):
            # For chunked retrieval, search spans within the matched
            # chunk (much more focused than the full parent).
            passage = h.get("chunk_token_ids") or h["token_ids"]
            # v13.1: graceful fallback when the passage has no query-rare
            # tokens. Previously this branch skipped the passage entirely,
            # producing empty extractions ~29% of the time on SQuAD dev
            # whenever a query used only common words or vocab mismatched.
            hit_positions = [i for i, t in enumerate(passage)
                              if t in q_content_rare]
            if not hit_positions:
                # Fall back to any query content token, rare filter relaxed
                hit_positions = [i for i, t in enumerate(passage)
                                  if t in q_content]
            content_idx = [i for i, t in enumerate(passage)
                            if t not in STRUCTURAL
                            and (t >= CHAR_CUTOFF or t in self._ALLOWED_PUNCT_IDS)]
            if not hit_positions:
                # Passage shares no content tokens with the query at all.
                # Rather than skipping (returning empty), treat every content
                # position as equally eligible so the scorer can still pick
                # a best-effort span based on qtype / rarity features.
                hit_positions = list(content_idx) or [0]
            dists = []
            for i in range(len(passage)):
                d = min(abs(i - p) for p in hit_positions)
                dists.append(d)

            for L in range(min_span, max_span + 1):
                for ci_start in range(len(content_idx) - L + 1):
                    start = content_idx[ci_start]
                    span = [passage[i] for i in content_idx[ci_start:ci_start + L]]
                    if all(t in q_content for t in span):
                        continue
                    if any(t == 1 for t in span):
                        continue

                    L_ = L
                    # v2 span scorer expects passage-agnostic features.
                    # Compute them in the same order as
                    # FEATURE_NAMES in genreg_span_scorer_v2.py.
                    rarity_sum = float(sum(tok_weight_np[t] for t in span))
                    rarity_max = float(max(tok_weight_np[t] for t in span))
                    length_f = float(L_)
                    semantic = 0.0
                    if q_vec is not None:
                        s_tens = torch.tensor(span, device=self.device)
                        s_w = self._tok_weight[s_tens].unsqueeze(1)
                        s_vec = (emb_n[s_tens] * s_w).sum(dim=0) / (s_w.sum() + 1e-8)
                        s_vec = F.normalize(s_vec.unsqueeze(0), dim=1).squeeze(0)
                        semantic = float((s_vec * q_vec).sum().item())
                    inv_u = -(abs(semantic - 0.5) ** 2) + 0.25
                    numeric = 1.0 if self._span_is_numeric(span) else 0.0
                    is_when = 1.0 if qtype in ("when", "year", "num") else 0.0
                    is_who = 1.0 if qtype in ("who", "whom") else 0.0
                    is_where = 1.0 if qtype == "where" else 0.0
                    is_what = 1.0 if qtype in ("what", "which") else 0.0
                    first_rare = float(tok_weight_np[span[0]])
                    last_rare = float(tok_weight_np[span[-1]])
                    # BM25-style match of span tokens against query IDF
                    bm25_span_q = 0.0
                    for t in span:
                        if t in q_content:
                            df = token_df_dict.get(t, 0)
                            if df > 0:
                                bm25_span_q += _math.log(
                                    (N_docs_val - df + 0.5) / (df + 0.5) + 1.0)
                    bm25_span_q /= max(L_, 1)

                    # v2 feature order (13 — retr_score dropped):
                    # bm25_span_q, rarity_sum, rarity_max, length,
                    # semantic_cos, inv_u_sem, numeric,
                    # is_when, is_who, is_where, is_what,
                    # first_rare, last_rare
                    feat_vec = [bm25_span_q, rarity_sum, rarity_max,
                                 length_f, semantic, inv_u, numeric,
                                 is_when, is_who, is_where, is_what,
                                 first_rare, last_rare]
                    retr_score = float(retr_score_by_h[h_idx])

                    if mlp_qf_t is not None:
                        # Stage 1: compute only heuristic score; defer
                        # MLP rerank until we have the full candidate
                        # pool to filter.
                        inv_u_m = -(abs(semantic - 0.5) ** 2) + 0.25
                        qt = 0.0
                        if is_when and numeric > 0.5: qt = 3.0
                        elif is_when: qt = -0.8
                        heur_sc = (rarity_sum + qt + inv_u_m + 0.5
                                    - 0.05 * max(0, L_ - 5))
                        mlp_stage_spans.append({
                            "h_idx": h_idx,
                            "span": list(span),
                            "start": start,
                            "L": L,
                            "heur_sc": heur_sc,
                            "sf": [bm25_span_q, rarity_sum, rarity_max,
                                    numeric, min(L_ / 8.0, 1.0),
                                    first_rare, last_rare, semantic],
                            "semantic": semantic,
                            "rarity_sum": rarity_sum,
                        })
                        continue
                    elif v3_weights is not None:
                        # v3 query-adaptive: 8-dim span feature vector
                        # matching SFEAT_NAMES order in
                        # genreg_span_scorer_v3.py
                        v3_feats = torch.tensor([
                            bm25_span_q,
                            rarity_sum,
                            rarity_max,
                            numeric,
                            min(L_ / 8.0, 1.0),
                            first_rare,
                            last_rare,
                            semantic,
                        ], device=self.device, dtype=torch.float32)
                        score = float((v3_weights * v3_feats).sum().item())
                    elif (self._span_scorer is not None
                            and self._span_scorer.get("version") == "v2_cross_passage"):
                        ft = torch.tensor(feat_vec, device=self.device,
                                           dtype=torch.float32)
                        score = float((ft * self._span_scorer_w).sum().item()
                                       + self._span_scorer_b)
                    else:
                        # Fallback heuristic (passage-relative features)
                        qtype_bonus = 0.0
                        if is_when and numeric:
                            qtype_bonus = 3.0
                        elif is_when:
                            qtype_bonus = -0.8
                        score = (rarity_sum + qtype_bonus + inv_u
                                  + (1.0 / (1.0 + h_idx)) * 0.5
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
                            "rarity_sum": rarity_sum,
                            "semantic": semantic,
                            "retr_score": retr_score,
                        }
                        text = self.detokenize(span)
                        best = (text, list(span), meta)

        # STAGE 2: if MLP is loaded, rerank top-FILTER_K heuristic candidates
        if mlp_qf_t is not None and mlp_stage_spans:
            # FILTER_K must match the v6 training value (100).
            FILTER_K = 100
            mlp_stage_spans.sort(key=lambda x: -x["heur_sc"])
            filtered = mlp_stage_spans[:FILTER_K]
            for cand in filtered:
                sf = torch.tensor(cand["sf"], device=self.device,
                                    dtype=torch.float32)
                x = torch.cat([sf, mlp_qf_t])                # (18,)
                hid = torch.tanh(
                    x @ self._span_mlp_W1 + self._span_mlp_b1)  # (16,)
                mlp_sc = float((hid @ self._span_mlp_W2 +
                                  self._span_mlp_b2).item())
                score = cand["heur_sc"] + self._span_mlp_beta * mlp_sc
                if score > best_score:
                    best_score = score
                    meta = {
                        "passage_title": hits[cand["h_idx"]].get("title"),
                        "passage_idx": cand["h_idx"],
                        "span_start": cand["start"],
                        "span_len": cand["L"],
                        "score": score,
                        "qtype": qtype,
                        "heur_sc": cand["heur_sc"],
                        "mlp_sc": mlp_sc,
                        "rarity_sum": cand["rarity_sum"],
                        "semantic": cand["semantic"],
                    }
                    text = self.detokenize(cand["span"])
                    best = (text, list(cand["span"]), meta)

        if return_passage:
            return best + (hits,)
        return best

    @torch.no_grad()
    def generate_qa(self, query, k_retrieve=3, max_span=100):
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
            # Use the full parent paragraph as copy source — more
            # coverage than the single chunk. Chunk tokens get an
            # extra SIF-weight boost below since they matched the
            # retrieval signal directly.
            chunk_set = set(h.get("chunk_token_ids") or [])
            for t in h["token_ids"]:
                if (t >= CHAR_CUTOFF or t in self._ALLOWED_PUNCT_IDS) \
                        and t not in (66, 67):
                    w = float(tok_weight_np[t]) if tok_weight_np is not None else 1.0
                    if w < 0.2:
                        continue
                    # Boost tokens that are in the matched chunk
                    if t in chunk_set:
                        w = min(1.0, w * 1.5)
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
            # Augment with rare passage tokens at the max n-gram prob
            # so attention cosine directly picks between "continue per
            # grammar" vs "copy from passage."
            existing = set(c[0] for c in cands)
            if cands:
                passage_base = max(c[1] for c in cands)
            else:
                passage_base = 0.2
            for t, w in passage_weighted.items():
                if t not in existing:
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
