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
