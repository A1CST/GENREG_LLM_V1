#!/usr/bin/env python3
"""retrieval_qadapt_v2 — per-query-adaptive retrieval reranker (take 2)

MODEL CARD (per GENREG_RULES §II)
  Name:      retrieval_qadapt_v2
  Purpose:   Learn query-adaptive blend weights over K=5 retrieval signals
             (bm25, dense, bm25_bigram, length_match, numeric_coin) so that
             per-query reranking pushes gold to r@1. Static blend caps at
             54.3 % r@1 on dev (bm25_w sweep); anything above that must
             come from per-query adaptation.
  Interface: q_feat ∈ R^10 (from model._compute_qfeats) → w ∈ Simplex^5
             At inference: rerank top-20 candidates with signal · w.
  Evolved parameters:
    W1 (10, H=16)   init 1/√10 · N(0,1)            160 params
    b1 (H,)         init 0.01 · N(0,1)              16
    W2 (H, 5)       init 1/√H  · N(0,1)             80
    b2 (5,)         init 0.01 · N(0,1)               5
    act_ids (H,)    random choice from 8-fn catalog (per-neuron)  16
    act_p1..p4 (H,) init N(0,1) each              4×16 = 64
    total ≈ 340 params
  Fitness:   mean log_softmax(signal @ w)[gold_idx]  (soft fitness, §IV.1)
             Multiplicative-equivalent via mean log-prob = log geo-mean.
  Energy:    delta = fitness - median_fitness
             DECAY=0.9 GAIN=2.0 FLOOR=0.2 E_MAX=1.5
             Target starved 3–15 % (§III).
  Selection: tournament with maturation gate=1; SURVIVAL_PCT=20; POP=300
  Mutation:  rate self-adapted [0.005, 0.2] start 0.05, floor 0.02
             scale self-adapted start 0.05
             anneal scale × 0.5 after 80 % of gens
  Hyperparams:
    N_GENERATIONS=2000   TRAIN_Q=8000   DEV_Q=300 (seed 7)
    PROBE_Q per gen=128  LOG_EVERY=50   DEV_EVAL_EVERY=100
  Success:   local: beat static-0.9 blend on train soft-fitness
             downstream: beat static-0.9 on dev r@1 by ≥1 pp
                         with train→dev drop < 5 % (§VII.2)
  Failure modes to watch:
    - train climbs, dev flat (distribution mismatch; v11 pattern)
    - mode collapse — all genomes output identical w
    - dev regression after train peak (overfit; stop early)
    - starved == 0 for 500+ gens (energy decorative)
  Baselines:
    - majority (shipped 0.85 blend):   53.0 % r@1 / 68.0 % r@3 on dev
    - best-static (0.9 blend):         54.3 % r@1 / 67.7 % r@3 on dev
    - v11 qadapt (historical):         +5 pp train, 0 pp dev — failed transfer
  Artifacts: checkpoints/retrieval_qadapt_v2.pkl, run_qadapt_v2.log,
             blend_sweep_results.json (prior baseline)
"""
import os, sys, json, time, pickle, random
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.abspath(__file__))
# Pull evolved-activation catalog from the main project root
for _ in range(4):
    _ROOT = os.path.dirname(_ROOT)
    if os.path.exists(os.path.join(_ROOT, "genreg_encoder_gpu.py")):
        sys.path.insert(0, _ROOT); break
from genreg_encoder_gpu import apply_evolved_activations, NUM_ACTIVATIONS

from lib.model import GenregLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QFEAT_D = 10
SIGNALS = 5               # bm25, dense, bm25_bigram, len_match, numeric
HIDDEN = 16
POP = 300
N_GEN = 2000
PROBE_Q = 128
DEV_EVAL_EVERY = 100
LOG_EVERY = 50
TRAIN_Q = 8000
DEV_Q = 300
SEED = 7
CACHE_FN = "qadapt_v2_cache.pkl"
OUT_CKPT = "checkpoints/retrieval_qadapt_v2.pkl"

ENERGY_DECAY = 0.9
ENERGY_GAIN = 2.0
ENERGY_FLOOR = 0.2
E_MAX = 1.5
SURVIVAL_PCT = 0.20
MUT_RATE_INIT = 0.05
MUT_RATE_MIN = 0.005
MUT_RATE_MAX = 0.20
MUT_SCALE_INIT = 0.05
ANNEAL_FRAC = 0.8


# ============================================================
# DATA PREP — precompute (qfeats, signal_matrix, gold_idx) per query
# ============================================================

def build_dataset(m, cases, n_keep, label):
    """For each case, retrieve top-20, keep only if gold is in pool, record
    (qfeats (10,), signals (20, 5), gold_idx (int))."""
    out = []
    skipped = 0
    t0 = time.time()
    STRUCTURAL = {66, 67}
    from lib.model import CHAR_CUTOFF
    for i, c in enumerate(cases):
        if len(out) >= n_keep:
            break
        hits = m.retrieve(c["q"], k=20)
        gold_idx = None
        for r, h in enumerate(hits):
            if h["text"] == c["ctx"]:
                gold_idx = r; break
        if gold_idx is None:
            skipped += 1; continue
        q_ids = m.tokenize(c["q"])
        qfeats_np = m._compute_qfeats(q_ids)
        # Compute 5 signals per candidate. Mirror model.py reranker logic.
        q_tok_ids = set(int(t) for t in q_ids.cpu().tolist())
        q_content = {t for t in q_tok_ids
                      if t >= CHAR_CUTOFF or t in m._ALLOWED_PUNCT_IDS}
        q_content.difference_update(STRUCTURAL)
        q_bigrams = set()
        ql = list(q_content)
        for k in range(len(ql) - 1):
            q_bigrams.add((ql[k], ql[k + 1]))
        digit_ids = {m.token_to_id.get(ch) for ch in "0123456789"} - {None}
        q_has_digit = 1.0 if any(t in digit_ids for t in q_ids.tolist()) else 0.0
        avgdl = m._rag["avgdl"]
        sig = np.zeros((20, SIGNALS), dtype=np.float32)
        for r, h in enumerate(hits):
            s1 = float(h.get("score", 0.0))                 # blended retrieval
            s2 = float(h.get("bm25", 0.0)) if isinstance(
                h.get("bm25", None), (int, float)) else 0.0
            # Simpler features we CAN compute from the hit:
            chunk_toks = h.get("chunk_token_ids") or h["token_ids"]
            c_bigrams = set()
            if isinstance(chunk_toks, list):
                for k in range(len(chunk_toks) - 1):
                    c_bigrams.add((chunk_toks[k], chunk_toks[k + 1]))
            s3 = float(len(q_bigrams & c_bigrams))
            cl = float(len(chunk_toks)) if isinstance(chunk_toks, list) else avgdl
            s4 = 1.0 - min(abs(cl - avgdl) / max(avgdl, 1e-6), 1.0)
            chunk_has_digit = 0.0
            if isinstance(chunk_toks, list):
                for t in chunk_toks[:200]:
                    if t in digit_ids:
                        chunk_has_digit = 1.0; break
            s5 = q_has_digit * chunk_has_digit
            sig[r] = [s1, s2, s3, s4, s5]
        # z-score each signal column across candidates (per-query norm)
        mu = sig.mean(axis=0, keepdims=True)
        sd = sig.std(axis=0, keepdims=True) + 1e-8
        sig_z = (sig - mu) / sd
        out.append({
            "qfeats": qfeats_np,
            "sig_z": sig_z,
            "gold_idx": gold_idx,
        })
        if (i + 1) % 200 == 0 or len(out) == n_keep:
            print(f"  [{label}] {i+1} seen, {len(out)} kept, "
                  f"{skipped} skipped  ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"  [{label}] final: {len(out)} kept / {i+1} seen "
          f"({skipped} skipped, {time.time()-t0:.0f}s)", flush=True)
    return out


def load_or_build_cache(m):
    if os.path.exists(CACHE_FN):
        print(f"loading cache {CACHE_FN}...", flush=True)
        with open(CACHE_FN, "rb") as f:
            return pickle.load(f)
    print("cache miss, building...", flush=True)
    # Train pool
    train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "../LLM/data_raw/squad_train.json")
    with open(train_path) as f:
        tr = json.load(f)
    train_cases = []
    for art in tr["data"]:
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                if qa.get("answers"):
                    train_cases.append({"q": qa["question"],
                                          "ctx": para["context"]})
    rng = np.random.default_rng(SEED)
    rng.shuffle(train_cases)

    # Dev pool — same 300 we've been using
    dev_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "../LLM/data_raw/squad_dev.json")
    with open(dev_path) as f:
        de = json.load(f)
    dev_cases = []
    for art in de["data"]:
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                if qa.get("answers"):
                    dev_cases.append({"q": qa["question"],
                                        "ctx": para["context"]})
    dev_rng = np.random.default_rng(SEED)
    if len(dev_cases) > DEV_Q:
        dev_cases = list(dev_rng.choice(dev_cases, size=DEV_Q, replace=False))

    train = build_dataset(m, train_cases, TRAIN_Q, "TRAIN")
    dev = build_dataset(m, dev_cases, DEV_Q, "DEV")
    cache = {"train": train, "dev": dev}
    with open(CACHE_FN, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"cached to {CACHE_FN}", flush=True)
    return cache


# ============================================================
# POPULATION + FORWARD
# ============================================================

def init_pop(P):
    std_in = 1.0 / (QFEAT_D ** 0.5)
    std_h = 1.0 / (HIDDEN ** 0.5)
    return {
        "W1": std_in * torch.randn(P, QFEAT_D, HIDDEN, device=DEVICE),
        "b1": 0.01 * torch.randn(P, HIDDEN, device=DEVICE),
        "W2": std_h * torch.randn(P, HIDDEN, SIGNALS, device=DEVICE),
        "b2": 0.01 * torch.randn(P, SIGNALS, device=DEVICE),
        "act_ids": torch.randint(0, NUM_ACTIVATIONS, (P, HIDDEN), device=DEVICE),
        "act_p1": torch.randn(P, HIDDEN, device=DEVICE),
        "act_p2": torch.randn(P, HIDDEN, device=DEVICE),
        "act_p3": torch.randn(P, HIDDEN, device=DEVICE),
        "act_p4": torch.randn(P, HIDDEN, device=DEVICE),
        # self-adapted mutation
        "mut_rate": torch.full((P,), MUT_RATE_INIT, device=DEVICE),
        "mut_scale": torch.full((P,), MUT_SCALE_INIT, device=DEVICE),
        # energy
        "energy": torch.ones(P, device=DEVICE),
    }


@torch.no_grad()
def forward_all(pop, qf_batch, sig_batch):
    """qf_batch (B, 10), sig_batch (B, 20, 5). Returns log_p_gold (P, B)."""
    B = qf_batch.shape[0]
    # h1 for each genome: (P, B, HIDDEN)
    #   qf_batch @ pop.W1 → (P, B, HIDDEN)
    h1 = torch.einsum('bi,pih->pbh', qf_batch, pop["W1"]) + \
         pop["b1"].unsqueeze(1)                       # (P, B, HIDDEN)
    # per-neuron evolved activation: apply_evolved_activations expects
    # (H_flat, N_flat). We have (P, B, HIDDEN) and per-genome per-neuron
    # activation ids + params. Flatten per-genome.
    P = pop["W1"].shape[0]
    # Reshape so the H dimension is neurons (for per-neuron act)
    h1_f = h1.permute(0, 2, 1).reshape(P * HIDDEN, B)  # (P*H, B)
    act_ids = pop["act_ids"].reshape(P * HIDDEN)
    p1 = pop["act_p1"].reshape(P * HIDDEN, 1).expand(P * HIDDEN, B).contiguous()
    p2 = pop["act_p2"].reshape(P * HIDDEN, 1).expand(P * HIDDEN, B).contiguous()
    p3 = pop["act_p3"].reshape(P * HIDDEN, 1).expand(P * HIDDEN, B).contiguous()
    p4 = pop["act_p4"].reshape(P * HIDDEN, 1).expand(P * HIDDEN, B).contiguous()
    a = apply_evolved_activations(h1_f, act_ids, p1, p2, p3, p4)
    a = a.view(P, HIDDEN, B).permute(0, 2, 1).contiguous()  # (P, B, HIDDEN)
    a = a.clamp(-10.0, 10.0)
    # Output weights: (P, B, 5) = a @ W2 + b2
    w = torch.einsum('pbh,phk->pbk', a, pop["W2"]) + pop["b2"].unsqueeze(1)
    # Softmax to force simplex weights
    w = F.softmax(w, dim=-1)                               # (P, B, 5)
    # Scores: signals (B, 20, 5) times weights (P, B, 5) → (P, B, 20)
    scores = torch.einsum('bcs,pbs->pbc', sig_batch, w)
    log_p = F.log_softmax(scores, dim=-1)                  # (P, B, 20)
    return log_p, w


@torch.no_grad()
def fitness_batch(pop, qf, sig, gold_idx):
    """qf (B, 10), sig (B, 20, 5), gold_idx (B,). Returns fit (P,) — higher better."""
    log_p, _ = forward_all(pop, qf, sig)
    B = gold_idx.shape[0]
    lp_gold = log_p.gather(2, gold_idx.view(1, B, 1).expand(log_p.shape[0], B, 1)
                            ).squeeze(2)                    # (P, B)
    return lp_gold.mean(dim=1)


@torch.no_grad()
def dev_r1(pop, qf, sig, gold_idx):
    """Return (P,) top-1 accuracy on dev."""
    log_p, _ = forward_all(pop, qf, sig)
    pred = log_p.argmax(dim=2)                              # (P, B)
    hits = (pred == gold_idx.unsqueeze(0)).float().mean(dim=1)
    return hits


def update_energy(pop, fit):
    median = fit.median()
    delta = fit - median
    pop["energy"] = (pop["energy"] * ENERGY_DECAY + delta).clamp(0.0, E_MAX)


def evolve(pop, fit):
    P = fit.shape[0]
    alive = pop["energy"] > ENERGY_FLOOR
    if alive.sum().item() < max(P // 4, 4):
        # too many starved; promote least-bad
        lift_n = max(P // 4, 4) - int(alive.sum().item())
        dead_fit = fit.clone()
        dead_fit[alive] = -1e9
        lift_idx = dead_fit.topk(lift_n).indices
        pop["energy"][lift_idx] = ENERGY_FLOOR + 0.01

    # Tournament: top SURVIVAL_PCT form elite pool
    n_elite = max(int(P * SURVIVAL_PCT), 2)
    # Blend fit with tiny noise to break ties
    fit_n = fit + 1e-4 * torch.randn_like(fit)
    # Only alive genomes eligible
    fit_eligible = fit_n.clone()
    fit_eligible[~alive] = -1e9
    elite_idx = fit_eligible.topk(n_elite).indices

    # Replace bottom (P - n_elite) genomes with mutated copies of random elites
    replace_targets = fit_eligible.topk(P - n_elite, largest=False).indices
    parents = elite_idx[torch.randint(0, n_elite, (replace_targets.shape[0],),
                                        device=DEVICE)]
    # Copy parents into replace_targets positions (except params; keep sep)
    for key in ("W1", "b1", "W2", "b2", "act_p1", "act_p2", "act_p3", "act_p4",
                 "act_ids", "mut_rate", "mut_scale"):
        pop[key][replace_targets] = pop[key][parents]
    pop["energy"][replace_targets] = 0.75  # child starts fresh but not full

    # Mutation (vectorized)
    # Self-adapted: perturb rate and scale first, then weights
    for idx in range(P):
        if idx in elite_idx.tolist():
            # Elite: mild mutation only (preserve basin)
            continue
        rate = pop["mut_rate"][idx].item()
        scale = pop["mut_scale"][idx].item()
        # mutate the mutation params themselves slightly
        new_rate = float(min(MUT_RATE_MAX,
                              max(MUT_RATE_MIN,
                                  rate * float(torch.exp(0.2 * torch.randn(1))))))
        new_scale = float(max(0.02,
                               scale * float(torch.exp(0.2 * torch.randn(1)))))
        pop["mut_rate"][idx] = new_rate
        pop["mut_scale"][idx] = new_scale
        if torch.rand(1).item() < new_rate:
            pop["W1"][idx] += new_scale * torch.randn_like(pop["W1"][idx])
            pop["W2"][idx] += new_scale * torch.randn_like(pop["W2"][idx])
            pop["b1"][idx] += new_scale * torch.randn_like(pop["b1"][idx])
            pop["b2"][idx] += new_scale * torch.randn_like(pop["b2"][idx])
            pop["act_p1"][idx] += new_scale * torch.randn_like(pop["act_p1"][idx])
            pop["act_p2"][idx] += new_scale * torch.randn_like(pop["act_p2"][idx])
            pop["act_p3"][idx] += new_scale * torch.randn_like(pop["act_p3"][idx])
            pop["act_p4"][idx] += new_scale * torch.randn_like(pop["act_p4"][idx])
            # occasionally flip one activation
            if torch.rand(1).item() < 0.1:
                flip = torch.randint(0, HIDDEN, (1,)).item()
                pop["act_ids"][idx, flip] = torch.randint(
                    0, NUM_ACTIVATIONS, (1,), device=DEVICE).item()


def static_baseline_fit(qf, sig, gold_idx, w_static):
    """Fitness for a FIXED static weight vector (majority baseline)."""
    w = torch.tensor(w_static, device=DEVICE, dtype=torch.float32)
    scores = torch.einsum('bcs,s->bc', sig, w)
    log_p = F.log_softmax(scores, dim=-1)
    B = gold_idx.shape[0]
    lp_gold = log_p.gather(1, gold_idx.unsqueeze(1)).squeeze(1)
    top1 = (log_p.argmax(dim=1) == gold_idx).float().mean().item()
    return lp_gold.mean().item(), top1


def to_gpu(batch):
    qf = torch.tensor(np.stack([b["qfeats"] for b in batch]),
                        device=DEVICE, dtype=torch.float32)
    sig = torch.tensor(np.stack([b["sig_z"] for b in batch]),
                        device=DEVICE, dtype=torch.float32)
    gi = torch.tensor([b["gold_idx"] for b in batch],
                        device=DEVICE, dtype=torch.long)
    return qf, sig, gi


def main():
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    print("Loading model...", flush=True)
    m = GenregLM("./checkpoints", device=DEVICE)
    cache = load_or_build_cache(m)
    train = cache["train"]; dev = cache["dev"]
    print(f"\ntrain={len(train)}  dev={len(dev)}", flush=True)

    # Baselines
    dev_qf, dev_sig, dev_gi = to_gpu(dev)
    # Signal index 0 = blended retrieval score (shipped 0.85 default) — use
    # that alone as "majority static baseline."
    for w_label, w_vec in [
        ("pure-s1 (shipped-retrieval)", [1.0, 0.0, 0.0, 0.0, 0.0]),
        ("uniform", [0.2]*5),
    ]:
        lp, t1 = static_baseline_fit(dev_qf, dev_sig, dev_gi, w_vec)
        print(f"  static [{w_label}]: dev log_p={lp:.3f} r@1={t1:.3f}",
              flush=True)

    # Train loop
    pop = init_pop(POP)
    t0 = time.time()
    best_dev = -1e9
    best_state = None
    best_gen = 0
    patience = 0
    MAX_PATIENCE = 8  # 8 dev evals without improvement → stop

    for gen in range(N_GEN):
        # Sample probe batch
        probe = random.sample(train, min(PROBE_Q, len(train)))
        qf, sig, gi = to_gpu(probe)
        fit = fitness_batch(pop, qf, sig, gi)
        update_energy(pop, fit)
        evolve(pop, fit)

        if gen % LOG_EVERY == 0:
            starved = int((pop["energy"] < ENERGY_FLOOR).sum().item())
            print(f"GEN {gen:4d}  best_fit={fit.max().item():+.4f}  "
                  f"med_fit={fit.median().item():+.4f}  "
                  f"mut_r_med={pop['mut_rate'].median().item():.3f}  "
                  f"mut_s_med={pop['mut_scale'].median().item():.3f}  "
                  f"starved={starved}/{POP} ({100*starved/POP:.0f}%)  "
                  f"{time.time()-t0:.0f}s", flush=True)

        if gen % DEV_EVAL_EVERY == 0 or gen == N_GEN - 1:
            # Evaluate each genome on full dev, pick best by dev fitness
            dev_fit = fitness_batch(pop, dev_qf, dev_sig, dev_gi)
            dev_r1_all = dev_r1(pop, dev_qf, dev_sig, dev_gi)
            bi = int(dev_fit.argmax().item())
            d_lp = dev_fit[bi].item()
            d_t1 = dev_r1_all[bi].item()
            t_lp = fit[bi].item()   # that same genome's training fit this batch
            if d_lp > best_dev:
                best_dev = d_lp
                best_gen = gen
                patience = 0
                best_state = {
                    k: (pop[k][bi].detach().cpu().numpy()
                         if isinstance(pop[k], torch.Tensor) and pop[k].dim() > 0
                         else pop[k][bi])
                    for k in pop
                }
                best_state.update({"gen": gen, "dev_lp": d_lp,
                                    "dev_r1": d_t1, "train_lp": t_lp})
            else:
                patience += 1
            print(f"  DEV @ gen {gen:4d}: lp={d_lp:+.4f} r@1={d_t1:.3f} "
                  f"(train_lp={t_lp:+.4f})  best_dev={best_dev:+.4f}@{best_gen}  "
                  f"patience={patience}/{MAX_PATIENCE}", flush=True)
            if patience >= MAX_PATIENCE:
                print(f"  early stop: no dev improvement in "
                      f"{patience} evals", flush=True)
                break

        # Anneal scale
        if gen == int(N_GEN * ANNEAL_FRAC):
            pop["mut_scale"] *= 0.5
            print(f"  [anneal] mut_scale × 0.5 at gen {gen}", flush=True)

    if best_state is None:
        raise RuntimeError("no improving gen")
    os.makedirs(os.path.dirname(OUT_CKPT), exist_ok=True)
    with open(OUT_CKPT, "wb") as f:
        pickle.dump(best_state, f)
    print(f"\n=== FINAL ===", flush=True)
    print(f"  best_dev lp: {best_dev:+.4f} @ gen {best_state['gen']}", flush=True)
    print(f"  best_dev r@1: {best_state['dev_r1']:.3f}", flush=True)
    print(f"  best_train lp (same genome): {best_state['train_lp']:+.4f}", flush=True)
    drop = ((best_state["train_lp"] - best_state["dev_lp"])
             / max(abs(best_state["train_lp"]), 1e-6))
    print(f"  train→dev relative drop: {drop*100:.1f}%  "
          f"(target < 5%)", flush=True)
    print(f"  saved → {OUT_CKPT}", flush=True)


if __name__ == "__main__":
    main()
