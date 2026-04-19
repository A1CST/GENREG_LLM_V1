# Changelog

A running log of what changed between snapshots so a reviewer (human
or AI) can tell at a glance what state the repo is in. Each version
block includes the headline accuracy number and the honest limitation
that defines the next move.

## v6-mlp-span-scorer-bigdata (2026-04-19, current)

**Headline:** retrieval recall@1 = 53.3 %, **extractive answer
containment = 9.0 %** (+1.3 pp over heuristic-only v4, +0.3 pp over
v5), RAG generation = 6.0 % (SQuAD v1.1 dev, 300-q sample, seed 7).

**What's new:** MLP span scorer retrained on 9.6K training QA pairs
(v5 used 2.4K). 1,894 effective train + 446 val examples after
top-100 heuristic filter. val top-1 = 29.6% (heuristic baseline
17.3%, +12.3 pp val lift). β = -1.05 ensemble weight.

Progression:
- Heuristic only (v4):   7.7 % extractive
- MLP 2.4K QA, top-50:   8.7 %  (v5)
- MLP 9.6K QA, top-100:  **9.0 %** (v6 — current)

## v5-mlp-span-scorer (2026-04-19)

**Headline:** retrieval recall@1 = 53.3 %, extractive answer
containment = 8.7 % (+1.0 pp over heuristic-only v4), RAG
generation = 6.0 % (SQuAD v1.1 dev, 300-q sample, seed 7, k=3).

**First learned span-scorer that actually improves dev performance
after four prior failures.** Key fixes over v1-v4 span-scorer
attempts:

1. **Containment labels instead of exact-match.** Earlier trainers
   labeled a span as positive only if span tokens exactly equal gold
   tokens. But inference measures whether *gold text appears as a
   substring* of the returned span. ~8% of SQuAD dev cases have
   SOME span containing gold at heuristic's top-1, vs ~0.3% that
   exactly match. Training on the containment signal matches the
   inference metric.
2. **Train/val split with early stopping on val fitness.** Snapshot
   the val-best genome, not the train-best. Previous runs overfit to
   99 % train / 0.3 % dev.
3. **Widened filter pool from top-20 to top-50.** Heuristic top-20
   only contains gold in 9 % of SQuAD dev cases; top-50 hits 13 %.
   MLP now reranks top-50 instead of top-20, raising the absolute
   ceiling.
4. **No gold substitution in training candidates.** Prior v1 run hit
   100 % train/val top-1 via a position leak — when gold wasn't in
   heuristic top-20, it got substituted at position -1, letting the
   MLP learn "pick lowest-heuristic = gold." Removed substitution;
   now skip training examples where heuristic filter misses gold.
5. **Two-stage inference path.** At inference, collect all spans with
   heuristic scores FIRST, filter to top-50 by heuristic, then apply
   the MLP to just those 50. Matches training distribution exactly.

**Training metrics (SQuAD train, 80/20 split, 2400/600 questions):**
- Effective training examples: 322 (gold found in heuristic top-50
  for only 13 % of QA pairs — corresponds to the dev ceiling)
- Heuristic baseline top-1 on val: 24.7 %
- MLP ensemble top-1 on val: **42.7 %** (+18.0 pp)
- β (MLP correction weight): −0.8 (MLP contribution to ensemble)
- Parameters: 305 (W1: 18×16, b1: 16, W2: 16×1, b2: 1, β: 1)

### Architecture

```
span_features (8) || query_features (10)   → 18
  ↓ W1 (18×16), tanh
hidden (16)
  ↓ W2 (16×1)
mlp_score (1)

final_score = heuristic_score + β * mlp_score
```

Query features: qtype one-hots {when, who, where, what, how},
has_digit, n_rare_tokens_norm, max_tok_sif, q_length_norm, bias.

Span features: BM25(span, query content), rarity_sum, rarity_max,
numeric, length_norm, first_rare, last_rare, semantic_cos.

The ensemble design (heuristic + β·mlp) means worst case the MLP
adds noise and β → 0, recovering heuristic. Actual learned β=−0.8
means the MLP adds signal orthogonal to the heuristic.

## v4-query-adaptive-retrieval (2026-04-18)

**What's new:**
- **Query-adaptive retrieval reranker** shipped at
  `checkpoints/retrieval_reranker.pkl` (50 params: 10 query features
  × 5 signal weights). Evolved against SQuAD train with
  inference-identical candidate pools. Training top-1 among top-20
  hybrid candidates = 62 %. **OFF BY DEFAULT** because on SQuAD dev
  the gain is marginal (+0.7 pp at recall@1) and recall@3 regresses
  slightly. Enable by setting `model._use_reranker = True`.
- `CHANGELOG.md` — version control for the auto-push cron.

### The query-adaptive signal attention design

Query features (10-dim): one-hots for {when, who, where, what, how},
has_digit, n_rare_tokens, max_tok_sif, q_length, bias.

Passage/span signals at retrieval time (5-dim): BM25, SIF cosine,
BM25-over-bigrams, length-match, numeric coincidence.

Scorer: `weights = softmax(query_features @ W)`, where W is the
evolved 10×5 matrix. Final score per candidate =
`dot(weights, signal_vector)`.

**Learned retrieval weights** (interpretable — reveal what the model
thinks each question type cares about):
```
                   bm25   sif   bigram  length  numeric
  is_when        -0.32  -1.38   +1.15   -0.37   +1.15
  is_who         -0.29  -0.44   -0.86   -0.99   +0.68
  is_where       +0.16  -0.55   +0.71   -1.15   -0.02
  is_what        +1.64  +2.06   +0.89   +0.32   -0.46
  is_how         +1.37  +1.88   -1.60   +0.22   -0.13
  has_digit      -0.75  -0.57   -0.66   -0.36   +1.02
```
"When" questions learn to lean on bigram-BM25 and numeric-match; "who"
questions lean on numeric-match (dates next to names); "what/how" lean
on plain BM25 + SIF.

### What did NOT work this session

1. **Span scorer v2** (single-passage training + full-passage
   features) — regressed extractive 8.3 % → 2-3 %. Diagnosis:
   feature distribution mismatch between full-passage training and
   chunked-retrieval inference.
2. **Span scorer v2-chunked** (retrained with chunk-matched inputs)
   — still regressed. The 50-candidate training pool didn't prepare
   the scorer for the 1000+ candidate inference pool.
3. **Span scorer v3-query-adaptive** (same pattern that worked for
   retrieval, applied to spans) — training top-1 only 3.8 % out of
   ~1000 candidates per question. Deployed: extractive dropped to
   2.0 %. 80 params can't linearly discriminate the gold span from
   1000+ plausible-looking spans. Discarded.
4. **Span scorer v3-two-stage** (heuristic filter top-20 →
   learned reranker). Training top-1 8.3 % on the filtered pool —
   still ≈ random within the 20 good-looking candidates the heuristic
   selected. When all candidates are heuristic-endorsed, the learnable
   features don't add more discriminating information. Discarded.
5. **Span MLP v1** (305-param MLP, ensemble = heur + β·mlp_score,
   one-shot training on all 1500 QA cases). Converged to 99.7 %
   train top-1 in 25 generations. Deployed: dev extractive
   **0.3 %**. Classic overfitting — small training set, big model,
   no validation signal.
6. **Span MLP v2** (val-split 80/20 + early-stopping checkpoint on
   val top-1). **Discovered a data leak**: when gold wasn't in the
   heuristic's top-20 filter, the trainer substituted gold at the
   lowest position. MLP learned "pick lowest-heuristic = gold" and
   reached 100 % val top-1 — but the signal evaporates at inference
   where no such substitution happens. Killed training at gen 300.
   **Lesson: evolutionary span scorers on small datasets require
   careful negative-sampling hygiene. Any systematic position signal
   in the candidate list is a leak.**
7. **Span MLP v3** (fix: drop training examples where gold isn't
   organically in heuristic top-20 — no substitution). Currently
   training. See below for outcome.
4. **Pseudo-relevance feedback** — amplifies the wrong topic when
   initial retrieval is wrong (32 % of top-3). Off by default.
5. **Query expansion via embedding neighbors** — flat (±0.3 pp).
   The evolved embedding's nearest neighbors are topical/co-
   occurrence-derived, not synonyms.

### Why the span-scorer attempts keep failing (and what would fix it)

The retrieval reranker works because it only has to distinguish
among ~20 candidate passages (the hybrid's top-20), and each
candidate has ~5 strong pre-computed signals. The learnable head
has enough capacity for 20-way discrimination.

The span scorer has to distinguish gold from ~1000 other spans in
3 chunks, and the per-span features are weaker. An 80-parameter
linear head can't do it — training top-1 plateaus at 3-4 %. This is
a capacity/complexity mismatch, not a training-data issue.

**Proper fix (not shipped, out of session scope):**

Option A — two-stage filtering. Use the heuristic to pick top-10
spans per chunk (30 total), then apply a query-adaptive reranker
to that filtered pool. Reduces the candidate space to something the
learnable head can handle.

Option B — nonlinear scorer. A small MLP (e.g., 8 features → 16
hidden → 1 score) has enough capacity for span-level ranking, but
evolution on ~150 parameters is slower to converge and wants more
training data.

Option C — learn to rank spans by COMPARING to gold span directly
(cosine similarity of full span to gold-answer embedding), not by
absolute feature-based score. This is a different fitness
formulation entirely.

The heuristic extractive (8 % answer containment) is the current
best. Rather than fight the linear-scorer wall further, the next
session should try Option A: two-stage filtering.

### Next-move ranking for the reviewing AI

1. **Two-stage span selection** (heuristic filter → adaptive rerank
   on filtered pool). Likely the biggest near-term extractive lift.
2. **Ensemble heuristic + learned scorer** — sum heuristic score and
   a learned scorer's score. Provides a floor of heuristic quality
   plus whatever upside the scorer adds.
3. **Multi-span extraction** — return top 3 scored spans as
   alternatives, not just top 1. Quick win for user experience.
4. **Better retrieval**: the reranker achieves 62 % on training
   top-20 but only +0.7 pp on dev. Either train on a larger/more
   diverse set, or train a full re-ranker (10 → 20 → 5 MLP) with
   more capacity.
5. **Cross-encoder over chunks + spans** — expensive but known to
   work. Would need to evolve a small transformer-style attention
   head over (query, span) pairs.

## v3-chunked-rag (2026-04-18, earlier today)

**Headline:** retrieval recall@1 = 52.3 %, extractive = 7.7 %, RAG
generation = 5.0 %.

**What changed since v2:**
- Paragraph-chunked retrieval index (20,958 paragraphs →
  46,586 × 80-content-token chunks with 20 overlap). Chunk
  embeddings int8-quantized to keep `rag_index.pkl` under GitHub's
  100 MB limit.
- `retrieve()` supports both chunked_v1 and legacy paragraph
  formats; returns the best chunk per parent, deduplicated.
- Span extraction searches inside the matched chunk scope, not the
  full parent (roughly 2× more focused).
- BM25 parameters retuned via 60-point sweep:
  `k1=1.2, b=0.5, bm25_weight=0.85`.
- Query expansion via embedding neighbors wired in (default on,
  weight 0.4). Neutral effect on aggregate metric.
- Pseudo-relevance feedback wired in but default off.

## v2 (2026-04-18, earlier today)

**Headline:** retrieval recall@1 = 51.3 %, extractive = 6.0 %, RAG
generation = 7.3 %.

**What changed since v1:**
- Vocabulary extended 51,641 → 61,641 (10 K new tokens: years,
  numbers, entity fragments). Embedding `hash_in` extended with
  random-Gaussian rows matching existing dimension statistics.
- N-gram tables recounted on the extended-vocab punctuated stream.
- RAG paragraph index rebuilt.

## v1 (2026-04-18, earlier today)

**First RAG release:**
- Hybrid BM25 + SIF dense retrieval on 20,958 paragraphs from
  SQuAD train + dev contexts.
- Copy-pool augmentation of rerank candidates.
- Heuristic extractive QA.

## chatbot-v1 (pre-RAG)

Sentence-shaped output with natural-stop 75-85 %, factual
accuracy effectively zero.

## gradient-free-clean (foundational)

Removed the ridge head and re-evolved CE attention against the
evolved embedding table. Full audit in `GRADIENT_AUDIT_REPORT.md`.
Foundation of the no-gradient claim.
