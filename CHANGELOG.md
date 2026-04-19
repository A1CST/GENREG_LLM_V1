# Changelog

A running log of what changed between snapshots so a reviewer (human
or AI) can tell at a glance what state the repo is in. Each version
block includes the headline accuracy number and the honest limitation
that defines the next move.

## v3-chunked-rag (2026-04-18, current)

**Headline:** retrieval recall@1 = 53.3 %, extractive answer
containment = 8.3 %, RAG generation answer containment = 5.7 %
(SQuAD v1.1 dev, 300 q sampled with seed 7, k=3 retrieval).

**What changed since v2:**
- Paragraph-chunked retrieval index (20,958 paragraphs →
  46,586 × 80-content-token chunks with 20 overlap). Chunk embeddings
  int8-quantized to keep `rag_index.pkl` under GitHub's 100 MB limit.
- `retrieve()` now supports both `chunked_v1` and legacy paragraph
  formats; returns the best chunk per parent, deduplicated.
- Span extraction searches inside the matched chunk scope, not the
  full parent (roughly 2× more focused).
- BM25 parameters retuned via 60-point sweep: `k1=1.2, b=0.5,
  bm25_weight=0.85` (was `k1=1.2, b=0.75, weight=0.7`). Adds 2 pp to
  recall@1 on 150-q dev sample.
- Query expansion via embedding neighbors wired in (default on,
  weight 0.4). Effect on SQuAD dev is nil; kept as a knob because
  it helps paraphrase cases even when it doesn't move the aggregate
  metric.
- Pseudo-relevance feedback wired in but default off — when initial
  retrieval is wrong, PRF amplifies the wrong topic. Useful only for
  confident top-1.
- `benchmark.py` 3-mode chatbot samples (no-RAG / RAG / extractive).

**What did NOT work and why:**
- **Span scorer v2**: evolved linear head trained on SQuAD train Q/A
  with cross-passage negative spans. Two training variants (with and
  without retrieval-score feature, full-passage and chunk-matched
  contexts). Training top-1 reached 39 % (vs 2 % random), but at
  inference extractive dropped to 2.0-2.7 %. Reverted to heuristic.
  Diagnosis: features computed on chunked retrieval at inference are
  distributionally different from features at training despite
  deliberate matching. See v2_report for weights + ablation. The v2
  scorer code remains in `LLM/components/attention/` but is NOT
  shipped in the repo.
- **Pseudo-relevance feedback (default on)**: hurt recall@1 by
  ~19 pp because top-3 is only 68 % correct; PRF amplifies the noise
  in the other 32 %.
- **Query expansion via embedding neighbors**: +/- 0.3 pp noise.
  Evolved embedding's nearest neighbors are co-occurrence-derived
  (topical), not synonyms, so expansion doesn't add true paraphrase
  coverage.

**Next-move ranking** (for the cron-firing AI that reads this):
1. Better span scorer. Current heuristic gets ~15 % conditional
   extraction rate when retrieval is correct. Need ~50 % to reach
   "usable" ~30 % containment. Candidate: cross-encoder-style span
   ranker trained against answer spans across MANY retrieval
   distributions. v2 attempts failed from distribution mismatch;
   fix requires training on same chunked retrieval the inference
   uses, with diverse retrieval confidences.
2. Retrieval ceiling. Recall@10 is 78 %; recall@1 is 53 %. A cheap
   re-ranker on top-20 would lift recall@1 significantly. Options:
   per-term-alignment score, BM25 over bigrams, or a tiny evolved
   re-rank head (gradient-free).
3. Multi-word entity handling. Many gold answers are 3-10 token
   spans ("1997 Treaty of Amsterdam", "Warsaw University of
   Technology building"). Extraction picks shorter spans because
   the heuristic penalizes length. A length prior conditioned on
   question type would help.

## v2 (2026-04-18, early)

**Headline:** retrieval recall@1 = 51.3 %, extractive answer
containment = 6.0 %, RAG generation answer containment = 7.3 %.

**What changed since v1:**
- Vocabulary extended 51,641 → 61,641 (10 K new tokens: years,
  numbers, common entity fragments). Embedding `hash_in` extended
  with random-Gaussian rows matching existing dimension statistics;
  all existing IDs preserved bit-identical.
- N-gram tables recounted on the extended-vocab punctuated stream.
- RAG paragraph index rebuilt with extended vocab so answer tokens
  like "2015" appear in candidate pools.

**What did NOT work:** span scorer v1 (single-passage training
distribution). See `RAG_V2_REPORT.md`.

## v1 (2026-04-18)

**Headline:** retrieval recall@1 = 48.7 %, RAG generation answer
containment = 3.7 %, extractive 4.0 %.

**First RAG release:**
- Hybrid BM25 + SIF-weighted dense retrieval on 20,958 paragraphs
  from SQuAD train + dev contexts (human-curated).
- Copy pool augmentation of rerank candidates with IDF-weighted
  passage tokens (the "copy mechanism").
- Extractive QA prototype with proximity / rarity / question-type
  heuristic.

## chatbot-v1 (earlier, pre-RAG)

Sentence-shaped output with natural-stop 75-85 %, but factual
accuracy effectively zero (0.3 % on SQuAD dev — only lexical-match
coincidences).

## gradient-free-clean (foundational)

Removed the ridge head and re-evolved CE attention against the
evolved embedding table. Full audit in `GRADIENT_AUDIT_REPORT.md`.
This is the release the no-gradient claim is built on.
