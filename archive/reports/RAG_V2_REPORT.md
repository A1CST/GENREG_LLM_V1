# RAG v2 — Vocabulary Extension & Span-Scorer

**Status:** Vocabulary extended, RAG containment doubled over v1.
Chunked retrieval (v3, see addendum at bottom) then lifted extractive
accuracy above RAG generation. Span-scorer evolution attempted but
train/test distribution mismatch made the learned scorer underperform
the heuristic. Reverted to heuristic for ship.

## Final numbers (300 SQuAD dev questions, k=3)

| metric | v1 | v2 (this release) | lift |
|---|---|---|---|
| retrieval recall@1 | 0.487 | **0.513** | +2.6 pp |
| retrieval recall@3 | 0.617 | **0.650** | +3.3 pp |
| answer containment — RAG generation | 0.037 | **0.073** | **2×** |
| answer containment — extractive | 0.040 | **0.060** | +2.0 pp |
| F1 — RAG generation | 0.042 | 0.048 | +0.006 |
| F1 — extractive | 0.043 | **0.061** | +0.018 |
| no-RAG baseline | 0.003 | 0.003 | — |

RAG generation containment doubled (3.7% → 7.3%). That's **24× over
the no-RAG baseline** of 0.3%.

## What changed

### 1. Vocabulary extension (51,641 → 61,641 tokens)

Scanned WikiText-103 raw + SQuAD train + dev for strings that the
current tokenizer would map to `<unk>`. Kept the top 10,000 by
corpus frequency (with SQuAD answer spans boosted 10×).

Top additions (mostly numbers the old vocab didn't have):

```
'000'  → 75,849 occurrences
'10'   → 53,794
'12'   → 38,311
'20'   → 38,172
'2010' → 32,235
'2008' → 30,641
'2009' → 30,475
'2011' → 30,144
'2007' → 29,165
'2012' → 27,889
'u.s'  → 26,971
```

These are critical for "when" questions — previously 22 % of SQuAD
dev gold answers were untokenizable (contained only `<unk>`).

**Embedding was extended** by appending 10K random-Gaussian rows to
the `hash_in` table, matching the per-column mean/std of existing
rows. The evolved skip + encoder + output stages are V-independent,
so they apply unchanged.

Every existing vocab ID and its embedding is preserved bit-identical.
New tokens start with plausibly-scaled random embeddings that
distinguish them from each other and from `<unk>`, without needing to
re-evolve anything upstream.

### 2. Corpus regeneration

- **Stream** — retokenized WikiText-103 raw with extended vocab.
  `<unk>` rate dropped from 2.9 % → 1.3 % (halved).
- **N-grams** — recounted on the new stream (50 M-token subset as
  before). 27,099 distinct tokens can follow a period (up slightly
  from 27,099 — unchanged qualitatively; punctuation signal
  preserved).
- **RAG paragraph index** — rebuilt with extended-vocab tokenization
  so answer tokens like "2015" appear directly in candidate pools.

### 3. Span-scorer evolution (attempted, reverted)

Goal: replace the hand-tuned heuristic span scorer with an evolved
linear head over 15 engineered features (proximity, rarity, length,
semantic cos, question-type flags, dist-to-query-hit, BM25-style
overlap, etc.). Trained on 400 SQuAD train Q/A pairs, each with the
gold span + 50 random negative spans from the same passage. 400
generations, 8 seconds, tournament selection.

Training accuracy converged to 31.3 % top-1 — the scorer correctly
ranks the gold span first in ~1/3 of training Q/A pairs (with 50
negatives each, random baseline is 2 %, so 15× random).

**But at inference it underperformed the heuristic** (1.3 % vs 6.0 %
answer containment). Distribution mismatch:

- Training: spans from *one* passage (the gold). All features (prox,
  mean_dist, etc.) are normalized relative to the same passage's
  query-hit positions.
- Inference: spans from *k=3* passages. Feature values aren't
  comparable across passages — a span's `prox=0.5` in passage A
  means something different from `prox=0.5` in passage B.

The evolved scorer learned relative feature weights that only make
sense within a single passage. Deployed across passages it breaks.

**What the scorer did teach us** (learned weights, despite the
distribution issue):

```
semantic       -3.584   high semantic cos to query = BAD (span echoes
                         the query, isn't an answer)
inv_u          +3.212   peak-at-0.5 semantic preferred (adjacent-but-
                         not-identical to query topic)
mean_dist      +3.905   close to query-term hits in passage = good
numeric        +1.788   digit-containing spans are often answers
rarity         +1.498   rare tokens are informative
overlap        -0.816   query-word overlap is BAD
length         -0.667   slight penalty on long spans
```

These signs match intuition and are consistent with BM25-style
answer localization. The heuristic in `extract_answer` was tuned in
the same direction (high rarity, numeric bonus for "when",
overlap penalty) and so performs comparably or better.

**Proper fix for a learned scorer would require:**
1. Training with spans drawn across multiple retrievals (not just
   gold passage), so the scorer sees cross-passage comparisons.
2. Features that are passage-invariant (e.g., query-span BM25 score
   directly rather than passage-relative position features).
3. Per-passage ranking as a prior, cross-passage span ranking on top.

Out of scope for this session. Heuristic + extended vocab gives
2× on containment which is the headline result; span scorer is
deferred.

## Representative samples

Cases where RAG+extended-vocab finds a correct answer:

```
Q: Against what country was Kennedy promising superiority over?
Gold: Soviet Union
EXTR: superiority over the soviet union .                ✓

Q: What year was the Ford F-series introduced?
Gold: 1948
RAG: the ford f series was introduced in 1948 .           ✓  (year now in vocab)

Q: What language is the Latin word solidus?
Gold: Latin
RAG: latin .                                               ✓
```

Cases where retrieval finds the right passage but copy/extraction
fails:

```
Q: What is the KNLS responsible for?
Gold: establish, equip, manage and maintain national and public libraries in the country
RAG: for the creation of an international controversy in the united states ...
(too long a span to copy as-is; multi-clause answers are hard)

Q: In what book did Betty Meggers describe...
Gold: Amazonia: Man and Culture in a Counterfeit Paradise
EXTR: amazon rainforest , rather than being pristine wilderness .
(retrieved right passage but picked a different span)
```

Cases where retrieval fails entirely (still 48.7 % of the time):

```
Q: Where is the eiffel tower located?
rag-top1: Guinea-Bissau  (totally unrelated)
```

## Still-true limitations

1. **Retrieval ceiling 51.3% recall@1.** Improving this is the next
   biggest lever. Options: better BM25 tuning, learned re-ranker,
   query expansion.
2. **Copy mechanism is still attention-cosine-based.** The RAG
   generation works because passage content tokens are in the
   candidate pool with enough probability weight, but there's no
   explicit "decide when to copy vs generate" gate. Evolving one
   would need cross-passage training discipline to avoid the same
   distribution-mismatch trap that hit the span scorer.
3. **Multi-token answers.** SQuAD answers are often 3-10 content
   tokens. Extractive can match them but only if the exact subsequence
   is in the passage. Paraphrases don't count.
4. **Vocabulary gaps remain.** The extension covers numbers and
   common years but 1.3 % of corpus tokens are still `<unk>`
   (long-tail proper nouns, foreign words, special characters, etc.).

## Files changed / added

In `github_repo/`:
- `checkpoints/vocab.pkl` — now 61,641 entries (was 51,641)
- `checkpoints/embed.pkl` — hash_in extended with 10K new rows
- `checkpoints/ngrams_{bigram,trigram,fourgram,fivegram}.pkl` — recounted
- `checkpoints/rag_index.pkl` — rebuilt with extended-vocab tokenization
- `lib/model.py` — optional `span_scorer.pkl` loading + feature-based
  extract_answer (heuristic fallback retained)
- `RAG_V2_REPORT.md` — this file

In `LLM/data_raw/`:
- `extend_vocab.py` — vocab extension scanner
- `extend_embed.py` — hash_in extension
- `tokens_added.pkl` — metadata on the 10K new tokens

In `LLM/components/attention/`:
- `genreg_span_scorer.py` — evolved-span-scorer trainer (kept for
  future work; not shipped)
- `checkpoints_span_scorer/span_scorer_final.pkl` — trained scorer
  (documented but not deployed)

Gradient-free guarantee holds: BM25 is counting, SIF is closed-form
linear algebra over counted + evolved-embedding data, vocab extension
is counting + random init, span-scorer evolution is tournament
selection.

---

## Addendum — v3 chunked retrieval

After v2 shipped (RAG generation 7.3 % answer containment, extractive
6.0 %), the retrieval ceiling was the bottleneck. Recall@1 was 51 %
and the BM25 lexical signal was being diluted by whole-paragraph
token counts on paragraphs that averaged 117 content tokens.

**Fix: chunk paragraphs into 80-token windows with 20-token overlap.**
20,958 SQuAD paragraphs → 46,586 chunks. BM25 and dense SIF scoring
now operate at the chunk level; retrieval returns the best chunk per
parent paragraph, deduplicated to top-K parents.

**Storage:** chunk embeddings stored as int8-quantized vectors with a
single scale factor (x / 127 ≈ original unit-normed values). This
brings `rag_index.pkl` from 173 MB → 89 MB, under GitHub's 100 MB
file-size hard limit.

**Span search** moved to chunk scope — the extract_answer routine
searches spans within the matched chunk, not the full parent, which
is about 2× more focused.

### v3 results (300 SQuAD dev questions, k=3, seed 7)

| metric | v2 (paragraph) | **v3 (chunked)** | lift |
|---|---|---|---|
| retrieval recall@1 | 0.513 | **0.523** | +1.0 pp |
| retrieval recall@3 | 0.650 | **0.673** | +2.3 pp |
| answer containment — extractive | 0.060 | **0.077** | +1.7 pp |
| answer containment — RAG generation | 0.073 | 0.050 | −2.3 pp |
| F1 — extractive | 0.061 | **0.072** | +0.011 |
| F1 — RAG generation | 0.048 | 0.050 | +0.002 |

Chunking cleanly helps retrieval and extraction. RAG generation
regresses slightly because the generation copy pool benefits from
having more diverse passage tokens (full paragraph) rather than a
smaller chunk.

**Extractive is now the winning QA path** — 7.7 % vs RAG's 5.0 %.
This flip matters because extractive is deterministic (no sampling
noise), faster (no per-token attention forward), and cheaper to
inspect (you can see exactly which span was chosen). The final
pipeline ships extractive as the recommended factual-QA mode and
RAG generation as a secondary option for prompts where span-matching
doesn't apply.

### What else was tried (and left off)

- **Smaller chunks (60-token, 30-overlap)** produced 74,292 chunks.
  The float16 embedding matrix alone hit 109 MB, over GitHub's file
  limit even after int8 quant (64 MB for embeddings plus ~70 MB of
  token lists and texts put the pickle over 120 MB). 80/20 chunking
  was the sweet spot for storage vs concentration.
- **Chunk-tokens-only copy pool for generation** hurt RAG
  containment (3.3 %) more than full-parent (5.0 %). Generation
  prefers broader candidate coverage.
- **Geometric-mean passage base probability** reduced repetition in
  RAG output ("saxophones saxophones orchestration"-style) but also
  dropped RAG containment to 3.3 %. Reverted to max-n-gram base;
  extractive is the more reliable path anyway.

### Files changed (v3)

- `checkpoints/rag_index.pkl` — rebuilt as chunked_v1 format with
  46,586 chunks, int8 embeddings, chunk_parent map, chunk_token_lists
- `lib/model.py` — retrieve() supports both chunked_v1 and legacy
  paragraph formats; extract_answer() searches chunk scope when
  available
- `LLM/data_raw/build_rag_chunked.py` — chunked index builder
