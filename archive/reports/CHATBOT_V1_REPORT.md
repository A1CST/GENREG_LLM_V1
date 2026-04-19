# Chatbot v1 — Build Report

**Status:** Chatbot-shape output achieved. Still gradient-free, still
evolution-only. Facts wrong, structure right.

## What changed from the "clean gradient-free" release

| layer of the stack | before (clean v1) | after (chatbot v1) |
|---|---|---|
| embedding | evolved, PPMI-SVD init | unchanged — reused |
| positional encoding | evolved, sinusoidal init | unchanged — reused |
| attention L0, L1 | evolved, CE fitness | unchanged — reused |
| attention rerank L1, L2 | evolved, rerank fitness | unchanged — reused |
| **rerank L3 (register)** | none | **new, evolved on SQuAD Q/A stream** |
| **n-gram tables** | word-only, 19M-token stream | **punctuated, 50M-token stream with sentence boundaries** |
| **candidate filter** | CHAR_CUTOFF bans punct | **allow . , ? ! as candidates** |
| **stop condition** | max_tokens only | **break on . / ? / ! / `<eos>`, with min-tokens gate** |

No existing component was re-evolved. We kept the clean gradient-free
stack, added one new evolved layer on top, and swapped in
punctuation-aware n-gram tables. The gradient-free claim holds — every
input to every fitness is either an evolved organism, counted
statistics, or closed-form math. Full audit still lives in
`GRADIENT_AUDIT_REPORT.md`.

## Data added

- **WikiText-103 raw** (182 MB, HuggingFace `Salesforce/wikitext`
  `wikitext-103-raw-v1`). The previous corpus used in this project
  had punctuation stripped; the raw version preserves `. , ? ! ( )
  " ' ;` etc. Human-curated by Merity et al., no model-generated
  text.
- **SQuAD v1.1** (87,599 Q/A pairs, 2.8 M tokens). Human-written
  questions over Wikipedia passages with human-verified answer spans.
  Stanford crowd workers, no gradient-distilled content. Source:
  `rajpurkar.github.io/SQuAD-explorer`.

Both clean per your no-gradient policy. Both downloaded in this
session, no prior ML lineage.

## Diagnostic: what was broken before

Before the punctuation fix, the diagnostic on 30 chatbot-style
prompts × 2 seeds showed:

- **Natural stop rate: 0 / 60** — the model never naturally
  terminated. Every generation ran to `max_tokens`.
- Reason: the old corpus had zero period tokens, so the n-gram
  cascade had never learned "what comes after a sentence." There was
  no EOS signal anywhere in the pipeline.

This is the root cause the user had me surface: a **data-missing
problem**, not an architecture problem.

## Build steps

1. **Download raw data.** WikiText-103 raw + SQuAD v1.1 →
   `LLM/data_raw/`.
2. **Rebuild token stream with punctuation preserved.**
   `build_punct_stream.py` parses the parquet, keeps `. , ? ! ; : " '
   ( ) -`, emits a 201 M-token training stream. OOV rate drops to
   2.9 %. **4.1 M period tokens** and **29 K article-`<eos>`
   markers** now exist in the data. (The existing 51,641-token vocab
   already had slots for all these punctuation chars — they just
   hadn't been populated in the stream. Embedding is reusable.)
3. **Recount n-grams on the punctuated stream.** 25 % of the stream
   (50 M tokens) with periodic pruning to fit in RAM (the first
   attempt OOM'd at 30 GB). Produced: 51 K bigram contexts, 1.28 M
   trigrams, 785 K 4-grams, 651 K 5-grams. **27,099 distinct tokens
   can follow a period.** That is the EOS signal path that was missing.
4. **Build SQuAD training stream.** Each Q/A pair tokenized as
   `[question] ? \n [answer] . <eos>`. 2.8 M tokens across 87,599
   pairs, avg pair length 32.1 tokens.
5. **Evolve the register layer.** `genreg_attn_register.py` — same
   rerank fitness as the existing clean rerank layers, evaluated on
   SQuAD windows with the new punctuated n-grams. 400 generations,
   108 seconds. Hot-started from clean rerank L2. Fitness moved
   `rerank-lp` from −6.05 → −5.99 and argmax-hit from 0.149 → 0.148
   on the SQuAD train stream; the absolute numbers are low because
   SQuAD answer tokens contain proper nouns the wiki n-gram cascade
   rarely proposes.
6. **Unban punctuation in the candidate pool.** `lib/model.py`:
   `_ngram_candidates` now allows `{. , ? !}` through its
   `CHAR_CUTOFF` filter.
7. **Add natural-stop trigger in `generate_rerank`.** Break on
   `. / ? / ! / <eos>` after a minimum-token gate (so the model
   can't collapse to an immediate ".").

## Chatbot-shape benchmark results

Diagnostic on 30 chatbot-style prompts × 2 seeds, max 40 tokens:

| metric | pre-punct (old stream) | chatbot v1 |
|---|---|---|
| natural stop rate | 0 / 60 (0 %) | **51 / 60 (85 %)** |
| answer-shape rate | 42 % | **50 %** |
| ran to max_tokens | 60 / 60 (100 %) | **9 / 60 (15 %)** |
| prompt→response topic cosine | 0.82 | 0.80 |
| response self-consistency | 0.86 | 0.80 |

And on the short benchmark subset (10 prompts × 2 seeds):

```
prompts x seeds: 10 x 2 = 20
natural stop rate: 15/20  (75%)
answer-shaped output: 12/20  (60%)
mean response length: 20.4 tokens
mean prompt->response topic cosine: 0.801
```

Representative outputs (chatbot v1, temp 0.7, α 5, top-k 30):

```
Q: napoleon bonaparte was born in
A: the city of brighton and hove .

Q: marie curie is known for
A: its high manoeuvrability and ability to perform tight turns .

Q: an atom is composed of
A: the city of san diego .

Q: democracy is a system of
A: the world in the early days of the war .

Q: albert einstein was a
A: awarded the distinguished service order , the city .

Q: a telescope is a device that
A: cost of the first in the world to have been named by the japanese
the following month , the battalion was ...

Q: the reason the sky is blue is
A: the album . it also includes the selection of the best in the
country .
```

## Honest assessment

**What works:**
- Output is sentence-shaped, not article-fragment-shaped.
- Responses terminate naturally ~85 % of the time.
- Average response length is ~20 tokens — chatbot scale, not article
  scale.
- Topic cosine between prompt and response stays ~0.80; the model is
  on-topic at the embedding level.
- The ridge head is still gone. No backprop anywhere. Pipeline is
  100 % gradient-free.
- Throughput went from 67 tok/s → 350 tok/s on CUDA because
  generations now terminate early.

**What does NOT work:**
- **Factual accuracy is zero.** The model treats "napoleon bonaparte
  was born in" as a prompt whose next words should sound
  Wikipedia-like, not as a factual query with a correct answer. It
  writes "brighton and hove" where "corsica" is correct.
- **Content is Wikipedia-style word-blending**, not knowledge
  retrieval. There is no mechanism to retrieve or condition on the
  "right Wikipedia article" for a given query.
- Answer-shape rate is 60 %, so 40 % of responses start with a
  non-answer word like a comma or a generic connector.
- Open-domain chatbot (arbitrary dialog, multi-turn, instruction
  following) is nowhere close.

**What's next, if you want to push this further:**

1. **Grounded retrieval.** For each query, find the nearest Wikipedia
   paragraph in embedding space and condition generation on it. This
   is classic RAG, gradient-free feasible.
2. **Re-evolve the CE and rerank layers on the punctuated stream.**
   Everything upstream of the register layer was evolved on the
   punctuation-stripped stream. Re-evolving should improve feature
   quality on sentence-shaped text. Maybe 1 hour.
3. **Scale the SQuAD training.** The register layer used only 20
   windows for fitness. Scaling to hundreds should sharpen
   answer-register shift without overfitting.
4. **Evolve an EOS-timing organism.** The current stop rule is
   hard-coded (break on `.`). Letting evolution control stop
   probability per position would give finer length control.

## File layout changes

**New in `github_repo/`:**
- `CHATBOT_V1_REPORT.md` — this file
- `diagnostic_chatbot.py` — chatbot-shape diagnostic harness
- `checkpoints/attn_rerank_L3.pkl` — the register layer

**Replaced:**
- `checkpoints/ngrams_{bigram,trigram,fourgram,fivegram}.pkl` — now
  count punctuation as valid tokens

**Still clean from the gradient-free release:**
- `checkpoints/{embed,posenc,vocab,attn_L0,attn_L1,attn_rerank_L1,attn_rerank_L2}.pkl`
- `lib/encoder.py`
- `GRADIENT_AUDIT_REPORT.md`

**Training scripts (in `LLM/components/attention/`, not shipped):**
- `genreg_attn_register.py` — register layer trainer
- `LLM/data_raw/build_punct_stream.py` — punctuated-stream builder
- `LLM/data_raw/build_ngrams.py` — n-gram recount
- `LLM/data_raw/build_squad_stream.py` — SQuAD → token stream
