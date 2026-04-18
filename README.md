# GENREG LM

A chatbot-shaped language model trained **without gradient descent,
without backpropagation, and without closed-form regression** against
supervised targets. Every learned parameter was produced by
tournament-selection neuroevolution. The n-gram statistics were
counted directly from the corpus.

Still a research artifact. **Output is now sentence-shaped and
terminates naturally 75 – 85 % of the time** (up from 0 % in the prior
release), but factual accuracy is zero. See `CHATBOT_V1_REPORT.md`
for the build path and honest numbers.

## The no-gradient claim, precisely

No component of this repository was produced by:

- stochastic gradient descent (SGD, Adam, etc.),
- backpropagation through a differentiable graph,
- or closed-form regression fitted against a supervised MSE
  objective (including ridge/OLS — the closed-form solution *is*
  what gradient descent converges to).

The pipeline is:

```
tokens
  -> Embedding               (PPMI-SVD init + evolved skip + encoder)
  -> Positional Encoding     (sinusoidal init + evolved gains/activation)
  -> Attention L0            (evolved, CE fitness via evolved embedding table)
  -> Attention L1            (evolved, on frozen L0)
  -> Rerank-evolved L1       (evolved, n-gram argmax fitness, wiki)
  -> Rerank-evolved L2       (evolved, same fitness, richer features)
  -> Register-evolved L3     (evolved, rerank fitness on SQuAD Q/A stream)
  -> N-gram cascade proposes K candidates  (2/3/4/5-gram, punctuated)
  -> Score = alpha * cos(attn_feat, cand_emb) + log ngram_prob
  -> Sample; break on . / ? / ! / <eos>
```

Closed-form math (SVD of the PPMI co-occurrence matrix, sinusoidal
position formulas) is allowed — those are unsupervised
decompositions over counted statistics, not fitted predictors.

A full audit of where a gradient-adjacent artifact had previously
leaked in (an SVD-compressed ridge head used during the CE fitness
stage of attention evolution) and how it was excised is in
`GRADIENT_AUDIT_REPORT.md`.

The core principle the audit surfaced: **"gradient-free" is a
property of the whole information supply chain, not of the
optimizer.** A tournament-selection training loop with no `.backward()`
can still be gradient-contaminated if its fitness function reads
from a supervised-fitted matrix. Auditing the training mechanism is
necessary but not sufficient — the fitness *inputs* have to be
audited too, and hot-start checkpoints and frozen base stacks inherit
contamination transitively.

## Calibrate your expectations

If you came here expecting GPT-2, you are in the wrong repo.

- Output is phrase-level, not sentence-level. You will recognize
  English words and short phrases. You will not get coherent answers
  or multi-sentence paragraphs.
- Topic drifts every 10–20 tokens. There's no long-range memory.
- No instruction-following, no dialog, no reasoning. A question gets
  text that sometimes looks like an answer and usually isn't.
- Numbers, rare words, proper nouns, and punctuation are weak spots.

## Install

```bash
pip install -r requirements.txt
```

PyTorch and NumPy. Python 3.8+. A GPU is not required.

## Quick start

Interactive:

```bash
python inference.py
```

One-shot:

```bash
python inference.py --prompt "the battle of waterloo was"
```

Pure n-gram (attention turned off, for comparison):

```bash
python inference.py --prompt "the battle of waterloo was" --alpha 0
```

Benchmark:

```bash
python benchmark.py
python benchmark.py --both     # CPU and GPU side by side
```

REPL commands: `/temp <f>`, `/topk <n>`, `/len <n>`, `/alpha <f>`
(0 = pure n-gram, 5 = default), `/rep <f>`, `/topp <f>`, `/quit`.

## Next-token accuracy

1,375 held-out word positions from WikiText-103. Character targets
excluded from scoring.

| method | top-1 |
|---|---|
| Random (1 / V) | 0.002 % |
| Always predict "the" | 8.4 % |
| Bigram argmax | 16.8 % |
| Trigram argmax | 19.4 % |
| 4-gram cascade argmax | 20.9 % |
| **This model top-1 (rerank path)** | **24.5 %** |
| **This model top-5 (rerank path)** | **51.8 %** |

The candidate pool from the n-gram cascade contains the true token
74% of the time on this split, so the theoretical ceiling for any
rerank-style approach on the current pool is 74% — we're at 33.1%
**conditioned on the true token being in the pool**.

## Generation diversity

8 held-out prompts × 2 seeds, 25 tokens each. `unique-frac` is the
fraction of distinct token IDs in each generation; `max-repeat` is
the longest contiguous exact-token repeat run.

| mode | unique-frac | max-repeat |
|---|---|---|
| Pure n-gram (alpha=0) | 0.86 | 1.00 |
| **Rerank, 5-layer stack (alpha=5)** | **0.80** | **1.00** |

The evolved attention stack narrows the candidate distribution toward
semantically-appropriate choices without introducing repetition.

## Chatbot shape

10 chatbot-style prompts × 2 seeds, 40 tokens max, default sampling:

| metric | chatbot v1 |
|---|---|
| natural stop rate (`. / ? / ! / <eos>`) | **75 %** |
| answer-shape rate (first word is answer-y) | **60 %** |
| mean response length | 20.4 tokens |
| mean prompt→response topic cosine | 0.80 |

Example outputs (temp 0.7, α 5):

```
Q: napoleon bonaparte was born in
A: the city of brighton and hove .

Q: marie curie is known for
A: its high manoeuvrability and ability to perform tight turns .

Q: democracy is a system of
A: the world in the early days of the war .
```

Sentence-shaped, self-terminating, wrong on facts. Wikipedia text-
shape is what it learned; it has never been given knowledge-retrieval
machinery. See `CHATBOT_V1_REPORT.md` for the build and honest
assessment.

## Model size

```
embedding parameters:            6,940,032
positional encoding:               397,824
attention CE layers:             4,718,664  (2 layers)
attention rerank layers:         7,077,996  (3 layers: 2 wiki + 1 register)
-------------------------------------------
ALL LEARNED PARAMETERS:         19,134,516
```

Plus counted (not learned) n-gram tables: 51 K bigrams, 111 K
trigrams, 283 K fourgrams, 174 K fivegrams. Total on-disk footprint:
114 MB. No single file exceeds 25 MB, so the repo ships without Git
LFS.

## CPU vs GPU

![cpu vs gpu](assets/cpu_vs_gpu.png)

The model is small enough that kernel-launch overhead dominates on
GPU for short generations. Run `python benchmark.py --both` to
reproduce on your hardware.

## Embedding space

![embedding 3D](assets/embedding_3d.png)

PCA of the evolved 768-dim embedding for the top 2,000 most common
word tokens, projected to 3D. Grey cloud is the full set; coloured
groups are hand-picked semantic clusters (countries, verbs, numbers,
colors, royalty, science). PC1–PC3 together explain ~8% of the
variance, so what you're seeing is a low-dimensional shadow of a
high-dimensional space. Additional angles in
`assets/embedding_3d_front.png` and `assets/embedding_3d_side.png`.

Numbers and colors form their own pockets, royalty words sit near
each other, countries clump on one side. None of this was supervised.
The embedding was evolved against a PPMI co-occurrence objective and
the grouping fell out as a side effect.

## What it actually outputs

Wiki-style continuation:

```
> the king sat on the
< the king sat on the bench . the first of these was the first time
  the following year .

> she was born in
< she was born in the adelaide rams in the united states and the
  soviet union .

> the film was directed by
< the film was directed by barry goldberg was the most successful of
  the two is now the site of the roman empire .
```

Chatbot-shape (prompted with questions and definitions):

```
> napoleon bonaparte was born in
< the city of brighton and hove .

> marie curie is known for
< its high manoeuvrability and ability to perform tight turns .

> democracy is a system of
< the world in the early days of the war .

> an atom is composed of
< the city of san diego .
```

Short, self-terminating, sentence-shaped — and almost always wrong on
facts. The pipeline learned Wikipedia-style text-shape, not knowledge
retrieval. Try `--alpha 0` (pure n-gram) to compare.

## Known failure modes

- **No FFN between attention and the rerank step.** The attention
  output goes straight to cosine-vs-candidate.
- **Rerank candidate set is bounded.** K=30 by default. If the true
  next token isn't proposed by the n-gram cascade (26% of held-out
  positions), it cannot be picked. Raise with `/topk`.
- **No long-range coherence.** N-gram window is 4 tokens. Attention
  context is 512, but the rerank fitness only scored next-token
  choice, so long-range structure is not under any selection
  pressure.
- **Word-level tokenizer.** Out-of-vocabulary becomes `<unk>`. No
  BPE.
- **Small training eval set.** The rerank fitness is computed on 6
  sequences (~660 word-target positions). A subsequent L3 layer kept
  improving that fitness but regressed generation diversity — the
  eval set is small enough to overfit. L3 was not shipped; the
  L2-capped 4-layer stack is the release.

## Repository layout

```
github_repo/
├── README.md
├── GRADIENT_AUDIT_REPORT.md    (how the no-gradient claim was verified)
├── requirements.txt
├── inference.py                 REPL + one-shot prompt
├── benchmark.py                 load / throughput / accuracy / diversity / samples
├── assets/                      comparison charts
├── lib/
│   ├── encoder.py               activation catalog
│   └── model.py                 frozen components + GenregLM wrapper
└── checkpoints/
    ├── vocab.pkl                token <-> id (V = 51,641)
    ├── embed.pkl                token embedding
    ├── posenc.pkl               positional encoding
    ├── attn_L0.pkl              evolved causal attention L0
    ├── attn_L1.pkl              evolved causal attention L1
    ├── attn_rerank_L1.pkl       evolved wiki-rerank layer 1
    ├── attn_rerank_L2.pkl       evolved wiki-rerank layer 2
    ├── attn_rerank_L3.pkl       evolved SQuAD-register layer
    ├── ngrams_bigram.pkl        2-gram table (punctuated)
    ├── ngrams_trigram.pkl       3-gram table (punctuated)
    ├── ngrams_fourgram.pkl      4-gram table (punctuated)
    ├── ngrams_fivegram.pkl      5-gram table (punctuated)
    └── heldout_sample.pkl       held-out token windows for benchmark
```

## Architecture notes

- **Vocabulary.** 51,641 tokens. 4 specials, ~92 characters and
  punctuation, 51,566 WikiText-103 words appearing ≥ 5 times. Char
  tokens (ids < 96) are banned at inference so the model stays
  word-level.
- **Embedding.** Each token id maps to a 768-dim vector through a
  fixed PPMI-SVD hash (closed-form, not learned against targets)
  plus an evolved head with a residual skip.
- **Positional encoding.** 512 positions. Sinusoidal formulas scaled
  per dimension by evolved gains, with a per-dimension evolved
  activation.
- **Attention.** 5 layers total. Layers 0 and 1 evolved against
  cross-entropy fitness computed via `features @ emb_table.T` — the
  projection is the evolved embedding table itself, so the CE
  landscape is gradient-free. Layers 2 and 3 are wiki-rerank-evolved:
  for each position, the n-gram cascade proposes K candidates, and
  fitness is the log-probability of the true token under
  `softmax(alpha * cos(attn_out, cand_emb) + log ngram_prob)`. Layer
  4 is the SQuAD-register layer: same rerank fitness, but evaluated on
  windows of concatenated SQuAD Q/A pairs so it learns the
  question → answer register shift. 6 heads × 128 dim, causal, evolved
  Q/K/V/O and per-head logit activation.
- **N-gram cascade.** 2-, 3-, 4-, 5-gram counts from the training
  corpus, with short-n fallback. Punctuation tokens (`. , ? !`) are
  counted as valid successors, so the cascade can propose sentence
  boundaries. The data source is WikiText-103 raw (punctuation
  preserved), 50 M-token slice.
- **Rerank sampling.** Top-K n-gram candidates scored by
  `alpha * cos(attn_last, cand_emb) + log ngram_prob`. Temperature,
  top-k, optional top-p, repetition penalty over the last 15 tokens.
  Non-sentence-punctuation chars hard-banned; `. , ? !` allowed as
  generation candidates. Generation breaks on `. / ? / ! / <eos>`
  after a minimum-token gate. `alpha=0` degenerates to pure n-gram
  sampling over the candidate pool.

## How was this trained

Evolutionary search. Populations of candidate configurations were
scored on component-specific fitness functions. The best reproduced
with mutation. Tournament selection is literal — no gradients, no
loss, no optimizer. Training scripts and fitness definitions are in
`LLM/components/` in the source tree but not shipped in this
artifact; the audit report describes the relevant flow.

## License

MIT.
