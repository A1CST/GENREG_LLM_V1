# GENREG LM

A language-model-shaped pipeline trained without gradient descent and
without backpropagation. Parameters were discovered by evolutionary
search. The n-gram statistics were counted directly from a corpus.

It is a research artifact, not a chatbot. Expect short English-shaped
fragments.

## Calibrate your expectations

If you came here expecting GPT-2, you are in the wrong repo.

- Output is phrase-level, not sentence-level. You will recognize
  English words and short phrases. You will not get coherent answers
  or multi-sentence paragraphs.
- Topic drifts every 10 to 20 tokens. The model has no long-range
  memory.
- The evolved attention stack actively makes generation worse when
  blended with the n-gram cascade. Best outputs come from pure n-gram
  (`/ngram 1.0`), which is mostly corpus statistics, not learned
  parameters.
- There is no instruction-following, no dialog, no reasoning. Asking
  it a question will produce text that sometimes looks like an answer
  and usually isn't.
- Numbers, rare words, proper nouns, and punctuation are all weak
  spots.

If that still sounds interesting: the pipeline runs at all, without a
gradient, on a single GPU, in hours of evolution instead of days of
backprop.

## Where this is going

This is a first working end-to-end build, not the final one. The
interesting result is that the evolutionary substrate is now
well-enough understood that I can pick a fitness signal, know roughly
how the population will move under it, and get a component that does
approximately the thing I asked for. The next iteration is expected
to be larger and better, not by brute-forcing more compute but by
applying what the last round of experiments taught about which
landscape pressures actually produce useful language behavior.

Treat this release as version 0.

## The pipeline

```
tokens
  -> Embedding             (51,641 x 768)
  -> Positional Encoding   (512 positions x 768)
  -> 2-layer Causal Attention (6 heads x 128 dim)
  -> Prediction Head       (SVD-compressed ridge, D=768 -> V=51,641)
  -> n-gram blend          (2/3/4-gram cascade)
  -> sample
```

Every learned weight was evolved. The n-gram tables are pre-computed
corpus statistics, not learned parameters.

## Install

```bash
pip install -r requirements.txt
```

PyTorch and NumPy. Python 3.8+. A GPU is not required and will slow
you down (see CPU vs GPU below).

## Quick start

Interactive:

```bash
python inference.py
```

One-shot:

```bash
python inference.py --prompt "the battle of waterloo was"
```

Benchmark:

```bash
python benchmark.py            # CPU
python benchmark.py --both     # CPU and GPU side by side
```

REPL commands: `/temp <f>`, `/topk <n>`, `/len <n>`, `/ngram <f>`
(0 for pure attention head, 1 for pure n-gram; 1.0 is best), `/rep <f>`,
`/quit`.

## Next-token accuracy

![accuracy](assets/accuracy.png)

855 held-out positions from the WikiText-103 tail. Character tokens
excluded from scoring.

| method | top-1 |
|---|---|
| Random (1 / V) | 0.002 % |
| Always predict "the" | 8.4 % |
| Bigram argmax | 16.8 % |
| Trigram argmax | 19.4 % |
| 4-gram cascade argmax | 20.9 % |
| This model top-1 | 27.0 % |
| This model top-5 | 51.5 % |

We beat the plain n-gram baselines by roughly 6 points on top-1. We
do not beat transformer LMs. A large share of the 27 % comes from the
n-gram cascade itself. The evolved attention and prediction head lift
things by another few points on top, which is the part that actually
required evolution.

## CPU vs GPU

![cpu vs gpu](assets/cpu_vs_gpu.png)

CPU is about 2x faster than a RTX 4080 for this workload. The model
is too small to amortize GPU kernel-launch and transfer overhead, and
the n-gram lookups are Python dict operations that a GPU contributes
nothing to. If you have a GPU, don't bother. Run
`python benchmark.py --both` to reproduce.

## Checkpoint breakdown

![checkpoint sizes](assets/checkpoint_sizes.png)

About 89 MB total. No single file exceeds 25 MB, so the repo ships
without Git LFS. Roughly 40 % of "the model" is actually the n-gram
tables (counted, not learned).

## Size context, with the obvious caveat

![size comparison](assets/size_comparison.png)

Yes, this is smaller than GPT-2 on every axis. No, that does not
mean this is "GPT-2 compressed." GPT-2 can hold a conversation, this
cannot. The chart is here because it's the first question people
ask, not because it's a fair comparison.

## What it actually outputs

Best mode (pure n-gram, temp 0.4, top-k 20, char-ban on):

```
> the battle of waterloo was
< less important agent for an agent the game the lowest level the
  aircraft is in lowest walks allowed at intervals over for military
  band over the game

> the film was directed by
< anthony freud aa mounts these events were incorporated elements
  incorporated his wife anne hathaway who later recalled the words
  is short the user and takes its title one review

> she was born in
< on an american on to lose interest is often confused the number
  and severity it often is for and won six world in and his complaint
  from number eight being
```

Fragments that could plausibly have appeared on Wikipedia are the
best case. Attention-only mode (`/ngram 0`) is noticeably worse. Try
it if you want to see function-word soup. That degradation tells you
what the evolved components currently contribute: a small lift above
the n-gram baseline. They cannot carry generation on their own yet.

## Known failure modes

- Attention hurts generation when blended in. It was evolved against
  masked-position reconstruction, not against next-token sampling.
  The fix is to re-evolve it with a distributional fitness instead of
  top-1 accuracy. That is a later iteration.
- No FFN between attention and head. GPT-2 has one. This doesn't.
- Depth does not scale. Stacking a third attention layer on the frozen
  pair did not help under any fitness variant tried so far. The issue
  is a property of sequentially freezing layers, not of attention
  itself.
- No long-range coherence. N-gram window is 4 tokens. Attention
  context is 512, but its features don't sample well.
- Word-level tokenizer. Out-of-vocabulary becomes `<unk>`. No BPE.

These are the open problems that the next iteration is aimed at.

## Repository layout

```
github_repo/
├── README.md
├── requirements.txt
├── inference.py           REPL + one-shot prompt
├── benchmark.py           load / throughput / accuracy / samples / --both
├── assets/                comparison charts
├── lib/
│   ├── encoder.py         activation catalog
│   └── model.py           frozen components + GenregLM wrapper
└── checkpoints/
    ├── vocab.pkl             token <-> id (V = 51,641)
    ├── embed.pkl             token embedding
    ├── posenc.pkl            positional encoding
    ├── attn_L0.pkl           causal attention layer 0
    ├── attn_L1.pkl           causal attention layer 1
    ├── predhead.pkl          prediction head (SVD-compressed)
    ├── ngrams_bigram.pkl     2-gram table
    ├── ngrams_trigram.pkl    3-gram table
    └── ngrams_fourgram.pkl   4-gram table
```

## Architecture notes

- **Vocabulary.** 51,641 tokens. 4 specials, ~92 characters and
  punctuation, 51,566 WikiText-103 words appearing >= 5 times. At
  inference the character tokens (ids < 96) are banned so the model
  stays word-level.
- **Embedding.** Each token id maps to a 768-dim vector through a
  fixed PPMI-SVD hash plus an evolved head with a residual skip.
- **Positional encoding.** 512 positions. Sinusoidal init scaled per
  dimension by evolved gains, with a per-dimension evolved activation.
  Preserves > 98 % cosine with the bare embedding.
- **Attention.** 2 layers, 6 heads, 128 per head. Evolved Q/K/V/O and
  per-head evolved activation on the logits. Causal masking.
- **Prediction head.** SVD-64 approximation of a ridge fit from the
  last attention output to 51,641 logits. About 13 MB on disk.
- **N-gram cascade.** 2-, 3-, 4-gram counts from the training corpus,
  with short-n fallback. Counted, not learned.
- **Sampling.** Attention logits and scaled n-gram log-probs are
  blended. Repetition penalty over the last 20 tokens. Character
  tokens are hard-banned. Temperature + top-k multinomial.

## How was this trained

Evolutionary search. Populations of candidate configurations were
scored on component-specific fitness functions. The best reproduced
with mutation. Training scripts, fitness definitions, and
hyperparameter schedules are not included in this repository.

## License

MIT.
