# GENREG LM

A language-model-shaped pipeline trained **without gradient descent
and without backpropagation**. Parameters were discovered by
evolutionary search; the n-gram statistics were tabulated directly
from a corpus.

It is a research artifact, not a useful chatbot. Expect short
English-shaped fragments — and a lot of honest failure modes along
the way.

## Be warned

If you come here expecting GPT-2, you will leave disappointed. A few
things to calibrate before trying:

- Output is **phrase-level, not sentence-level**. You will recognize
  English words and short phrases; you will not get coherent answers
  or multi-sentence paragraphs.
- Topic drifts every 10–20 tokens. The model has no real long-range
  memory.
- The learned attention stack **actively makes generation worse**
  when blended with the n-gram cascade (see below). Best outputs come
  from pure n-gram (`/ngram 1.0`), which is almost entirely corpus
  statistics, not learned parameters.
- There is no instruction-following, no dialog, no reasoning. Asking
  it a question will produce text that sometimes looks like an answer
  and usually isn't.
- Numbers, rare words, proper nouns, and punctuation are all weak
  spots.

What is interesting is that the pipeline runs at all: every learned
weight was produced without gradients, on a single GPU, in hours of
evolution rather than days of backprop.

## What you get

A tiny end-to-end LM that you can poke at interactively:

```
tokens
  -> Embedding             (51,641 x 768)
  -> Positional Encoding   (512 positions x 768)
  -> 2-layer Causal Attention (6 heads x 128 dim)
  -> Prediction Head       (SVD-compressed ridge, D=768 -> V=51,641)
  -> n-gram blend          (2/3/4-gram cascade)
  -> sample
```

All weights were evolved. The n-gram tables are pre-computed corpus
statistics — not learned, just counted.

## Install

```bash
pip install -r requirements.txt
```

PyTorch and NumPy only. Python 3.8+. A GPU is **not** required and
will in fact slow you down (see below).

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

Commands inside the REPL: `/temp <f>`, `/topk <n>`, `/len <n>`,
`/ngram <f>` (0 = pure attention head, 1 = pure n-gram; 1.0 gives
the best output), `/rep <f>`, `/quit`.

## Actual performance

### Next-token accuracy (honest)

![accuracy](assets/accuracy.png)

855 held-out positions from the WikiText-103 tail, character tokens
excluded from scoring.

| method | top-1 |
|---|---|
| Random (1 / V) | 0.002 % |
| Always predict "the" | 8.4 % |
| Bigram argmax | 16.8 % |
| Trigram argmax | 19.4 % |
| 4-gram cascade argmax | 20.9 % |
| **This model top-1** | **27.0 %** |
| **This model top-5** | **51.5 %** |

We beat the plain n-gram baselines by ~6 pp on top-1, but we do not
beat the best transformer LMs trained with backprop. Nothing in this
repo can match those numbers.

A large share of "this model"'s score comes from the n-gram cascade
itself. The evolved attention + head contribute something on top,
but the gap between "n-gram only" (20.9 %) and "full pipeline"
(27.0 %) is where all the evolution actually pays off.

### CPU vs GPU

![cpu vs gpu](assets/cpu_vs_gpu.png)

CPU is about 2× faster than a RTX 4080 for this workload. The model
is too small to amortize GPU kernel-launch and transfer overhead, and
the n-gram lookups are pure Python dict operations that a GPU
contributes nothing to. If you have a GPU, don't bother using it.

Run `python benchmark.py --both` to reproduce on your own hardware.

### Checkpoint breakdown

![checkpoint sizes](assets/checkpoint_sizes.png)

~89 MB total. No single file exceeds 25 MB so the repository ships
cleanly to GitHub without LFS. Notice that ~40 % of the "model" is
actually just the n-gram tables — the counted, not the learned part.

### Size context (with caveats)

![size comparison](assets/size_comparison.png)

Yes, this is smaller than GPT-2 on every axis. No, that does not
mean it is "GPT-2 compressed". GPT-2 can hold a conversation; this
cannot. The chart is here because people ask, not because it's a
fair comparison.

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

Fragments that feel like "might have appeared on Wikipedia somewhere"
are the best case. The attention-only mode (`/ngram 0`) is noticeably
worse — try it and you will see function-word soup. That degradation
is real and it tells you what the evolved components currently
contribute: they lift the n-gram baseline by a few points, but they
cannot carry generation on their own.

## What didn't work (and is still open)

We kept this honest list so the project isn't oversold:

- **Attention hurts generation.** Trained for masked-position
  reconstruction (cloze); not for autoregressive sampling. Blending
  in any amount of attention-head logits degrades output under
  sampling. Open problem: re-evolve with a distributional (KL) fitness
  instead of top-1 accuracy.
- **No FFN between attention and head.** GPT-2 has one; we don't.
  Attempted versions so far had diagnostic bugs.
- **Depth doesn't scale.** Stacking a third attention layer on the
  frozen pair didn't pay off under any of the fitness variants we
  tried (residual cloze, curriculum, multi-objective). Sequential
  frozen attention has a real ceiling around 2 layers.
- **No long-range coherence.** The n-gram window is 4 tokens; the
  attention context is 512 but its features don't sample well. Neither
  component tracks "what sentence is this about".
- **Vocabulary is word-level.** Out-of-vocabulary words become
  `<unk>`. No BPE.

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

- **Vocabulary.** 51,641 tokens. 4 specials + ~92 characters and
  punctuation + 51,566 WikiText-103 words appearing >= 5 times. At
  inference the character tokens (ids < 96) are hard-banned so the
  model stays word-level.

- **Embedding.** Each token id maps to a 768-dim vector through a
  fixed PPMI-SVD hash plus an evolved linear-plus-activation head
  with a residual skip. The hash is frozen; the head is evolved.

- **Positional encoding.** 512 positions. Sinusoidal initialization
  scaled per-dimension by evolved gains, then a per-dimension evolved
  activation. Preserves >98 % cosine with the bare embedding.

- **Attention.** 2 layers, 6 heads, 128 per head. Each layer has
  evolved Q/K/V/O projections plus a per-head evolved activation on
  the logits. Causal masking.

- **Prediction head.** SVD-64 approximation of a ridge-regression
  fit from attention output to 51,641 logits. Reconstructed as
  `U * S @ V^T` at load. ~13 MB on disk.

- **N-gram cascade.** 2-, 3-, and 4-gram counts from the training
  corpus, with short-n fallback. Not learned — tabulated.

- **Sampling.** Attention logits and scaled n-gram log-probs are
  blended by a user-set weight. Repetition penalty over the last
  20 tokens. Character tokens hard-banned. Temperature + top-k
  multinomial.

## How was this trained

Evolutionary search. Populations of candidate configurations were
scored on component-specific fitness functions, the best reproduced
with mutation. Full training scripts, fitness definitions, and
hyperparameter schedules are not included in this repository. The
n-gram tables are a direct count over the training corpus.

## License

MIT.
