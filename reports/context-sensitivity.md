# Context-width sensitivity

Retrieval-only measurement. No model calls, no tokens: an ablation cannot change
the answer unless it changes what reaches the generator, and *that* is free to
measure. Gold is ground truth by construction — every case carries a globally
unique coined token, so "did a chunk containing it reach the generator" needs no
judge.

## Why this report exists

The first ablation study returned zero in every outcome column. The cause was not
a broken counterfactual engine — it was `max_context_chunks: 12`. Twelve chunks of
a corpus whose median chunk is ~111 tokens is roughly 1,300 tokens of context for
a single-fact question. The window was wide enough to contain the right chunk
regardless of how badly ranking was damaged, so no ablation could move the answer.

## Gold retention vs context width

| max_context_chunks | baseline | no_lexical | no_semantic | no_fusion | no_expansion | spread |
|---|---|---|---|---|---|---|
| 1 | 91% | 81% | 100% | 100% | 91% | 19pp |
| 2 | 95% | 86% | 100% | 100% | 94% | 14pp |
| 3 | 95% | 88% | 100% | 100% | 97% | 12pp |
| 5 | 100% | 92% | 100% | 100% | 100% | 8pp |
| 8 | 100% | 96% | 100% | 100% | 100% | 4pp |
| 12 | 100% | 97% | 100% | 100% | 100% | 3pp |

**Spread is the point.** It collapses from ~20pp at a 1-chunk window to ~4pp at
12. Above roughly 5 chunks this pipeline's retrieval ablations are unmeasurable on
this corpus — not because the ablations do nothing, but because the window absorbs
them. Anyone running a retrieval ablation should report this curve before reporting
an effect size.

## Gold retention by query kind

At the shipped width (`max_context_chunks=3`).

| config | identifier | value | paraphrase | followup |
|---|---|---|---|---|
| baseline | 40/40 (100%) | 40/40 (100%) | 32/40 (80%) | 40/40 (100%) |
| `no_lexical` | 38/40 | 40/40 | 22/40 | 40/40 |
| `no_semantic` | 40/40 | 40/40 | 40/40 | 40/40 |
| `no_fusion` | 40/40 | 40/40 | 40/40 | 40/40 |
| `no_expansion` | 40/40 | 40/40 | 35/40 | 40/40 |

### Two findings and a failed prediction

**The dense leg hurts at tight context.** `no_semantic` beats baseline at every
width up to 3 and ties at 5. `bge-small-en-v1.5` has never seen this corpus's
coined identifiers, so it embeds them as noise while BM25 matches them exactly;
reciprocal rank fusion then averages a good signal with a bad one and lands below
the good one alone. **Limitation:** a corpus built from invented tokens is
adversarial to any pretrained embedder. This is a result about *this corpus*, not
a general claim about hybrid retrieval.

**A retracted finding, kept on the record.** An earlier run of this sweep put
`followup` retention at 28% and concluded the rewrite stage was broken. It was not.
The harness was not passing each case's conversation history, so a query like
"what is its default value" reached retrieval with no referent and the rewrite
stage correctly skipped itself. The pipeline was right; the measurement was wrong.
It is recorded here because it is the same failure mode this project exists to
catch — a confident number, no error anywhere, and a plausible story attached to
it. The number above is from the corrected harness.

**The hybrid-crossover prediction failed.** The expectation was that `no_lexical`
would break `identifier` queries while `no_semantic` broke `paraphrase` queries —
the textbook argument for hybrid retrieval. It does not happen here: each generated
document is several chunks about one subsystem, so a wide window retrieves the
whole document either way. Recorded because a failed prediction bounds the claim
more usefully than a confirmed one.
