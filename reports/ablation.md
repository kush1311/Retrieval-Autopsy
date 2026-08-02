# Ablation study

code src@7882e00f · corpus seed@cab5c9bd · embed_model BAAI/bge-small-en-v1.5 · provider groq

> **Provenance, and why this file is not regenerated with the others.**
>
> This is the **real-model** run: `openai/gpt-oss-120b` on Groq's free tier with
> `bge-small-en-v1.5` embeddings. It is **not reproducible in a checkout without a
> `GROQ_API_KEY`**, so `make reports` cannot rebuild it and it therefore carries an older
> `code` stamp than the offline reports beside it. That difference is provenance, not
> drift. The offline equivalent is
> [`ablation-simulated.md`](ablation-simulated.md), which writes to its own filename
> precisely so a simulator sweep can never overwrite a real-model one.
>
> **Two known defects in this artifact, both fixed in code after it was written:**
>
> 1. The worked example below reads *"fell from fused rank 1 to rank 1"*. An unchanged
>    rank is not a fall — `explain()` now distinguishes held / fell / rose / left the
>    candidate list (`tests/test_explain_rank_movement.py`). Regenerating this file with a
>    key will produce the corrected wording.
> 2. Only the three core ablations were run. The three composites the spec calls for have
>    still never been run against a real model.

25 queries × 3 ablations = 75 variant runs, each against a baseline run of the same query.

## Findings

| ablation | identical | equivalent | improved | degraded | now refuses | now answers | now wrong | now confident wrong | Δ tokens | Δ cost |
|---|---|---|---|---|---|---|---|---|---|---|
| `no_gate` | 11 | 10 | 1 | 0 | 0 | 0 | 0 | 3 | +34,567 | +0.0000 |
| `no_lexical` | 10 | 13 | 0 | 0 | 1 | 0 | 0 | 1 | +3,942 | +0.0000 |
| `no_rerank` | 18 | 6 | 1 | 0 | 0 | 0 | 0 | 0 | -199 | +0.0000 |

**Read the last column first.** `now confident wrong` counts answers that were correct at baseline and became incorrect *with no hedge and no refusal* — a wrong answer a user has no way to distinguish from a right one. `now refuses` and `now wrong` are also regressions, but they are loud ones.

## What each ablation was predicted to do

| ablation | predicted failure | confident-wrong rate | verdict |
|---|---|---|---|
| `no_gate` | hallucinated answer where a refusal was correct | 12.0% | confirmed |
| `no_lexical` | exact identifiers become unfindable | 4.0% | confirmed |
| `no_rerank` | gray-zone queries degrade | 0.0% | changed the answer, not its correctness |

## A worked example

Ablation `no_gate`, case `tenant_atlas:checkpointing:para`:

> `c_29870e4099cef1c0` (Atlas checkpointing > Overview) fell from fused rank 1 to rank 1 and dropped out of context. It had lexical rank 1 (score 16.15) and semantic rank 3 (score 0.78). 1 other context chunks also dropped.

baseline trace `3GK7WE75Y6EPJ21K7N8GDQ3D98` · variant trace `4SYWP07C1HJYSHQX2RJHJY80T7`
