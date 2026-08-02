# Ablation study

code src@62fb6c18 · corpus seed@cab5c9bd · embed_model offline-concept-86d01716 · provider offline

> **These numbers describe the offline simulator, not a language model.**
> The `offline` provider is a deterministic rule-based stand-in that exists so the
> whole system runs in CI with no API key. It reproduces the *mechanism* behind each
> failure class faithfully, but its absolute rates are properties of the simulator.
> Re-run with `AUTOPSY_PROVIDER=live` for numbers about a real model.


60 queries × 14 ablations = 840 variant runs, each against a baseline run of the same query.

## Findings

| ablation | identical | equivalent | improved | degraded | now refuses | now answers | now wrong | now confident wrong | Δ tokens | Δ cost |
|---|---|---|---|---|---|---|---|---|---|---|
| `no_lexical+no_discriminator_guard` | 34 | 4 | 2 | 0 | 0 | 0 | 0 | 20 | -797 | +0.0000 |
| `no_rewrite` | 50 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | +10,303 | +0.0000 |
| `no_discriminator_guard` | 50 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | -868 | +0.0000 |
| `no_gate+no_discriminator_guard` | 35 | 13 | 2 | 0 | 0 | 0 | 0 | 10 | +48,187 | +0.0000 |
| `no_rerank` | 51 | 0 | 0 | 0 | 0 | 0 | 2 | 7 | -21,904 | +0.0000 |
| `no_lexical` | 44 | 4 | 2 | 0 | 0 | 0 | 10 | 0 | -408 | +0.0000 |
| `no_semantic` | 39 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | -21,485 | +0.0000 |
| `no_fusion` | 59 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | -642 | +0.0000 |
| `no_gate` | 45 | 13 | 2 | 0 | 0 | 0 | 0 | 0 | +48,381 | +0.0000 |
| `force_rerank` | 55 | 3 | 2 | 0 | 0 | 0 | 0 | 0 | +36,028 | +0.0000 |
| `no_expansion` | 50 | 9 | 1 | 0 | 0 | 0 | 0 | 0 | -954 | +0.0000 |
| `top_k_1` | 50 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | -12,089 | +0.0000 |
| `gate_on_fused` | 26 | 23 | 2 | 0 | 9 | 0 | 0 | 0 | +21,690 | +0.0000 |
| `no_lexical+no_gate` | 33 | 14 | 3 | 0 | 0 | 0 | 10 | 0 | +49,018 | +0.0000 |

**Read the last column first.** `now confident wrong` counts answers that were correct at baseline and became incorrect *with no hedge and no refusal* — a wrong answer a user has no way to distinguish from a right one. `now refuses` and `now wrong` are also regressions, but they are loud ones.

## What each ablation was predicted to do

| ablation | predicted failure | confident-wrong rate | verdict |
|---|---|---|---|
| `no_lexical+no_discriminator_guard` | the same retrieval failure as no_lexical, but silent instead of flagged | 33.3% | confirmed |
| `no_rewrite` | follow-up questions lose their referent | 16.7% | confirmed |
| `no_discriminator_guard` | near-miss identifiers answered confidently instead of flagged | 16.7% | confirmed |
| `no_gate+no_discriminator_guard` | both refusal mechanisms removed at once | 16.7% | confirmed |
| `no_rerank` | gray-zone queries degrade | 11.7% | confirmed |
| `no_lexical` | exact identifiers become unfindable | 0.0% | broke 10 answers, but all of them loudly |
| `no_semantic` | paraphrased queries stop matching | 0.0% | no measurable effect on this corpus |
| `no_fusion` | worse ranking, subtler than either leg alone | 0.0% | no measurable effect on this corpus |
| `no_gate` | hallucinated answer where a refusal was correct | 0.0% | changed the answer, not its correctness |
| `force_rerank` | cost rises, quality usually does not — a useful negative result | 0.0% | changed the answer, not its correctness |
| `no_expansion` | answers truncate mid-context | 0.0% | no measurable effect on this corpus |
| `top_k_1` | confident single-source wrongness | 0.0% | no measurable effect on this corpus |
| `gate_on_fused` | threshold becomes uninterpretable and drifts with top_k | 0.0% | broke 9 answers, but all of them loudly |
| `no_lexical+no_gate` | a retrieval failure and a missing refusal producing a hallucination together | 0.0% | broke 10 answers, but all of them loudly |

## Interactions — where the single-ablation numbers mislead

Each row below is a composite whose damage is more than the sum of its parts.

- **`no_lexical+no_discriminator_guard`**: 20 confidently-wrong answers (33.3%), against 10 for the parts measured separately (`no_lexical` 0 + `no_discriminator_guard` 10). The two mechanisms guard the same failures, so removing either one alone looks safe.

This is the practical warning in the whole study. A team that removes one of a redundant pair sees its eval stay green, concludes the component was dead weight, and ships. The exposure only appears when the second one goes — often in a later change, by a different person, with no failing test in between.

## A worked example

Ablation `no_lexical+no_discriminator_guard`, case `tenant_atlas:quiescing:id:4132`:

> `c_8e48a23ff4ef9570` (Errors > ATS-4132 — operation operation aborted) left the candidate list entirely (was fused rank 2). It had lexical rank 1 (score 4.79). Without the lexical leg, only the semantic signal remained. 2 other context chunks also dropped.

baseline trace `464XQBS41AT90FGCVHEEFV23PQ` · variant trace `09R1W00X3EGEJ4DMP52GQ84KRP`
