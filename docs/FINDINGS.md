# Findings

What this pipeline was measured to do, and where the measurements stop being load bearing. Moved out of the README so the front page stays a front page; nothing here has been edited.

See also [ENGINEERING.md](ENGINEERING.md) for the decisions behind these numbers, and [`../reports/`](../reports/) for the generated artifacts they are drawn from.

---

## The finding

Measured with real models — `openai/gpt-oss-120b` generating, `bge-small-en-v1.5`
embedding, on Groq's free tier.

| ablation | identical | equivalent | improved | now refuses | **now confident wrong** | Δ tokens |
|---|---|---|---|---|---|---|
| `no_gate` | 11 | 10 | 1 | 0 | **3 (12.0%)** | +34,567 |
| `no_lexical` | 10 | 13 | 0 | 1 | **1 (4.0%)** | +3,942 |
| `no_rerank` | 18 | 6 | 1 | 0 | **0 (0.0%)** | **−199** |

25 queries × 3 ablations. Full table and per-ablation predictions:
[`reports/ablation.md`](../reports/ablation.md).

**Removing the refusal gate turned 12% of correct answers into confidently wrong ones** —
no hedge, no refusal, a citation attached. `no_rerank` is the useful negative result: the
reranker cost 199 tokens and changed nothing, because on this corpus it never fired at all.

### Why only 25 queries

A free tier caps at roughly 100k tokens per model per day. A 220-query × 14-ablation sweep
is 3,080 runs and needs about 300× that, so the honest options are a small real sample or a
large simulated one. This is the small real sample.

The three composite ablations the spec calls for — removing two redundant guards at once to
expose interactions a single-ablation study cannot see — are implemented and tested, but
have not been run against a real model at any useful sample size. That gap is real and is
listed in [What is and isn't verified](ENGINEERING.md#what-is-and-isnt-verified).

### Why this table was empty until it wasn't

The first real-model sweep returned **zero in every outcome column**. Nothing errored; the
counterfactual engine looked like it worked. The cause was one config value:
`max_context_chunks: 12`. Twelve chunks of a corpus whose median chunk is ~111 tokens is
~1,300 tokens of context for a single-fact question — wide enough to hold the right chunk
however badly ranking was damaged, so no ablation could move the answer.

Gold-chunk retention by context width. Gold needs no judge: every generated case carries a
globally unique coined token, so "did a chunk containing it reach the generator" is a
substring check.

| max_context_chunks | baseline | `no_lexical` | `no_semantic` | `no_fusion` | spread |
|---|---|---|---|---|---|
| 1 | 91% | 81% | **100%** | **100%** | 19pp |
| 2 | 95% | 86% | **100%** | **100%** | 14pp |
| **3** (shipped) | 95% | 88% | **100%** | **100%** | **12pp** |
| 5 | 100% | 92% | 100% | 100% | 8pp |
| 8 | 100% | 96% | 100% | 100% | 4pp |
| 12 | 100% | 97% | 100% | 100% | 3pp |

Above ~5 chunks the ablations are unmeasurable on this corpus. The shipped width is 3,
chosen from this curve; the full curve is in
[`reports/context-sensitivity.md`](../reports/context-sensitivity.md) so the choice is
auditable rather than convenient. **Anyone running a retrieval ablation should report this
curve before reporting an effect size.**

There is a second finding in that table, and it is not the one the spec predicted:
**BM25 alone reaches 100% retention at every width.** `no_semantic` and `no_fusion` never
lose a gold chunk, at any context size, while `no_lexical` costs 3–19pp. The dense leg is
not contributing on this corpus — it is displacing lexical hits. `bge-small-en-v1.5` has
never seen `pellshale` or `KLV-4021`, so it embeds coined identifiers as noise while BM25
matches them exactly, and reciprocal rank fusion then averages a good signal with a bad one.

**Limitation, stated plainly:** a corpus built from invented tokens is adversarial to any
pretrained embedder. This is a result about *this corpus*, not a general claim about hybrid
retrieval. The textbook prediction — `no_lexical` breaking identifier queries while
`no_semantic` breaks paraphrase queries — did **not** reproduce here, and a failed
prediction bounds the claim more usefully than a confirmed one would.

That measurement costs nothing. An ablation cannot change the answer unless it changes what
reaches the generator, and comparing context sets needs no model call —
`Pipeline.run(..., generate=False)` exists for exactly this. It is also what makes a full
sweep affordable: spend model calls only on the cases whose evidence actually moved.

### The sweep that isn't here

An earlier version of this README carried a 220 x 14 table with cells like "67 confidently
wrong". It has been removed rather than relabelled, because it was never a result.

That sweep was run with the offline concept embedder against an index built from dense
vectors. The combination is invalid, and it failed on **every one of the 3,080 runs**. The
pipeline writes a failure into the answer text, so baseline and variant matched
byte-for-byte, and the classifier's byte-equality shortcut filed all 3,080 as `identical`.
The findings table came back with zeros everywhere and no errors reported anywhere -- which
reads exactly like "this pipeline is robust to ablation".

The build spec predicted this class of bug and named the wrong cause: it warned that a
cache keyed without the config hash would make every diff come back `IDENTICAL`. The cache
was fine. Byte-equality applied to error messages did it instead.

Fixed in three places: `classify()` now returns `ERROR` when either run failed
(`tests/test_failed_runs_are_not_agreement.py` pins the exact shape), `ERROR` counts as a
regression so an errored sweep cannot read as a passing one, and simulator sweeps write to
their own filename so they cannot overwrite a real-model run.

---

## The demo

> **This transcript does not reproduce under the current default configuration.**
>
> It was recorded with a different embedder. Re-running the same command today under
> `provider=groq` with `embedder=fastembed` gives:
>
> ```
>  * #1   L1   S1   fused_top_k   KLV-4021 — batch operation aborted
> ```
>
> — the correct chunk at **semantic rank 1**, not absent from the dense leg, and the
> reranker skipped entirely because the gate was already confident. `bge-small` handles
> `KLV-4021` fine, so removing the lexical leg does **not** break this particular query.
>
> The transcript is kept because the failure *mode* it demonstrates is real and is what the
> discriminator guard exists to catch — but it is an artifact of an older configuration, and
> the per-leg ranks below should not be read as current. The measured, aggregate version of
> this claim is in [The finding](#the-finding): the lexical leg is worth 3–19pp of gold-chunk
> retention depending on context width. That number holds; this anecdote does not.

```
$ python -m autopsy.cli query "what does KLV-4021 mean" --tenant tenant_kelvin

  retrieval competition (top 8)
   * #2   L1   S-   rr=100.0  rerank_promoted     KLV-4021 — batch operation aborted
   * #14  L-   S13            fused_top_k         KLV-4022 — batch operation aborted
   * #1   L-   S1   rr=60.0   fused_top_k         KLV-4003 — segment operation aborted
   * #8   L-   S7   rr=60.0   fused_top_k         KLV-4002 — segment operation aborted

  answer [grounded]
    KLV-4021 means the pellshale budget was exhausted while accepting incoming
    batches. [1]
```

The dense leg (`S`) retrieves the whole `KLV-40xx` family and ranks the right one 13th,
because it cannot see the difference between the codes. The lexical leg (`L1`) finds it
exactly. The reranker promotes it to the top. Now remove the lexical leg:

```
$ ... --ablation no_lexical

  answer [ungrounded]
    The sources do not document `klv-4021`. The closest documented behaviour is:
    KLV-4003 means the murrvale budget was exhausted while merging segments. [1]
```

Wrong, but *flagged* — the discriminator guard noticed that the identifier being asked
about appears in none of the sources. Remove that too:

```
$ ... --ablation no_lexical --ablation no_discriminator_guard

  answer [grounded]
    KLV-4003 means the murrvale budget was exhausted while merging segments. [1]
```

Fluent, cited, marked grounded, and about a completely different error code. That is
the 20-second recording.

---

## What this is

Three artifacts over one instrumentation layer.

| Artifact | Input | Output |
|---|---|---|
| **Inspector** | one query | live visual of the whole pipeline |
| **Eval runner** | a test set | pass/fail report, CI exit code |
| **Ablation study** | test set × configs | the findings table above |

They are one system. The pipeline emits a rich trace; rendering one trace is the
inspector, asserting over many is the eval, re-running with stages disabled is the
study. `POST /counterfactual` and `make ablate` call the *same* classifier, so the
thing the demo shows and the thing the report measures cannot diverge.

```
rewrite → embed → retrieve(dense ∥ sparse) → fuse → gate → rerank? → expand → generate
```

Every stage is optional and described by config. An ablation is
`replace(cfg, lexical=None)` — a data transform, not a code path. Nothing in the
pipeline reads an environment variable or a global.

---

## The three suites

### Isolation — 12 probes, and why 10 of them would otherwise prove nothing

Querying tenant A and checking you got tenant A's documents proves nothing, because
nothing was competing. This suite plants a document on the *same topic* in every
tenant, with different values and a per-run random canary, then queries that topic as
one tenant. Foreign documents are strong retrieval candidates and the boundary is the
only thing holding them back. Every output surface is scanned for foreign canaries —
answer, citations, candidate set, rewritten query, stage skip reasons, error text.

The documents deliberately share a `doc_id` across tenants, because neighbour expansion
resolves documents by identifier and that is where the boundary breaks.

**Two positive controls are mandatory and CRITICAL.** A tenant must still see its own
documents, and global documents must stay reachable. Without them a system that returns
nothing to anybody passes all ten leak probes.

Two properties this suite has that most don't:

- **It is proven able to fail.** `tests/test_evals.py` deliberately widens the tenant
  scope to the whole corpus and asserts the probes go red, then empties the scope
  entirely and asserts the *positive controls* go red.
- **A probe that exercises nothing reports MEDIUM, not PASS.** The neighbour-expansion
  probe fails itself if expansion added zero chunks, because a run that could not have
  detected a leak is not evidence there wasn't one.

One probe is honest about being weak: `prompt_level_override` cannot fail under the
offline provider, because a simulator that doesn't follow instructions cannot be
prompt-injected. It says so in its own output.

### Silent failure — wrong-and-confident rate

Every wrong answer is one of two things:

- **wrong, flagged** — hedged, refused, or uncertain. A system that is 70% accurate and
  flags every uncertain answer is deployable.
- **wrong, confident** — no hedge, clean citation. A system that is 85% accurate and
  never flags anything is not, because nothing distinguishes the 85% from the 15%.

Nine hand-built traps across eight failure classes: `near_miss_vocabulary`,
`phantom_value` (×2), `stale_deprecated`, `cross_version`, `partial_evidence`,
`ambiguous_scope`, `negation`, `superlative`. Each names the documents that collide and
what the plausible wrong answer looks like, so a reader can open the corpus and check
the trap is fair in about a minute. Current result: 3 of 9 wrong-and-confident.

### Ablation regression

The study's own output distribution becomes a test. If `no_lexical` used to break 34%
and now breaks 3%, either the pipeline changed, the corpus changed, or the measurement
broke — all three worth an alert. Tolerance is in percentage points, not relative
change, because 1 case out of 220 becoming 2 is a 100% relative swing and means nothing.

---

## Judge calibration

Every judge-derived number is reported next to the judge's measured agreement. Current:

- agreement **0.57**, Cohen's kappa **0.39**, n=106
- order instability **0.00** (every comparison is run in both directions)
- verbosity bias **+0.129**

Full confusion matrix and every disagreement: [`reports/judge-calibration.md`](../reports/judge-calibration.md).

The report names its own largest systematic error rather than leaving it in a matrix
for someone to notice. Three bias controls, each measured rather than assumed:
**position** (both orders, disagreement published as an instability rate rather than
averaged away), **verbosity** (correlation between length delta and verdict),
**self-preference** (the live judge is OpenAI; generation is Anthropic).

Two honest caveats:

1. The shipped labels are **derived, not human** — computed from the synthetic corpus's
   ground truth. Objective and reproducible, but a human also judges whether
   differently-worded answers mean the same thing. The report prints a warning saying
   this until you add `evals/golden/human.jsonl`. See
   [`evals/golden/README.md`](../evals/golden/README.md).
2. Because the corpus carries ground truth, **the headline ablation column doesn't
   depend on the judge at all.** Every generated fact has a globally unique token, so
   "did this answer come from the right chunk" is a substring check. The judge is only
   consulted where ground truth is absent.

---

## The corpus

Two corpora, kept separate on purpose.

**`corpus/seed/`** — 15 committed documents for three **fictional** systems (Kelvin, a
key-value store; Atlas, a relational database; Vela, a vector database), plus a shared
global tenant. Fictional rather than real products deliberately: benchmark fixtures that
assert things about real software are misinformation waiting to be quoted out of
context, and the property the corpus actually needs — dense technical vocabulary, exact
identifiers, plausibly overlapping domains — doesn't require the products to exist.
These are what the traps are built against and what a human reads in the demo.

**`corpus/generated/`** — 54 generated documents (434 chunks) with **ground truth**.
Every fact carries a globally unique token, so correctness is a substring check with no
judge in the loop. Four query shapes, each designed so a different ablation breaks it:
`identifier` (breaks under `no_lexical`), `paraphrase` (breaks under `no_semantic`,
worded with synonyms so BM25 has nothing to match), `value`, `absent` (a real family, an
unassigned code), `out_of_scope` (a subsystem that exists nowhere — the only shape that
makes the gate measurable), and `followup` (exercises the rewrite path). Seeded and
deterministic. 540 cases total.

**`corpus/manifest.yaml`** also declares real sources — Qdrant, DuckDB, PostgreSQL docs
— fetched by `make fetch`. Every source starts `licence_verified: false` and the fetcher
refuses it until a human has read the licence and flipped the flag. Provenance is pinned
*on fetch* into `corpus/manifest.lock.yaml` rather than hand-written into the manifest:
a SHA typed into a config file is a claim nobody checked.
