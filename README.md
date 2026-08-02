<h1 align="center">Retrieval Autopsy</h1>

<p align="center">
  <b>A RAG pipeline built to be <i>observed</i>.</b><br>
  Every stage is instrumented and every stage is switchable, so one system powers a live
  visual debugger, a headless eval suite, and an ablation study.
</p>

<p align="center">
  <a href="#quickstart"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="LICENSE"><img alt="MIT" src="https://img.shields.io/badge/licence-MIT-blue"></a>
  <a href="#quickstart"><img alt="No API key required" src="https://img.shields.io/badge/API%20key-not%20required-brightgreen"></a>
  <a href="#what-is-and-isnt-verified"><img alt="221 tests collected" src="https://img.shields.io/badge/tests-220%20passed%20%C2%B7%201%20skipped-brightgreen"></a>
  <a href="reports/"><img alt="Reports" src="https://img.shields.io/badge/reports-committed-informational"></a>
</p>

<p align="center">
  Built by <b>Kushal Desai</b>
</p>

---

**Runs with no API key, no Docker, and no Node.** `pip install -e .` then `make ingest`
gives you a working hybrid retriever, a 3D embedding-space inspector, and an eval suite
that fails the build when the system starts lying. Jump to [Quickstart](#quickstart).

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
[`reports/ablation.md`](reports/ablation.md).

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
listed in [What is and isn't verified](#what-is-and-isnt-verified).

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
[`reports/context-sensitivity.md`](reports/context-sensitivity.md) so the choice is
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

## Quickstart

**No API key is needed for anything below.** Nothing here reaches the network or costs
money. Requires Python 3.12+ and nothing else.

```bash
git clone https://github.com/kushaldesai/retrieval-autopsy.git
cd retrieval-autopsy

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt    # or: pip install -e ".[all]"
export PYTHONPATH=.                # Windows: $env:PYTHONPATH = "."

python -m autopsy.cli ingest       # generate the corpus + build the index (~10s)
python -m autopsy.cli query "what does KLV-4021 mean" --tenant tenant_kelvin
```

That last command prints a full trace: every stage, both retrieval legs with their ranks
*and* raw scores, what the gate decided, and the answer. No key was involved.

### Then open the inspector

```bash
python -m uvicorn api.main:app --port 8000     # then visit http://localhost:8000
```

Three tabs. **RAG** shows the nine stages executing with per-stage timings. **SPACE** puts
all 449 chunks in a 3D projection of the embedding space and animates the query through
both retrieval legs into the model. **EVAL** runs the trap suites and plots each probe
where its question lands in the corpus, coloured by outcome.

<details>
<summary><b>Task-runner equivalents</b> — <code>make</code> on Unix, <code>run.ps1</code> on Windows</summary>

```bash
make install     # venv + dependencies
make ingest      # generate the corpus and build the index (idempotent)
make test        # 221 tests collected by pytest, including negative controls
make eval        # isolation + silent-failure, exits non-zero on a new failure
make ablate      # the counterfactual sweep -> reports/ablation-simulated.md
make calibrate   # judge calibration -> reports/judge-calibration.md
make api         # FastAPI + inspector on :8000
```

Windows has no `make` by default. `run.ps1` takes the same targets and, unlike the raw
commands above, **loads `.env` for you**:

```powershell
.\run.ps1 install
.\run.ps1 demo-query    # baseline vs ablated, side by side — start here
.\run.ps1 api           # inspector on :8000
.\run.ps1 all           # ingest, test, eval, ablate in sequence
.\run.ps1 help
```

Also available: `make web` (Vite dev server, needs Node 20+ — see the
[unverified](#what-is-and-isnt-verified) note) and `docker compose up` (keyless demo on
:3000, replaying pre-recorded traces).
</details>

<details>
<summary><b>Troubleshooting</b> — the three things that actually go wrong</summary>

**`index/config mismatch — the index was built by a different provider`**

The index stores which embedder built it, and refuses to be searched by a different one.
Either re-run `ingest` under the current settings, or point the process back at the
provider that built it. This is the guard working: left to fail lazily, the mismatch shows
up as a per-chunk error mid-query and reads like a corrupt corpus.

**`embedded Qdrant at ./corpus/index/qdrant is locked by another process`**

Embedded Qdrant allows one process at a time. Usually the API server is still running. Stop
it, or sidestep the lock entirely with `AUTOPSY_VECTOR_BACKEND=local` — numpy in-process,
same results.

**`provider='groq' needs GROQ_API_KEY, which is empty`**

`run.ps1` fails fast rather than letting you discover it four stages into a query. Either
paste a key into `.env` or set `AUTOPSY_PROVIDER=offline`. Note that a variable already set
in your shell beats `.env` — standard dotenv semantics, and `run.ps1` tells you when it
happens, because editing `.env` and watching the old value survive is otherwise
indistinguishable from the edit not saving.
</details>

### Configuration

Copy [`.env.example`](.env.example) to `.env` — it documents every variable the
project reads. One thing worth knowing up front:

| Consumer | Reads `.env`? |
|---|---|
| `docker compose` | **yes** — auto-loaded from the repo root |
| Python / `make` | **no** — the process environment only; nothing calls `load_dotenv()` |
| Vite / `npm` | **no** — reads `web/.env`, and only `VITE_`-prefixed names |

The Python side does not auto-load a dotenv file on purpose. `AUTOPSY_PROVIDER`
decides whether the pipeline spends money, and a file quietly flipping it to `live`
because it happened to be on disk is the kind of surprise that arrives as a bill.

The defaults need no configuration at all. To run against **real models for free**:

```bash
export GROQ_API_KEY=…                 # console.groq.com/keys — free tier
AUTOPSY_PROVIDER=groq make ingest     # rebuilds the index with real local embeddings
AUTOPSY_PROVIDER=groq make eval
```

| | `offline` | `groq` | `openai` | `live` |
|---|---|---|---|---|
| generation | simulator | `openai/gpt-oss-120b` | `gpt-4o-mini` | `claude-sonnet-4-6` |
| rerank | simulator | `llama-3.1-8b-instant` | `gpt-4o-mini` | `claude-haiku-4-5` |
| judge | rule-based | `qwen/qwen3.6-27b` | `gpt-4o-mini` | `gpt-4o-mini` |
| embeddings | concept bags | `bge-small-en-v1.5` (local ONNX) | `text-embedding-3-small` | `text-embedding-3-small` |
| cost | none | none | ~$0.0003/question | per token |

The judge is a different model family from the generator wherever the provider serves more
than one. That is the only real control for self-preference bias, and Groq happens to serve
three families, so it survives the move. Under `openai` it does not — one family judging
itself, and the calibration number should be read with that in mind.

**`provider` and `embedder` are independent axes**, not one setting. Groq serves no
embeddings endpoint at all, so a Groq run pairs a remote chat model with a local ONNX
embedder and the trace has to be able to say so. The useful consequence:

```bash
AUTOPSY_PROVIDER=offline AUTOPSY_EMBEDDER=fastembed make ingest
```

Simulated answers, **real** 384-dimensional embeddings, no key. The 3D view then shows true
geometry instead of a simulation, for free.

**Measured Groq free-tier limits** (response headers, 2026-08-02): **8,000 tokens per
minute** and 1,000 requests per day. At ~2k tokens of retrieved context per call that is
about four calls a minute; a full `ablate -n 220` sweep will not fit, so use `--core -n 15`.

There is also a **daily** token cap that Groq exposes in no header. When it is exhausted
the per-minute counters still read healthy, so the failure looks like a rate limit that
never clears. This cost me an afternoon of chasing the wrong counter; the provider now
checks the response body text instead.

Anthropic + OpenAI instead:

```bash
export ANTHROPIC_API_KEY=… OPENAI_API_KEY=…
AUTOPSY_PROVIDER=live make reports
```

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

Full confusion matrix and every disagreement: [`reports/judge-calibration.md`](reports/judge-calibration.md).

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
   [`evals/golden/README.md`](evals/golden/README.md).
2. Because the corpus carries ground truth, **the headline ablation column doesn't
   depend on the judge at all.** Every generated fact has a globally unique token, so
   "did this answer come from the right chunk" is a substring check. The judge is only
   consulted where ground truth is absent.

---

## Engineering decisions worth knowing about

**Cache keys include the config hash.** Without it an ablated run hits the baseline's
cached answer, every diff returns `IDENTICAL`, and the counterfactual engine appears to
work perfectly while measuring nothing. It's the subtlest bug in the project and there
is a test for it. Key: `(config_hash, tenant_id, stage, input_hash)`.

**Embedding caches are shared; answer caches are not.** An embedding is a function of
text and model, so `(model_id, text)` is safe and saves most of the cost of a sweep.
Answer caches keyed on query text leak across tenants — that's the `cache_namespace`
probe.

**The offline embedder's model ID is a hash of its own rules.** The "model" here is a
synonym lexicon and a stemmer that live in the repo and change under you. Editing a
synonym while the ID stayed `offline-concept-v1` meant ingest reused vectors built by
the *previous* rules, with no error anywhere. Fingerprinting the lexicon into the ID
makes stale vectors impossible.

**BM25 IDF is the Lucene form, not the classic one.** `log(N − df + 0.5) − log(df + 0.5)`
is exactly zero when a term appears in half the documents, so a query for the one
identifier that matters can score 0.0 against every chunk and return nothing — and it
presents as "retrieval found nothing", not "the weighting collapsed". Never fired on
449 chunks; fired immediately on a 4-chunk test fixture.

**Temperature 0 pins the model generation.** Sampling parameters were *removed* on
Claude Opus 4.7+, Sonnet 5, and Fable 5 — `temperature=0.0` is a 400 there, not a no-op.
Determinism requires the pin, so the generator and reranker come from
`ACCEPTS_SAMPLING_PARAMS` (`claude-sonnet-4-6`, `claude-haiku-4-5`). Moving to a newer
model is a research decision, not a version bump.

**The counterfactual engine raises, not warns, on provenance mismatch.** A diff computed
across a model upgrade looks exactly as meaningful as a real one.

**`skip_reason` is prose with numbers**, because it renders straight into the UI:
`"dense_top1 0.71 > 0.60 and margin 0.19 > 0.10 — retrieval is confident, skipping the
reranker"`. A test asserts skip reasons are at least four words and don't start with
`cond`.

**Ablating a leg does not silently disable the gate.** `no_semantic` removes the gate's
input signal; rather than passing everything, the gate records that it could not run and
says so in the trace. There's a test for that too.

**Baselined failures don't fail the build; new ones do.** Failing unconditionally on
every HIGH finding sounds stricter and is worse — this corpus has four reproducible
silent-failure hits, so an unconditional rule leaves CI permanently red, and a
permanently red build is one nobody reads. Acknowledged failures are listed in the
report under their own heading. This is a deliberate deviation from the spec's literal
"exit 1 on any HIGH".

**No LangChain or LlamaIndex.** The entire point is fine-grained control of stage
boundaries and tracing; a framework's abstractions fight you on exactly that. Hand-rolled
so every stage boundary is observable.

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

---

## What is and isn't verified

Being specific about this, because "it's built" and "it runs" are different claims.

**Verified in this environment** — Python 3.12, all offline:

- `make ingest` — 449 chunks from 69 documents, and idempotent (a second run re-embeds 0)
- `make test` — 221 tests collected by pytest, from 172 `def test_` functions (the
  difference is parametrisation): unit, property tests on the fusion maths (hypothesis), API,
  websocket streaming, and the negative controls described above
- `make eval` — 22 findings across two suites. **Exits 1 on this corpus**, because the
  offline simulator confidently answers 5 of the 9 traps. That is the suite working, not
  failing: see the offline caveat below.
- `make ablate` — writes [`reports/ablation-simulated.md`](reports/ablation-simulated.md)
  under its own filename so a simulator sweep can never overwrite a real-model one
- `make calibrate` — kappa **0.33** over 107 derived pairs
  ([`reports/judge-calibration.md`](reports/judge-calibration.md))
- `make schema` — Zod types generated from the Pydantic models
- The API and websocket, via `fastapi.testclient` (10 tests)
- isolation **12/12**, under both the concept and the fastembed embedder

**The offline numbers are worse than the real-model ones, and that is expected.** The
simulator is rule-based; hedging is exactly what it is worst at. Offline silent-failure is
**4/9 traps handled** versus 9/10 on a real model. Do not read the offline figure as a
property of the pipeline.

**Verified against real models** — Groq free tier plus local ONNX embeddings. **None of
this is reproducible in a checkout without a `GROQ_API_KEY`**, so `make reports` cannot
rebuild it:

- `openai/gpt-oss-120b` generation, `llama-3.1-8b-instant` rerank,
  `qwen/qwen3.6-27b` judge, `bge-small-en-v1.5` embeddings
- The headline ablation table at the top of this file, and
  [`reports/ablation.md`](reports/ablation.md) — which carries a provenance banner naming
  two known defects in that artifact
- isolation **12/12 findings**, silent-failure **9/10 findings** (9 traps plus 1 aggregate)
- gate calibration across all three candidate signals
- Retry behaviour under real 429s and a real dropped TLS handshake

**Written but NOT verified here** — this environment has no Node and no Docker:

- **`web/`** — the React inspector has never been built, typechecked, or rendered. The
  four panels, the streaming reducer, and demo-mode replay are written against the
  generated types, but treat the whole directory as unreviewed until `npm run build`
  passes. CI runs `typecheck` and `build`, so the first push will say.
- **`autopsy/providers/live.py`** — the Anthropic + OpenAI path. Never executed; no
  keys for it. The Groq provider is a separate module and *is* verified.
- **Qdrant backend** — implemented in `autopsy/store/vectors.py`, never run. The default
  local backend is what every committed number used.
- **`docker compose`** and both Dockerfiles.

**Known limitations of the numbers themselves:**

- Everything above is the offline simulator (see the callout at the top).
- Chunks average ~100 tokens, not the 400–800 the spec targets, because structural
  splitting on headings comes first and the documents have short sections. This is
  deliberate — it keeps each error code individually retrievable, which is what the
  identifier queries need — but it means chunk-size behaviour is untested at realistic
  sizes.
- The judge's reference labels are derived, not human (see above).

---

## Deliberate deviations from the build spec

Five, each with a reason:

1. **The seed corpus is fictional systems, not Postgres/Redis/Qdrant.** Benchmark
   fixtures that assert things about real software are misinformation waiting to be
   quoted out of context. The real-docs path exists and is licence-gated (`make fetch`);
   the offline corpus is what makes CI work with no network.
2. **A generated corpus with ground truth was added.** 66 hand-written chunks are too
   few to measure anything — with `top_k` at 20, retrieval returns most of the corpus
   and every ablation comes back identical because ranking never mattered. The
   generated corpus also removes the judge from the headline number.
3. **Baselined HIGH failures don't fail the build** (the spec says any HIGH exits 1).
   Reasoning in the engineering-decisions section above.
4. **`no_discriminator_guard` and three composites were added to the ablation set.**
   The guard is the difference between wrong and wrong-and-confident, and the
   composites are where the interesting result lives.
5. **Caches are in-process, not Redis.** The pipeline is a single process and the
   caches are keyed correctly; adding a Redis dependency would buy nothing here and
   would put a network hop inside the thing being timed. `Outcome` also gained
   `IMPROVED` and `ERROR` — without them, an ablation that *helps* is miscounted as a
   regression and a crashed variant vanishes from the table.

---

## Repo map

```
autopsy/
  config.py          PipelineConfig — every stage optional, None means ablated
  ablations.py       named config transforms, including composites
  trace.py           the schema everything else depends on; TS is generated from it
  determinism.py     config hashing, version stamps, the comparability guard
  pipeline.py        composes stages, emits Trace
  counterfactual.py  Outcome classification, computed explanations, the findings table
  cache.py           the config-hash key, and the semantic cache's discriminator guard
  textutil.py        one definition of "is this claim supported", shared by all consumers
  chunking.py        structural-first, never splits a code block
  ingest.py          content-addressed and idempotent
  tsgen.py           Pydantic models -> Zod schemas -> TS types
  stages/            nine stages, one file each, uniform protocol
  providers/         offline simulator | live SDKs, chosen by config
  store/             chunks, BM25 (per-tenant index), vectors (local | Qdrant)

evals/
  runner.py          suites, findings, baseline diff, exit code
  judge.py           two-way judge, calibration, kappa, bias controls
  suites/            isolation | silent_failure | ablation regression
  traps/             nine trap definitions, one YAML each
  golden/            judge labels — derived shipped, human slot documented

corpus/              seed (handwritten) + synthetic (generated, ground truth) + manifest
api/                 FastAPI: /query, /counterfactual, /trace/{id}, WS /stream
web/                 React inspector — four panels, demo mode  (unbuilt, see above)
reports/             generated and committed
tests/               221 collected by pytest, including the negative controls
```

---

## Author

**Kushal Desai** — design, implementation, and the measurements in `reports/`.

If you use this in your own work, a link back is appreciated. If you find a number in here
that you cannot reproduce, that is a bug worth an issue: every figure is supposed to name
the configuration it came from, and one that does not is a defect in the reporting, not a
detail.

## Licence and scope

Code is [MIT](LICENSE) — © 2026 Kushal Desai.

The seed corpus is authored for this repository (CC0). No client data, no production
systems, no real tenant names — open documentation and fictional systems only. Findings
are framed as failure *classes*, not verdicts on any vendor: a benchmark that reads as
fair gets shared by the people it criticises.

### Contributing

Issues and pull requests are welcome. Two things make a PR easy to merge:

1. **`make test` and `make eval` both pass.** The eval compares against
   `evals/baseline.offline.json` and exits non-zero on any change to the failure set. If
   your change legitimately alters that set, run `python -m autopsy.cli eval
   --update-baseline` and say so in the description — an unexplained baseline update is the
   one diff that gets questioned every time.
2. **A new finding comes with the test that catches it.** Most of the tests here exist
   because something was wrong and nothing noticed.
