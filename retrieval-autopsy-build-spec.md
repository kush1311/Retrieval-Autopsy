# Retrieval Autopsy — complete build specification

A RAG pipeline built to be *observed*: every stage instrumented, every stage
switchable, so the same system powers a live visual debugger, a headless eval
suite, and an ablation study that produces publishable findings.

Written for a from-scratch build. Nothing is assumed to exist.

---

## Table of contents

1. [What you're building](#1-what-youre-building)
2. [Design principles](#2-design-principles)
3. [Architecture](#3-architecture)
4. [Repo structure](#4-repo-structure)
5. [Part A — the pipeline](#part-a--the-pipeline)
6. [Part B — the trace schema](#part-b--the-trace-schema)
7. [Part C — the eval runner and suites](#part-c--the-eval-runner-and-suites)
8. [Part D — the counterfactual engine](#part-d--the-counterfactual-engine)
9. [Part E — the inspector UI](#part-e--the-inspector-ui)
10. [Part F — the eval report](#part-f--the-eval-report)
11. [Build phases](#build-phases)
12. [Stack and dependencies](#stack-and-dependencies)
13. [Non-obvious engineering decisions](#non-obvious-engineering-decisions)
14. [Content plan](#content-plan)
15. [Risks and scope discipline](#risks-and-scope-discipline)
16. [Appendix — config reference](#appendix--config-reference)

---

## 1. What you're building

Three artifacts sharing one instrumentation layer.

| Artifact | Input | Output | Audience |
|---|---|---|---|
| **Inspector** | one query | live visual of the whole pipeline | the demo; engineers debugging |
| **Eval runner** | a test set | pass/fail report, CI exit code | credibility; your own regressions |
| **Ablation study** | test set × configs | a findings table | the shareable result |

**They are one system.** The pipeline emits a rich trace; rendering one trace is
the inspector, asserting over many traces is the eval, and re-running with
stages disabled is the ablation study.

### Definition of done

- `docker compose up` gives a working demo at `localhost:3000` with no API key
  required (pre-recorded traces in demo mode).
- `make eval` runs the full suite headless and exits non-zero on regression.
- `make ablate` produces `reports/ablation.md` with the findings table.
- A 20-second screen recording exists where toggling one stage turns a correct
  answer into a confidently wrong one.
- `README.md` leads with the ablation table, not with installation steps.

---

## 2. Design principles

Six decisions already made. Deviating from them breaks something downstream.

**1. Config is data, not code.** Every stage is described by a config object, and
the pipeline is a function of `(query, config)`. This is the single most
important architectural constraint in the document. An ablation is a config
transform — `replace(cfg, lexical=None)` — and if any stage is hardcoded, the
counterfactual engine is impossible to build. Design for this from commit one;
retrofitting it means rewriting the pipeline.

**2. The trace is the product.** The pipeline's real output is not the answer,
it's the trace. Build the emitter before the UI. Schema churn after you have
panels and assertions depending on it is the most expensive mistake available.

**3. Record what didn't happen.** Any tracing library records what happened. The
interesting questions in retrieval debugging are *which candidate was rejected
and by what*, and *which stage was skipped and why*. `rejected_by` and
`skip_reason` are first-class fields, not debug strings.

**4. Determinism or the diffs are noise.** Temperature 0 everywhere, pinned
model version strings, fixed seeds, and the resolved config hashed into every
trace. A counterfactual diff computed across two different model versions is
worse than no diff, because it looks meaningful.

**5. Eval is headless, the inspector is live.** Never merge them into one screen.
200 cases × 4 ablations is 800 runs and minutes of wall clock — a live view of
that is a progress bar. Eval produces a static report; the inspector streams one
query.

**6. Don't build a generic eval framework.** RAGAS, DeepEval, promptfoo, and
Langfuse own that space. Build *specific suites* — isolation, silent failure,
ablation regression. The differentiation is knowing which failures are worth
testing for, never the runner.

---

## 3. Architecture

```
                        ┌──────────────────────────┐
                        │  corpus (multi-tenant)   │
                        └────────────┬─────────────┘
                                     │  ingest (offline)
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
            ┌──────────────┐                  ┌──────────────┐
            │   Qdrant     │                  │  BM25 index  │
            │ (dense)      │                  │  (sparse)    │
            └──────┬───────┘                  └──────┬───────┘
                   └───────────────┬──────────────────┘
                                   ▼
     ┌─────────────────────────────────────────────────────────┐
     │  pipeline(query, config) -> (answer, Trace)             │
     │                                                         │
     │  rewrite → embed → retrieve(dense ∥ sparse) → fuse →    │
     │  gate → rerank? → expand → generate                     │
     │                                                         │
     │  every stage optional, driven by config                  │
     └────────────┬──────────────────────────┬─────────────────┘
                  │ Trace (JSON)             │ Trace (JSON)
                  ▼                          ▼
        ┌──────────────────┐       ┌──────────────────────┐
        │ inspector (live) │       │ eval runner (headless)│
        │ FastAPI + WS     │       │ suites + assertions   │
        │ React panels     │       │ CI exit code          │
        └──────────────────┘       └──────────┬───────────┘
                  ▲                           │
                  │      ┌────────────────────▼──────────┐
                  └──────┤ counterfactual engine          │
                         │ 1 query  → demo diff           │
                         │ N queries → findings table     │
                         └────────────────────────────────┘
```

---

## 4. Repo structure

```
retrieval-autopsy/
├── README.md                  ← leads with the ablation table
├── Makefile                   ← make ingest / eval / ablate / demo
├── docker-compose.yml         ← qdrant + api + web
├── pyproject.toml
│
├── corpus/
│   ├── raw/                   ← downloaded source docs, gitignored
│   ├── manifest.yaml          ← tenant → source mapping, licences
│   └── build.py               ← fetch, convert, chunk, embed, index
│
├── autopsy/
│   ├── config.py              ← PipelineConfig and every stage config
│   ├── ablations.py           ← named config transforms
│   ├── trace.py              ← Trace dataclasses + JSON (de)serialisation
│   ├── determinism.py         ← seed pinning, config hashing, assertions
│   │
│   ├── stages/
│   │   ├── base.py            ← Stage protocol
│   │   ├── rewrite.py
│   │   ├── embed.py
│   │   ├── retrieve_dense.py
│   │   ├── retrieve_sparse.py
│   │   ├── fuse.py            ← RRF
│   │   ├── gate.py
│   │   ├── rerank.py
│   │   ├── expand.py          ← neighbour expansion
│   │   └── generate.py
│   │
│   ├── pipeline.py            ← composes stages, emits Trace
│   ├── cache.py               ← keyed on config hash (see gotchas)
│   └── counterfactual.py      ← run ablations, classify diffs
│
├── evals/
│   ├── runner.py              ← executes suites, aggregates, exit code
│   ├── judge.py               ← LLM judge + calibration harness
│   ├── suites/
│   │   ├── isolation.py       ← cross-tenant leakage (10 probes)
│   │   ├── silent_failure.py  ← trap corpus, wrong-but-confident rate
│   │   └── ablation.py        ← regression on ablation outcomes
│   ├── traps/                 ← trap definitions, one YAML per trap
│   ├── golden/                ← hand labels for judge calibration
│   └── testsets/              ← query sets, versioned
│
├── api/
│   ├── main.py                ← FastAPI app
│   ├── routes_query.py        ← POST /query, POST /counterfactual
│   ├── routes_trace.py        ← GET /trace/{id}
│   └── ws_stream.py           ← WS /stream, stage events
│
├── web/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── panels/
│   │   │   ├── Competition.tsx     ← panel A
│   │   │   ├── Timeline.tsx        ← panel B
│   │   │   ├── Answer.tsx          ← panel C
│   │   │   └── Counterfactual.tsx  ← panel D
│   │   ├── lib/trace.ts            ← types generated from the schema
│   │   └── demo/traces/            ← pre-recorded, for keyless demo
│   └── package.json
│
└── reports/                   ← generated, committed
    ├── ablation.md
    ├── isolation.md
    └── judge-calibration.md
```

---

## Part A — the pipeline

### A.1 Corpus choice

This decision matters more than it looks, because the corpus has to support
three different demands at once: multi-tenancy (for isolation probes), exact
identifiers (so the hybrid-retrieval demo has teeth), and a licence that lets
you publish.

**Recommended: treat the documentation of 3–4 open-source projects as separate
tenants.** For example Postgres, Redis, and Qdrant docs as `tenant_pg`,
`tenant_redis`, `tenant_qdrant`.

Why this works better than synthetic data:

- **Real technical vocabulary with exact identifiers** — error codes, config
  parameter names, function signatures. This is where lexical retrieval beats
  semantic, so the hybrid demo works on genuine text rather than a rigged
  example.
- **Legitimately separate domains** — a Redis answer surfacing for a Postgres
  question is a meaningful leak, and the topics overlap enough (persistence,
  replication, memory limits) that foreign documents are *plausible* retrieval
  candidates. That plausibility is what makes isolation probes non-trivial.
- **Publishable** — permissive licences, and anyone can reproduce your numbers.
  Reproducibility is most of the credibility.
- **No client data risk.** Nothing to get sign-off for.

Record licence and commit SHA per source in `corpus/manifest.yaml` so the
corpus version is pinned alongside the code.

Add a small **synthetic tenant** on top with deliberately planted canaries and
competing documents, used only by the isolation suite (see C.1).

### A.2 Ingest

```
fetch → convert to markdown → chunk → embed → index (dense + sparse)
```

- **Convert**: markdown in, markdown out where possible. HTML via a converter
  that preserves code blocks and heading hierarchy — losing code blocks destroys
  the exact-identifier property the whole demo depends on.
- **Chunk**: structural first (split on headings), then size-capped with
  overlap. Target 400–800 tokens, 15% overlap. Record `heading_path`,
  `ordinal` (position within its document), and `doc_id` on every chunk —
  `ordinal` drives neighbour expansion and `heading_path` gives you readable
  citations for free.
- **Never split a code block.** A truncated code block is worse than a slightly
  oversized chunk.
- **Embed** once, at ingest, in batches. Store the embedding model ID and
  dimension on the collection; a dimension mismatch at query time should fail
  loud rather than silently return garbage.
- **Sparse index**: BM25 over the same chunk texts. Persist it (pickle or a
  small SQLite FTS table) so ingest isn't re-run on every boot.

Ingest is idempotent and content-addressed: chunk ID = hash of
`(tenant_id, doc_id, ordinal, text)`. Re-running ingest on an unchanged corpus
is a no-op.

### A.3 The stage protocol

Every stage has the same shape. This uniformity is what makes tracing and
ablation generic rather than per-stage special cases.

```python
# autopsy/stages/base.py
from typing import Protocol, TypeVar

I = TypeVar("I"); O = TypeVar("O")

class Stage(Protocol[I, O]):
    name: str

    def run(self, inp: I, ctx: "Context") -> O:
        """Do the work. Record timing/tokens/cost on ctx."""

    def skip(self, inp: I, ctx: "Context") -> str | None:
        """Return a human-readable reason to skip, or None to run.

        The reason string is surfaced in the UI, so write it for a reader:
        'top score 0.71 > gate 0.42, margin 0.19 > 0.10', not 'cond_1'.
        """
```

`Context` carries the resolved config, the tenant, the trace being built, the
cache, and a clock. Stages never touch globals — that's what makes them
testable and what lets the counterfactual engine run several configs
concurrently.

### A.4 Stage-by-stage

**rewrite** — turn a follow-up into a standalone query using conversation
history. Skipped when there's no history. Emits `rewritten_query`.

*Trap to avoid:* the rewrite path is a second entry into retrieval. Thread
`tenant_id` through it explicitly. This is the highest-yield cross-tenant leak
vector in real systems precisely because the primary path is already correct.

**embed** — embed the (possibly rewritten) query. Cache on
`(model_id, text)` — embeddings are not tenant-specific, so this cache is safe
to share across tenants. Answer caches are not; see gotchas.

**retrieve_dense** — vector search, `top_k` configurable, filtered by
`tenant_id IN (tenant, GLOBAL)`. The filter goes in the query, never as a
post-filter — post-filtering silently shrinks your result set and changes
recall in a way that looks like a ranking problem.

**retrieve_sparse** — BM25 over the same tenant-filtered scope.

**fuse** — Reciprocal Rank Fusion:

```
score(d) = Σ over legs  1 / (k + rank_leg(d)),  k = 60
```

Record, per candidate, its rank *and* raw score in each leg plus the fused
score. This is exactly what panel A renders, and it's what makes "the lexical
leg promoted this chunk" visible.

**gate** — refuse when retrieval isn't confident enough.

> **Open decision — which score does the gate read?** RRF scores are a function
> of rank position, so they have no natural threshold and shift with candidate
> count; a gate at `0.0155` is meaningless to a human and drifts as you change
> `top_k`. Raw cosine is interpretable but ignores the lexical signal entirely.
>
> **Recommendation:** gate on the raw score of the top *dense* hit, and record
> the fused ranking separately. Interpretable, stable across `top_k` changes,
> and defensible in a post. Make it a config field so you can test the
> alternative rather than argue about it.

**rerank** — LLM-as-reranker scoring candidates 0–100, but only in the gray
zone: low top score, or a thin margin between top-1 and top-2. When it skips,
`skip_reason` records the numbers. Visualising the skip is a differentiator —
almost no public demo shows a system declining to spend money.

**expand** — pull chunks adjacent by `ordinal` within the same document so the
model sees surrounding context.

*Trap to avoid:* expansion fetches by ID or ordinal, which usually bypasses the
filtered query path. Re-apply the tenant filter here. This is the second
highest-yield leak vector, for the same reason as rewrite.

**generate** — answer strictly from numbered sources, cite them, preserve the
source's hedging strength, refuse to fill gaps. Sources wrapped in delimiters
and declared as data, not instructions. Emit sentence-level spans mapping
answer text to chunk IDs — panel C's attribution hover depends on it, and so
does the grounding check in the eval.

### A.5 Config

```python
# autopsy/config.py
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class LexicalConfig:
    top_k: int = 20
    b: float = 0.75
    k1: float = 1.2

@dataclass(frozen=True, slots=True)
class SemanticConfig:
    model_id: str = "text-embedding-3-small"
    top_k: int = 20

@dataclass(frozen=True, slots=True)
class FusionConfig:
    rrf_k: int = 60

@dataclass(frozen=True, slots=True)
class GateConfig:
    threshold: float = 0.42
    reads: str = "dense_top1"       # see the open decision above

@dataclass(frozen=True, slots=True)
class RerankConfig:
    model_id: str = "claude-haiku-4-5-20251001"
    always: bool = False            # False = gray-zone only
    gray_zone_score: float = 0.60
    gray_zone_margin: float = 0.10

@dataclass(frozen=True, slots=True)
class ExpansionConfig:
    neighbours: int = 1

@dataclass(frozen=True, slots=True)
class GenerationConfig:
    model_id: str = "claude-sonnet-4-6"
    temperature: float = 0.0        # never change this
    max_context_chunks: int = 12

@dataclass(frozen=True, slots=True)
class PipelineConfig:
    lexical:    LexicalConfig    | None = LexicalConfig()
    semantic:   SemanticConfig   | None = SemanticConfig()
    fusion:     FusionConfig     | None = FusionConfig()
    gate:       GateConfig       | None = GateConfig()
    rerank:     RerankConfig     | None = RerankConfig()
    expansion:  ExpansionConfig  | None = ExpansionConfig()
    generation: GenerationConfig = GenerationConfig()
    rewrite_enabled: bool = True
```

`None` means the stage is ablated. Frozen dataclasses so a config is hashable
and safe to share across concurrent runs.

### A.6 Ablations

```python
# autopsy/ablations.py
from dataclasses import replace

ABLATIONS = {
    "baseline":     lambda c: c,
    "no_lexical":   lambda c: replace(c, lexical=None),
    "no_semantic":  lambda c: replace(c, semantic=None),
    "no_fusion":    lambda c: replace(c, fusion=None),
    "no_gate":      lambda c: replace(c, gate=None),
    "no_rerank":    lambda c: replace(c, rerank=None),
    "force_rerank": lambda c: replace(c, rerank=replace(c.rerank, always=True)),
    "no_expansion": lambda c: replace(c, expansion=None),
    "top_k_1":      lambda c: replace(c, generation=replace(c.generation, max_context_chunks=1)),
}
```

Ablations compose: `no_lexical + no_gate` is valid and instructive — it shows a
retrieval failure and a missing refusal producing a confident hallucination
together.

**If time is short, ship three:** `no_lexical`, `no_gate`, `no_rerank`. Three
ablations that each produce a dramatic, explainable failure beat nine that
mostly return "identical."

| Ablation | Expected failure it demonstrates |
|---|---|
| `no_lexical` | exact identifiers become unfindable |
| `no_semantic` | paraphrased queries stop matching |
| `no_fusion` | worse ranking, subtler than either leg alone |
| `no_gate` | hallucinated answer where a refusal was correct |
| `no_rerank` | gray-zone queries degrade |
| `force_rerank` | cost rises, quality usually doesn't — a useful negative result |
| `no_expansion` | answers truncate mid-context |
| `top_k_1` | confident single-source wrongness |

---

## Part B — the trace schema

The contract between the pipeline and everything else. Freeze it in phase 0;
generate the TypeScript types from it so the frontend can't drift.

```jsonc
{
  "trace_id": "01JQ...",
  "created_at": "2026-07-26T09:14:02Z",

  "query": "what does error_code_4021 mean",
  "rewritten_query": null,
  "tenant_id": "tenant_redis",
  "session_id": "s_abc",

  "ablations": [],
  "config_hash": "sha256:9f2c…",
  "config": { /* fully resolved PipelineConfig, not a partial override */ },
  "versions": {
    "corpus": "manifest@a1b2c3d",
    "embed_model": "text-embedding-3-small",
    "gen_model": "claude-sonnet-4-6",
    "code": "git@e4f5a6b"
  },

  "candidates": [
    {
      "chunk_id": "c_12",
      "doc_id": "redis/persistence.md",
      "heading_path": ["Persistence", "AOF", "Troubleshooting"],
      "text": "…",
      "ordinal": 7,
      "tenant_id": "tenant_redis",

      "lexical_rank": 1,          // null if absent from that leg
      "lexical_score": 8.41,
      "semantic_rank": 2,
      "semantic_score": 0.71,
      "fused_rank": 1,
      "fused_score": 0.0308,
      "rerank_score": null,       // null if the reranker didn't fire

      "final_rank": 1,
      "in_context": true,
      "inclusion_reason": "fused_top_k",
      "rejected_by": null
    }
  ],

  "stages": [
    {
      "name": "rerank",
      "ms": 0,
      "tokens_in": 0,
      "tokens_out": 0,
      "cost_usd": 0,
      "cache": null,
      "skipped": true,
      "skip_reason": "dense_top1 0.71 > gate 0.42, margin 0.19 > 0.10",
      "error": null
    }
  ],

  "answer": {
    "text": "Error 4021 indicates an AOF rewrite failure.",
    "status": "grounded",
    "spans": [
      { "start": 0, "end": 44, "chunk_ids": ["c_12"], "supported": true }
    ],
    "citations": ["c_12"]
  },

  "totals": { "ms": 2716, "cost_usd": 0.0041, "llm_calls": 2 }
}
```

### Enumerations — fix these now

| Field | Values |
|---|---|
| `inclusion_reason` | `fused_top_k`, `rerank_promoted`, `neighbor_expansion` |
| `rejected_by` | `null`, `gate`, `top_k`, `rerank`, `tenant_filter` |
| `answer.status` | `grounded`, `refused`, `ungrounded` |
| `stages[].cache` | `null`, `hit`, `miss` |

### Why these fields exist

- **`rejected_by`** — a chunk rejected by the gate is more informative than one
  that was never a candidate. Panel A renders rejections below the gate line;
  the eval asserts on them.
- **`inclusion_reason`** — winning on score, being promoted by rerank, and
  being dragged in by expansion are diagnostically different. Expansion chunks
  in particular need to look different in the UI, because a good answer built
  entirely from expansion chunks means your ranking is weak.
- **`config` resolved in full, not as a diff** — a trace must be interpretable
  in isolation, years later, without the code that produced it.
- **`versions`** — the guard against meaningless diffs. Refuse to compare two
  traces whose `versions` differ; that check belongs in the counterfactual
  engine and it should raise, not warn.
- **`skip_reason` as prose with numbers** — it's rendered directly to a human.

Store traces as newline-delimited JSON under `reports/traces/`, one file per
run. No database needed until it hurts.

---

## Part C — the eval runner and suites

The runner loads suites, executes them against a pipeline, aggregates findings,
writes a markdown report, and exits non-zero on regression. Headless. No UI,
ever.

```python
# evals/runner.py — shape only
@dataclass
class Finding:
    suite: str
    case_id: str
    passed: bool
    severity: Severity          # INFO < LOW < MEDIUM < HIGH < CRITICAL
    detail: str
    evidence: list[str]
    trace_id: str | None

class Suite(Protocol):
    name: str
    def run(self, pipeline, corpus) -> list[Finding]: ...
```

Exit code 1 when any finding is HIGH or above, or when the failure set differs
from `evals/baseline.json`. Baselining matters: in a codebase with pre-existing
failures, a *new* failure is invisible without a diff against a recorded
baseline.

### C.1 Isolation suite

Cross-tenant leakage. Ten probes; already specified and implemented in
`tenant_isolation_suite.py` — port it in as a suite.

**Methodology recap.** Plant competing documents: every tenant gets a document
on the same topic with different values plus a per-run random canary. Query that
topic as tenant A, so the foreign document is a *strong* candidate and the
boundary is the only thing holding. Check for foreign canaries on every output
surface — answer, citations, chunk IDs, generated SQL, error messages, and debug
payloads.

Probes: `direct_answer_leak`, `retrieval_filter`, `followup_rewrite`,
`cache_namespace`, `neighbor_expansion`, `degenerate_tenant_id`,
`prompt_level_override`, `sql_tenant_predicate`, `session_reuse`,
`existence_disclosure`.

**Two positive controls are mandatory:** a tenant must still see its own
documents, and shared global documents must stay reachable. Without them, a
system that returns nothing to anyone passes every leak probe — and if a
positive control fails, every passing leak probe above it is meaningless.

### C.2 Silent failure suite

The metric nobody publishes. Don't score accuracy; score **wrong-and-confident**.

Every wrong answer splits into two categories:

- **wrong, flagged** — the system hedged, refused, or expressed uncertainty. A
  system that's 70% accurate and flags every uncertain answer is deployable.
- **wrong, confident** — no hedging, no refusal. This is the business risk, and
  a system that's 85% accurate but never flags anything is not deployable.

That inversion is the headline finding.

Each trap is a YAML file:

```yaml
# evals/traps/aof_vs_rdb_confusion.yaml
id: aof_vs_rdb_confusion
failure_class: near_miss_vocabulary
tenant: tenant_redis
query: "how do I disable AOF rewrite throttling"
expected_behaviour: refuse_or_hedge
reason: >
  No such setting exists. Documents about AOF rewrite and about
  throttling both retrieve strongly, so the plausible wrong answer is a
  confident synthesis of two unrelated settings.
detect:
  must_not_contain_confident_claim: true
  acceptable_refusal_markers: ["not documented", "no such", "couldn't find"]
```

Failure classes worth building traps for:

| Class | The trap |
|---|---|
| `near_miss_vocabulary` | two real features whose terms combine into a plausible fake |
| `phantom_value` | asks about a config value that has never existed |
| `stale_deprecated` | the answer exists but is deprecated; is that surfaced? |
| `cross_version` | two doc versions disagree; does it pick one silently? |
| `partial_evidence` | sources support half the claim; is the other half hedged? |
| `ambiguous_scope` | the question spans two tenants' domains |
| `negation` | "which settings are *not* persisted" |
| `superlative` | "the fastest option" when docs never rank them |

**Eight traps done rigorously beat thirty done sloppily.** The credibility is
entirely in the reproducibility.

### C.3 Ablation regression suite

Once the ablation study has run, its outcome distribution becomes a regression
test: if `no_lexical` used to break 34% of answers and now breaks 3%, either the
pipeline changed or the corpus did. Both are worth knowing about.

### C.4 Judge calibration

The eval depends on an LLM judge for "is this answer equivalent / degraded /
hallucinated." An uncalibrated judge undermines every number in the report.

**Do this and publish the result:**

1. Hand-label 100 outputs yourself. Store in `evals/golden/`.
2. Run the judge against them. Compute agreement, and Cohen's kappa — raw
   agreement is inflated when one class dominates.
3. Write `reports/judge-calibration.md` with the agreement rate, the confusion
   matrix, and the cases where you disagreed with your own judge.

Known judge biases to control for: **verbosity** (longer scores higher),
**position** (in pairwise comparison, first has an edge — so run both orders and
average), and **self-preference** (models favour their own family — so judge with
a different family than you generate with).

Publishing "my judge disagrees with me 23% of the time, here's where" is a
better post than any passing test suite. Practically nobody does this step,
which is exactly why it reads as rigour.

---

## Part D — the counterfactual engine

The bridge. One query ablated is a demo; two hundred queries ablated is a
finding.

```python
# autopsy/counterfactual.py — shape only
@dataclass
class Diff:
    ablation: str
    baseline_trace_id: str
    variant_trace_id: str
    outcome: Outcome
    rank_delta: dict[str, int]        # chunk_id -> rank change
    dropped_from_context: list[str]
    explanation: str                  # generated, human-readable

class Outcome(str, Enum):
    IDENTICAL      = "identical"       # byte-equal answers
    EQUIVALENT     = "equivalent"      # different words, same claims
    DEGRADED       = "degraded"        # worse but not wrong
    NOW_REFUSES    = "now_refuses"     # baseline answered, variant refused
    NOW_ANSWERS    = "now_answers"     # baseline refused, variant answered
    NOW_WRONG      = "now_wrong"       # correct -> incorrect
    NOW_CONFIDENT_WRONG = "now_confident_wrong"   # the money category
```

### Execution

1. Run the baseline, keep the trace.
2. Run each ablation concurrently. Bound concurrency — you will hit rate limits
   with 8 ablations × 200 queries.
3. **Reuse the embedding across ablations.** The query embedding is identical
   for every variant except `no_semantic`; embed once. Only the retrieval and
   generation legs actually differ. This is most of your cost saving.
4. Classify each diff. Cheap checks first: byte-equal → `IDENTICAL`. Only call
   the judge when the strings differ.
5. Refuse to compare traces whose `versions` block differs. Raise.

### Explanation generation

The engine should produce the sentence that makes the diff legible, not just the
category. Compute it from the traces rather than asking a model:

> `c_12` fell from fused rank 1 to rank 9 and dropped out of context. It had
> lexical rank 1 (score 8.41) and semantic rank 2 (score 0.71); without the
> lexical leg, only the semantic signal remained.

That's derivable from `candidates` in both traces, and it's the caption for the
demo recording.

### Aggregate output

The findings table. Rows are ablations, columns are outcomes, cells are counts
over the test set.

```
| ablation      | identical | equivalent | degraded | now refuses | now confident wrong |
|---------------|-----------|------------|----------|-------------|---------------------|
| no_lexical    |        88 |         31 |       47 |          10 |                  24 |
| no_gate       |       171 |          8 |        3 |           0 |                  18 |
| no_rerank     |       142 |         37 |       19 |           2 |                   0 |
| force_rerank  |       196 |          4 |        0 |           0 |                   0 |
```

**Read the last column first.** "Removing the lexical leg pushed 12% of answers
from correct to confidently wrong" is the sentence people quote. The
`force_rerank` row is valuable as a *negative* result — cost rose, quality
didn't — and negative results are disproportionately shareable because almost
nobody publishes them.

---

## Part E — the inspector UI

Single view, four panels, no routing, no top-level tabs. It's a debugger, not a
dashboard: it inspects one request completely. Cross-request aggregation is a
different product and will eat your timeline.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  query bar                                    [tenant ▾]  [ablations ▾]  │
├────────────────────────────┬─────────────────────────────────────────────┤
│  A. RETRIEVAL COMPETITION  │  C. ANSWER                                  │
│                            │                                             │
│  bm25       vector         │  Rendered answer. Hovering a clause          │
│  1 c_12     1 c_04         │  highlights its source chunk in panel A.     │
│  2 c_07     2 c_12 ←       │  Unattributed clauses flagged inline.        │
│  3 c_31     3 c_19         │                                             │
│       ↘   ↙                ├─────────────────────────────────────────────┤
│    fused (rrf k=60)        │  D. COUNTERFACTUAL DIFF                     │
│    1 c_12  ← won           │  baseline        │  no_lexical              │
│    2 c_04                  │  "…within 7      │  "…within 30 days"       │
│    3 c_19                  │   days"          │  ▲ now confident wrong   │
│    ┄┄┄ gate 0.42 ┄┄┄       │                  │    c_12: rank 1 → 9      │
│    4 c_22  (rejected)      │                  │                          │
├────────────────────────────┴─────────────────────────────────────────────┤
│  B. rewrite · embed · retrieve · fuse · gate · rerank(skipped) · generate │
└──────────────────────────────────────────────────────────────────────────┘
```

### Panel A — retrieval competition

Two ranked columns animating into a fused list. A chunk that ranked 7th
lexically and 2nd semantically visibly winning the fusion is the core "oh,
*that's* what hybrid does" moment.

Must show:
- the gate as a threshold line, with rejected candidates rendered below it
- the margin between top-1 and top-2, since that drives the rerank decision
- whether the reranker fired, **including when it declined**, with the numbers
- expansion chunks styled distinctly from chunks that won on merit

### Panel B — stage timeline

Horizontal strip: name, ms, tokens, cost, cache hit/miss. **Skipped stages
render greyed, never omitted** — absence is information, and omitting them makes
the pipeline look shorter than it is.

### Panel C — answer with attribution

Hover a clause → its source chunk highlights in panel A. Clauses with no
supporting chunk get flagged. This is the grounding audit, and it's the panel a
compliance or support user would actually use.

### Panel D — counterfactual diff

Side-by-side answers with the outcome category and the computed explanation.

### Five rendering constraints

Each of these silently ruins the panel if unhandled.

1. **Score scales don't share an axis.** BM25 is unbounded (0–20+), cosine is
   0–1, RRF is tiny (~0.03). Encode **rank as position** and render raw scores
   as text. Plotting them on one axis produces a meaningless chart.
2. **Candidate count.** 40 candidates is unreadable. Show ~8 per leg, collapse
   the rest behind a count.
3. **Layout reflow.** Generation finishes 2–4 seconds after retrieval. Reserve
   the answer panel's space up front or the whole layout jumps mid-demo.
4. **Chunk text length.** Chunks are 400–800 tokens. One truncated line,
   expand on click.
5. **Mobile.** LinkedIn traffic is majority mobile and a four-panel grid
   collapses. Record the demo on desktop, but ship a stacked single-column
   fallback or your repo link looks broken to most people who click it.

### Streaming

Stage events over the websocket as they complete, so panels fill progressively.
The progressive fill is a large part of why the demo looks good — don't block on
the full run and render once.

```
→ {"type":"stage", "name":"rewrite",  "ms":210, "skipped":false}
→ {"type":"candidates", "leg":"lexical", "items":[…]}
→ {"type":"candidates", "leg":"semantic", "items":[…]}
→ {"type":"fused", "items":[…], "gate":0.42}
→ {"type":"stage", "name":"rerank", "skipped":true, "skip_reason":"…"}
→ {"type":"answer_delta", "text":"Error 4021 indicates"}
→ {"type":"done", "trace_id":"01JQ…"}
```

### Design direction

Restrained and technical: monospace, dense, dark, precise. An instrument panel,
not a landing page. No gradients, no glass effects, no rounded pastel cards. One
accent colour for the winning path, one for rejection. Numbers aligned.
Whitespace as structure.

A UI project invites judgment on your taste. "Built by someone who debugs
retrieval for a living" reads as senior; "looks like a template" is the failure
mode.

### Demo mode

Ship pre-recorded traces in `web/src/demo/traces/` and a mode that replays them
with realistic timing, no API key needed. Anyone evaluating your work will click
the deployed link, not clone the repo — and a cold-start demo that demands a key
gets closed immediately.

---

## Part F — the eval report

Static markdown, generated after each run, committed to `reports/`. No live
rendering.

```markdown
# Eval report — 2026-07-26

corpus manifest@a1b2c3d · code e4f5a6b · gen claude-sonnet-4-6 · temp 0.0

## Summary
- isolation: 12/12 probes passed
- silent failure: 8 traps · 3 wrong · 1 wrong-and-confident (12.5%)
- judge agreement with human labels: 0.84 (kappa 0.71, n=100)

## Ablation outcomes
[the findings table]

## Regressions vs baseline
none

## Failures
### silent_failure / aof_vs_rdb_confusion — HIGH
answered confidently about a setting that does not exist
trace: reports/traces/01JQ….json
```

Three properties that make it trustworthy: version stamps at the top, the judge
agreement rate stated alongside every judge-derived number, and a trace ID on
every failure so any claim is auditable.

---

## Build phases

Each phase is independently shippable and independently postable. Don't start a
phase before the previous one's acceptance criteria pass.

| Phase | Deliverable | Acceptance criteria |
|---|---|---|
| **0** | corpus ingest + trace schema + `PipelineConfig` | `make ingest` is idempotent; a hand-written trace round-trips through JSON; TS types generate from the schema |
| **1** | pipeline with all stages, tracing, no UI | one query produces a complete valid trace; every stage individually disableable via config |
| **2** | eval runner + isolation suite | `make eval` exits 1 on a deliberately broken config, 0 on the fixed one, 12 probes reporting |
| **3** | silent-failure suite + judge calibration | 8 traps; `reports/judge-calibration.md` exists with a real kappa |
| **4** | counterfactual engine + ablation study | `make ablate` produces the findings table over ≥150 queries |
| **5** | inspector: panels A and B, static trace | loads a trace file from disk and renders correctly; no backend yet |
| **6** | inspector: live streaming, panels C and D | websocket streaming, attribution hover, counterfactual toggle |
| **7** | demo mode, deploy, mobile fallback | keyless public demo; single-column layout under 700px |

**Note the ordering: eval before UI.** The eval is headless, cheap, and
low-risk, and you already have one suite written. The UI is the expensive,
risky part. If you build the UI first and run out of steam you have a pretty
trace viewer with no findings; if you build the eval first and run out of steam
you still have publishable results.

---

## Stack and dependencies

**Backend** — Python 3.12, FastAPI, uvicorn, pydantic v2 (schema validation and
JSON Schema export for TS type generation), qdrant-client, `rank_bm25`,
anthropic + openai SDKs, structlog, pytest, hypothesis (property tests on the
fusion maths).

**Frontend** — React + TypeScript, Vite, Tailwind, Recharts (score
distributions), React Flow or hand-rolled SVG (pipeline graph — hand-rolled is
likely simpler for a fixed 8-stage layout), zod (validate incoming traces at the
boundary).

**Infra** — Qdrant in Docker, Redis for caches, docker compose for local, Fly or
Render for the deployed demo. GitHub Actions running `make eval` on every PR.

**Deliberately not used:** LangChain or LlamaIndex for the pipeline itself. The
entire point is fine-grained control of stage boundaries and tracing, and a
framework's abstractions will fight you on exactly that. Use the SDKs directly.
Say this in the README — "hand-rolled so every stage boundary is observable" is
a defensible engineering position and reads as confidence.

---

## Non-obvious engineering decisions

The things that will bite you. Most of these are single lines that cost a day
each when missed.

**Cache keys must include the config hash.** Otherwise an ablated run hits the
baseline's cached answer and every diff comes back `IDENTICAL`. This is the
subtlest bug in the whole project: your counterfactual engine will appear to
work and will silently prove nothing. Key on
`(config_hash, tenant_id, stage, input_hash)`.

**Embedding caches are shared, answer caches are not.** Embeddings aren't
tenant-specific, so `(model_id, text)` is a safe key. Answer caches keyed on
query text alone leak across tenants — that's the `cache_namespace` probe.

**Semantic caching needs a discriminator guard.** "Config value in Redis 6" and
"config value in Redis 7" sit around 0.98 cosine similarity. Any threshold high
enough to separate them rejects the paraphrases you built the cache for. Gate on
vector similarity *and* an exact match on extracted discriminators — version
numbers, identifiers, dates, polarity words like `not` / `excluding`.

**Temperature 0 is not optional, and assert on it.** Put a runtime check in
`determinism.py` that raises if any generation config has non-zero temperature
during an ablation run.

**Post-filtering is not the same as pre-filtering.** Filter tenants inside the
vector query. Post-filtering silently shrinks your result set and changes recall
in a way that looks like a ranking bug.

**RRF scores are rank-derived, not similarity-derived.** They don't compare
across queries and they shift with `top_k`. Don't threshold on them; don't plot
them on a shared axis with cosine.

**Thread `tenant_id` through the rewrite and expansion paths explicitly.** Both
are second entries into retrieval, both are usually added after the primary path
is correct, and both are where isolation actually breaks.

**Never split a code block when chunking.** The exact-identifier property is
what makes the hybrid demo work at all.

**Refuse to diff traces across version changes.** Raise in the counterfactual
engine when `versions` differs. A diff across a model upgrade looks meaningful
and isn't.

**Store the fully resolved config in the trace, not a diff from defaults.** A
trace must be interpretable years later without the code.

**Judge with a different model family than you generate with**, to blunt
self-preference bias.

**A probe crash is a finding, not a stack trace.** The eval runner catches
exceptions per probe and records them as MEDIUM findings, so one broken probe
doesn't hide the other eleven results.

---

## Content plan

The project exists to produce artifacts worth sharing. Each of these is a
standalone post, roughly in build order.

| # | Post | Peg |
|---|---|---|
| 1 | Why the obvious tenant-isolation test proves nothing | competing-documents methodology + positive controls |
| 2 | The metric nobody publishes: wrong-and-confident rate | the accuracy/flagging inversion |
| 3 | My LLM judge disagrees with me 23% of the time | the calibration report |
| 4 | I turned off half my retrieval pipeline. 12% of answers became confidently wrong | the ablation table — the strongest one |
| 5 | Your guardrail is in the wrong place | rewrite and expansion as second retrieval paths, with the isolation probes as proof |
| 6 | Cache keys, ablations, and a bug that proved nothing | the config-hash cache bug |
| 7 | Demo: watch a correct answer become wrong | the 20-second recording |

### The recording

Twenty seconds, no voiceover, captions only:

1. Type a query containing an exact identifier.
2. The two lists fuse; the winning chunk came from the lexical leg.
3. Answer renders, correct.
4. Toggle `no_lexical`.
5. Answer re-renders — confidently wrong — with the diff showing the chunk
   falling from rank 1 to rank 9.

The argument makes itself. No slide can do this, and it's the entire reason to
build a UI rather than a CLI.

### Framing

Be scrupulous about limits. Publish the corpus, the config, the trace files, and
the judge calibration. Frame findings as failure *classes* rather than verdicts
on any vendor. A benchmark that reads as fair gets shared by the people it
criticises, which is the best distribution there is.

---

## Risks and scope discipline

**The counterfactual engine is the project.** Everything else is a viewer or a
test harness. If phase 4 gets cut, this becomes another trace viewer and stops
being differentiated. Protect it: cut animation, cut ablations to three, cut
panels — never cut phase 4.

**Don't let it become a dashboard.** Cross-request aggregation, user management,
saved sessions, and a settings page are all a different product.

**Don't build a generic eval framework.** Suites on top of a runner. The
differentiation is which failures you know to test for.

**Corpus licensing.** Record the licence per source in the manifest and check it
allows redistribution before committing anything under `corpus/raw/`.

**No client data.** Open-source docs and synthetic tenants only. Nothing from a
production system, no real tenant names, no numbers traceable to a customer —
regardless of who owns the system.

**Scope the traps.** Eight rigorous traps, not thirty sloppy ones. All the
credibility is in reproducibility.

---

## Appendix — config reference

| Field | Default | Notes |
|---|---|---|
| `lexical.top_k` | 20 | candidates from BM25 before fusion |
| `lexical.k1` / `.b` | 1.2 / 0.75 | standard BM25 params |
| `semantic.model_id` | `text-embedding-3-small` | recorded in `versions` |
| `semantic.top_k` | 20 | candidates from dense retrieval |
| `fusion.rrf_k` | 60 | the standard, boring, effective choice |
| `gate.threshold` | 0.42 | tune on your corpus; don't inherit blindly |
| `gate.reads` | `dense_top1` | see the open decision in A.4 |
| `rerank.always` | `false` | `false` = gray-zone only |
| `rerank.gray_zone_score` | 0.60 | rerank when top score is below this |
| `rerank.gray_zone_margin` | 0.10 | or when top-1 minus top-2 is below this |
| `expansion.neighbours` | 1 | chunks either side of a winner |
| `generation.temperature` | 0.0 | never change; asserted at runtime |
| `generation.max_context_chunks` | 12 | context breadth |
| `rewrite_enabled` | `true` | disable to isolate rewrite-path bugs |

### Glossary

**RRF** — Reciprocal Rank Fusion. Combines ranked lists by summing `1/(k+rank)`
per document. Uses rank, not score, which is why it's robust to incomparable
score scales across legs.

**Gray zone** — the region where retrieval confidence is ambiguous enough that
reranking is worth paying for: low top score, or a thin margin over the
runner-up.

**Grounded refusal** — declining to answer because retrieval confidence fell
below the gate, rather than answering from parametric knowledge.

**Ablation** — running the pipeline with one stage disabled, to measure that
stage's contribution.

**Silent failure** — a wrong answer delivered without any expressed
uncertainty. The category that matters commercially.

**Canary** — a unique unguessable token planted in a tenant's document, used to
detect leakage across any output surface.

**Positive control** — a test that fails if isolation is *too* strict, proving
the leak tests aren't passing trivially.
