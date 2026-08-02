"""The stage protocol, the shared context, and the pipeline state.

Every stage has the same shape:

* ``skip(state, ctx) -> str | None`` — a human-readable reason to skip, or ``None``
  to run. The reason string is rendered straight into the UI, so it is written for a
  reader: *"dense_top1 0.71 > gate 0.42, margin 0.19 > 0.10"*, never *"cond_1"*.
* ``run(state, ctx) -> state`` — do the work.

That uniformity is what makes tracing and ablation generic instead of nine special
cases. Stages never touch globals: everything they need arrives on ``Context``, which
is what lets the counterfactual engine run several configs concurrently against one
index without them seeing each other.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol

from autopsy.cache import EmbeddingCache, SemanticAnswerCache, StageCache
from autopsy.config import PipelineConfig
from autopsy.providers import Embedding, Providers
from autopsy.store import Index, LexicalIndex, VectorStore
from autopsy.trace import Answer, CacheState, Candidate, StageEvent, StageRecord

Emitter = Callable[[Any], None]


def _noop(_event: Any) -> None:
    pass


@dataclass
class Context:
    """Everything a stage is allowed to reach."""

    cfg: PipelineConfig
    config_hash: str
    tenant_id: str
    index: Index
    lexical: LexicalIndex
    vectors: VectorStore
    providers: Providers
    stage_cache: StageCache
    embed_cache: EmbeddingCache
    answer_cache: SemanticAnswerCache | None = None
    session_id: str | None = None
    emit: Emitter = _noop
    records: list[StageRecord] = field(default_factory=list)
    clock: Callable[[], float] = time.perf_counter
    #: The record for the stage currently executing. Set by ``stage_span`` so a stage
    #: can annotate itself without reaching into ``records`` and guessing an index.
    current: StageRecord | None = None

    def record(self, name: str) -> StageRecord | None:
        for r in self.records:
            if r.name == name:
                return r
        return None


@dataclass
class State:
    """Data flowing through the pipeline."""

    query: str
    history: list[str] = field(default_factory=list)
    rewritten_query: str | None = None
    embedding: Embedding | None = None

    candidates: dict[str, Candidate] = field(default_factory=dict)
    lexical_order: list[str] = field(default_factory=list)
    semantic_order: list[str] = field(default_factory=list)
    fused_order: list[str] = field(default_factory=list)

    gate_signal: str | None = None
    gate_value: float | None = None
    gate_passed: bool = True

    rerank_scores: dict[str, float] = field(default_factory=dict)
    context_ids: list[str] = field(default_factory=list)
    answer: Answer | None = None

    @property
    def effective_query(self) -> str:
        """The query retrieval actually uses. Rewrite is a *second entry* into
        retrieval, which is why it gets its own name rather than quietly overwriting
        ``query`` — a trace has to show both."""
        return self.rewritten_query or self.query

    def ordered(self, ids: list[str]) -> list[Candidate]:
        return [self.candidates[i] for i in ids if i in self.candidates]


class Stage(Protocol):
    name: str

    def skip(self, state: State, ctx: Context) -> str | None: ...

    def run(self, state: State, ctx: Context) -> State: ...


@contextmanager
def stage_span(ctx: Context, name: str) -> Iterator[StageRecord]:
    """Time a stage, append its record, and emit its stream event.

    The record is appended even when the stage raises. A stage that blew up must still
    appear in the timeline with its error attached — an exception that deletes the
    evidence of where it happened is the least useful kind.
    """
    record = StageRecord(name=name)
    # Appended up front, not in the finally block. A stage annotates itself via
    # ``ctx.current`` while it runs, and a stage that raises must still leave its
    # record — and everything it managed to record before failing — in the trace.
    ctx.records.append(record)
    ctx.current = record
    start = ctx.clock()
    try:
        yield record
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
        record.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record.ms = round((ctx.clock() - start) * 1000, 3)
        ctx.current = None
        ctx.emit(
            StageEvent(
                name=record.name,
                ms=record.ms,
                skipped=record.skipped,
                skip_reason=record.skip_reason,
                cache=CacheState(record.cache) if record.cache else None,
                cost_usd=record.cost_usd,
            )
        )


def execute(stage: Stage, state: State, ctx: Context) -> State:
    """Run one stage, honouring its own skip decision and recording either way."""
    with stage_span(ctx, stage.name) as record:
        reason = stage.skip(state, ctx)
        if reason is not None:
            record.skipped = True
            record.skip_reason = reason
            return state
        return stage.run(state, ctx)


def ensure_candidate(state: State, chunk, ctx: Context) -> Candidate:
    """Get or create the accumulator for a chunk.

    Asserts the tenant boundary at the point of creation. Every candidate in a trace
    passes through here, so this single check covers both retrieval legs, neighbour
    expansion, and anything added later — which is the point of funnelling them.
    """
    existing = state.candidates.get(chunk.chunk_id)
    if existing is not None:
        return existing
    if chunk.tenant_id not in (ctx.tenant_id, "tenant_global"):
        raise PermissionError(
            f"chunk {chunk.chunk_id} belongs to {chunk.tenant_id} but the query is for "
            f"{ctx.tenant_id}. A candidate reached the pipeline through an unfiltered "
            "path — this is a tenant isolation failure, not a ranking bug."
        )
    candidate = Candidate(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        heading_path=list(chunk.heading_path),
        text=chunk.text,
        ordinal=chunk.ordinal,
        tenant_id=chunk.tenant_id,
    )
    state.candidates[chunk.chunk_id] = candidate
    return candidate


__all__ = [
    "Context",
    "Emitter",
    "Stage",
    "State",
    "ensure_candidate",
    "execute",
    "stage_span",
]
