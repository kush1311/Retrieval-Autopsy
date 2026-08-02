"""The pipeline: ``(query, tenant, config) -> (answer, Trace)``.

Nine stages, every one optional, every one described by config. There is no branch in
this file that reads an environment variable or a global; if you want different
behaviour you pass a different config, which is what makes an ablation a data
transform rather than a code path.

The stage order is fixed because the data dependencies are real:

    rewrite → embed → retrieve(dense ∥ sparse) → fuse → gate → rerank? → expand → generate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from autopsy.ablations import normalise
from autopsy.cache import EmbeddingCache, SemanticAnswerCache, StageCache
from autopsy.config import PipelineConfig, default_config
from autopsy.determinism import (
    assert_deterministic,
    config_hash,
    content_id,
    resolved_config,
    set_seeds,
    versions_block,
)
from autopsy.providers import build_providers
from autopsy.stages import (
    Context,
    EmbedStage,
    ExpandStage,
    FuseStage,
    FusedEventEmitter,
    GateStage,
    GenerateStage,
    RerankStage,
    RetrieveDenseStage,
    RetrieveSparseStage,
    RewriteStage,
    State,
    execute,
    finalize_context,
)
from autopsy.store import Index, LexicalIndex, build_vector_store
from autopsy.trace import (
    STAGE_ORDER,
    Answer,
    AnswerStatus,
    Candidate,
    DoneEvent,
    ErrorEvent,
    StageRecord,
    Totals,
    Trace,
    utc_now_iso,
)


class PipelineError(RuntimeError):
    """A stage failed. Carries the partial trace, because a failure you cannot see
    inside is a failure you cannot fix."""

    def __init__(self, message: str, trace: Trace) -> None:
        super().__init__(message)
        self.trace = trace


@dataclass
class Pipeline:
    index: Index
    answer_cache: SemanticAnswerCache | None = None
    _lexical: LexicalIndex = field(init=False, repr=False)
    _vectors: Any = field(init=False, repr=False)
    #: Shared across every run and every ablation. See ``cache.py`` for why this one
    #: is safe to share and the stage cache is not.
    embed_cache: EmbeddingCache = field(default_factory=EmbeddingCache)
    stage_cache: StageCache = field(default_factory=StageCache)

    def __post_init__(self) -> None:
        self._lexical = LexicalIndex(self.index)
        self._vectors = build_vector_store(self.index, self.index.stats)

    def close(self) -> None:
        """Release the vector store's resources.

        Only the embedded-Qdrant backend holds anything that needs releasing — an
        exclusive file lock — but callers should not have to know which backend they got.
        Safe to call more than once, and a no-op for the in-process store.
        """
        close = getattr(self._vectors, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ----------------------------------------------------------------------------

    def run(
        self,
        query: str,
        *,
        tenant_id: str,
        cfg: PipelineConfig | None = None,
        history: list[str] | None = None,
        session_id: str | None = None,
        ablations: list[str] | None = None,
        emit: Callable[[Any], None] | None = None,
        generate: bool = True,
    ) -> Trace:
        """Run the pipeline and return its trace.

        ``generate=False`` stops after context assembly. That is not a debug
        convenience — it is the only affordable way to run a full ablation sweep on a
        free tier.

        An ablation can only change the answer if it changes what reaches the
        generator. If the context set is identical, the answer is identical by
        construction, and paying a model call to rediscover that wastes quota that is
        capped per day. Retrieval-only runs cost nothing, so the sweep can measure
        every case's *sensitivity* first and spend generation tokens only where the
        evidence actually moved.
        """
        cfg = cfg or default_config()
        assert_deterministic(cfg)
        set_seeds(cfg.runtime.seed)

        chash = config_hash(cfg)
        names = normalise(ablations)
        emitter = emit or (lambda _e: None)

        ctx = Context(
            cfg=cfg,
            config_hash=chash,
            tenant_id=tenant_id,
            index=self.index,
            lexical=self._lexical,
            vectors=self._vectors,
            providers=build_providers(cfg, self.index.stats),
            stage_cache=self.stage_cache if cfg.runtime.cache_enabled else StageCache(enabled=False),
            embed_cache=self.embed_cache if cfg.runtime.cache_enabled else EmbeddingCache(enabled=False),
            answer_cache=self.answer_cache,
            session_id=session_id,
            emit=emitter,
        )
        state = State(query=query, history=list(history or []))

        error: Exception | None = None
        try:
            for stage in (
                RewriteStage(),
                EmbedStage(),
                RetrieveDenseStage(),
                RetrieveSparseStage(),
                FuseStage(),
                GateStage(),
            ):
                state = execute(stage, state, ctx)

            FusedEventEmitter.emit(state, ctx)

            for stage in (RerankStage(), ExpandStage()):
                state = execute(stage, state, ctx)

            if state.gate_passed:
                finalize_context(state, ctx)
            if generate:
                state = execute(GenerateStage(), state, ctx)
        except Exception as exc:  # noqa: BLE001 - re-raised with the trace attached
            error = exc

        trace = self._build_trace(
            state=state, ctx=ctx, cfg=cfg, chash=chash, ablations=names,
            session_id=session_id, error=error,
        )
        if error is not None:
            emitter(ErrorEvent(message=str(error)))
            raise PipelineError(str(error), trace) from error

        emitter(DoneEvent(trace_id=trace.trace_id, trace=trace))
        return trace

    # ----------------------------------------------------------------------------

    def _build_trace(
        self, *, state: State, ctx: Context, cfg: PipelineConfig, chash: str,
        ablations: list[str], session_id: str | None, error: Exception | None,
    ) -> Trace:
        records = _ordered_records(ctx.records)
        answer = state.answer or Answer(
            text=(
                f"The pipeline failed before producing an answer: {error}"
                if error
                else "No answer was produced."
            ),
            status=AnswerStatus.UNGROUNDED,
            refusal_reason=str(error) if error else None,
        )
        candidates = sorted(
            state.candidates.values(),
            key=lambda c: (
                c.final_rank if c.final_rank is not None else 10**6,
                c.fused_rank if c.fused_rank is not None else 10**6,
                c.chunk_id,
            ),
        )
        totals = Totals(
            ms=round(sum(r.ms for r in records), 3),
            cost_usd=round(sum(r.cost_usd for r in records), 8),
            llm_calls=sum(1 for r in records if r.tokens_out and not r.skipped),
            tokens_in=sum(r.tokens_in for r in records),
            tokens_out=sum(r.tokens_out for r in records),
        )
        return Trace(
            trace_id=content_id(chash, ctx.tenant_id, state.query, ablations, state.history),
            created_at=utc_now_iso(),
            query=state.query,
            rewritten_query=state.rewritten_query,
            tenant_id=ctx.tenant_id,
            session_id=session_id,
            ablations=ablations,
            config_hash=chash,
            config=resolved_config(cfg),
            versions=versions_block(
                cfg,
                self.index.meta.get("corpus_version", "unknown"),
                # The model the vectors were actually built with, not the one the
                # config requested — those differ whenever the index is stale.
                embed_model=self.index.meta.get("embed_model"),
            ),
            candidates=candidates,
            stages=records,
            answer=answer,
            totals=totals,
        )


def _ordered_records(records: list[StageRecord]) -> list[StageRecord]:
    """Canonical stage order, with any stage that never ran filled in as skipped.

    Skipped stages are rendered greyed rather than omitted, so the timeline always has
    the same nine slots. A pipeline that *looks* shorter under an ablation is exactly
    the wrong impression to leave — the stage did not disappear, it declined to run,
    and the reason is the interesting part.
    """
    seen = {r.name: r for r in records}
    out: list[StageRecord] = []
    for name in STAGE_ORDER:
        record = seen.pop(name, None)
        out.append(
            record
            if record is not None
            else StageRecord(name=name, skipped=True, skip_reason="stage did not run")
        )
    out.extend(seen.values())
    return out


__all__ = ["Pipeline", "PipelineError"]
