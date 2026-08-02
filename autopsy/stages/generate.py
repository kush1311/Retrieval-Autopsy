"""Generation: answer strictly from numbered sources, and prove it.

The stage's real output is not the prose — it is the prose *plus* sentence-level spans
mapping answer text back to chunk IDs. One field, two consumers: panel C's attribution
hover renders it, and the eval's grounding check asserts on it. Building the demo
feature and the test on the same data is what keeps them honest about each other.

``answer.status`` distinguishes three outcomes that are often collapsed into one:

* ``grounded``   — every sentence has a supporting chunk.
* ``refused``    — the system declined. A good outcome when evidence is thin.
* ``ungrounded`` — it answered, and at least one sentence has no support. This is the
  one that matters commercially, and it is the reason ``ungrounded`` is a first-class
  status rather than a warning attached to ``grounded``.
"""

from __future__ import annotations

from autopsy.providers import SourceChunk
from autopsy.stages.base import Context, State
from autopsy.trace import Answer, AnswerStatus, Span
from autopsy.textutil import is_hedged


class GenerateStage:
    name = "generate"

    def skip(self, state: State, ctx: Context) -> str | None:
        if state.answer is not None:
            return "gate already produced a grounded refusal; no generation needed"
        return None

    def run(self, state: State, ctx: Context) -> State:
        record = ctx.current
        gen = ctx.cfg.generation
        sources = [
            SourceChunk(
                n=i,
                chunk_id=c.chunk_id,
                heading_path=list(c.heading_path),
                text=c.text,
            )
            for i, c in enumerate(state.ordered(state.context_ids), start=1)
        ]

        key = ctx.stage_cache.key(
            config_hash=ctx.config_hash,
            tenant_id=ctx.tenant_id,
            stage=self.name,
            payload=[state.effective_query, [s.chunk_id for s in sources]],
        )
        result, cache_state = ctx.stage_cache.get_or_compute(
            key,
            lambda: ctx.providers.llm.generate(
                query=state.effective_query,
                sources=sources,
                model_id=gen.model_id,
                temperature=gen.temperature,
                max_tokens=gen.max_tokens,
                discriminator_guard=gen.discriminator_guard,
            ),
        )

        spans = [
            Span(
                start=start,
                end=end,
                chunk_ids=list(chunk_ids),
                supported=bool(chunk_ids),
            )
            for start, end, chunk_ids in result.spans
        ]
        unsupported = [s for s in spans if not s.supported]

        if result.refused:
            status = AnswerStatus.REFUSED
        elif unsupported:
            status = AnswerStatus.UNGROUNDED
        else:
            status = AnswerStatus.GROUNDED

        state.answer = Answer(
            text=result.text,
            status=status,
            spans=spans,
            citations=list(result.citations),
            hedged=result.hedged or is_hedged(result.text),
            refusal_reason="generator found no supporting evidence" if result.refused else None,
        )

        if record is not None:
            record.cache = cache_state
            record.tokens_in = result.usage.tokens_in
            record.tokens_out = result.usage.tokens_out
            record.cost_usd = result.usage.cost_usd
            record.detail = {
                "sources": len(sources),
                "status": status.value,
                "unsupported_spans": len(unsupported),
                "temperature": gen.temperature,
            }
        return state


__all__ = ["GenerateStage"]
