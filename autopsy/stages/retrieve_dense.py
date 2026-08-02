"""Dense retrieval leg.

The tenant filter lives inside the query (see ``store/vectors.py``), never as a
post-filter over the results. Post-filtering silently shrinks the result set and
changes recall in a way that presents as a ranking problem — you go looking for a
scoring bug and the actual fault is that ``top_k`` was spent on documents the caller
was never allowed to see.
"""

from __future__ import annotations

from autopsy.stages.base import Context, State, ensure_candidate
from autopsy.trace import CandidatesEvent


class RetrieveDenseStage:
    name = "retrieve_dense"

    def skip(self, state: State, ctx: Context) -> str | None:
        if ctx.cfg.semantic is None:
            return "semantic leg ablated by config"
        if state.embedding is None:
            return "no query embedding available"
        return None

    def run(self, state: State, ctx: Context) -> State:
        record = ctx.current
        assert ctx.cfg.semantic is not None and state.embedding is not None
        hits = ctx.vectors.search(
            query=state.embedding, tenant_id=ctx.tenant_id, top_k=ctx.cfg.semantic.top_k
        )
        for rank, (chunk, score) in enumerate(hits, start=1):
            candidate = ensure_candidate(state, chunk, ctx)
            candidate.semantic_rank = rank
            candidate.semantic_score = round(float(score), 6)
            state.semantic_order.append(chunk.chunk_id)

        if record is not None:
            record.detail = {
                "top_k": ctx.cfg.semantic.top_k,
                "returned": len(hits),
                "scope": f"{ctx.tenant_id} + tenant_global",
            }
        ctx.emit(CandidatesEvent(leg="semantic", items=state.ordered(state.semantic_order)))
        return state


__all__ = ["RetrieveDenseStage"]
