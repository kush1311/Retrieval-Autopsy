"""Sparse (BM25) retrieval leg, over the same tenant-scoped chunk set.

This is the leg that finds ``KV-4022`` when the query says ``KV-4022``. The dense
leg's tokenizer smears digit runs inside identifiers, so it retrieves the right family
of documents and cannot rank within it. Removing this leg is what turns a correct
answer into a confidently wrong one about the neighbouring error code — which is the
whole ``no_lexical`` demonstration.
"""

from __future__ import annotations

from autopsy.stages.base import Context, State, ensure_candidate
from autopsy.trace import CandidatesEvent


class RetrieveSparseStage:
    name = "retrieve_sparse"

    def skip(self, state: State, ctx: Context) -> str | None:
        if ctx.cfg.lexical is None:
            return "lexical leg ablated by config"
        return None

    def run(self, state: State, ctx: Context) -> State:
        record = ctx.current
        assert ctx.cfg.lexical is not None
        hits = ctx.lexical.search(
            query=state.effective_query,
            tenant_id=ctx.tenant_id,
            top_k=ctx.cfg.lexical.top_k,
            k1=ctx.cfg.lexical.k1,
            b=ctx.cfg.lexical.b,
        )
        for rank, (chunk, score) in enumerate(hits, start=1):
            candidate = ensure_candidate(state, chunk, ctx)
            candidate.lexical_rank = rank
            candidate.lexical_score = round(float(score), 6)
            state.lexical_order.append(chunk.chunk_id)

        if record is not None:
            record.detail = {
                "top_k": ctx.cfg.lexical.top_k,
                "returned": len(hits),
                "k1": ctx.cfg.lexical.k1,
                "b": ctx.cfg.lexical.b,
            }
        ctx.emit(CandidatesEvent(leg="lexical", items=state.ordered(state.lexical_order)))
        return state


__all__ = ["RetrieveSparseStage"]
