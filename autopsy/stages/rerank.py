"""LLM-as-reranker, fired only in the gray zone.

The gray zone is where retrieval confidence is ambiguous enough that paying for a
model call is worth it: a low top score, **or** a thin margin between top-1 and top-2.
A confident, well-separated result gets no reranker and no bill.

**Visualising the skip is the differentiator.** Almost no public demo shows a system
declining to spend money, and "we ran the expensive model on 11% of queries, here are
the numbers it used to decide" is a more interesting thing to show an engineer than a
reranked list. That is why ``skip_reason`` carries the actual figures.
"""

from __future__ import annotations

from autopsy.providers import SourceChunk
from autopsy.stages.base import Context, State
from autopsy.stages.gate import read_signal
from autopsy.trace import InclusionReason


class RerankStage:
    name = "rerank"

    def skip(self, state: State, ctx: Context) -> str | None:
        if ctx.cfg.rerank is None:
            return "reranker ablated by config"
        if not state.gate_passed:
            return "gate refused; nothing to rerank"
        if not state.fused_order:
            return "no candidates to rerank"
        if ctx.cfg.rerank.always:
            return None

        reads = ctx.cfg.gate.reads if ctx.cfg.gate else "dense_top1"
        top1, top2 = read_signal(state, reads)
        if top1 is None:
            return f"no '{reads}' signal available to judge the gray zone; reranking anyway"

        cfg = ctx.cfg.rerank
        # Both bounds are relative, so they stay meaningful whichever signal
        # `gate.reads` names. See RerankConfig for why absolutes break here.
        gate_threshold = ctx.cfg.gate.threshold if ctx.cfg.gate else top1
        score_bound = gate_threshold * cfg.gray_zone_ratio
        margin = (top1 - top2) if top2 is not None else float("inf")
        margin_bound = abs(top1) * cfg.gray_zone_margin_ratio

        if top1 < score_bound or margin < margin_bound:
            return None
        margin_text = "n/a (single candidate)" if top2 is None else f"{margin:.2f}"
        return (
            f"{reads} {top1:.2f} > {score_bound:.2f} (gate {gate_threshold:.2f} × "
            f"{cfg.gray_zone_ratio}) and margin {margin_text} > {margin_bound:.2f} — "
            "retrieval is confident, skipping the reranker"
        )

    def run(self, state: State, ctx: Context) -> State:
        record = ctx.current
        assert ctx.cfg.rerank is not None
        cfg = ctx.cfg.rerank

        shortlist_ids = state.fused_order[: cfg.top_n]
        shortlist = [
            SourceChunk(
                n=i,
                chunk_id=c.chunk_id,
                heading_path=list(c.heading_path),
                text=c.text,
            )
            for i, c in enumerate(state.ordered(shortlist_ids), start=1)
        ]

        key = ctx.stage_cache.key(
            config_hash=ctx.config_hash,
            tenant_id=ctx.tenant_id,
            stage=self.name,
            payload=[state.effective_query, shortlist_ids],
        )
        (scores, usage), cache_state = ctx.stage_cache.get_or_compute(
            key,
            lambda: ctx.providers.llm.rerank(
                query=state.effective_query, candidates=shortlist, model_id=cfg.model_id
            ),
        )

        before = {cid: i for i, cid in enumerate(state.fused_order)}
        for chunk_id, score in scores.items():
            candidate = state.candidates.get(chunk_id)
            if candidate is not None:
                candidate.rerank_score = round(float(score), 3)

        # Reranked candidates sort ahead of everything the reranker never saw, so a
        # shortlist cut-off can never accidentally promote an unscored chunk above a
        # scored one.
        state.fused_order = sorted(
            state.fused_order,
            key=lambda cid: (
                state.candidates[cid].rerank_score is None,
                -(state.candidates[cid].rerank_score or 0.0),
                -(state.candidates[cid].fused_score or 0.0),
                cid,
            ),
        )
        promoted = [
            cid
            for i, cid in enumerate(state.fused_order)
            if state.candidates[cid].rerank_score is not None and i < before.get(cid, 10**6)
        ]
        for cid in promoted:
            state.candidates[cid].inclusion_reason = InclusionReason.RERANK_PROMOTED
        state.rerank_scores = dict(scores)

        if record is not None:
            record.cache = cache_state
            record.tokens_in = usage.tokens_in
            record.tokens_out = usage.tokens_out
            record.cost_usd = usage.cost_usd
            record.detail = {
                "scored": len(scores),
                "shortlist": len(shortlist),
                "promoted": promoted,
                "trigger": "always" if cfg.always else "gray_zone",
            }
        return state


__all__ = ["RerankStage"]
