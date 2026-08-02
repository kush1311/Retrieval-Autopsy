"""Neighbour expansion, and final context assembly.

Expansion pulls chunks adjacent by ``ordinal`` within the same document, so the
generator sees the sentence before and after the one that matched.

**This is the second high-yield tenant leak vector**, for the same structural reason
as rewrite: expansion fetches by identifier rather than by query, which routes around
the filtered query path entirely. ``Index.neighbours`` is keyed on
``(tenant_id, doc_id)`` so a cross-tenant neighbour is not merely unlikely but
unrepresentable, and ``ensure_candidate`` re-asserts the boundary on the way in. The
``neighbor_expansion`` isolation probe exists to keep both true.

Context assembly lives here too, because expansion is what makes it a trade-off:
neighbours compete with lower-ranked winners for the same ``max_context_chunks``
budget. Depth versus breadth is a real decision and the trace shows which one each
chunk won on, via ``inclusion_reason``.
"""

from __future__ import annotations

from autopsy.stages.base import Context, State, ensure_candidate
from autopsy.trace import InclusionReason, RejectedBy


class ExpandStage:
    name = "expand"

    def skip(self, state: State, ctx: Context) -> str | None:
        if ctx.cfg.expansion is None:
            return "expansion ablated by config"
        if not state.gate_passed:
            return "gate refused; no context to expand"
        if not state.fused_order:
            return "no winners to expand around"
        return None

    def run(self, state: State, ctx: Context) -> State:
        record = ctx.current
        assert ctx.cfg.expansion is not None
        span = ctx.cfg.expansion.neighbours
        budget = ctx.cfg.generation.max_context_chunks

        added = 0
        for chunk_id in state.fused_order[:budget]:
            chunk = ctx.index.get(chunk_id)
            if chunk is None:
                continue
            for neighbour in ctx.index.neighbours(chunk, span):
                if neighbour.chunk_id in state.candidates:
                    continue
                candidate = ensure_candidate(state, neighbour, ctx)
                candidate.inclusion_reason = InclusionReason.NEIGHBOR_EXPANSION
                added += 1

        if record is not None:
            record.detail = {"neighbours": span, "added": added}
        return state


def finalize_context(state: State, ctx: Context) -> None:
    """Choose what the generator actually sees, and label why each chunk is there.

    Winners are taken in rank order; each winner's expansion neighbours follow
    immediately after it, so a document's surrounding context stays adjacent in the
    prompt rather than scattered. The whole list is then cut to
    ``max_context_chunks``.

    Everything that did not make the cut gets ``rejected_by = top_k``. A chunk that
    was a candidate and lost is far more informative than one that was never
    retrieved, which is the entire reason the field exists.
    """
    budget = ctx.cfg.generation.max_context_chunks
    selected: list[str] = []

    for chunk_id in state.fused_order:
        if len(selected) >= budget:
            break
        if chunk_id not in selected:
            selected.append(chunk_id)
        chunk = ctx.index.get(chunk_id)
        if chunk is None or ctx.cfg.expansion is None:
            continue
        for neighbour in ctx.index.neighbours(chunk, ctx.cfg.expansion.neighbours):
            if len(selected) >= budget:
                break
            if neighbour.chunk_id in state.candidates and neighbour.chunk_id not in selected:
                selected.append(neighbour.chunk_id)

    state.context_ids = selected
    chosen = set(selected)
    for rank, chunk_id in enumerate(selected, start=1):
        candidate = state.candidates[chunk_id]
        candidate.in_context = True
        candidate.final_rank = rank
        candidate.rejected_by = None
        if candidate.inclusion_reason is None:
            candidate.inclusion_reason = InclusionReason.FUSED_TOP_K
    for chunk_id, candidate in state.candidates.items():
        if chunk_id in chosen:
            continue
        candidate.in_context = False
        if candidate.rejected_by is None:
            candidate.rejected_by = RejectedBy.TOP_K


__all__ = ["ExpandStage", "finalize_context"]
