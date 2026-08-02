"""Reciprocal Rank Fusion, and what happens without it.

    score(d) = Σ over legs  1 / (k + rank_leg(d)),   k = 60

RRF combines ranked lists using **rank**, not score. That is the whole trick: BM25 is
unbounded and roughly 0–20 on this corpus, cosine is 0–1, and the two have no shared
unit. Rank is the only quantity both legs agree on.

Three modes, all recorded in the trace so a reader can tell which one produced the
ordering they are looking at:

* ``rrf``        — both legs present and ``fusion`` configured. The normal path.
* ``single_leg`` — one leg ablated. Nothing to fuse; the surviving ranking passes through.
* ``naive_sum``  — ``fusion`` ablated with both legs present. Scores are added
  directly, on their incomparable scales. BM25's unbounded magnitude swamps cosine
  entirely, so this is functionally lexical-only with extra steps. That is not a
  strawman: adding raw scores is the first thing most people try, and its failure is
  invisible because the output still looks like a sensibly ordered list.
"""

from __future__ import annotations

from autopsy.stages.base import Context, State
from autopsy.trace import FusedEvent


def rrf_score(ranks: list[int], k: int) -> float:
    """Σ 1/(k + rank) over the legs a document appeared in."""
    return sum(1.0 / (k + r) for r in ranks)


class FuseStage:
    name = "fuse"

    def skip(self, state: State, ctx: Context) -> str | None:
        if not state.candidates:
            return "no candidates from either leg"
        return None

    def run(self, state: State, ctx: Context) -> State:
        record = ctx.current
        both_legs = ctx.cfg.lexical is not None and ctx.cfg.semantic is not None

        if ctx.cfg.fusion is not None and both_legs:
            mode = "rrf"
            k = ctx.cfg.fusion.rrf_k
            for candidate in state.candidates.values():
                ranks = [
                    r
                    for r in (candidate.lexical_rank, candidate.semantic_rank)
                    if r is not None
                ]
                candidate.fused_score = round(rrf_score(ranks, k), 8)
            detail = {"mode": mode, "rrf_k": k}
        elif both_legs:
            mode = "naive_sum"
            for candidate in state.candidates.values():
                candidate.fused_score = round(
                    (candidate.lexical_score or 0.0) + (candidate.semantic_score or 0.0), 8
                )
            detail = {
                "mode": mode,
                "warning": "raw scores summed across incomparable scales; "
                "BM25 magnitude dominates cosine",
            }
        else:
            mode = "single_leg"
            leg = "lexical" if ctx.cfg.lexical is not None else "semantic"
            for candidate in state.candidates.values():
                rank = candidate.lexical_rank if leg == "lexical" else candidate.semantic_rank
                # Invert rank into a descending score so downstream ordering is
                # uniform regardless of mode.
                candidate.fused_score = round(1.0 / (1 + (rank or 10**6)), 8)
            detail = {"mode": mode, "leg": leg}

        ordered = sorted(
            state.candidates.values(),
            key=lambda c: (-(c.fused_score or 0.0), c.chunk_id),
        )
        for rank, candidate in enumerate(ordered, start=1):
            candidate.fused_rank = rank
        state.fused_order = [c.chunk_id for c in ordered]

        if record is not None:
            record.detail = detail | {"candidates": len(ordered)}
        return state


class FusedEventEmitter:
    """Emitted after the gate so the event can carry the threshold line panel A draws."""

    @staticmethod
    def emit(state: State, ctx: Context) -> None:
        ctx.emit(
            FusedEvent(
                items=state.ordered(state.fused_order),
                gate=ctx.cfg.gate.threshold if ctx.cfg.gate else None,
                gate_reads=state.gate_signal,
                gate_value=state.gate_value,
            )
        )


__all__ = ["FuseStage", "FusedEventEmitter", "rrf_score"]
