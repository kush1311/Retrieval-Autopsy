"""The retrieval-confidence gate: refuse rather than answer from nothing.

**Which score does the gate read?** This was an open question in the spec and it is
settled here as a config field rather than a hardcoded choice, so the alternative can
be measured instead of argued about.

The default is ``dense_top1``: the raw cosine (or, offline, the raw query-coverage) of
the best dense hit. It is interpretable — a human can look at 0.42 and reason about
it — and it is stable when you change ``top_k``.

The alternative, ``fused_top1``, is available and is instructive to run: RRF scores
are derived from rank position, so they have no natural threshold, they shift as the
candidate count changes, and a gate set at ``0.0155`` means nothing to anyone. The
``gate_on_fused`` ablation exists to show that failure rather than assert it.

**One behaviour worth knowing about before it surprises you.** If the configured
signal is unavailable — ``dense_top1`` under ``no_semantic``, for instance — the gate
does *not* silently pass everything. It records that it could not run, and says so in
the trace. Ablating a retrieval leg quietly disabling your refusal logic is itself a
finding, and the trace is where it should become visible.
"""

from __future__ import annotations

from autopsy.stages.base import Context, State
from autopsy.trace import AnswerStatus, RejectedBy

SIGNALS = {
    "dense_top1": ("semantic_score", "semantic_rank"),
    "lexical_top1": ("lexical_score", "lexical_rank"),
    "fused_top1": ("fused_score", "fused_rank"),
}


def read_signal(state: State, reads: str) -> tuple[float | None, float | None]:
    """Return ``(top1, top2)`` for the configured signal, or ``(None, None)``."""
    score_attr, rank_attr = SIGNALS[reads]
    scored = [
        (getattr(c, score_attr), c.chunk_id)
        for c in state.candidates.values()
        if getattr(c, score_attr) is not None and getattr(c, rank_attr) is not None
    ]
    if not scored:
        return None, None
    scored.sort(key=lambda r: (-r[0], r[1]))
    top1 = float(scored[0][0])
    top2 = float(scored[1][0]) if len(scored) > 1 else None
    return top1, top2


class GateStage:
    name = "gate"

    def skip(self, state: State, ctx: Context) -> str | None:
        if ctx.cfg.gate is None:
            return "gate ablated by config — every retrieval will be answered"
        if not state.candidates:
            return "no candidates to score"
        top1, _ = read_signal(state, ctx.cfg.gate.reads)
        if top1 is None:
            return (
                f"gate reads '{ctx.cfg.gate.reads}' but that signal is absent under this "
                "config, so the gate could not run. Retrieval confidence was NOT checked."
            )
        return None

    def run(self, state: State, ctx: Context) -> State:
        record = ctx.current
        assert ctx.cfg.gate is not None
        gate = ctx.cfg.gate
        top1, top2 = read_signal(state, gate.reads)
        assert top1 is not None

        state.gate_signal = gate.reads
        state.gate_value = round(top1, 6)
        state.gate_passed = top1 >= gate.threshold
        margin = round(top1 - top2, 6) if top2 is not None else None

        if record is not None:
            record.detail = {
                "reads": gate.reads,
                "value": state.gate_value,
                "threshold": gate.threshold,
                "margin": margin,
                "passed": state.gate_passed,
            }

        if not state.gate_passed:
            for candidate in state.candidates.values():
                candidate.rejected_by = RejectedBy.GATE
                candidate.in_context = False
            if record is not None:
                record.skip_reason = (
                    f"{gate.reads} {top1:.3f} < gate {gate.threshold:.2f} — refusing "
                    "rather than answering from low-confidence retrieval"
                )
            state.answer = _refusal(gate.reads, top1, gate.threshold)
        return state


def _refusal(reads: str, value: float, threshold: float):
    from autopsy.trace import Answer

    text = (
        "I could not find documentation that answers this with enough confidence to "
        "respond. The closest matches scored below the retrieval threshold, so "
        "answering would mean guessing rather than citing."
    )
    return Answer(
        text=text,
        status=AnswerStatus.REFUSED,
        spans=[],
        citations=[],
        refusal_reason=f"{reads}={value:.3f} below threshold {threshold:.2f}",
        hedged=True,
    )


__all__ = ["GateStage", "SIGNALS", "read_signal"]
