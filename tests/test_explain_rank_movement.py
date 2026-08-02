"""`explain()` must not narrate a fall that did not happen.

The committed report carried this sentence:

    `c_29870e4099cef1c0` (Atlas checkpointing > Overview) fell from fused rank 1 to
    rank 1 and dropped out of context.

Rank 1 to rank 1 is not a fall. The single template covered every case, so an unchanged
rank was still described as falling — which points a reader at the wrong cause. The chunk
was displaced by something promoted above it, or by expansion filling the window; it was
not out-ranked.

There was already a guard for the budget-shrank case, with a comment saying exactly this.
It only fired when `max_context_chunks` changed, and `no_gate` does not change it.
"""

from __future__ import annotations

import pytest

from autopsy.counterfactual import explain
from autopsy.trace import Answer, Candidate, Totals, Trace


def _trace(cands: list[Candidate], *, budget: int = 3) -> Trace:
    return Trace(
        trace_id="T" * 26, created_at="2026-08-02T00:00:00Z", query="q",
        tenant_id="tenant_kelvin", config_hash="sha256:x",
        config={"generation": {"max_context_chunks": budget}},
        versions={"corpus": "c", "code": "d", "provider": "offline"},
        candidates=cands,
        answer=Answer(text="a", status="grounded"), totals=Totals(),
    )


def _cand(cid: str, *, fused: int | None, ctx: bool, lex: int | None = 1,
          sem: int | None = 2) -> Candidate:
    return Candidate(
        chunk_id=cid, doc_id="reference/checkpointing.md",
        heading_path=["Atlas checkpointing", "Overview"], text="…", ordinal=0,
        tenant_id="tenant_kelvin",
        lexical_rank=lex, lexical_score=16.15 if lex else None,
        semantic_rank=sem, semantic_score=0.71 if sem else None,
        fused_rank=fused, fused_score=0.03,
        final_rank=fused, in_context=ctx,
    )


def test_an_unchanged_rank_is_never_described_as_a_fall():
    """The exact shape from reports/ablation.md line 29."""
    base = _trace([_cand("c_29870e4099cef1c0", fused=1, ctx=True)])
    var = _trace([_cand("c_29870e4099cef1c0", fused=1, ctx=False)])

    text, _delta, dropped = explain(base, var)

    assert "c_29870e4099cef1c0" in text
    assert "fell from fused rank 1 to rank 1" not in text
    assert "fell" not in text, f"unchanged rank still narrated as a fall: {text}"
    assert "held fused rank 1" in text
    assert "displaced" in text
    assert dropped == ["c_29870e4099cef1c0"]


def test_a_genuine_fall_still_says_fell():
    base = _trace([_cand("c_a", fused=1, ctx=True)])
    var = _trace([_cand("c_a", fused=9, ctx=False)])
    text, _d, _dr = explain(base, var)
    assert "fell from fused rank 1 to rank 9" in text


def test_leaving_the_candidate_list_is_named_as_such():
    base = _trace([_cand("c_a", fused=1, ctx=True)])
    var = _trace([])
    text, _d, _dr = explain(base, var)
    assert "left the candidate list entirely" in text
    assert "fell" not in text


def test_an_unranked_survivor_is_distinguished_from_a_disappearance():
    base = _trace([_cand("c_a", fused=1, ctx=True)])
    var = _trace([_cand("c_a", fused=None, ctx=False)])
    text, _d, _dr = explain(base, var)
    assert "went unranked" in text
    assert "left the candidate list" not in text


def test_a_rank_that_improved_but_still_dropped_is_not_called_a_fall():
    """Possible when a shorter window cuts a chunk that rose. Rare, and the old template
    would have said 'fell from rank 4 to rank 2', which is simply false."""
    base = _trace([_cand("c_a", fused=4, ctx=True)])
    var = _trace([_cand("c_a", fused=2, ctx=False)])
    text, _d, _dr = explain(base, var)
    assert "rose from fused rank 4 to rank 2" in text
    assert "fell" not in text


def test_the_budget_guard_still_takes_precedence():
    """When max_context_chunks shrank, the cause is the budget, not the ranking."""
    base = _trace([_cand("c_a", fused=1, ctx=True), _cand("c_b", fused=2, ctx=True)],
                  budget=3)
    var = _trace([_cand("c_a", fused=1, ctx=True), _cand("c_b", fused=2, ctx=False)],
                 budget=1)
    text, _d, _dr = explain(base, var)
    assert "context budget shrank" in text
    assert "fell" not in text


@pytest.mark.parametrize("b_rank,v_rank", [(1, 1), (1, 9), (4, 2), (1, None)])
def test_no_wording_contradicts_the_numbers_it_quotes(b_rank, v_rank):
    """Whatever the case, the sentence must not claim a direction the ranks deny."""
    base = _trace([_cand("c_a", fused=b_rank, ctx=True)])
    var = _trace([_cand("c_a", fused=v_rank, ctx=False)])
    text, _d, _dr = explain(base, var)
    if v_rank is not None and b_rank is not None:
        if v_rank == b_rank:
            assert "fell" not in text and "rose" not in text
        elif v_rank > b_rank:
            assert "rose" not in text
        else:
            assert "fell" not in text
