"""POST /query and POST /counterfactual."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from autopsy.ablations import ABLATIONS, EXPECTED_FAILURE, all_ablations, compose
from autopsy.counterfactual import classify, explain
from autopsy.pipeline import PipelineError
from autopsy.trace import Trace
from api.state import get_state

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    tenant_id: str = Field(min_length=1, max_length=100)
    ablations: list[str] = Field(default_factory=list, max_length=6)
    history: list[str] = Field(default_factory=list, max_length=20)
    session_id: str | None = None


class CounterfactualResponse(BaseModel):
    baseline: Trace
    variant: Trace
    outcome: str
    explanation: str
    dropped_from_context: list[str]
    rank_delta: dict[str, int]


def _validate(names: list[str]) -> None:
    unknown = [n for part in names for n in part.split("+") if n not in ABLATIONS]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"unknown ablation(s): {unknown}. Known: {sorted(ABLATIONS)}",
        )


def _run(req: QueryRequest, ablations: list[str]) -> Trace:
    state = get_state()
    cfg = compose(ablations, state.config) if ablations else state.config
    try:
        return state.pipeline.run(
            req.query,
            tenant_id=req.tenant_id,
            cfg=cfg,
            history=req.history,
            session_id=req.session_id,
            ablations=ablations,
        )
    except PipelineError as exc:
        # The partial trace is the useful part of a failure; returning only a 500 with
        # a message throws away the record of where it broke.
        raise HTTPException(
            status_code=500,
            detail={"message": str(exc), "trace": exc.trace.model_dump(mode="json")},
        ) from exc


@router.post("/query", response_model=Trace)
def post_query(req: QueryRequest) -> Trace:
    _validate(req.ablations)
    return _run(req, req.ablations)


@router.post("/counterfactual", response_model=CounterfactualResponse)
def post_counterfactual(req: QueryRequest) -> CounterfactualResponse:
    """Run the same query with and without an ablation and diff the two.

    This is the single-query form of the ablation study — identical code path, so the
    thing the demo shows and the thing the report measures cannot diverge.
    """
    if not req.ablations:
        raise HTTPException(status_code=400, detail="counterfactual needs at least one ablation")
    _validate(req.ablations)
    baseline = _run(req, [])
    variant = _run(req, req.ablations)
    outcome = classify(baseline, variant, case=None, judge=None)
    text, rank_delta, dropped = explain(baseline, variant)
    return CounterfactualResponse(
        baseline=baseline,
        variant=variant,
        outcome=outcome.value,
        explanation=text,
        dropped_from_context=dropped,
        rank_delta=rank_delta,
    )


@router.get("/meta")
def get_meta() -> dict[str, Any]:
    state = get_state()
    return {
        "tenants": [t for t in state.index.tenants() if t != "tenant_global"],
        "ablations": [
            {"name": name, "expected": EXPECTED_FAILURE.get(name, "")}
            for name in all_ablations()
        ],
        "corpus": state.index.meta.get("corpus_version"),
        "provider": state.config.runtime.provider,
        "embedder": state.config.runtime.embedder,
        "embed_model": state.index.meta.get("embed_model"),
        "gen_model": state.config.generation.model_id,
        # Surfaced so the header states the gate's signal and threshold rather than
        # leaving a reader to assume a default that may not apply to this embedder.
        "gate_reads": state.config.gate.reads if state.config.gate else None,
        "gate_threshold": state.config.gate.threshold if state.config.gate else None,
        "chunks": len(state.index),
        "vector_backend": os.environ.get("AUTOPSY_VECTOR_BACKEND", "local"),
    }
