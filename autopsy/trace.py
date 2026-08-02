"""The trace schema — the contract between the pipeline and everything else.

Design principle #2: the trace is the product. The pipeline's real output is not the
answer, it is this object. The inspector renders one; the eval asserts over many; the
counterfactual engine diffs two.

Design principle #3: record what *didn't* happen. ``rejected_by`` and ``skip_reason``
are first-class typed fields, not debug strings, because the interesting questions in
retrieval debugging are which candidate was rejected and by what, and which stage was
skipped and why.

Frozen in phase 0. TypeScript types are generated from the JSON Schema exported here
(``python -m autopsy.cli schema``) so the frontend cannot drift from the backend.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"


# --------------------------------------------------------------------------------------
# Enumerations — fixed in phase 0, per the spec
# --------------------------------------------------------------------------------------


class InclusionReason(str, Enum):
    FUSED_TOP_K = "fused_top_k"
    RERANK_PROMOTED = "rerank_promoted"
    NEIGHBOR_EXPANSION = "neighbor_expansion"


class RejectedBy(str, Enum):
    GATE = "gate"
    TOP_K = "top_k"
    RERANK = "rerank"
    TENANT_FILTER = "tenant_filter"


class AnswerStatus(str, Enum):
    GROUNDED = "grounded"
    REFUSED = "refused"
    UNGROUNDED = "ungrounded"


class CacheState(str, Enum):
    HIT = "hit"
    MISS = "miss"


STAGE_ORDER: tuple[str, ...] = (
    "rewrite",
    "embed",
    "retrieve_dense",
    "retrieve_sparse",
    "fuse",
    "gate",
    "rerank",
    "expand",
    "generate",
)


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


class _Model(BaseModel):
    # ``validate_assignment`` matters here: stages assign enum members directly
    # (``candidate.rejected_by = RejectedBy.GATE``), and without it the in-memory value
    # is an enum while the serialised value is a string. Two representations of the
    # same field is how a UI ends up rendering ``RejectedBy.GATE`` at a user.
    model_config = ConfigDict(
        extra="forbid", use_enum_values=True, populate_by_name=True, validate_assignment=True
    )


class Candidate(_Model):
    """One retrieval candidate, with its standing in every leg it appeared in.

    Per-leg rank *and* raw score are both recorded. Rank is what panel A positions by;
    the raw score is rendered as text beside it. They are not interchangeable — see
    the note on score scales in the rendering constraints.
    """

    chunk_id: str
    doc_id: str
    heading_path: list[str] = Field(default_factory=list)
    text: str
    ordinal: int
    tenant_id: str

    lexical_rank: int | None = None
    lexical_score: float | None = None
    semantic_rank: int | None = None
    semantic_score: float | None = None
    fused_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

    final_rank: int | None = None
    in_context: bool = False
    inclusion_reason: InclusionReason | None = None
    rejected_by: RejectedBy | None = None


class StageRecord(_Model):
    """One stage's execution record. Skipped stages are recorded, never omitted —
    absence is information, and an omitted stage makes the pipeline look shorter than
    it is."""

    name: str
    ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cache: CacheState | None = None
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class Span(_Model):
    """A sentence-level slice of the answer mapped back to its supporting chunks.

    Panel C's attribution hover reads this, and so does the grounding check in the
    eval — one field serving the demo and the test suite is the whole point.
    """

    start: int
    end: int
    chunk_ids: list[str] = Field(default_factory=list)
    supported: bool = False


class Answer(_Model):
    text: str
    status: AnswerStatus
    spans: list[Span] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    refusal_reason: str | None = None
    hedged: bool = False


class Totals(_Model):
    ms: float = 0.0
    cost_usd: float = 0.0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


class Trace(_Model):
    schema_version: str = SCHEMA_VERSION
    trace_id: str
    created_at: str

    query: str
    rewritten_query: str | None = None
    tenant_id: str
    session_id: str | None = None

    ablations: list[str] = Field(default_factory=list)
    config_hash: str
    config: dict[str, Any]
    versions: dict[str, str]

    candidates: list[Candidate] = Field(default_factory=list)
    stages: list[StageRecord] = Field(default_factory=list)
    answer: Answer
    totals: Totals = Field(default_factory=Totals)

    # ---- convenience accessors used by the eval suites and the diff engine ----

    def stage(self, name: str) -> StageRecord | None:
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def context_chunks(self) -> list[Candidate]:
        """Candidates that actually reached the generator, in final rank order."""
        inc = [c for c in self.candidates if c.in_context]
        return sorted(inc, key=lambda c: (c.final_rank if c.final_rank is not None else 10**6))

    def candidate(self, chunk_id: str) -> Candidate | None:
        for c in self.candidates:
            if c.chunk_id == chunk_id:
                return c
        return None

    def foreign_chunks(self) -> list[Candidate]:
        """Anything in context whose tenant is neither the query tenant nor global."""
        return [
            c
            for c in self.context_chunks()
            if c.tenant_id not in (self.tenant_id, "tenant_global")
        ]

    # ---- serialisation ----

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "Trace":
        return cls.model_validate(json.loads(raw))

    def write(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.trace_id}.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: str | Path) -> "Trace":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Streaming events — the websocket wire format
# --------------------------------------------------------------------------------------


class StageEvent(_Model):
    type: Literal["stage"] = "stage"
    name: str
    ms: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None
    cache: CacheState | None = None
    cost_usd: float = 0.0


class CandidatesEvent(_Model):
    type: Literal["candidates"] = "candidates"
    leg: Literal["lexical", "semantic"]
    items: list[Candidate]


class FusedEvent(_Model):
    type: Literal["fused"] = "fused"
    items: list[Candidate]
    gate: float | None = None
    gate_reads: str | None = None
    gate_value: float | None = None


class AnswerDeltaEvent(_Model):
    type: Literal["answer_delta"] = "answer_delta"
    text: str


class DoneEvent(_Model):
    type: Literal["done"] = "done"
    trace_id: str
    trace: Trace


class ErrorEvent(_Model):
    type: Literal["error"] = "error"
    message: str


StreamEvent = (
    StageEvent | CandidatesEvent | FusedEvent | AnswerDeltaEvent | DoneEvent | ErrorEvent
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def export_json_schema() -> dict[str, Any]:
    """The schema the TypeScript types are generated from."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "RetrievalAutopsyTrace",
        "schema_version": SCHEMA_VERSION,
        "definitions": {
            "Trace": Trace.model_json_schema(ref_template="#/definitions/{model}"),
            "StageEvent": StageEvent.model_json_schema(ref_template="#/definitions/{model}"),
            "CandidatesEvent": CandidatesEvent.model_json_schema(
                ref_template="#/definitions/{model}"
            ),
            "FusedEvent": FusedEvent.model_json_schema(ref_template="#/definitions/{model}"),
            "AnswerDeltaEvent": AnswerDeltaEvent.model_json_schema(
                ref_template="#/definitions/{model}"
            ),
            "DoneEvent": DoneEvent.model_json_schema(ref_template="#/definitions/{model}"),
        },
    }


__all__ = [
    "SCHEMA_VERSION",
    "STAGE_ORDER",
    "Answer",
    "AnswerDeltaEvent",
    "AnswerStatus",
    "CacheState",
    "Candidate",
    "CandidatesEvent",
    "DoneEvent",
    "ErrorEvent",
    "FusedEvent",
    "InclusionReason",
    "RejectedBy",
    "Span",
    "StageEvent",
    "StageRecord",
    "StreamEvent",
    "Totals",
    "Trace",
    "export_json_schema",
    "utc_now_iso",
]
