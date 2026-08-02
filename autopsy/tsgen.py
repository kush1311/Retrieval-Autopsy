"""Generate the frontend's types from the Python models.

The trace schema is the contract between the pipeline and everything else, so the
frontend must not hold a second, hand-maintained copy of it. Adding a field in
``trace.py`` and forgetting to add it in TypeScript is a silent, compiling bug: the
panel just renders nothing where the new field should be.

This emits **Zod schemas**, with the TypeScript types inferred from them
(``z.infer``). Emitting bare types would give compile-time safety and nothing at
runtime — and the boundary that matters here is runtime, because the frontend also
loads pre-recorded trace files from disk in demo mode, where nothing has validated
them since they were written. One generated artifact, both jobs, no duplication.

    python -m autopsy.cli schema --ts web/src/lib/trace.ts
"""

from __future__ import annotations

import types as pytypes
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from autopsy import trace as trace_module

HEADER = """// GENERATED FILE — do not edit.
//
// Regenerate with:  python -m autopsy.cli schema --ts web/src/lib/trace.ts
//
// Source of truth is autopsy/trace.py. Zod schemas rather than bare types because
// demo mode loads pre-recorded trace JSON straight off disk, and nothing has
// validated those since the moment they were written.

import { z } from "zod";
"""

#: Emitted in dependency order — Zod is eager, so a schema must exist before it is
#: referenced.
MODELS: list[type[BaseModel]] = [
    trace_module.Candidate,
    trace_module.StageRecord,
    trace_module.Span,
    trace_module.Answer,
    trace_module.Totals,
    trace_module.Trace,
    trace_module.StageEvent,
    trace_module.CandidatesEvent,
    trace_module.FusedEvent,
    trace_module.AnswerDeltaEvent,
    trace_module.DoneEvent,
    trace_module.ErrorEvent,
]

ENUMS: list[type[Enum]] = [
    trace_module.InclusionReason,
    trace_module.RejectedBy,
    trace_module.AnswerStatus,
    trace_module.CacheState,
]


class UnsupportedAnnotation(TypeError):
    """Raised rather than guessed.

    A silently mistyped field is worse than a failed build: the generator would emit
    something plausible, the frontend would compile, and the mismatch would only show
    up as a panel rendering nothing.
    """


def _zod(annotation: Any) -> str:
    if annotation is Any:
        return "z.unknown()"
    if annotation is type(None):
        return "z.null()"
    if annotation is str:
        return "z.string()"
    if annotation in (int, float):
        return "z.number()"
    if annotation is bool:
        return "z.boolean()"

    origin = get_origin(annotation)

    if origin is Literal:
        values = get_args(annotation)
        if len(values) == 1:
            return f'z.literal("{values[0]}")'
        return "z.enum([" + ", ".join(f'"{v}"' for v in values) + "])"

    if origin in (Union, pytypes.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        nullable = len(args) != len(get_args(annotation))
        inner = _zod(args[0]) if len(args) == 1 else (
            "z.union([" + ", ".join(_zod(a) for a in args) + "])"
        )
        return f"{inner}.nullable()" if nullable else inner

    if origin in (list, set, tuple):
        (item,) = get_args(annotation) or (Any,)
        return f"z.array({_zod(item)})"

    if origin is dict:
        args = get_args(annotation) or (str, Any)
        return f"z.record(z.string(), {_zod(args[1])})"

    if isinstance(annotation, type):
        if issubclass(annotation, Enum):
            return f"{annotation.__name__}Schema"
        if issubclass(annotation, BaseModel):
            return f"{annotation.__name__}Schema"

    raise UnsupportedAnnotation(
        f"tsgen cannot express {annotation!r}. Add a case rather than letting the "
        "generator guess — a wrong type here compiles and fails silently."
    )


def _enum_block(enum: type[Enum]) -> str:
    values = ", ".join(f'"{member.value}"' for member in enum)
    return (
        f"export const {enum.__name__}Schema = z.enum([{values}]);\n"
        f"export type {enum.__name__} = z.infer<typeof {enum.__name__}Schema>;\n"
    )


def _model_block(model: type[BaseModel]) -> str:
    lines = [f"export const {model.__name__}Schema = z.object({{"]
    for name, field in model.model_fields.items():
        expr = _zod(field.annotation)
        if not field.is_required():
            expr += f".default({_default(field.get_default(call_default_factory=True))})"
        lines.append(f"  {name}: {expr},")
    lines.append("});")
    lines.append(
        f"export type {model.__name__} = z.infer<typeof {model.__name__}Schema>;"
    )
    return "\n".join(lines) + "\n"


def _default(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        return "[]"
    if isinstance(value, dict):
        return "{}"
    if isinstance(value, Enum):
        return f'"{value.value}"'
    return "undefined"


def generate() -> str:
    parts = [HEADER, ""]
    for enum in ENUMS:
        parts.append(_enum_block(enum))
    for model in MODELS:
        parts.append(_model_block(model))
    parts.append(
        "export const StreamEventSchema = z.discriminatedUnion(\"type\", [\n"
        "  StageEventSchema,\n"
        "  CandidatesEventSchema,\n"
        "  FusedEventSchema,\n"
        "  AnswerDeltaEventSchema,\n"
        "  DoneEventSchema,\n"
        "  ErrorEventSchema,\n"
        "]);\n"
        "export type StreamEvent = z.infer<typeof StreamEventSchema>;\n"
    )
    parts.append(
        "/** Parse a trace from an untrusted source (a demo file, a paste, an old run). */\n"
        "export function parseTrace(raw: unknown): Trace {\n"
        "  return TraceSchema.parse(raw);\n"
        "}\n"
    )
    return "\n".join(parts)


__all__ = ["MODELS", "ENUMS", "UnsupportedAnnotation", "generate"]
