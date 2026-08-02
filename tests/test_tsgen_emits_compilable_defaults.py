"""The TypeScript generator must not emit code that cannot compile.

There is no Node in this project's toolchain, so `make schema` was the *only* check on
`tsgen.py` output — and `make schema` only checks that a file was written. It happily wrote

    totals: TotalsSchema.default(undefined),

which Zod rejects outright:

    error TS2769: No overload matches this call.
      Argument of type 'undefined' is not assignable to parameter of type '{...}'

The cause was a `return "undefined"` catch-all in `_default()`. `totals` is declared
`Field(default_factory=Totals)`, so its default is a `BaseModel` instance — a type the
function had no case for — and the fallback turned that into a string that looks like code.
Generation succeeded, `make schema` reported success, and the breakage surfaced only when
CI compiled a frontend nobody had ever built.

These tests are the substitute for a compiler. They cannot type-check TypeScript; they can
establish that every default is a real literal, that it matches the Python default it claims
to represent, and that an unhandled type fails loudly at generation time instead of
producing plausible-looking garbage.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pytest

from autopsy import tsgen
from autopsy.trace import Totals, Trace

GENERATED = Path(__file__).resolve().parent.parent / "web" / "src" / "lib" / "trace.ts"


def test_no_default_is_the_literal_undefined() -> None:
    """THE regression, checked against the committed artifact."""
    source = GENERATED.read_text(encoding="utf-8")
    offenders = [
        f"line {i}: {line.strip()}"
        for i, line in enumerate(source.splitlines(), 1)
        if "default(undefined)" in line
    ]
    assert not offenders, (
        "`.default(undefined)` does not compile under Zod:\n  " + "\n  ".join(offenders)
    )


def test_every_model_field_renders_without_the_fallback() -> None:
    """Checks the generator directly, so a new model with an awkward default is caught
    before anyone regenerates and commits."""
    bad = []
    for model in tsgen.MODELS:
        for name, field in model.model_fields.items():
            if field.is_required():
                continue
            rendered = tsgen._default(
                field.get_default(call_default_factory=True),
                where=f"{model.__name__}.{name}",
            )
            if "undefined" in rendered:
                bad.append(f"{model.__name__}.{name} -> {rendered}")
    assert not bad, "these defaults render as undefined: " + "; ".join(bad)


def test_an_unsupported_default_raises_instead_of_emitting_undefined() -> None:
    """The behaviour change that matters. A generator that emits broken code and exits 0
    is worse than one that refuses, because the failure lands somewhere else entirely."""
    with pytest.raises(TypeError, match="cannot render the default"):
        tsgen._default(dt.timedelta(seconds=1), where="Fake.field")


def test_a_nested_model_default_renders_all_of_its_fields() -> None:
    """`{}` would also compile here, since every Totals field has its own default. It would
    also be a lie: the schema would no longer state what the default *is*."""
    rendered = tsgen._default(Totals(), where="Trace.totals")
    for name in Totals.model_fields:
        assert name in rendered, f"{name} missing from the rendered default: {rendered}"


def test_the_rendered_default_equals_the_python_default() -> None:
    """Parse the emitted object literal back and compare. Catches a renderer that produces
    compilable TypeScript with the wrong values in it — `[]` for a non-empty list default
    was exactly that."""
    field = Trace.model_fields["totals"]
    rendered = tsgen._default(field.get_default(call_default_factory=True), where="x")
    # Bare JS identifier keys -> quoted, so json can read it.
    as_json = re.sub(r"([{,]\s*)([A-Za-z_$][A-Za-z0-9_$]*)\s*:", r'\1"\2":', rendered)
    assert json.loads(as_json) == Totals().model_dump(mode="json")


def test_non_empty_container_defaults_keep_their_contents() -> None:
    assert tsgen._default([1, 2], where="x") == "[1, 2]"
    assert tsgen._default({"a": 1}, where="x") == "{a: 1}"
    assert tsgen._default([], where="x") == "[]"
    assert tsgen._default({}, where="x") == "{}"


def test_keys_that_are_not_identifiers_get_quoted() -> None:
    """An unquoted `content-type:` is a syntax error, not a type error — it would take the
    whole module down rather than one field."""
    assert tsgen._default({"content-type": "x"}, where="k") == '{"content-type": "x"}'
    assert tsgen._default({"ok_key": 1}, where="k") == "{ok_key: 1}"


def test_strings_are_escaped() -> None:
    assert tsgen._default('say "hi"', where="s") == '"say \\"hi\\""'


def test_the_committed_file_is_what_the_generator_produces_now() -> None:
    """CI runs `make schema` then `git diff --exit-code`. Failing here first names the
    cause instead of leaving a bare non-zero exit."""
    assert GENERATED.read_text(encoding="utf-8") == tsgen.generate(), (
        "web/src/lib/trace.ts is stale — regenerate with `make schema` and commit"
    )
