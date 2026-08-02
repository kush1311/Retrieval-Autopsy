"""Two runs that failed the same way are not "identical".

This is the highest-consequence bug found in the project, and it was silent.

A 3,080-run ablation sweep hit an index/embedder mismatch that failed *every single run*.
The pipeline writes the failure into the answer text, so baseline and variant matched
byte-for-byte; `classify`'s byte-equality shortcut filed all 3,080 as agreement; and the
findings table came back with zeros in every outcome column and no errors reported
anywhere. It looked exactly like "this pipeline is robust to ablation."

The build spec names this failure directly: *"your counterfactual engine will appear to
work and will silently prove nothing."* It was right, and the cause was not the cache it
warned about — it was byte-equality applied to error messages.
"""

from __future__ import annotations

from dataclasses import replace

from autopsy.counterfactual import Outcome, classify
from autopsy.trace import Answer, StageRecord, Totals, Trace


def _trace(text: str, *, stage_error: str | None = None, status: str = "grounded") -> Trace:
    return Trace(
        trace_id="T" * 26,
        created_at="2026-08-01T00:00:00Z",
        query="what does KLV-4001 mean",
        tenant_id="tenant_kelvin",
        config_hash="sha256:deadbeef",
        config={},
        versions={
            "corpus": "seed@abc", "embed_model": "m", "gen_model": "g",
            "rerank_model": "r", "provider": "groq", "embedder": "fastembed",
            "code": "src@1",
        },
        stages=[StageRecord(name="retrieve_dense", error=stage_error)],
        answer=Answer(text=text, status=status),  # type: ignore[arg-type]
        totals=Totals(),
    )


ERROR_TEXT = (
    "The pipeline failed before producing an answer: chunk c_648592bcb065a344 was "
    "indexed with dense vectors but the query is a concept bag."
)


def test_identical_error_text_is_not_identical():
    """The exact shape of the real bug: same error both sides, byte-equal answers."""
    baseline = _trace(ERROR_TEXT, stage_error="ProviderError")
    variant = _trace(ERROR_TEXT, stage_error="ProviderError")
    assert baseline.answer.text == variant.answer.text  # the trap
    assert classify(baseline, variant) is Outcome.ERROR


def test_a_stage_error_on_either_side_is_an_error():
    ok = _trace("KLV-4001 means the vetchcrest budget was exhausted. [1]")
    broken = _trace("KLV-4001 means the vetchcrest budget was exhausted. [1]",
                    stage_error="ProviderError")
    assert classify(ok, broken) is Outcome.ERROR
    assert classify(broken, ok) is Outcome.ERROR


def test_placeholder_answer_text_is_an_error_even_without_a_stage_error():
    """A pipeline-level abort leaves no stage error, only the placeholder answer."""
    ok = _trace("KLV-4001 means the vetchcrest budget was exhausted. [1]")
    aborted = _trace("No answer was produced.")
    assert classify(ok, aborted) is Outcome.ERROR


def test_genuinely_identical_answers_still_classify_as_identical():
    """The fix must not swallow the legitimate case it sits in front of."""
    text = "KLV-4001 means the vetchcrest budget was exhausted. [1]"
    assert classify(_trace(text), _trace(text)) is Outcome.IDENTICAL


def test_error_counts_as_a_regression():
    """An errored sweep must not read as a passing one."""
    assert Outcome.ERROR.is_regression
    assert not Outcome.IDENTICAL.is_regression
