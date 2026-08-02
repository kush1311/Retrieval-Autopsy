"""A suite with N failures can never report 100% accuracy.

Same class of bug as `test_failed_runs_are_not_agreement.py`, found in a different place:
a failure silently dropping out of a metric's denominator so the metric reads clean.

`SilentFailureSuite` appended to `outcomes` *inside* the probe, after the pipeline call. A
trap whose pipeline run raised was caught by `guard()` and recorded as a failure, but never
reached `outcomes`. The aggregate then divided by the survivors:

    reports/silent-failure.md   "silent_failure: 2/10 passed"        <- suite level
    reports/silent-failure.md   "1 traps · 0 wrong · 0 wrong-and-    <- same file
                                 confident (0.0%) · raw accuracy 100.0%"

Ten traps attempted, one scored, nine errored, and the headline claimed perfection. The
committed report contained both numbers and nobody reading it would reconcile them.
"""

from __future__ import annotations

import pytest

from evals.runner import Severity
from evals.suites.silent_failure import Outcome, SilentFailureSuite, Trap


def _trap(tid: str) -> Trap:
    return Trap(
        id=tid, failure_class="phantom_value", tenant="tenant_kelvin",
        query=f"what is {tid}", expected_behaviour="refuse_or_hedge",
        reason="does not exist",
    )


def _outcome(trap: Trap, *, correct: bool, flagged: bool) -> Outcome:
    # `trace` is unused by _summary; None keeps the fixture honest about that rather
    # than fabricating a Trace that implies the metric reads it.
    return Outcome(trap=trap, trace=None, correct=correct, flagged=flagged, problems=[])


def _suite(n_traps: int) -> SilentFailureSuite:
    s = SilentFailureSuite.__new__(SilentFailureSuite)
    object.__setattr__(s, "traps", [_trap(f"t{i}") for i in range(n_traps)])
    object.__setattr__(s, "name", "silent_failure")
    return s


def test_errored_traps_cannot_vanish_from_the_denominator():
    """The exact committed-report shape: 10 attempted, 1 scored, 9 lost."""
    suite = _suite(10)
    scored = [_outcome(suite.traps[0], correct=True, flagged=True)]

    f = suite._summary(scored, attempted=10)

    assert "10 traps attempted" in f.detail
    assert "9 errored" in f.detail
    # The old code produced exactly this string. It must not be reachable.
    assert "raw accuracy is 100.0%" not in f.detail
    assert "1 traps ·" not in f.detail


def test_a_partial_run_fails_loudly_rather_than_reporting_a_rate():
    """MEDIUM does not trip the exit code, so an all-errored suite used to exit 0."""
    suite = _suite(10)
    f = suite._summary([], attempted=10)
    assert f.passed is False
    assert f.severity >= Severity.HIGH, "a partial denominator must trip the exit code"
    assert "NOT reportable" in f.detail


@pytest.mark.parametrize("n_wrong", [1, 3, 9, 10])
def test_a_suite_with_failures_never_reports_perfect_accuracy(n_wrong):
    suite = _suite(10)
    outcomes = (
        [_outcome(t, correct=False, flagged=False) for t in suite.traps[:n_wrong]]
        + [_outcome(t, correct=True, flagged=True) for t in suite.traps[n_wrong:]]
    )
    f = suite._summary(outcomes, attempted=10)
    assert "100.0%" not in f.detail.split("raw accuracy is")[-1], (
        f"{n_wrong} wrong answers still reported perfect accuracy: {f.detail}"
    )


def test_a_clean_full_run_still_reports_normally():
    """The fix must not make a genuinely complete run look broken."""
    suite = _suite(4)
    outcomes = [_outcome(t, correct=True, flagged=True) for t in suite.traps]
    f = suite._summary(outcomes, attempted=4)
    assert f.passed is True
    assert f.severity == Severity.INFO
    assert "4 traps ·" in f.detail
    assert "raw accuracy is 100.0%" in f.detail   # legitimate here: 4 attempted, 4 scored


def test_percentages_are_taken_over_attempted_not_scored():
    """2 confident-wrong out of 10 attempted is 20%, not 100% of the 2 that scored."""
    suite = _suite(10)
    outcomes = [_outcome(t, correct=False, flagged=False) for t in suite.traps[:2]]
    f = suite._summary(outcomes, attempted=10)
    # 8 errored, so this must refuse to quote a rate at all.
    assert "NOT reportable" in f.detail
