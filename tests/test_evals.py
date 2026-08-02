"""Tests for the eval machinery — including negative controls.

**A suite that cannot fail is worth nothing.** Every green isolation run in this repo
is only meaningful if the same probes go red when isolation is actually broken, so the
tests below deliberately break it and assert that they do. The same logic applies to
the trap suite and the baseline diff.
"""

from __future__ import annotations

import pytest

from autopsy.config import default_config
from autopsy.ingest import Document, build_index
from autopsy.pipeline import Pipeline
from autopsy.store.chunks import GLOBAL_TENANT, Index
from evals.judge import (
    GoldenCase, RuleJudge, TwoWayJudge, Verdict, calibrate, cohens_kappa,
)
from evals.runner import (
    Finding, Report, Severity, diff_baseline, exit_code, guard, render_report,
)
from evals.suites.isolation import IsolationSuite, build_corpus
from evals.suites.silent_failure import Trap, evaluate, load_traps

CFG = default_config()


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------


def test_a_crashing_probe_becomes_a_finding_not_a_traceback():
    def boom() -> Finding:
        raise RuntimeError("probe exploded")

    finding = guard("suite", "case", boom)
    assert not finding.passed
    assert finding.severity == Severity.MEDIUM
    assert "probe exploded" in finding.detail


def _report(*findings: Finding) -> Report:
    return Report(findings=list(findings), versions={"corpus": "c"}, provider="offline")


def test_high_severity_failure_fails_the_build():
    report = _report(
        Finding(suite="s", case_id="a", passed=False, severity=Severity.HIGH, detail="x")
    )
    assert exit_code(report, baseline=set()) == 1


def test_a_known_failure_in_the_baseline_does_not_fail_the_build():
    """Without a baseline diff, a new failure is invisible in a codebase that already
    has red. With one, only the *change* matters."""
    finding = Finding(suite="s", case_id="a", passed=False, severity=Severity.MEDIUM, detail="x")
    report = _report(finding)
    assert exit_code(report, baseline={"s/a"}) == 0
    assert exit_code(report, baseline=set()) == 1


def test_a_new_failure_fails_even_below_the_severity_bar():
    findings = [
        Finding(suite="s", case_id="a", passed=False, severity=Severity.LOW, detail="x"),
        Finding(suite="s", case_id="b", passed=False, severity=Severity.LOW, detail="y"),
    ]
    report = _report(*findings)
    new, fixed = diff_baseline(report, {"s/a"})
    assert new == ["s/b"] and fixed == []
    assert exit_code(report, {"s/a"}) == 1


def test_report_stamps_the_offline_provider_warning():
    text = render_report(_report(), baseline=set())
    assert "offline simulator" in text


def test_report_omits_the_warning_for_a_live_run():
    report = Report(findings=[], versions={"corpus": "c"}, provider="live")
    assert "offline simulator" not in render_report(report, baseline=set())


# --------------------------------------------------------------------------------------
# Isolation — positive path and negative controls
# --------------------------------------------------------------------------------------


def test_isolation_suite_passes_on_a_correct_pipeline():
    findings = IsolationSuite(cfg=CFG, run_token="testtoken").run()
    failures = [f for f in findings if not f.passed]
    assert not failures, [f"{f.case_id}: {f.detail}" for f in failures]
    assert len(findings) >= 12


def test_isolation_suite_detects_a_deliberately_broken_boundary(monkeypatch):
    """NEGATIVE CONTROL. Widen `scope` to the whole corpus — the exact bug a
    post-filter implementation has — and assert the suite goes red. Without this test,
    twelve green probes prove only that the probes ran."""
    corpus = build_corpus(CFG, "leaktoken")

    def leaky_scope(self, tenant_id: str):
        return list(self.chunks)  # every tenant sees everything

    monkeypatch.setattr(Index, "scope", leaky_scope)

    suite = IsolationSuite(cfg=CFG, run_token="leaktoken")
    monkeypatch.setattr(
        "evals.suites.isolation.build_corpus", lambda cfg, token: corpus
    )
    findings = suite.run()
    failures = [f for f in findings if not f.passed]
    assert failures, "a wide-open scope must be caught by at least one probe"
    assert any(f.severity == Severity.CRITICAL for f in failures)
    caught = {f.case_id for f in failures}
    assert "retrieval_filter" in caught or "scope_predicate" in caught


def test_positive_control_fails_when_a_tenant_cannot_see_its_own_documents(monkeypatch):
    """NEGATIVE CONTROL for the positive control. A system that returns nothing to
    anybody passes every leak probe; the positive controls exist to catch exactly
    that, so they must themselves be able to fail."""
    def empty_scope(self, tenant_id: str):
        return []

    monkeypatch.setattr(Index, "scope", empty_scope)
    findings = IsolationSuite(cfg=CFG, run_token="emptytoken").run()
    failed = {f.case_id for f in findings if not f.passed}
    assert "positive_control_own_documents" in failed
    assert "positive_control_global_documents" in failed


# --------------------------------------------------------------------------------------
# Traps
# --------------------------------------------------------------------------------------


def test_every_trap_declares_its_reasoning():
    traps = load_traps()
    assert len(traps) >= 8, "eight rigorous traps beat thirty sloppy ones, but not four"
    classes = {t.failure_class for t in traps}
    assert len(classes) >= 6, f"traps cluster on too few failure classes: {classes}"
    for trap in traps:
        assert len(trap.reason.split()) >= 20, f"{trap.id}: reason is too thin to audit"
        assert trap.detect, f"{trap.id}: no detection criteria"


def _trace_with(text: str, *, hedged: bool, status: str = "grounded"):
    from autopsy.trace import Answer, Totals, Trace

    return Trace(
        trace_id="t", created_at="2026-07-26T00:00:00Z", query="q", tenant_id="tenant_x",
        config_hash="h", config={}, versions={},
        answer=Answer(text=text, status=status, hedged=hedged), totals=Totals(),
    )


def test_trap_scores_a_confident_wrong_answer_as_high_severity():
    trap = Trap(
        id="t", failure_class="phantom_value", tenant="tenant_x", query="q",
        expected_behaviour="refuse_or_hedge", reason="r",
        detect={"must_not_contain_confident_claim": True},
    )
    outcome = evaluate(trap, _trace_with("The default is 42.", hedged=False))
    assert outcome.verdict == "wrong_confident"


def test_trap_scores_a_hedged_wrong_answer_as_merely_wrong():
    """The whole point of the metric: the same wrong answer is a different problem
    depending on whether the system flagged it."""
    trap = Trap(
        id="t", failure_class="phantom_value", tenant="tenant_x", query="q",
        expected_behaviour="refuse_or_hedge", reason="r",
        detect={"must_not_contain_confident_claim": True,
                "must_contain_any": ["not documented"]},
    )
    outcome = evaluate(trap, _trace_with("This may not be documented.", hedged=True))
    assert outcome.flagged
    assert outcome.verdict in ("ok", "wrong_flagged")


# --------------------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------------------


def test_kappa_is_zero_for_a_judge_that_always_says_the_same_thing():
    """Raw agreement is inflated when one class dominates; kappa is not fooled."""
    pairs = [("equivalent", "equivalent")] * 80 + [("degraded", "equivalent")] * 20
    assert sum(1 for a, b in pairs if a == b) / len(pairs) == 0.8
    assert cohens_kappa(pairs) == pytest.approx(0.0, abs=1e-9)


def test_kappa_is_one_for_perfect_agreement():
    pairs = [("equivalent", "equivalent")] * 5 + [("degraded", "degraded")] * 5
    assert cohens_kappa(pairs) == pytest.approx(1.0)


def test_two_refusals_are_equivalent_however_differently_worded():
    judge = RuleJudge()
    verdict, _ = judge.compare(
        "q",
        "I could not find documentation that answers this with enough confidence.",
        "The provided sources do not document this; no supporting evidence was found.",
    )
    assert verdict == Verdict.EQUIVALENT


def test_disjoint_numbers_are_contradictory():
    verdict, _ = RuleJudge().compare("q", "The default is 900.", "The default is 300.")
    assert verdict == Verdict.CONTRADICTORY


def test_symmetric_judge_is_not_reported_as_order_unstable():
    """A judge that never claims a direction cannot contradict itself by swapping the
    arguments. Counting that as instability measures the harness, not the judge."""
    two_way = TwoWayJudge(RuleJudge())
    two_way.compare("q", "the flange budget was exhausted", "the spindle limit was reached")
    assert two_way.instability_rate == 0.0


def test_calibration_reports_agreement_and_disagreements():
    golden = [
        GoldenCase("a", "q", "The default is 900.", "The default is 300.",
                   Verdict.CONTRADICTORY, "test"),
        GoldenCase("b", "q", "same text", "same text", Verdict.EQUIVALENT, "test"),
    ]
    cal = calibrate(TwoWayJudge(RuleJudge()), golden)
    assert cal.n == 2
    assert cal.agreement == pytest.approx(1.0)
    assert not cal.disagreements
