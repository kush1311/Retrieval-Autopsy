"""The eval runner: executes suites, aggregates findings, writes a report, exits non-zero.

Headless. No UI, ever. 200 cases across 8 ablations is 1600 runs; a live view of that
is a progress bar, and building one costs time that should go into knowing which
failures are worth testing for.

Two deliberate design choices:

**A probe crash is a finding, not a stack trace.** Exceptions are caught per case and
recorded as MEDIUM findings, so one broken probe cannot hide the other eleven results.
An eval harness that dies on the first error tells you less than one that reports
eleven passes and one crash.

**Regressions are measured against a recorded baseline.** In any codebase with
pre-existing failures, a *new* failure is invisible without a diff. Exit code 1 means
either something HIGH or above failed, or the failure set changed — not merely that
something is red.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Protocol

from autopsy.determinism import REPO_ROOT, code_version

REPORTS_DIR = REPO_ROOT / "reports"


def baseline_path(provider: str = "offline") -> Path:
    """Baselines are per-provider, and that is not a convenience.

    The failure set is a property of the model, not just the code: llama-3.3-70b
    refuses the superlative trap that the offline simulator answers confidently. One
    shared baseline means switching provider reports every genuine difference as a
    regression, the diff becomes noise, and people start passing
    ``--update-baseline`` reflexively — at which point the mechanism that was supposed
    to catch new failures catches nothing.
    """
    return REPO_ROOT / "evals" / f"baseline.{provider}.json"


#: Kept for the default provider so existing call sites and docs stay valid.
BASELINE_PATH = baseline_path("offline")


class Severity(IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.name


#: Anything at or above this fails the build on its own.
FAIL_AT = Severity.HIGH


@dataclass
class Finding:
    suite: str
    case_id: str
    passed: bool
    severity: Severity
    detail: str
    evidence: list[str] = field(default_factory=list)
    trace_id: str | None = None
    #: Structured extras for consumers that need more than prose — the query, which
    #: chunks reached the model, the outcome class. Deliberately not folded into
    #: ``evidence``: that is human-readable text rendered into the report, and stuffing
    #: machine-readable fields there would mean parsing the report to get them back.
    #:
    #: Never part of ``key``, so adding fields here can never change baseline diffing.
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.case_id}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.name
        return d


class Suite(Protocol):
    name: str

    def run(self) -> list[Finding]: ...


@dataclass
class Report:
    findings: list[Finding]
    versions: dict[str, str]
    provider: str
    notes: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if not f.passed]

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.failures if f.severity >= FAIL_AT]

    def by_suite(self) -> dict[str, list[Finding]]:
        out: dict[str, list[Finding]] = {}
        for finding in self.findings:
            out.setdefault(finding.suite, []).append(finding)
        return out


def run_suites(
    suites: list[Suite],
    *,
    versions: dict[str, str],
    provider: str,
    on_progress: Callable[[str], None] | None = None,
    on_finding: Callable[[Finding], None] | None = None,
) -> Report:
    """Execute suites and collect findings.

    ``on_finding`` fires as each result lands, which is what lets the inspector render
    an eval run live instead of showing a spinner for two minutes. Suites that expose an
    ``on_finding`` attribute report per probe; the rest are reported in one batch when
    ``run()`` returns, so a suite that has not been taught to stream still shows up.
    """
    findings: list[Finding] = []
    for suite in suites:
        if on_progress:
            on_progress(f"running {suite.name}")
        streams = hasattr(suite, "on_finding")
        if on_finding is not None and streams:
            suite.on_finding = on_finding  # type: ignore[attr-defined]
        try:
            produced = suite.run()
            findings.extend(produced)
            if on_finding is not None and not streams:
                for finding in produced:
                    on_finding(finding)
        except Exception as exc:  # noqa: BLE001 - a broken suite is a finding
            broken = Finding(
                suite=suite.name,
                case_id="__suite__",
                passed=False,
                severity=Severity.HIGH,
                detail=f"the suite itself raised: {type(exc).__name__}: {exc}",
                evidence=traceback.format_exc().splitlines()[-6:],
            )
            findings.append(broken)
            if on_finding is not None:
                on_finding(broken)
    return Report(findings=findings, versions=versions, provider=provider)


def guard(
    suite: str,
    case_id: str,
    fn: Callable[[], Finding],
    emit: Callable[[Finding], None] | None = None,
) -> Finding:
    """Run one case, converting a crash into a MEDIUM finding rather than a traceback.

    A probe that raises is a finding, not a stack trace: one broken probe must not hide
    the other eleven results. ``emit`` reports the outcome the moment it is known.
    """
    try:
        finding = fn()
    except Exception as exc:  # noqa: BLE001
        finding = Finding(
            suite=suite,
            case_id=case_id,
            passed=False,
            severity=Severity.MEDIUM,
            detail=f"probe crashed: {type(exc).__name__}: {exc}",
            evidence=traceback.format_exc().splitlines()[-6:],
        )
    if emit is not None:
        emit(finding)
    return finding


# --------------------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------------------


def load_baseline(path: Path | None = None, provider: str = "offline") -> set[str]:
    path = path or baseline_path(provider)
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("known_failures", []))


def write_baseline(report: Report, path: Path | None = None) -> None:
    path = path or baseline_path(report.provider)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "known_failures": sorted(f.key for f in report.failures),
                "recorded_against": report.versions,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def diff_baseline(report: Report, baseline: set[str]) -> tuple[list[str], list[str]]:
    """``(new_failures, newly_fixed)``."""
    current = {f.key for f in report.failures}
    return sorted(current - baseline), sorted(baseline - current)


def exit_code(report: Report, baseline: set[str]) -> int:
    """Non-zero on a *new* blocking failure, or on any change to the failure set.

    Note the "new". Failing unconditionally on every HIGH finding sounds stricter and
    is worse: this corpus has four genuine, reproducible silent-failure hits, so an
    unconditional rule leaves the build permanently red, and a permanently red build
    is one nobody reads. Baselined failures stay visible — the report lists them under
    "Acknowledged failures" with their severity — but they do not drown out the signal
    that something *changed*, which is the thing CI is actually good at detecting.

    Deleting a line from ``evals/baseline.json`` is how you promote an acknowledged
    failure back to a blocking one, and that is a reviewable diff.
    """
    new_failures, _ = diff_baseline(report, baseline)
    new_blocking = [f for f in report.blocking if f.key not in baseline]
    return 1 if (new_blocking or new_failures) else 0


# --------------------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------------------

PROVIDER_WARNING = (
    "> **These numbers describe the offline simulator, not a language model.**\n"
    "> The `offline` provider is a deterministic rule-based stand-in that exists so the\n"
    "> whole system runs in CI with no API key. It reproduces the *mechanism* behind each\n"
    "> failure class faithfully, but its absolute rates are properties of the simulator.\n"
    "> Re-run with `AUTOPSY_PROVIDER=live` for numbers about a real model.\n"
)


def render_report(report: Report, baseline: set[str], title: str = "Eval report") -> str:
    new_failures, fixed = diff_baseline(report, baseline)
    lines: list[str] = [f"# {title}", ""]
    stamp = " · ".join(f"{k} {v}" for k, v in sorted(report.versions.items()))
    lines += [stamp, ""]
    if report.provider == "offline":
        lines += [PROVIDER_WARNING, ""]

    lines += ["## Summary", ""]
    for suite, findings in sorted(report.by_suite().items()):
        passed = sum(1 for f in findings if f.passed)
        worst = max((f.severity for f in findings if not f.passed), default=Severity.INFO)
        suffix = "" if passed == len(findings) else f" · worst severity {worst.name}"
        # Say "findings", not a bare fraction. `findings` includes any aggregate a suite
        # emits, so silent_failure's 9 traps produce 10 findings — and "5/10 passed"
        # printed beside "9 traps ·" in the same report is exactly the internal
        # inconsistency an external reviewer flagged. The numbers were both right; only
        # the label was missing.
        lines.append(f"- **{suite}**: {passed}/{len(findings)} findings passed{suffix}")
    for note in report.notes:
        lines.append(f"- {note}")
    lines.append("")

    # Aggregate findings — case ids like `__rate__` — carry the headline metrics. They
    # are passing INFO records, so the failures-only sections below skip them entirely:
    # the wrong-and-confident rate was being computed and then dropped on the floor,
    # which for a report whose whole subject is that number is the one thing it must
    # not do.
    aggregates = [f for f in report.findings if f.case_id.startswith("__")]
    if aggregates:
        lines += ["## Headline metrics", ""]
        for finding in sorted(aggregates, key=lambda f: f.suite):
            lines.append(f"**{finding.suite}** — {finding.detail}")
            if finding.evidence:
                lines.append("")
                for item in finding.evidence:
                    lines.append(f"- {item}")
            lines.append("")

    lines += ["## Regressions vs baseline", ""]
    if not baseline:
        lines.append("No baseline recorded yet. Run `make eval-baseline` to pin one.")
    elif not new_failures and not fixed:
        lines.append("none")
    else:
        for key in new_failures:
            lines.append(f"- **NEW FAILURE** `{key}`")
        for key in fixed:
            lines.append(f"- fixed `{key}`")
    lines.append("")

    acknowledged = [f for f in report.failures if f.key in baseline]
    if acknowledged:
        lines += [
            "## Acknowledged failures",
            "",
            "Reproducible failures recorded in `evals/baseline.json`. They do **not** fail "
            "the build — a permanently red build is one nobody reads — but they are real, "
            "and they are listed here so that stays uncomfortable. Remove a line from the "
            "baseline to promote one back to blocking.",
            "",
        ]
        for finding in sorted(acknowledged, key=lambda f: (-f.severity, f.case_id)):
            lines.append(f"- `{finding.key}` — **{finding.severity.name}** · {finding.detail}")
        lines.append("")

    failures = sorted(report.failures, key=lambda f: (-f.severity, f.suite, f.case_id))
    lines += ["## Failures", ""]
    if not failures:
        lines.append("none")
    for finding in failures:
        lines.append(f"### {finding.suite} / {finding.case_id} — {finding.severity.name}")
        lines.append("")
        lines.append(finding.detail)
        if finding.trace_id:
            lines.append("")
            lines.append(f"trace: `reports/traces/{finding.trace_id}.json`")
        if finding.evidence:
            lines.append("")
            lines.append("```")
            lines.extend(finding.evidence[:12])
            lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report(text: str, name: str, directory: Path = REPORTS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def base_versions(index_meta: dict[str, Any], provider: str) -> dict[str, str]:
    return {
        "corpus": str(index_meta.get("corpus_version", "unknown")),
        "code": code_version(),
        "provider": provider,
        "embed_model": str(index_meta.get("embed_model", "unknown")),
    }


__all__ = [
    "BASELINE_PATH",
    "FAIL_AT",
    "Finding",
    "PROVIDER_WARNING",
    "REPORTS_DIR",
    "Report",
    "Severity",
    "Suite",
    "base_versions",
    "diff_baseline",
    "exit_code",
    "guard",
    "load_baseline",
    "render_report",
    "run_suites",
    "write_baseline",
    "write_report",
]
