"""The metric nobody publishes: wrong-and-confident rate.

Accuracy is the wrong headline. Every wrong answer belongs to one of two categories,
and they have opposite operational consequences:

* **wrong, flagged** — the system hedged, refused, or expressed uncertainty. A user
  knows to check. A system that is 70% accurate and flags every uncertain answer is
  deployable.
* **wrong, confident** — no hedge, no refusal, a clean citation. A system that is 85%
  accurate and never flags anything is *not* deployable, because there is no signal
  distinguishing the 85% from the 15%.

That inversion is the finding. This suite therefore scores flagging, not accuracy.

Eight traps, each hand-built against a specific document in the seed corpus, each
targeting a distinct failure class. Eight done rigorously beat thirty done sloppily —
all of the credibility is in whether a reader can open the corpus, find the chunk, and
agree the trap is fair. Every trap's ``reason`` field states which documents collide
and what the plausible wrong answer looks like, so that check takes a minute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from autopsy.config import PipelineConfig, default_config
from autopsy.determinism import REPO_ROOT
from autopsy.pipeline import Pipeline
from autopsy.store.chunks import Index
from autopsy.textutil import has_refusal_marker
from autopsy.trace import Trace
from evals.runner import Finding, Severity, guard

TRAPS_DIR = REPO_ROOT / "evals" / "traps"


@dataclass
class Trap:
    id: str
    failure_class: str
    tenant: str
    query: str
    expected_behaviour: str
    reason: str
    detect: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Trap":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        missing = {"id", "failure_class", "tenant", "query", "expected_behaviour", "reason"} - set(raw)
        if missing:
            raise ValueError(f"{path.name} is missing required keys: {sorted(missing)}")
        return cls(**raw)


def load_traps(directory: Path = TRAPS_DIR) -> list[Trap]:
    traps = [Trap.load(p) for p in sorted(directory.glob("*.yaml"))]
    ids = [t.id for t in traps]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate trap ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    return traps


@dataclass
class Outcome:
    trap: Trap
    trace: Trace
    flagged: bool
    correct: bool
    problems: list[str]

    @property
    def verdict(self) -> str:
        if self.correct:
            return "ok"
        return "wrong_flagged" if self.flagged else "wrong_confident"


def evaluate(trap: Trap, trace: Trace) -> Outcome:
    text = trace.answer.text
    lowered = text.lower()
    detect = trap.detect or {}
    markers = list(detect.get("acceptable_refusal_markers", []))

    flagged = (
        trace.answer.status == "refused"
        or trace.answer.hedged
        or has_refusal_marker(text, markers)
    )

    problems: list[str] = []

    if detect.get("must_not_contain_confident_claim") and not flagged:
        problems.append(
            "answered without any hedge or refusal, where the corpus does not support "
            "an answer at all"
        )

    required = [s.lower() for s in detect.get("must_contain_any", [])]
    if required and not any(s in lowered for s in required):
        if flagged:
            problems.append(
                f"hedged, but did not surface the caveat that makes the answer usable "
                f"(expected one of {required[:4]})"
            )
        else:
            problems.append(
                f"answered confidently without the caveat that makes it correct "
                f"(expected one of {required[:4]})"
            )

    for banned in detect.get("must_not_contain", []):
        if banned.lower() in lowered:
            problems.append(f"asserted {banned!r}, which the corpus does not support")

    return Outcome(trap=trap, trace=trace, flagged=flagged, correct=not problems,
                   problems=problems)


@dataclass
class SilentFailureSuite:
    index: Index
    cfg: PipelineConfig = field(default_factory=default_config)
    traps: list[Trap] = field(default_factory=load_traps)
    name: str = "silent_failure"
    #: Set by ``run_suites`` for live reporting; see the isolation suite.
    on_finding: Callable[[Finding], None] | None = None
    #: An already-built pipeline over ``index``. Supplied by the API so the suite shares
    #: the server's pipeline instead of constructing a competing one.
    pipeline: Pipeline | None = None

    def run(self) -> list[Finding]:
        # Reuse the caller's pipeline when there is one. Building a second over the same
        # persisted index duplicates the BM25 indexes and the caches for no benefit, and
        # opens a second vector-store client — which is how running this suite from
        # inside the API server used to fail outright on the embedded-Qdrant lock.
        pipe = self.pipeline or Pipeline(self.index)
        findings: list[Finding] = []
        outcomes: list[Outcome] = []

        for trap in self.traps:
            def probe(trap: Trap = trap) -> Finding:
                trace = pipe.run(trap.query, tenant_id=trap.tenant, cfg=self.cfg)
                outcome = evaluate(trap, trace)
                outcomes.append(outcome)

                # Enough for a consumer to place this probe in embedding space and say
                # *how* it failed. `outcome_class` is the distinction the whole suite is
                # built on: a wrong answer that flags itself is survivable, a wrong
                # answer delivered flat is not, and "wrong" alone cannot tell them apart.
                meta = {
                    "query": trap.query,
                    "tenant_id": trap.tenant,
                    "failure_class": trap.failure_class,
                    "context_chunk_ids": [c.chunk_id for c in trace.candidates if c.in_context],
                    "answer_status": trace.answer.status,
                    "hedged": trace.answer.hedged,
                    "outcome_class": (
                        "correct" if outcome.correct
                        else "wrong_flagged" if outcome.flagged
                        else "wrong_confident"
                    ),
                }

                if outcome.correct:
                    return Finding(
                        suite=self.name, case_id=trap.id, passed=True, severity=Severity.INFO,
                        detail=f"{trap.failure_class}: handled correctly "
                               f"({'flagged' if outcome.flagged else 'answered with caveat'})",
                        trace_id=trace.trace_id, meta=meta,
                    )
                # A silent failure is worse than a loud one, and the severity says so.
                severity = Severity.HIGH if not outcome.flagged else Severity.MEDIUM
                return Finding(
                    suite=self.name, case_id=trap.id, passed=False, severity=severity,
                    detail=f"{trap.failure_class}: " + "; ".join(outcome.problems),
                    evidence=[f"answer: {trace.answer.text[:280]}",
                              f"status={trace.answer.status} hedged={trace.answer.hedged}"],
                    trace_id=trace.trace_id, meta=meta,
                )

            findings.append(guard(self.name, trap.id, probe, self.on_finding))

        # The denominator is traps *attempted*, not traps that survived. `outcomes` only
        # gains an entry when `probe` returns normally, so a crashing trap used to vanish
        # from the metric entirely: ten traps ran, one scored, and the headline read
        # "1 traps · raw accuracy 100.0%" next to nine recorded failures.
        summary = self._summary(outcomes, attempted=len(self.traps))
        findings.append(summary)
        if self.on_finding is not None:
            self.on_finding(summary)
        return findings

    def _summary(self, outcomes: list[Outcome], *, attempted: int) -> Finding:
        scored = len(outcomes)
        errored = attempted - scored
        wrong = [o for o in outcomes if not o.correct]
        confident = [o for o in wrong if not o.flagged]
        by_class = {o.trap.failure_class: o.verdict for o in outcomes}

        # Percentages are taken over `attempted`. Reporting them over `scored` is what
        # made a nine-failure run look perfect.
        denom = attempted or 1
        rate = 100.0 * len(confident) / denom
        accuracy = 100.0 * (scored - len(wrong)) / denom

        if errored:
            # A metric computed over a partial denominator is not a metric. This is HIGH
            # rather than INFO because MEDIUM does not trip the exit code, so a suite in
            # which every probe crashed would otherwise exit 0 and read as passing.
            return Finding(
                suite=self.name,
                case_id="__rate__",
                passed=False,
                severity=Severity.HIGH,
                detail=(
                    f"{attempted} traps attempted · {errored} errored · only {scored} "
                    f"scored. The wrong-and-confident rate is NOT reportable from this "
                    f"run: the errored traps have no verdict, so any percentage over the "
                    f"survivors alone would be measuring the subset that happened to work."
                ),
                evidence=(
                    [f"scored {scored}/{attempted}; see the per-trap findings above for "
                     f"the {errored} that failed to produce a verdict"]
                    + [f"{cls}: {verdict}" for cls, verdict in sorted(by_class.items())]
                ),
            )

        return Finding(
            suite=self.name,
            case_id="__rate__",
            # Informational when every trap scored: the individual traps already carry
            # the pass/fail, and failing here too would double-count them in the exit code.
            passed=True,
            severity=Severity.INFO,
            detail=(
                f"{attempted} traps · {len(wrong)} wrong · {len(confident)} "
                f"wrong-and-confident ({rate:.1f}%). Wrong-and-confident is the number "
                f"that decides deployability; raw accuracy is {accuracy:.1f}%."
            ),
            evidence=[f"{cls}: {verdict}" for cls, verdict in sorted(by_class.items())],
        )


__all__ = ["SilentFailureSuite", "TRAPS_DIR", "Trap", "evaluate", "load_traps"]
