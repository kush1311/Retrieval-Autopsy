"""Ablation regression: the study's own output, turned into a test.

Once the ablation sweep has run, its outcome distribution is a fingerprint of the
pipeline. If ``no_lexical`` used to push 34% of answers into confidently-wrong and now
pushes 3%, one of three things happened: the pipeline changed, the corpus changed, or
the measurement broke. All three are worth an alert, and none of them show up in a
suite that only asks "does the pipeline still return an answer".

The tolerance is in percentage points rather than relative change on purpose. A
relative threshold makes small columns hypersensitive — 1 case out of 200 moving to 2
is a 100% relative swing and means nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopsy.determinism import REPO_ROOT
from evals.runner import Finding, Severity

SNAPSHOT_PATH = REPO_ROOT / "evals" / "ablation-snapshot.json"

#: Percentage points of drift tolerated per (ablation, outcome) cell before it is a
#: finding. Wide enough to absorb a handful of cases moving, narrow enough to catch a
#: leg that silently stopped contributing.
TOLERANCE_PP = 8.0


def write_snapshot(
    table: dict[str, dict[str, int]], versions: dict[str, str], path: Path = SNAPSHOT_PATH
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"table": table, "versions": versions}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rates(row: dict[str, int]) -> dict[str, float]:
    total = sum(row.values()) or 1
    return {k: 100.0 * v / total for k, v in row.items()}


@dataclass
class AblationRegressionSuite:
    table: dict[str, dict[str, int]]
    versions: dict[str, str]
    snapshot: dict[str, Any] | None = field(default_factory=load_snapshot)
    tolerance_pp: float = TOLERANCE_PP
    name: str = "ablation_regression"

    def run(self) -> list[Finding]:
        if self.snapshot is None:
            return [
                Finding(
                    suite=self.name, case_id="__snapshot__", passed=True,
                    severity=Severity.INFO,
                    detail="no ablation snapshot recorded yet; run `make ablate-snapshot` "
                           "to pin the current distribution as the reference",
                )
            ]

        findings: list[Finding] = []
        old_versions = self.snapshot.get("versions", {})
        drifted = {
            k: (old_versions.get(k), self.versions.get(k))
            for k in ("corpus", "provider", "embed_model")
            if old_versions.get(k) != self.versions.get(k)
        }
        if drifted:
            # Not a failure. A corpus change is expected to move these numbers, and
            # reporting it as a regression would train people to ignore the suite.
            findings.append(
                Finding(
                    suite=self.name, case_id="__provenance__", passed=True,
                    severity=Severity.INFO,
                    detail="the snapshot was recorded against different provenance, so "
                           "drift below is expected rather than a regression",
                    evidence=[f"{k}: {a} -> {b}" for k, (a, b) in drifted.items()],
                )
            )

        old_table: dict[str, dict[str, int]] = self.snapshot.get("table", {})
        for ablation, row in sorted(self.table.items()):
            old_row = old_table.get(ablation)
            if old_row is None:
                findings.append(
                    Finding(
                        suite=self.name, case_id=f"{ablation}:new", passed=True,
                        severity=Severity.INFO,
                        detail=f"`{ablation}` is not in the snapshot; nothing to compare",
                    )
                )
                continue
            now, before = _rates(row), _rates(old_row)
            moves = []
            for outcome in sorted(set(now) | set(before)):
                delta = now.get(outcome, 0.0) - before.get(outcome, 0.0)
                if abs(delta) > self.tolerance_pp:
                    moves.append(
                        f"{outcome}: {before.get(outcome, 0.0):.1f}% -> "
                        f"{now.get(outcome, 0.0):.1f}% ({delta:+.1f}pp)"
                    )
            if moves:
                findings.append(
                    Finding(
                        suite=self.name, case_id=ablation, passed=False,
                        severity=Severity.MEDIUM if drifted else Severity.HIGH,
                        detail=(
                            f"the outcome distribution for `{ablation}` moved by more than "
                            f"{self.tolerance_pp:.0f} percentage points. Either the pipeline "
                            "changed, the corpus changed, or the measurement broke."
                        ),
                        evidence=moves,
                    )
                )
            else:
                findings.append(
                    Finding(
                        suite=self.name, case_id=ablation, passed=True, severity=Severity.INFO,
                        detail=f"`{ablation}` distribution stable within "
                               f"{self.tolerance_pp:.0f}pp",
                    )
                )
        return findings


__all__ = [
    "AblationRegressionSuite", "SNAPSHOT_PATH", "TOLERANCE_PP", "load_snapshot", "write_snapshot",
]
