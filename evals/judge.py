"""The LLM judge, and the harness that says how much to trust it.

Every judge-derived number in this repo is reported next to the judge's measured
agreement rate. An uncalibrated judge silently undermines every figure downstream, and
publishing "my judge disagrees with the reference labels 23% of the time, here is
where" is a better artifact than any passing test suite — practically nobody does it,
which is exactly why doing it reads as rigour.

Three known biases, each controlled for explicitly rather than hoped away:

* **Position.** In pairwise comparison the first item has an edge. Every comparison is
  run in both orders and the results reconciled; disagreement between the two orders is
  counted and published as an instability rate rather than averaged into invisibility.
* **Verbosity.** Longer answers score higher. The calibration report correlates verdict
  against length difference so the effect is measurable rather than assumed absent.
* **Self-preference.** Models favour their own family's output. The live judge is
  OpenAI while generation is Anthropic, so the judge is never grading its own relatives.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from autopsy.determinism import REPO_ROOT
from autopsy.textutil import concept_set, has_refusal_marker, is_hedged

GOLDEN_DIR = REPO_ROOT / "evals" / "golden"


class Verdict(str, Enum):
    EQUIVALENT = "equivalent"      # different words, same claims
    DEGRADED = "degraded"          # still defensible, but worse or thinner
    CONTRADICTORY = "contradictory"  # the two answers cannot both be right
    UNSTABLE = "unstable"          # the judge changed its mind when the order flipped


class Judge(Protocol):
    name: str
    #: Can this judge tell *which* of two answers is worse, or only that they differ
    #: in quality? An asymmetric judge reporting DEGRADED in both directions is
    #: contradicting itself; a symmetric one is just being consistent. Conflating the
    #: two inflates the measured instability rate with an artefact of the harness,
    #: which is a much worse outcome than not measuring it at all.
    directional: bool

    def compare(self, question: str, a: str, b: str) -> tuple[Verdict, str]: ...


# --------------------------------------------------------------------------------------
# Order-swapped wrapper — the position-bias control
# --------------------------------------------------------------------------------------

_MIRROR = {
    Verdict.EQUIVALENT: Verdict.EQUIVALENT,
    Verdict.CONTRADICTORY: Verdict.CONTRADICTORY,
}


@dataclass
class TwoWayJudge:
    """Runs every comparison in both orders and reconciles.

    ``EQUIVALENT`` and ``CONTRADICTORY`` are symmetric relations, so flipping the
    arguments must not change them. ``DEGRADED`` is directional: if A→B is degraded,
    B→A should not also be degraded. Where the two runs disagree, the result is
    ``UNSTABLE`` — surfaced, not silently resolved, because a coin-flip verdict
    reported as a verdict is worse than no verdict.
    """

    inner: Judge
    swaps: int = 0
    unstable: int = 0
    name: str = "two-way"
    directional: bool = True

    def __post_init__(self) -> None:
        self.name = f"two-way({self.inner.name})"
        self.directional = getattr(self.inner, "directional", True)

    def compare(self, question: str, a: str, b: str) -> tuple[Verdict, str]:
        forward, why_f = self.inner.compare(question, a, b)
        reverse, why_r = self.inner.compare(question, b, a)
        self.swaps += 1

        expected = _MIRROR.get(forward)
        if expected is not None and reverse == expected:
            return forward, why_f
        if forward == reverse == Verdict.DEGRADED:
            if self.directional:
                # An order-sensitive judge saying "B is worse" and "A is worse" about
                # the same pair has contradicted itself. That is the position bias
                # this wrapper exists to catch.
                self.unstable += 1
                return Verdict.UNSTABLE, "both orders reported degradation; direction unresolved"
            return Verdict.DEGRADED, f"{why_f} (symmetric judge; direction not claimed)"
        if forward == Verdict.DEGRADED and reverse != Verdict.DEGRADED:
            return forward, why_f
        if forward != reverse:
            self.unstable += 1
            return (
                Verdict.UNSTABLE,
                f"order-dependent: {forward.value} vs {reverse.value} ({why_f} / {why_r})",
            )
        return forward, why_f

    @property
    def instability_rate(self) -> float:
        return (self.unstable / self.swaps) if self.swaps else 0.0


# --------------------------------------------------------------------------------------
# Offline judge
# --------------------------------------------------------------------------------------

_NUM_RE = re.compile(r"\b\d[\d,._]*\b")


@dataclass
class RuleJudge:
    """A deterministic stand-in used under the offline provider.

    It is not a language model and does not pretend to be. It compares the concept
    sets, the numeric literals, and the hedging strength of two answers. That is
    enough to separate "same claims, different words" from "different numbers", which
    is the distinction the counterfactual engine actually needs — and it makes the
    calibration harness testable without a key.
    """

    equivalent_threshold: float = 0.72
    name: str = "rule-judge"
    #: Symmetric by construction: concept overlap and numeric-set comparison do not
    #: change when the arguments swap, so this judge never claims a direction.
    directional: bool = False

    def compare(self, question: str, a: str, b: str) -> tuple[Verdict, str]:
        a_norm, b_norm = a.strip().lower(), b.strip().lower()
        if a_norm == b_norm:
            return Verdict.EQUIVALENT, "identical text"

        a_refused = has_refusal_marker(a) or is_hedged(a)
        b_refused = has_refusal_marker(b) or is_hedged(b)

        if a_refused and b_refused:
            # Two refusals are the same answer however differently they are worded.
            # Falling through to the overlap check here scored them CONTRADICTORY,
            # because refusal boilerplate shares almost no vocabulary — 18 of the 106
            # calibration disagreements were this one bug.
            return Verdict.EQUIVALENT, "both answers decline"

        a_nums = set(_NUM_RE.findall(a))
        b_nums = set(_NUM_RE.findall(b))
        if a_nums and b_nums and not (a_nums & b_nums):
            return Verdict.CONTRADICTORY, f"disjoint numeric claims: {sorted(a_nums)[:3]} vs {sorted(b_nums)[:3]}"

        overlap = _jaccard(concept_set(a), concept_set(b))
        if a_refused != b_refused:
            side = "second" if b_refused else "first"
            return Verdict.DEGRADED, f"the {side} answer hedges and the other does not"
        if overlap >= self.equivalent_threshold:
            return Verdict.EQUIVALENT, f"concept overlap {overlap:.2f}"
        if overlap >= 0.35:
            return Verdict.DEGRADED, f"partial overlap {overlap:.2f}"
        return Verdict.CONTRADICTORY, f"little shared content (overlap {overlap:.2f})"


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


# --------------------------------------------------------------------------------------
# Live judge
# --------------------------------------------------------------------------------------

_JUDGE_SYSTEM = """You compare two answers to the same question and classify their \
relationship. Judge the *claims*, not the prose.

Reply with exactly one word on the first line:

EQUIVALENT   - they make the same claims, however differently worded
DEGRADED     - the second is thinner, vaguer, or hedges where the first committed, \
but does not contradict it
CONTRADICTORY - they cannot both be true

Then one short line of justification.

Ignore length. A longer answer is not a better one, and a shorter answer that makes \
the same claims is EQUIVALENT, not DEGRADED."""


@dataclass
class LLMJudge:
    model_id: str = "gpt-4o-mini"
    #: ``openai`` | ``groq``. Which client carries the request; the model must belong
    #: to a different family than the generator either way.
    backend: str = "openai"
    name: str = "llm-judge"
    directional: bool = True

    def __post_init__(self) -> None:
        self.name = f"llm-judge({self.model_id})"

    def _chat(self):
        if self.backend == "groq":
            from autopsy.providers.groq import GroqLLM

            return GroqLLM()
        from autopsy.providers.live import OpenAIChat

        return OpenAIChat()

    def compare(self, question: str, a: str, b: str) -> tuple[Verdict, str]:
        user = f"Question: {question}\n\n--- Answer A ---\n{a}\n\n--- Answer B ---\n{b}"
        text = self._chat().complete(
            system=_JUDGE_SYSTEM, user=user, model_id=self.model_id, max_tokens=160
        ).text
        head, _, rest = text.strip().partition("\n")
        word = head.strip().upper()
        mapping = {
            "EQUIVALENT": Verdict.EQUIVALENT,
            "DEGRADED": Verdict.DEGRADED,
            "CONTRADICTORY": Verdict.CONTRADICTORY,
        }
        if word not in mapping:
            # An unparseable verdict is unstable, not a silent EQUIVALENT. Defaulting
            # to the benign class would quietly suppress real differences.
            return Verdict.UNSTABLE, f"unparseable judge output: {head[:80]!r}"
        return mapping[word], rest.strip()[:200]


def build_judge(provider: str) -> Judge:
    """Pick a judge whose family differs from the generator's.

    Self-preference is the bias with no workaround other than choosing a different
    model: a Llama judge scoring Llama answers inflates every number it produces, and
    no prompt fixes that. Groq happens to serve three separate families, so the rule
    survives the move off Anthropic/OpenAI — generation runs on Llama, judging on Qwen.
    """
    if provider == "offline":
        return TwoWayJudge(RuleJudge())
    if provider == "groq":
        from autopsy.providers.groq import GROQ_JUDGE_MODEL

        return TwoWayJudge(LLMJudge(model_id=GROQ_JUDGE_MODEL, backend="groq"))
    return TwoWayJudge(LLMJudge())


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------


@dataclass
class GoldenCase:
    case_id: str
    question: str
    answer_a: str
    answer_b: str
    label: Verdict
    source: str  # "human" | "derived"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoldenCase":
        return cls(
            case_id=d["case_id"], question=d["question"], answer_a=d["answer_a"],
            answer_b=d["answer_b"], label=Verdict(d["label"]), source=d.get("source", "derived"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "question": self.question, "answer_a": self.answer_a,
            "answer_b": self.answer_b, "label": self.label.value, "source": self.source,
        }


@dataclass
class Calibration:
    n: int
    agreement: float
    kappa: float
    confusion: dict[tuple[str, str], int]
    disagreements: list[tuple[GoldenCase, Verdict, str]]
    instability_rate: float
    verbosity_bias: float
    label_source: str
    judge_name: str
    unstable: int = 0
    notes: list[str] = field(default_factory=list)


def load_golden(directory: Path = GOLDEN_DIR) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(GoldenCase.from_dict(json.loads(line)))
    return cases


def cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    """Chance-corrected agreement.

    Raw agreement is inflated whenever one class dominates, and in this data
    ``EQUIVALENT`` dominates heavily — a judge that answered "equivalent" to
    everything would score above 70% raw. Kappa is the number to read.
    """
    if not pairs:
        return 0.0
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    gold = Counter(a for a, _ in pairs)
    pred = Counter(b for _, b in pairs)
    expected = sum((gold[c] / n) * (pred[c] / n) for c in set(gold) | set(pred))
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1 - expected)


def calibrate(judge: Judge, golden: list[GoldenCase]) -> Calibration:
    pairs: list[tuple[str, str]] = []
    confusion: Counter[tuple[str, str]] = Counter()
    disagreements: list[tuple[GoldenCase, Verdict, str]] = []
    length_deltas: list[tuple[float, int]] = []
    unstable = 0

    for case in golden:
        verdict, why = judge.compare(case.question, case.answer_a, case.answer_b)
        if verdict == Verdict.UNSTABLE:
            unstable += 1
        pairs.append((case.label.value, verdict.value))
        confusion[(case.label.value, verdict.value)] += 1
        if verdict != case.label:
            disagreements.append((case, verdict, why))
        length_deltas.append(
            (len(case.answer_b) - len(case.answer_a), 1 if verdict == Verdict.EQUIVALENT else 0)
        )

    agreement = sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else 0.0
    sources = {c.source for c in golden}
    return Calibration(
        n=len(golden),
        agreement=agreement,
        kappa=cohens_kappa(pairs),
        confusion=dict(confusion),
        disagreements=disagreements,
        instability_rate=getattr(judge, "instability_rate", 0.0),
        verbosity_bias=_point_biserial(length_deltas),
        label_source="+".join(sorted(sources)) if sources else "none",
        judge_name=judge.name,
        unstable=unstable,
    )


def _point_biserial(rows: list[tuple[float, int]]) -> float:
    """Correlation between "answer B is longer" and "judge called them equivalent".

    A positive value means the judge is more forgiving of longer answers, which is the
    verbosity bias in its measurable form.
    """
    if len(rows) < 3:
        return 0.0
    xs = [r[0] for r in rows]
    ys = [float(r[1]) for r in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return round(num / (dx * dy), 4) if dx and dy else 0.0


def render_calibration(cal: Calibration) -> str:
    labels = sorted({k[0] for k in cal.confusion} | {k[1] for k in cal.confusion})
    lines = [
        "# Judge calibration",
        "",
        f"judge: `{cal.judge_name}` · n={cal.n} · labels: **{cal.label_source}**",
        "",
    ]
    if "human" not in cal.label_source:
        lines += [
            "> **These are derived labels, not human labels.** They come from the synthetic",
            "> corpus's ground truth: each generated fact carries a globally unique token, so",
            "> whether an answer is correct is a substring check, and the reference label for",
            "> a pair follows from the two correctness values. That makes the number below",
            "> objective and reproducible, but it is *not* the number the spec asks for.",
            "> Agreement with a human is a different and harder question, because a human",
            "> also judges whether a differently-worded answer means the same thing.",
            "> Drop human-labelled rows into `evals/golden/human.jsonl` and re-run; the",
            "> harness will use them and this warning will disappear.",
            "",
        ]
    lines += [
        "## Headline",
        "",
        f"- agreement: **{cal.agreement:.2f}**",
        f"- Cohen's kappa: **{cal.kappa:.2f}** — read this one, not the raw agreement; "
        "one class dominates and inflates the raw figure",
        f"- order instability: **{cal.instability_rate:.2f}** of comparisons changed verdict "
        "when the two answers were swapped",
        f"- verbosity bias: **{cal.verbosity_bias:+.3f}** correlation between "
        "'B is longer' and 'judge said equivalent' (0 is unbiased)",
        f"- unparseable or unstable verdicts: {cal.unstable}",
        "",
        "## Confusion matrix",
        "",
        "| reference \\ judge | " + " | ".join(labels) + " |",
        "|---" * (len(labels) + 1) + "|",
    ]
    for gold in labels:
        row = [f"| **{gold}** "]
        for pred in labels:
            row.append(f"| {cal.confusion.get((gold, pred), 0)} ")
        lines.append("".join(row) + "|")
    off_diagonal = sorted(
        ((n, g, p) for (g, p), n in cal.confusion.items() if g != p), reverse=True
    )
    if off_diagonal:
        n, gold, pred = off_diagonal[0]
        share = 100.0 * n / cal.n if cal.n else 0.0
        lines += [
            "",
            "## The systematic error",
            "",
            f"The single largest disagreement is **reference `{gold}` → judge "
            f"`{pred}`**, {n} cases ({share:.0f}% of the set). That is not noise; it is "
            "a direction. Read it as a standing correction to apply to every "
            f"judge-derived number: this judge under-reports `{gold}` and over-reports "
            f"`{pred}`.",
        ]
        if gold == "contradictory" and pred == "degraded":
            lines += [
                "",
                "Concretely: when two answers assert *different facts* with similar "
                "vocabulary — the same sentence shape with one identifier swapped — a "
                "judge scoring lexical overlap sees a near-match and calls it a mild "
                "degradation. Distinguishing them needs domain knowledge the rule "
                "judge does not have. This is the main reason the ablation table's "
                "headline column is computed from ground truth rather than from the "
                "judge wherever ground truth exists.",
            ]

    lines += ["", "## Where the judge and the reference disagree", ""]
    if not cal.disagreements:
        lines.append("none")
    for case, verdict, why in cal.disagreements[:25]:
        lines += [
            f"### `{case.case_id}` — reference `{case.label.value}`, judge `{verdict.value}`",
            "",
            f"> {case.question}",
            "",
            f"- **A**: {case.answer_a[:220]}",
            f"- **B**: {case.answer_b[:220]}",
            f"- judge said: {why}",
            "",
        ]
    if len(cal.disagreements) > 25:
        lines.append(f"_...and {len(cal.disagreements) - 25} more._")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "Calibration",
    "GOLDEN_DIR",
    "GoldenCase",
    "Judge",
    "LLMJudge",
    "RuleJudge",
    "TwoWayJudge",
    "Verdict",
    "build_judge",
    "calibrate",
    "cohens_kappa",
    "load_golden",
    "render_calibration",
]
