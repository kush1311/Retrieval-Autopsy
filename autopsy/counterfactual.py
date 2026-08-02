"""The counterfactual engine — the bridge between the demo and the finding.

One query ablated is a demo. Two hundred queries ablated is a result. Same code path,
same trace schema, same classifier; only the loop is different.

Four properties this module is responsible for, in order of how expensive they are to
get wrong:

1. **It refuses to compare traces from different provenance.** ``assert_comparable``
   raises, it does not warn, when the corpus, models, provider, or code differ. A diff
   across a model upgrade looks exactly as meaningful as a real one and is worthless.
2. **Cheap checks run before expensive ones.** Byte-equal answers short-circuit to
   ``IDENTICAL`` without touching the judge. Most cells in the findings table are
   decided by string comparison.
3. **Ground truth beats the judge wherever it exists.** The generated corpus attaches
   a globally unique token to every fact, so "did this answer come from the right
   chunk" is a substring check. The judge is called only for the narrower question of
   whether two *differently worded* answers say the same thing — which shrinks both the
   cost and the amount of the headline number that rests on a model's opinion.
4. **The explanation is computed, not generated.** "``c_12`` fell from fused rank 1 to
   rank 9 and dropped out of context" is derivable from the two traces. Asking a model
   to narrate a diff it cannot verify would put a plausible sentence where a checkable
   one belongs.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from autopsy.ablations import EXPECTED_FAILURE, apply
from autopsy.config import PipelineConfig
from autopsy.determinism import assert_comparable
from autopsy.pipeline import Pipeline, PipelineError
from autopsy.trace import Trace


class Outcome(str, Enum):
    IDENTICAL = "identical"                      # byte-equal answers
    EQUIVALENT = "equivalent"                    # different words, same claims
    DEGRADED = "degraded"                        # worse but not wrong
    NOW_REFUSES = "now_refuses"                  # baseline answered, variant refused
    NOW_ANSWERS = "now_answers"                  # baseline refused, variant answered
    NOW_WRONG = "now_wrong"                      # correct -> incorrect, but flagged
    NOW_CONFIDENT_WRONG = "now_confident_wrong"  # the money category
    IMPROVED = "improved"                        # incorrect -> correct
    ERROR = "error"                              # the variant did not complete

    @property
    def is_regression(self) -> bool:
        return self in (
            Outcome.DEGRADED, Outcome.NOW_REFUSES, Outcome.NOW_WRONG,
            Outcome.NOW_CONFIDENT_WRONG, Outcome.ERROR,
        )


#: Column order for the findings table. ``now confident wrong`` sits last because it is
#: the column people read first, and a table is read right-to-left when the rightmost
#: column is the one that matters.
TABLE_COLUMNS = [
    Outcome.IDENTICAL, Outcome.EQUIVALENT, Outcome.IMPROVED, Outcome.DEGRADED,
    Outcome.NOW_REFUSES, Outcome.NOW_ANSWERS, Outcome.NOW_WRONG,
    Outcome.NOW_CONFIDENT_WRONG,
]


@dataclass
class Diff:
    ablation: str
    case_id: str
    baseline_trace_id: str
    variant_trace_id: str
    outcome: Outcome
    rank_delta: dict[str, int] = field(default_factory=dict)
    dropped_from_context: list[str] = field(default_factory=list)
    explanation: str = ""
    cost_delta_usd: float = 0.0
    #: Token delta matters even when cost is zero. Under the offline provider every
    #: price is 0.00, so a cost column alone would report `force_rerank` as free —
    #: burying the entire point of that ablation, which is that it spends more and
    #: buys nothing.
    tokens_delta: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation": self.ablation,
            "case_id": self.case_id,
            "baseline_trace_id": self.baseline_trace_id,
            "variant_trace_id": self.variant_trace_id,
            "outcome": self.outcome.value,
            # Only the largest moves. The full map is one entry per context chunk per
            # diff, which turns the evidence file into 2MB of mostly zeros — and an
            # audit artifact nobody opens is not an audit artifact.
            "rank_delta": dict(
                sorted(self.rank_delta.items(), key=lambda kv: -abs(kv[1]))[:10]
            ),
            "dropped_from_context": self.dropped_from_context,
            "explanation": self.explanation,
            "cost_delta_usd": round(self.cost_delta_usd, 8),
            "tokens_delta": self.tokens_delta,
        }


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------


def _flagged(trace: Trace) -> bool:
    return trace.answer.status == "refused" or trace.answer.hedged


def _correct(trace: Trace, case: dict[str, Any] | None) -> bool | None:
    """Objective correctness from ground truth, or ``None`` when there is none."""
    if not case:
        return None
    if case.get("expect") == "refuse_or_hedge":
        return _flagged(trace)
    key = case.get("answer_key")
    if not key:
        return None
    return str(key).lower() in trace.answer.text.lower()


def _failed(trace: Trace) -> bool:
    """Did this run fail to produce a real answer?

    Checked two ways because a stage error and a pipeline-level abort surface
    differently: an ``error`` on any stage record, or the placeholder text the trace
    builder substitutes when no answer was produced.
    """
    if any(s.error for s in trace.stages):
        return True
    return trace.answer.text.startswith(
        ("The pipeline failed before producing an answer", "No answer was produced")
    )


def classify(
    baseline: Trace,
    variant: Trace,
    *,
    case: dict[str, Any] | None = None,
    judge: Any | None = None,
) -> Outcome:
    assert_comparable(baseline.versions, variant.versions)

    # A run that did not complete is not a result, and two runs that failed the same way
    # are not "identical". Without this check a sweep in which *every* run raised the same
    # exception reports a clean table of 100% identical: the pipeline writes the error into
    # the answer text, both sides match byte-for-byte, and the byte-equality shortcut below
    # files it as agreement.
    #
    # That is not hypothetical. A 3,080-run sweep here did exactly that — an index/embedder
    # mismatch failed every single run, and the findings table came back all zeros with no
    # errors reported anywhere. The engine appeared to work and proved nothing, which is the
    # specific failure this project exists to make visible.
    if _failed(baseline) or _failed(variant):
        return Outcome.ERROR

    if baseline.answer.text == variant.answer.text:
        return Outcome.IDENTICAL

    b_refused = baseline.answer.status == "refused"
    v_refused = variant.answer.status == "refused"

    b_ok = _correct(baseline, case)
    v_ok = _correct(variant, case)

    if b_ok is not None and v_ok is not None:
        # Ground truth available: no model opinion is involved in the headline column.
        if b_ok and v_ok:
            return Outcome.EQUIVALENT
        if not b_ok and v_ok:
            return Outcome.IMPROVED
        if b_ok and not v_ok:
            if v_refused:
                return Outcome.NOW_REFUSES
            return Outcome.NOW_WRONG if _flagged(variant) else Outcome.NOW_CONFIDENT_WRONG
        # Both wrong. The interesting sub-case is losing the *flag* rather than losing
        # the answer: a system that was wrong-but-hedging and is now wrong-and-certain
        # got materially more dangerous without its accuracy changing at all.
        if _flagged(baseline) and not _flagged(variant):
            return Outcome.NOW_CONFIDENT_WRONG
        if not _flagged(baseline) and _flagged(variant):
            return Outcome.NOW_REFUSES if v_refused else Outcome.DEGRADED
        return Outcome.EQUIVALENT

    # No ground truth: fall back to structure, then to the judge.
    if b_refused and not v_refused:
        return Outcome.NOW_ANSWERS
    if v_refused and not b_refused:
        return Outcome.NOW_REFUSES
    if judge is None:
        return Outcome.DEGRADED

    from evals.judge import Verdict

    verdict, _why = judge.compare(baseline.query, baseline.answer.text, variant.answer.text)
    if verdict == Verdict.EQUIVALENT:
        return Outcome.EQUIVALENT
    if verdict == Verdict.CONTRADICTORY:
        return Outcome.NOW_CONFIDENT_WRONG if not _flagged(variant) else Outcome.NOW_WRONG
    return Outcome.DEGRADED


# --------------------------------------------------------------------------------------
# Explanation
# --------------------------------------------------------------------------------------


def explain(baseline: Trace, variant: Trace) -> tuple[str, dict[str, int], list[str]]:
    """Derive the sentence that makes the diff legible, from the two traces alone."""
    b_ctx = {c.chunk_id: c for c in baseline.context_chunks()}
    v_ctx = {c.chunk_id: c for c in variant.context_chunks()}
    dropped = [cid for cid in b_ctx if cid not in v_ctx]

    rank_delta: dict[str, int] = {}
    for chunk_id, b in b_ctx.items():
        v = variant.candidate(chunk_id)
        if b.fused_rank is None:
            continue
        v_rank = v.fused_rank if v is not None and v.fused_rank is not None else None
        rank_delta[chunk_id] = (v_rank - b.fused_rank) if v_rank is not None else 999

    if not dropped:
        stage_changes = [
            s.name
            for s in variant.stages
            if s.skipped and not (baseline.stage(s.name) or s).skipped
        ]
        if stage_changes:
            reasons = "; ".join(
                f"{n}: {(variant.stage(n).skip_reason or 'skipped')}" for n in stage_changes[:2]
            )
            return (
                f"The context was unchanged, but {', '.join(stage_changes)} no longer ran "
                f"({reasons}).",
                rank_delta,
                dropped,
            )
        return ("Same context, same ranking; only the generated wording differs.",
                rank_delta, dropped)

    # A shrunken context budget drops chunks that did not move at all. Reporting
    # "fell from rank 4 to rank 4 and dropped out of context" is technically true and
    # actively misleading about the cause, so budget changes are named as such.
    b_budget = baseline.config.get("generation", {}).get("max_context_chunks")
    v_budget = variant.config.get("generation", {}).get("max_context_chunks")
    if b_budget and v_budget and v_budget < b_budget:
        kept = sorted(v_ctx.values(), key=lambda c: c.final_rank or 10**6)
        survivor = kept[0].chunk_id if kept else "nothing"
        return (
            f"The context budget shrank from {b_budget} chunks to {v_budget}, so "
            f"{len(dropped)} chunks were cut on budget rather than on rank. Only "
            f"`{survivor}` survived.",
            rank_delta,
            dropped,
        )

    # Lead with the highest-ranked chunk that fell out: it is the one that changed the
    # answer, and naming a mid-list casualty buries the cause.
    lead_id = min(dropped, key=lambda cid: b_ctx[cid].final_rank or 10**6)
    lead = b_ctx[lead_id]
    v_lead = variant.candidate(lead_id)

    # Word the movement from what actually happened to the rank.
    #
    # This used to be one template — "fell from fused rank {b} to {where}" — which
    # produced "fell from fused rank 1 to rank 1 and dropped out of context" in the
    # committed report. An unchanged rank narrated as a fall points the reader at the
    # wrong cause: the chunk was displaced by something promoted above it or by
    # expansion filling the window, not out-ranked. The budget guard above catches only
    # the case where max_context_chunks itself shrank.
    b_rank, v_rank = lead.fused_rank, (v_lead.fused_rank if v_lead else None)
    if v_lead is None:
        movement = f"left the candidate list entirely (was fused rank {b_rank})"
    elif v_rank is None:
        movement = f"was still a candidate but went unranked (was fused rank {b_rank})"
    elif b_rank is None:
        movement = f"reached fused rank {v_rank}"
    elif v_rank == b_rank:
        movement = (
            f"held fused rank {b_rank} and still dropped out of context — it was "
            f"displaced by another chunk rather than out-ranked"
        )
    elif v_rank > b_rank:
        movement = f"fell from fused rank {b_rank} to rank {v_rank} and dropped out of context"
    else:
        movement = (
            f"rose from fused rank {b_rank} to rank {v_rank} and dropped out of context "
            f"anyway"
        )

    legs = []
    if lead.lexical_rank is not None:
        legs.append(f"lexical rank {lead.lexical_rank} (score {lead.lexical_score:.2f})")
    if lead.semantic_rank is not None:
        legs.append(f"semantic rank {lead.semantic_rank} (score {lead.semantic_score:.2f})")
    legs_text = " and ".join(legs) if legs else "no per-leg signal recorded"

    tail = ""
    if lead.lexical_rank is not None and (
        v_lead is None or v_lead.lexical_rank is None
    ):
        tail = " Without the lexical leg, only the semantic signal remained."
    elif lead.semantic_rank is not None and (
        v_lead is None or v_lead.semantic_rank is None
    ):
        tail = " Without the semantic leg, only the lexical signal remained."

    heading = " > ".join(lead.heading_path[-2:]) if lead.heading_path else lead.doc_id
    return (
        f"`{lead_id}` ({heading}) {movement}. It had {legs_text}.{tail}"
        + (f" {len(dropped) - 1} other context chunks also dropped." if len(dropped) > 1 else ""),
        rank_delta,
        dropped,
    )


# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------


@dataclass
class CounterfactualEngine:
    pipeline: Pipeline
    base_config: PipelineConfig
    judge: Any | None = None
    #: Concurrency is a network-latency optimisation, so it defaults to serial for the
    #: offline provider: threads there add scheduling noise and contend on the shared
    #: caches without hiding any latency, because there is none to hide.
    max_workers: int = 1
    trace_sink: Callable[[Trace], None] | None = None

    def _run(self, case: dict[str, Any], ablation: str) -> Trace:
        cfg = apply(ablation, self.base_config) if ablation != "baseline" else self.base_config
        try:
            trace = self.pipeline.run(
                case["query"],
                tenant_id=case["tenant_id"],
                cfg=cfg,
                history=case.get("history") or [],
                ablations=[] if ablation == "baseline" else [ablation],
            )
        except PipelineError as exc:
            trace = exc.trace
        if self.trace_sink:
            self.trace_sink(trace)
        return trace

    def compare_one(self, case: dict[str, Any], ablation: str) -> Diff:
        baseline = self._run(case, "baseline")
        variant = self._run(case, ablation)
        return self._diff(case, ablation, baseline, variant)

    def _diff(self, case, ablation: str, baseline: Trace, variant: Trace) -> Diff:
        outcome = classify(baseline, variant, case=case, judge=self.judge)
        text, rank_delta, dropped = explain(baseline, variant)
        return Diff(
            ablation=ablation,
            case_id=case.get("case_id", case["query"][:40]),
            baseline_trace_id=baseline.trace_id,
            variant_trace_id=variant.trace_id,
            outcome=outcome,
            rank_delta=rank_delta,
            dropped_from_context=dropped,
            explanation=text,
            cost_delta_usd=variant.totals.cost_usd - baseline.totals.cost_usd,
            tokens_delta=(
                (variant.totals.tokens_in + variant.totals.tokens_out)
                - (baseline.totals.tokens_in + baseline.totals.tokens_out)
            ),
        )

    def sweep(
        self,
        cases: list[dict[str, Any]],
        ablations: list[str],
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[Diff]:
        """Run every case against every ablation.

        The baseline is executed once per case and reused across all ablations — with
        eight ablations that is a ~47% reduction in work on its own, before the shared
        embedding cache saves the query embedding for every variant except
        ``no_semantic``.
        """
        diffs: list[Diff] = []
        total = len(cases)

        def one_case(case: dict[str, Any]) -> list[Diff]:
            baseline = self._run(case, "baseline")
            out = []
            for ablation in ablations:
                if ablation == "baseline":
                    continue
                try:
                    variant = self._run(case, ablation)
                except Exception as exc:  # noqa: BLE001 - a broken variant is a data point
                    out.append(
                        Diff(
                            ablation=ablation,
                            case_id=case.get("case_id", ""),
                            baseline_trace_id=baseline.trace_id,
                            variant_trace_id="",
                            outcome=Outcome.ERROR,
                            explanation=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    continue
                out.append(self._diff(case, ablation, baseline, variant))
            return out

        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for i, result in enumerate(pool.map(one_case, cases), start=1):
                    diffs.extend(result)
                    if on_progress:
                        on_progress(i, total)
        else:
            for i, case in enumerate(cases, start=1):
                diffs.extend(one_case(case))
                if on_progress:
                    on_progress(i, total)
        return diffs


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


def tabulate(diffs: Iterable[Diff]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for diff in diffs:
        row = table.setdefault(diff.ablation, {})
        row[diff.outcome.value] = row.get(diff.outcome.value, 0) + 1
    return table


def render_table(table: dict[str, dict[str, int]], diffs: list[Diff]) -> str:
    header = "| ablation | " + " | ".join(c.value.replace("_", " ") for c in TABLE_COLUMNS)
    header += " | Δ tokens | Δ cost |"
    sep = "|---" * (len(TABLE_COLUMNS) + 3) + "|"
    lines = [header, sep]
    costs: dict[str, float] = {}
    tokens: dict[str, int] = {}
    for diff in diffs:
        costs[diff.ablation] = costs.get(diff.ablation, 0.0) + diff.cost_delta_usd
        tokens[diff.ablation] = tokens.get(diff.ablation, 0) + diff.tokens_delta
    for ablation in sorted(table, key=lambda a: -table[a].get("now_confident_wrong", 0)):
        row = table[ablation]
        cells = [str(row.get(c.value, 0)) for c in TABLE_COLUMNS]
        lines.append(
            f"| `{ablation}` | " + " | ".join(cells)
            + f" | {tokens.get(ablation, 0):+,} | {costs.get(ablation, 0.0):+.4f} |"
        )
    return "\n".join(lines)


def _interactions_section(
    table: dict[str, dict[str, int]], totals: dict[str, int]
) -> list[str]:
    """Find composites whose damage exceeds the sum of their parts, and say why.

    This is the analysis that makes the whole table honest. When two mechanisms guard
    the same failure, ablating either one alone looks harmless — the other silently
    covers for it — and a study that only reports single ablations concludes both are
    unnecessary. Only the composite reveals the exposure. Leaving a reader to spot a
    0.5%% row next to an 18%% row and infer that themselves is leaving the finding on
    the floor.
    """
    def cw(name: str) -> int:
        return table.get(name, {}).get("now_confident_wrong", 0)

    rows: list[str] = []
    for name in table:
        if "+" not in name:
            continue
        parts = name.split("+")
        if not all(p in table for p in parts):
            continue
        parts_sum = sum(cw(p) for p in parts)
        combined = cw(name)
        if combined <= parts_sum + 2:
            continue
        n = totals.get(name, 1) or 1
        detail = " + ".join(f"`{p}` {cw(p)}" for p in parts)
        rows.append(
            f"- **`{name}`**: {combined} confidently-wrong answers ({100 * combined / n:.1f}%), "
            f"against {parts_sum} for the parts measured separately ({detail}). "
            f"The two mechanisms guard the same failures, so removing either one alone "
            f"looks safe."
        )

    if not rows:
        return []
    return [
        "",
        "## Interactions — where the single-ablation numbers mislead",
        "",
        "Each row below is a composite whose damage is more than the sum of its parts.",
        "",
        *rows,
        "",
        "This is the practical warning in the whole study. A team that removes one of a "
        "redundant pair sees its eval stay green, concludes the component was dead "
        "weight, and ships. The exposure only appears when the second one goes — often "
        "in a later change, by a different person, with no failing test in between.",
    ]


def render_report(
    diffs: list[Diff], *, versions: dict[str, str], n_cases: int, provider: str
) -> str:
    table = tabulate(diffs)
    total_per_ablation = {a: sum(row.values()) for a, row in table.items()}

    lines = ["# Ablation study", ""]
    lines.append(" · ".join(f"{k} {v}" for k, v in sorted(versions.items())))
    lines.append("")
    if provider == "offline":
        from evals.runner import PROVIDER_WARNING

        lines += [PROVIDER_WARNING, ""]
    lines += [
        f"{n_cases} queries × {len(table)} ablations = {sum(total_per_ablation.values())} "
        "variant runs, each against a baseline run of the same query.",
        "",
        "## Findings",
        "",
        render_table(table, diffs),
        "",
        "**Read the last column first.** `now confident wrong` counts answers that were "
        "correct at baseline and became incorrect *with no hedge and no refusal* — a "
        "wrong answer a user has no way to distinguish from a right one. `now refuses` "
        "and `now wrong` are also regressions, but they are loud ones.",
        "",
        "## What each ablation was predicted to do",
        "",
        "| ablation | predicted failure | confident-wrong rate | verdict |",
        "|---|---|---|---|",
    ]
    for ablation in sorted(table, key=lambda a: -table[a].get("now_confident_wrong", 0)):
        row = table[ablation]
        n = total_per_ablation[ablation] or 1
        cw = row.get("now_confident_wrong", 0)
        loud = row.get("now_wrong", 0) + row.get("now_refuses", 0) + row.get("degraded", 0)
        unchanged = row.get("identical", 0) + row.get("equivalent", 0)
        rate = 100.0 * cw / n
        if cw:
            verdict = "confirmed"
        elif loud:
            verdict = f"broke {loud} answers, but all of them loudly"
        elif unchanged >= n * 0.98:
            verdict = "no measurable effect on this corpus"
        else:
            verdict = "changed the answer, not its correctness"
        lines.append(
            f"| `{ablation}` | {EXPECTED_FAILURE.get(ablation, '—')} | {rate:.1f}% | {verdict} |"
        )

    lines += _interactions_section(table, total_per_ablation)

    # Pick the example from the ablation that did the most damage, and prefer a case
    # whose explanation tells the *retrieval* story — a chunk falling out of the
    # ranking is legible; "the budget shrank" is true but explains nothing about why
    # hybrid retrieval matters.
    worst_ablation = max(
        table, key=lambda a: table[a].get("now_confident_wrong", 0), default=None
    )
    candidates = [
        d
        for d in diffs
        if d.outcome == Outcome.NOW_CONFIDENT_WRONG and d.ablation == worst_ablation
    ]
    # Prefer an example whose explanation tells a *ranking* story over one that only
    # says the budget shrank. Matching on any of the rank phrasings rather than the
    # single string "fell from fused rank", which was the only wording before the
    # movement cases were split out and would now silently skip the displaced-but-
    # unmoved case.
    def _tells_a_ranking_story(d: "Diff") -> bool:
        return any(
            phrase in d.explanation
            for phrase in ("fused rank", "candidate list", "went unranked")
        )

    worst = max(
        candidates,
        key=lambda d: (_tells_a_ranking_story(d), len(d.dropped_from_context)),
        default=None,
    )
    if worst is not None:
        lines += [
            "",
            "## A worked example",
            "",
            f"Ablation `{worst.ablation}`, case `{worst.case_id}`:",
            "",
            f"> {worst.explanation}",
            "",
            f"baseline trace `{worst.baseline_trace_id}` · "
            f"variant trace `{worst.variant_trace_id}`",
        ]

    errors = [d for d in diffs if d.outcome == Outcome.ERROR]
    if errors:
        lines += ["", "## Runs that failed outright", ""]
        for diff in errors[:10]:
            lines.append(f"- `{diff.ablation}` / `{diff.case_id}`: {diff.explanation}")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "CounterfactualEngine",
    "Diff",
    "Outcome",
    "TABLE_COLUMNS",
    "classify",
    "explain",
    "render_report",
    "render_table",
    "tabulate",
]
