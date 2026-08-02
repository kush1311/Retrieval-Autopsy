"""Derive the gate threshold from the corpus instead of inheriting it.

The gate reads a raw similarity score, and every embedding model puts that score on a
different scale. The concept simulator is query-coverage in [0, 1] with a true floor
at 0. BGE-small puts *unrelated* text at roughly 0.55 and relevant matches at
0.72–0.85. Carrying a threshold across models is how a gate silently stops existing:
set it too low and everything passes, too high and everything refuses. Either way the
trace still says `gate: passed`, and nothing fails.

The corpus already contains the labels needed to pick one honestly. Every generated
case is either **answerable** (`identifier`, `value`, `paraphrase`, `followup` — the
answer is in the corpus) or **out of scope** (`out_of_scope` — the subsystem exists
nowhere). Score both populations, then choose the threshold that best separates them.

Reported alongside the number: the achieved separation, and what it costs. A threshold
with a 30% false-refusal rate is not a good threshold no matter how few hallucinations
it stops, and the report says so rather than presenting one number as an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autopsy.config import PipelineConfig
from autopsy.providers import build_providers
from autopsy.store.chunks import Index
from autopsy.store.vectors import build_vector_store

#: Answerable query shapes. `absent` is deliberately excluded: it asks about an
#: unassigned code inside a *real* family, so retrieval confidence is legitimately
#: high and the gate is not the mechanism that should catch it. Including it would
#: push the threshold up until it started refusing answerable questions too.
ANSWERABLE = {"identifier", "value", "paraphrase", "followup"}
OUT_OF_SCOPE = {"out_of_scope"}


@dataclass
class GateCalibration:
    threshold: float
    embed_model: str
    reads: str
    n_answerable: int
    n_out_of_scope: int
    #: Answerable queries the threshold would wrongly refuse.
    false_refusal_rate: float
    #: Out-of-scope queries it would wrongly admit.
    false_admit_rate: float
    separation: float
    answerable_scores: list[float] = field(default_factory=list)
    oos_scores: list[float] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Is there a threshold worth having at all?

        Below ~0.15 of separation the two populations overlap so much that any choice
        trades one error for the other roughly one-for-one, and the honest answer is
        that this signal cannot gate on this corpus.
        """
        return self.separation >= 0.15


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(p * (len(ordered) - 1)))))
    return ordered[idx]


def score_population(
    index: Index, cfg: PipelineConfig, cases: list[dict[str, Any]], reads: str = "dense_top1"
) -> tuple[list[float], list[float], str]:
    """Return ``(answerable_scores, out_of_scope_scores, embed_model)`` for one signal.

    Scores the retrieval legs directly rather than running the pipeline: the gate reads
    its number before any later stage touches it, so going through the pipeline would
    fold the reranker into the measurement.
    """
    from autopsy.store.lexical import LexicalIndex

    providers = build_providers(cfg, index.stats)
    store = build_vector_store(index, index.stats)
    lexical = LexicalIndex(index)
    answerable: list[float] = []
    oos: list[float] = []

    for case in cases:
        kind = case.get("kind")
        if kind not in ANSWERABLE and kind not in OUT_OF_SCOPE:
            continue
        tenant = case["tenant_id"]

        if reads == "lexical_top1":
            hits = lexical.search(
                query=case["query"], tenant_id=tenant, top_k=1,
                k1=cfg.lexical.k1 if cfg.lexical else 1.2,
                b=cfg.lexical.b if cfg.lexical else 0.75,
            )
            top = float(hits[0][1]) if hits else 0.0
        else:
            embedding, _usage = providers.embedder.embed_query(case["query"])
            dense = store.search(query=embedding, tenant_id=tenant, top_k=20)
            if reads == "dense_top1":
                top = float(dense[0][1]) if dense else 0.0
            else:  # fused_top1
                sparse = lexical.search(
                    query=case["query"], tenant_id=tenant, top_k=20,
                    k1=cfg.lexical.k1 if cfg.lexical else 1.2,
                    b=cfg.lexical.b if cfg.lexical else 0.75,
                )
                k = cfg.fusion.rrf_k if cfg.fusion else 60
                fused: dict[str, float] = {}
                for rank, (chunk, _s) in enumerate(dense, start=1):
                    fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
                for rank, (chunk, _s) in enumerate(sparse, start=1):
                    fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
                top = max(fused.values(), default=0.0)

        (answerable if kind in ANSWERABLE else oos).append(top)

    return answerable, oos, providers.embedder.model_id


SIGNALS = ("dense_top1", "lexical_top1", "fused_top1")


def calibrate_all(
    index: Index, cfg: PipelineConfig, cases: list[dict[str, Any]]
) -> list[GateCalibration]:
    """Calibrate every available gate signal and rank them by separation.

    The spec left "which score does the gate read?" as an open decision and made it a
    config field so it could be measured rather than argued about. This is the
    measurement. The answer is not universal — it depends on the embedding model and
    on what the out-of-scope queries look like — which is exactly why it should not be
    a hardcoded constant in anyone's pipeline.
    """
    results = [calibrate(index, cfg, cases, reads=r) for r in SIGNALS]
    return sorted(results, key=lambda c: -c.separation)


def calibrate(
    index: Index, cfg: PipelineConfig, cases: list[dict[str, Any]],
    reads: str = "dense_top1",
) -> GateCalibration:
    answerable, oos, embed_model = score_population(index, cfg, cases, reads=reads)
    if not answerable or not oos:
        raise ValueError(
            "need both answerable and out_of_scope cases to calibrate; regenerate the "
            "test set with `python -m corpus.synthetic`"
        )

    # Sweep every midpoint between observed scores and keep the one with the best
    # balanced error. Optimising raw accuracy would let the larger population decide
    # the threshold on its own.
    candidates = sorted({round((a + b) / 2, 4) for a in answerable for b in oos})
    best = (1e9, _percentile(oos, 0.95))
    for threshold in candidates:
        fr = sum(1 for s in answerable if s < threshold) / len(answerable)
        fa = sum(1 for s in oos if s >= threshold) / len(oos)
        cost = max(fr, fa)  # balanced: the worse of the two errors
        if cost < best[0]:
            best = (cost, threshold)
    threshold = round(best[1], 3)

    fr = sum(1 for s in answerable if s < threshold) / len(answerable)
    fa = sum(1 for s in oos if s >= threshold) / len(oos)
    return GateCalibration(
        threshold=threshold,
        embed_model=embed_model,
        reads=reads,
        n_answerable=len(answerable),
        n_out_of_scope=len(oos),
        false_refusal_rate=round(fr, 4),
        false_admit_rate=round(fa, 4),
        separation=round(_percentile(answerable, 0.10) - _percentile(oos, 0.90), 4),
        answerable_scores=answerable,
        oos_scores=oos,
    )


def render_comparison(results: list[GateCalibration]) -> str:
    """The signal bake-off, then a full report on the winner."""
    best = results[0]
    lines = [
        "# Gate calibration",
        "",
        f"embedding model: `{best.embed_model}`",
        "",
        "## Which signal should the gate read?",
        "",
        "The spec left this open and made `gate.reads` a config field so it could be",
        "measured instead of argued about. This is the measurement, on this corpus with",
        "this embedding model — it is not a universal answer, which is the point.",
        "",
        "| `gate.reads` | threshold | false refusals | false admits | separation |",
        "|---|---|---|---|---|",
    ]
    for cal in results:
        mark = " **←**" if cal is best else ""
        lines.append(
            f"| `{cal.reads}`{mark} | {cal.threshold:g} | {cal.false_refusal_rate:.1%} | "
            f"{cal.false_admit_rate:.1%} | {cal.separation:+.3f} |"
        )
    lines += [
        "",
        "*Separation is answerable-p10 minus out-of-scope-p90: how much daylight sits",
        "between the two populations. Negative means they overlap and no threshold",
        "separates them.*",
        "",
        "---",
        "",
    ]
    return "\n".join(lines) + render(best)


def render(cal: GateCalibration) -> str:
    lines = [
        f"## Winning signal: `{cal.reads}`",
        "",
        f"embedding model: `{cal.embed_model}`",
        "",
        "### Recommended threshold",
        "",
        f"    gate.reads     = {cal.reads}",
        f"    gate.threshold = {cal.threshold}",
        "",
        "| | answerable | out of scope |",
        "|---|---|---|",
        f"| n | {cal.n_answerable} | {cal.n_out_of_scope} |",
        f"| min | {min(cal.answerable_scores):.3f} | {min(cal.oos_scores):.3f} |",
        f"| p10 | {_percentile(cal.answerable_scores, 0.10):.3f} | "
        f"{_percentile(cal.oos_scores, 0.10):.3f} |",
        f"| median | {_percentile(cal.answerable_scores, 0.50):.3f} | "
        f"{_percentile(cal.oos_scores, 0.50):.3f} |",
        f"| p90 | {_percentile(cal.answerable_scores, 0.90):.3f} | "
        f"{_percentile(cal.oos_scores, 0.90):.3f} |",
        f"| max | {max(cal.answerable_scores):.3f} | {max(cal.oos_scores):.3f} |",
        "",
        "### What this threshold costs",
        "",
        f"- **false refusals**: {cal.false_refusal_rate:.1%} of answerable queries score "
        "below it and would be refused despite the answer being in the corpus",
        f"- **false admits**: {cal.false_admit_rate:.1%} of out-of-scope queries score "
        "above it and reach the generator anyway",
        f"- **separation** (answerable p10 − out-of-scope p90): {cal.separation:+.3f}",
        "",
    ]
    if not cal.usable:
        lines += [
            "> **The two populations overlap too much for this to be a real gate.**",
            "> Separation below 0.15 means every threshold trades one error for the other",
            "> roughly one-for-one. Raising it does not make the system safer, it makes it",
            "> quieter about answerable questions. Either the embedding model is not",
            "> discriminating on this corpus, or the gate needs a different signal —",
            "> `gate.reads` accepts `fused_top1` and `lexical_top1` as well.",
            "",
        ]
    lines += [
        "### Why this is not a constant",
        "",
        "Every embedding model puts similarity on its own scale. The offline concept",
        "simulator scores query-coverage in [0, 1] with a genuine floor at 0;",
        "`bge-small-en-v1.5` scores *unrelated* text around 0.55, because a normalised",
        "cosine between two English sentences is never near zero. A threshold moved",
        "between them without re-deriving it either passes everything or refuses",
        "everything — and in both cases the trace still reports `gate: passed` or",
        "`gate: refused` exactly as if it were working.",
    ]
    return "\n".join(lines) + "\n"


__all__ = ["ANSWERABLE", "GateCalibration", "OUT_OF_SCOPE", "calibrate", "render", "score_population"]
