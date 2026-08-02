"""Named config transforms.

An ablation is not a code path — it is a function ``PipelineConfig -> PipelineConfig``.
That is the entire mechanism, and it only works because no stage is hardcoded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from autopsy.config import PipelineConfig

Ablation = Callable[[PipelineConfig], PipelineConfig]


def _force_rerank(c: PipelineConfig) -> PipelineConfig:
    if c.rerank is None:
        raise ValueError("force_rerank cannot compose with no_rerank")
    return replace(c, rerank=replace(c.rerank, always=True))


def _top_k_1(c: PipelineConfig) -> PipelineConfig:
    return replace(c, generation=replace(c.generation, max_context_chunks=1))


def _gate_on_fused(c: PipelineConfig) -> PipelineConfig:
    if c.gate is None:
        raise ValueError("gate_on_fused cannot compose with no_gate")
    # 1/(60+1) + 1/(60+1) ~= 0.0328 for a doc topping both legs; a doc topping one leg
    # only scores ~0.0164. The threshold sits between them. It is also completely
    # opaque to a human, which is the point being demonstrated.
    return replace(c, gate=replace(c.gate, reads="fused_top1", threshold=0.0250))


ABLATIONS: dict[str, Ablation] = {
    "baseline": lambda c: c,
    "no_lexical": lambda c: replace(c, lexical=None),
    "no_semantic": lambda c: replace(c, semantic=None),
    "no_fusion": lambda c: replace(c, fusion=None),
    "no_gate": lambda c: replace(c, gate=None),
    "no_rerank": lambda c: replace(c, rerank=None),
    "force_rerank": _force_rerank,
    "no_expansion": lambda c: replace(c, expansion=None),
    "top_k_1": _top_k_1,
    "no_rewrite": lambda c: replace(c, rewrite_enabled=False),
    "gate_on_fused": _gate_on_fused,
    "no_discriminator_guard": lambda c: replace(
        c, generation=replace(c.generation, discriminator_guard=False)
    ),
}

#: What each ablation is supposed to demonstrate. Rendered into the ablation report so
#: a reader can check the observed outcome against the predicted one — a prediction
#: that fails is more interesting than one that holds.
EXPECTED_FAILURE: dict[str, str] = {
    "baseline": "control",
    "no_lexical": "exact identifiers become unfindable",
    "no_semantic": "paraphrased queries stop matching",
    "no_fusion": "worse ranking, subtler than either leg alone",
    "no_gate": "hallucinated answer where a refusal was correct",
    "no_rerank": "gray-zone queries degrade",
    "force_rerank": "cost rises, quality usually does not — a useful negative result",
    "no_expansion": "answers truncate mid-context",
    "top_k_1": "confident single-source wrongness",
    "no_rewrite": "follow-up questions lose their referent",
    "gate_on_fused": "threshold becomes uninterpretable and drifts with top_k",
    "no_discriminator_guard": "near-miss identifiers answered confidently instead of flagged",
    "no_lexical+no_discriminator_guard":
        "the same retrieval failure as no_lexical, but silent instead of flagged",
    "no_lexical+no_gate":
        "a retrieval failure and a missing refusal producing a hallucination together",
    "no_gate+no_discriminator_guard": "both refusal mechanisms removed at once",
}

#: If time is short, ship these three. Each produces a dramatic, explainable failure.
CORE_THREE = ("no_lexical", "no_gate", "no_rerank")

#: Composites worth running as their own rows.
#:
#: These are not redundant with their parts. ``no_lexical`` alone loses the exact
#: identifier and the discriminator guard catches it, so the failures are *loud*.
#: Remove the guard as well and the same retrieval failure becomes silent. The single
#: ablations measure "did it break"; the composite measures "did anyone notice", and
#: those are different questions with different answers.
COMPOSITES = (
    "no_lexical+no_discriminator_guard",
    "no_lexical+no_gate",
    "no_gate+no_discriminator_guard",
)


def all_ablations(include_composites: bool = True) -> list[str]:
    names = [n for n in ABLATIONS if n != "baseline"]
    return names + list(COMPOSITES) if include_composites else names


def apply(name: str, cfg: PipelineConfig) -> PipelineConfig:
    """Apply one named ablation. Composite names use ``+``."""
    if "+" in name:
        return compose(name.split("+"), cfg)
    try:
        fn = ABLATIONS[name]
    except KeyError:
        raise KeyError(
            f"unknown ablation {name!r}; known: {', '.join(sorted(ABLATIONS))}"
        ) from None
    return fn(cfg)


def compose(names: list[str] | tuple[str, ...], cfg: PipelineConfig) -> PipelineConfig:
    """Ablations compose. ``no_lexical+no_gate`` shows a retrieval failure and a
    missing refusal producing a confident hallucination *together*, which is a more
    honest picture of how production systems actually fail than either alone."""
    out = cfg
    for n in names:
        out = apply(n.strip(), out)
    return out


def normalise(names: list[str] | tuple[str, ...] | None) -> list[str]:
    """Canonical ordering, ``baseline`` dropped, duplicates removed.

    Canonical because the ablation list goes into the trace ID; ``a+b`` and ``b+a``
    are the same experiment and must not produce two trace files.
    """
    if not names:
        return []
    seen: list[str] = []
    for raw in names:
        for part in str(raw).split("+"):
            part = part.strip()
            if part and part != "baseline" and part not in seen:
                seen.append(part)
    return sorted(seen)


__all__ = [
    "ABLATIONS", "COMPOSITES", "CORE_THREE", "EXPECTED_FAILURE", "Ablation",
    "all_ablations", "apply", "compose", "normalise",
]
