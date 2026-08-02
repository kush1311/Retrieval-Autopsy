"""Pipeline configuration.

Design principle #1: config is data, not code. The pipeline is a pure function of
``(query, tenant, PipelineConfig)``. An ablation is a config transform, nothing more.

Every stage-shaped field is ``Optional``; ``None`` means the stage is ablated out.
Everything is a frozen dataclass so a config is hashable, safe to share across
concurrently running variants, and cheaply transformed with ``dataclasses.replace``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal

# --------------------------------------------------------------------------------------
# Stage configs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LexicalConfig:
    """BM25 leg."""

    top_k: int = 20
    b: float = 0.75
    k1: float = 1.2


@dataclass(frozen=True, slots=True)
class SemanticConfig:
    """Dense vector leg."""

    model_id: str = "text-embedding-3-small"
    top_k: int = 20


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Reciprocal Rank Fusion."""

    rrf_k: int = 60


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Refuse when retrieval confidence is too low.

    ``reads`` is a config field rather than a hardcoded choice specifically so the
    alternative can be measured instead of argued about. See the open decision in
    the spec, A.4:

    * ``dense_top1``  — raw cosine of the best dense hit. Interpretable, stable
      across ``top_k`` changes. The default, and the defensible one.
    * ``fused_top1``  — RRF score of the fused winner. Rank-derived, so the
      threshold is meaningless to a human and drifts with candidate count.
    * ``lexical_top1`` — raw BM25 of the best sparse hit. Unbounded scale.
    """

    threshold: float = 0.42
    reads: Literal["dense_top1", "fused_top1", "lexical_top1"] = "dense_top1"


@dataclass(frozen=True, slots=True)
class RerankConfig:
    """LLM-as-reranker, fired only in the gray zone unless ``always``.

    The gray zone is expressed **relative to the gate threshold**, not as an absolute
    score, because it reads whatever signal ``gate.reads`` names and those signals do
    not share a scale. An absolute ``0.60`` is a sensible midpoint for cosine and
    meaningless against BM25, where top-1 is routinely 5–20: the reranker silently
    never fires, `no_rerank` becomes a no-op, and that row of the ablation table
    reports "no measurable effect" for a component that was switched off the whole
    time. Nothing errors. Relative thresholds are scale-free and stay correct when the
    signal changes.
    """

    model_id: str = "claude-haiku-4-5-20251001"
    always: bool = False
    #: Rerank when top-1 is below ``gate.threshold × this``. At 1.6, a gate of 0.42
    #: makes the gray zone anything under 0.67.
    gray_zone_ratio: float = 1.6
    #: Rerank when the top-1/top-2 gap is under this fraction of top-1 — a thin margin
    #: means the ordering is close to arbitrary regardless of absolute score.
    gray_zone_margin_ratio: float = 0.15
    top_n: int = 10


@dataclass(frozen=True, slots=True)
class ExpansionConfig:
    """Neighbour expansion by ``ordinal`` within the same document."""

    neighbours: int = 1


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    model_id: str = "claude-sonnet-4-6"
    temperature: float = 0.0  # never change this; asserted at runtime

    #: How many chunks reach the generator. **Derived from measurement, not inherited.**
    #:
    #: This was 12, and 12 silently made the entire ablation study unmeasurable. Twelve
    #: chunks of a corpus whose median chunk is ~111 tokens is ~1,300 tokens of context
    #: for a single-fact question: wide enough to contain the right chunk however badly
    #: ranking was damaged, so every ablation returned "identical" and the findings table
    #: was all zeros. Nothing errored. The engine looked like it worked.
    #:
    #: Measured gold-retention spread across ablations (see
    #: reports/context-sensitivity.md):
    #:
    #:     width  1 → 20pp    width  5 →  7pp
    #:     width  2 → 14pp    width  8 →  4pp
    #:     width  3 → 12pp    width 12 →  4pp
    #:
    #: 3 is the smallest width that keeps baseline retention reasonable (71%) while
    #: leaving ranking load-bearing, and it matches what production RAG actually uses.
    #: The full curve is published so the choice is auditable rather than convenient —
    #: picking the width that maximises an effect and reporting only that width would be
    #: tuning the benchmark to produce a result.
    max_context_chunks: int = 3
    max_tokens: int = 700
    #: Hedge when the query names an identifier, version, or polarity word that appears
    #: in none of the retrieved sources.
    #:
    #: This is the difference between "wrong" and "wrong and confident", and it is a
    #: config field precisely so the ablation study can put a number on it rather than
    #: asserting it. Ask about `KLV-4213` when only `KLV-4212` and `KLV-4214` were
    #: retrieved: with the guard on you get a flagged non-answer, with it off you get a
    #: fluent description of the wrong error code. Most systems ship without it.
    discriminator_guard: bool = True


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Not a pipeline stage, but it changes pipeline behaviour, so it is part of the
    config hash.

    Leaving the provider out of the hash was tempting and would have been a bug: the
    cache is keyed on the config hash, so an offline-provider answer would be served
    to a real-provider run.

    ``provider`` and ``embedder`` are separate fields because they are separate
    decisions. Groq serves chat models and has no embeddings endpoint at all, so a
    single "provider" axis cannot describe a Groq run — you would have to either
    pretend the dense leg is Groq's or hide which model produced it. Two fields keep
    both halves visible in every trace.

    * ``concept``   — the offline simulator's concept bags. Free, deterministic, not
      a real embedding model.
    * ``fastembed`` — BAAI/bge-small-en-v1.5 via ONNX, running locally. Real
      embeddings, no API key, no GPU, ~130MB of model on first use.
    * ``openai``    — text-embedding-3-small. Costs money.
    """

    provider: Literal["offline", "live", "groq", "openai"] = "offline"
    embedder: Literal["concept", "fastembed", "openai"] = "concept"
    seed: int = 1729
    cache_enabled: bool = True


# --------------------------------------------------------------------------------------
# Root config
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    lexical: LexicalConfig | None = field(default_factory=LexicalConfig)
    semantic: SemanticConfig | None = field(default_factory=SemanticConfig)
    fusion: FusionConfig | None = field(default_factory=FusionConfig)
    gate: GateConfig | None = field(default_factory=GateConfig)
    rerank: RerankConfig | None = field(default_factory=RerankConfig)
    expansion: ExpansionConfig | None = field(default_factory=ExpansionConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    rewrite_enabled: bool = True

    def __post_init__(self) -> None:
        if self.lexical is None and self.semantic is None:
            raise ValueError(
                "ablating both retrieval legs leaves no candidates at all; "
                "that is a broken config, not an interesting ablation"
            )


#: Per-provider defaults. Model IDs and the gate threshold both change with the
#: provider, and neither transfers.
#:
#: The gate especially. It reads a raw similarity score, and every embedding model puts
#: that score on its own scale: the concept simulator is query-coverage in [0,1] with a
#: true floor at 0, while BGE-small puts *unrelated* text at ~0.55 and relevant matches
#: at 0.72–0.85. Carrying 0.42 over to BGE means the gate passes literally everything
#: and silently stops existing. These are starting points — derive yours with
#: `python -m autopsy.cli calibrate-gate`.
PROVIDER_DEFAULTS: dict[str, dict[str, object]] = {
    "offline": {
        "embedder": "concept",
        "embed_model": "offline-concept",
        "generation_model": "claude-sonnet-4-6",
        "rerank_model": "claude-haiku-4-5-20251001",
        "gate_reads": "dense_top1",
        "gate_threshold": 0.42,
    },
    "live": {
        "embedder": "openai",
        "embed_model": "text-embedding-3-small",
        "generation_model": "claude-sonnet-4-6",
        "rerank_model": "claude-haiku-4-5-20251001",
        "gate_reads": "dense_top1",
        "gate_threshold": 0.42,
    },
    "groq": {
        # Groq has no embeddings endpoint, so the dense leg runs locally on ONNX.
        "embedder": "fastembed",
        "embed_model": "BAAI/bge-small-en-v1.5",
        # gpt-oss-120b rather than llama-3.3-70b: each Groq model has its own daily
        # token budget, and the 70b's 100k/day is ~43 questions. See groq.py.
        "generation_model": "openai/gpt-oss-120b",
        "rerank_model": "llama-3.1-8b-instant",
        # Both values derived with `calibrate-gate`, not guessed — and the guess would
        # have been wrong in a way that produced no error. BGE-small scores unrelated
        # English at ~0.55 and cannot separate "configure wavefront pinning in Kelvin"
        # (fictional) from "configure compaction in Kelvin" (real), because the shared
        # phrasing dominates the embedding. Measured separation on `dense_top1` was
        # *negative*. BM25 separates them cleanly, because the fictional subsystem's
        # tokens appear in no document at all.
        # Measured: lexical_top1 separates +1.675 with 0% false refusals and 4% false
        # admits; dense_top1 separates -0.037 and fused_top1 -0.002 (both overlap
        # completely). The spec recommended dense_top1; on this corpus with this
        # embedder that recommendation is wrong, and the config field is what let us
        # find out instead of arguing.
        #
        # Caveat worth carrying: BM25 wins here partly because the out-of-scope queries
        # name invented subsystems, so their tokens appear in no document. An
        # out-of-scope question phrased entirely in in-corpus vocabulary would slip
        # past this signal. Re-run `calibrate-gate` against your own negatives.
        "gate_reads": "lexical_top1",
        "gate_threshold": 1.738,
    },
    "openai": {
        # gpt-4o-mini for every chat role. Cheap — about $0.0003 a query — but not free,
        # which is why `groq` is the default rather than this.
        #
        # The judge shares a family with the generator here. Models prefer their own
        # output, so every judge-derived number under this provider is an upper bound;
        # see autopsy/providers/openai_chat.py and the calibration report. Set
        # AUTOPSY_JUDGE_MODEL to something outside the family to restore independence.
        "embedder": "openai",
        "embed_model": "text-embedding-3-small",
        "generation_model": "gpt-4o-mini",
        "rerank_model": "gpt-4o-mini",
        # text-embedding-3-small scores unrelated English at ~0.17 cosine against
        # bge-small's ~0.55, so dense_top1 is plausible here where it was useless on
        # Groq. Provisional until `calibrate-gate` has actually run against it — the
        # Groq numbers above are measured, these two are not.
        "gate_reads": "dense_top1",
        "gate_threshold": 0.42,
    },
}


#: The model each embedder actually loads, used when AUTOPSY_EMBEDDER is overridden away
#: from its provider's default. Keys must match RuntimeConfig.embedder exactly.
EMBEDDER_MODELS: dict[str, str] = {
    "concept": "offline-concept",
    "fastembed": "BAAI/bge-small-en-v1.5",
    "openai": "text-embedding-3-small",
}


def default_config() -> PipelineConfig:
    """The resolved baseline for the selected provider.

    Provider defaults to ``offline``. Silently upgrading to a paid provider because a
    key happened to be exported is the kind of surprise that arrives as a bill, so the
    opt-in is explicit even when credentials are sitting right there.
    """
    provider = os.environ.get("AUTOPSY_PROVIDER", "offline").strip().lower()
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(
            f"AUTOPSY_PROVIDER must be one of {sorted(PROVIDER_DEFAULTS)}, got {provider!r}"
        )
    d = PROVIDER_DEFAULTS[provider]

    embedder = os.environ.get("AUTOPSY_EMBEDDER", str(d["embedder"])).strip().lower()

    # The embedding model belongs to the *embedder*, not the provider.
    #
    # This used to read `d["embed_model"]` unconditionally, which meant overriding only
    # AUTOPSY_EMBEDDER produced an impossible pair: `provider=offline` +
    # `embedder=fastembed` handed fastembed the model name "offline-concept" and crashed
    # with `Model offline-concept is not supported in TextEmbedding`. The two axes are
    # documented as independent, and that made them silently coupled — found while
    # auditing the isolation suite under a non-default embedder.
    if embedder == str(d["embedder"]):
        default_embed_model = str(d["embed_model"])       # provider's own pairing
    else:
        default_embed_model = EMBEDDER_MODELS[embedder]   # overridden: use its own model
    embed_model = os.environ.get("AUTOPSY_EMBED_MODEL", default_embed_model)
    gate_threshold = float(os.environ.get("AUTOPSY_GATE_THRESHOLD", d["gate_threshold"]))
    gate_reads = os.environ.get("AUTOPSY_GATE_READS", str(d["gate_reads"])).strip()

    return PipelineConfig(
        semantic=SemanticConfig(model_id=embed_model),
        gate=GateConfig(threshold=gate_threshold, reads=gate_reads),  # type: ignore[arg-type]
        rerank=RerankConfig(model_id=str(d["rerank_model"])),
        generation=GenerationConfig(model_id=str(d["generation_model"])),
        runtime=RuntimeConfig(provider=provider, embedder=embedder),  # type: ignore[arg-type]
    )


def stage_enabled(cfg: PipelineConfig, stage: str) -> bool:
    """Is this stage present in the config at all?"""
    match stage:
        case "rewrite":
            return cfg.rewrite_enabled
        case "embed" | "retrieve_dense":
            return cfg.semantic is not None
        case "retrieve_sparse":
            return cfg.lexical is not None
        case "fuse":
            return cfg.fusion is not None
        case "gate":
            return cfg.gate is not None
        case "rerank":
            return cfg.rerank is not None
        case "expand":
            return cfg.expansion is not None
        case "generate":
            return True
        case _:
            raise KeyError(stage)


__all__ = [
    "LexicalConfig",
    "SemanticConfig",
    "FusionConfig",
    "GateConfig",
    "RerankConfig",
    "ExpansionConfig",
    "GenerationConfig",
    "RuntimeConfig",
    "PipelineConfig",
    "default_config",
    "stage_enabled",
    "replace",
]
