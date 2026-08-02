"""Pipeline behaviour, trace integrity, and the cache-key bug that would prove nothing."""

from __future__ import annotations

from dataclasses import replace

import pytest

from autopsy.ablations import apply
from autopsy.cache import SemanticAnswerCache, StageCache
from autopsy.config import GenerationConfig, default_config
from autopsy.ingest import Document, build_index
from autopsy.pipeline import Pipeline
from autopsy.store.chunks import GLOBAL_TENANT, Index
from autopsy.trace import STAGE_ORDER, Trace

CFG = default_config()


@pytest.fixture(scope="module")
def index() -> Index:
    docs = [
        Document(
            tenant_id="tenant_a",
            doc_id="guide.md",
            markdown=(
                "# Widget guide\n\n"
                "## Error ERR-4021\n\n"
                "ERR-4021 means the flange budget was exhausted during assembly. The "
                "operation is abandoned and retried later; nothing is lost, but the "
                "backlog grows until a retry succeeds and the queue drains again.\n\n"
                "## Error ERR-4022\n\n"
                "ERR-4022 means the spindle budget was exhausted during assembly. This "
                "is distinct from ERR-4021 and has a different remedy; raising the "
                "spindle allowance resolves it and restarting the node does not.\n\n"
                "## Throughput\n\n"
                "Throughput is bounded by the configured cap per cycle. Each cycle "
                "processes the oldest eligible item to completion before moving on, so "
                "one oversized item delays everything behind it in the queue.\n"
            ),
        ),
        Document(
            tenant_id="tenant_b",
            doc_id="guide.md",
            markdown=(
                "# Gadget guide\n\n"
                "## Error ERR-4021\n\n"
                "ERR-4021 means the tenant_b bobbin budget was exhausted during "
                "assembly. This document belongs to a different tenant entirely and "
                "must never appear in a tenant_a answer under any circumstances.\n"
            ),
        ),
        Document(
            tenant_id=GLOBAL_TENANT,
            doc_id="shared.md",
            markdown=(
                "# Shared policy\n\n"
                "## Support\n\n"
                "Every tenant may read this shared policy document. It describes the "
                "escalation path and applies uniformly regardless of which product the "
                "question concerns.\n"
            ),
        ),
    ]
    ix, _ = build_index(docs, cfg=CFG, label="test")
    return ix


@pytest.fixture()
def pipe(index: Index) -> Pipeline:
    return Pipeline(index)


# --------------------------------------------------------------------------------------
# Trace integrity
# --------------------------------------------------------------------------------------


def test_trace_round_trips_through_json(pipe: Pipeline):
    trace = pipe.run("what does ERR-4021 mean", tenant_id="tenant_a", cfg=CFG)
    restored = Trace.from_json(trace.to_json())
    assert restored.model_dump() == trace.model_dump()


def test_every_stage_appears_even_when_skipped(pipe: Pipeline):
    """Absence is information. A pipeline that looks shorter under an ablation is the
    wrong impression to leave — the stage declined to run, and the reason is the point."""
    trace = pipe.run("what does ERR-4021 mean", tenant_id="tenant_a",
                     cfg=apply("no_rerank", CFG))
    assert [s.name for s in trace.stages] == list(STAGE_ORDER)
    rerank = trace.stage("rerank")
    assert rerank.skipped and rerank.skip_reason


def test_skip_reasons_are_written_for_a_reader(pipe: Pipeline):
    trace = pipe.run("what does ERR-4021 mean", tenant_id="tenant_a", cfg=CFG)
    for stage in trace.stages:
        if stage.skipped and stage.skip_reason:
            assert len(stage.skip_reason.split()) >= 4, stage.skip_reason
            assert not stage.skip_reason.startswith("cond")


def test_rejected_candidates_are_recorded_not_dropped(pipe: Pipeline):
    trace = pipe.run(
        "what does ERR-4021 mean", tenant_id="tenant_a",
        cfg=replace(CFG, generation=GenerationConfig(max_context_chunks=1)),
    )
    rejected = [c for c in trace.candidates if c.rejected_by is not None]
    assert rejected, "a chunk that lost is more informative than one never retrieved"
    assert all(c.rejected_by == "top_k" for c in rejected)


def test_context_chunks_carry_an_inclusion_reason(pipe: Pipeline):
    trace = pipe.run("what does ERR-4021 mean", tenant_id="tenant_a", cfg=CFG)
    for candidate in trace.context_chunks():
        assert candidate.inclusion_reason in (
            "fused_top_k", "rerank_promoted", "neighbor_expansion",
        )


def test_trace_id_is_stable_for_identical_inputs(pipe: Pipeline):
    a = pipe.run("what does ERR-4021 mean", tenant_id="tenant_a", cfg=CFG)
    b = pipe.run("what does ERR-4021 mean", tenant_id="tenant_a", cfg=CFG)
    assert a.trace_id == b.trace_id


def test_config_in_trace_is_fully_resolved(pipe: Pipeline):
    trace = pipe.run("what does ERR-4021 mean", tenant_id="tenant_a", cfg=CFG)
    assert trace.config["generation"]["temperature"] == 0.0
    assert trace.config["fusion"]["rrf_k"] == 60
    assert set(trace.versions) >= {"corpus", "code", "provider", "gen_model"}


# --------------------------------------------------------------------------------------
# The cache bug
# --------------------------------------------------------------------------------------


def test_cache_key_includes_the_config_hash(pipe: Pipeline):
    """The subtlest bug available: without the config hash in the key, an ablated run
    serves the baseline's cached answer, every diff returns IDENTICAL, and the
    counterfactual engine appears to work while measuring nothing."""
    query = "what does ERR-4021 mean"
    base = pipe.run(query, tenant_id="tenant_a", cfg=CFG)
    ablated = pipe.run(query, tenant_id="tenant_a", cfg=apply("top_k_1", CFG))
    assert base.config_hash != ablated.config_hash
    assert len(base.context_chunks()) != len(ablated.context_chunks())


def test_stage_cache_key_separates_tenants():
    key_a = StageCache.key(config_hash="h", tenant_id="tenant_a", stage="generate", payload=["q"])
    key_b = StageCache.key(config_hash="h", tenant_id="tenant_b", stage="generate", payload=["q"])
    assert key_a != key_b


def test_semantic_cache_rejects_a_discriminator_mismatch():
    """'config value in Kelvin 6' and '... Kelvin 7' sit at ~0.98 similarity and have
    different answers. No threshold separates them; only an exact discriminator match
    does."""
    from autopsy.textutil import concept_set

    cache = SemanticAnswerCache(threshold=0.5)
    q6 = "what is the default config value in Kelvin 6"
    q7 = "what is the default config value in Kelvin 7"
    cache.store(config_hash="h", tenant_id="t", query=q6, concepts=concept_set(q6), value="six")

    hit, _ = cache.lookup(config_hash="h", tenant_id="t", query=q7, concepts=concept_set(q7))
    assert hit is None
    assert cache.rejected_by_discriminator == 1

    hit, state = cache.lookup(config_hash="h", tenant_id="t", query=q6, concepts=concept_set(q6))
    assert hit == "six" and state == "hit"


def test_semantic_cache_is_namespaced_by_tenant():
    from autopsy.textutil import concept_set

    cache = SemanticAnswerCache(threshold=0.5)
    q = "how long is data retained"
    cache.store(config_hash="h", tenant_id="tenant_a", query=q, concepts=concept_set(q), value="30")
    hit, _ = cache.lookup(config_hash="h", tenant_id="tenant_b", query=q, concepts=concept_set(q))
    assert hit is None, "an answer cache keyed on query text alone leaks across tenants"


# --------------------------------------------------------------------------------------
# Tenant boundary
# --------------------------------------------------------------------------------------


def test_no_foreign_chunk_ever_becomes_a_candidate(pipe: Pipeline):
    trace = pipe.run("what does ERR-4021 mean", tenant_id="tenant_a", cfg=CFG)
    assert all(c.tenant_id in ("tenant_a", GLOBAL_TENANT) for c in trace.candidates)
    assert "bobbin" not in trace.answer.text


def test_global_documents_stay_reachable(index: Index):
    assert any(c.tenant_id == GLOBAL_TENANT for c in index.scope("tenant_a"))


def test_neighbours_never_cross_tenants_despite_a_shared_doc_id(index: Index):
    """Both tenants use doc_id 'guide.md'. Keying neighbours on doc_id alone would
    cross the boundary here."""
    chunk = next(c for c in index.chunks if c.tenant_id == "tenant_a")
    for neighbour in index.neighbours(chunk, 3):
        assert neighbour.tenant_id == "tenant_a"


def test_degenerate_tenant_ids_narrow_rather_than_widen(pipe: Pipeline):
    for bad in ("", "*", "tenant_a' OR '1'='1", "TENANT_A"):
        trace = pipe.run("ERR-4021", tenant_id=bad, cfg=CFG)
        assert all(c.tenant_id in (bad, GLOBAL_TENANT) for c in trace.candidates)


# --------------------------------------------------------------------------------------
# Ablations actually change something
# --------------------------------------------------------------------------------------


def test_ablating_the_lexical_leg_removes_the_sparse_signal(pipe: Pipeline):
    trace = pipe.run("what does ERR-4021 mean", tenant_id="tenant_a",
                     cfg=apply("no_lexical", CFG))
    assert trace.stage("retrieve_sparse").skipped
    assert all(c.lexical_rank is None for c in trace.candidates)


def test_gate_refuses_when_retrieval_is_weak(pipe: Pipeline):
    trace = pipe.run("how do I configure the quantum flux capacitor",
                     tenant_id="tenant_a", cfg=CFG)
    assert trace.answer.status == "refused"


def test_removing_the_gate_removes_the_refusal(pipe: Pipeline):
    query = "how do I configure the quantum flux capacitor"
    gated = pipe.run(query, tenant_id="tenant_a", cfg=CFG)
    ungated = pipe.run(query, tenant_id="tenant_a", cfg=apply("no_gate", CFG))
    if gated.answer.status == "refused":
        assert ungated.stage("gate").skipped


def test_gate_records_that_it_could_not_run_rather_than_passing_silently(pipe: Pipeline):
    """Ablating the dense leg removes the gate's input signal. Silently passing every
    query would be a security-relevant regression disguised as a retrieval change."""
    trace = pipe.run("what does ERR-4021 mean", tenant_id="tenant_a",
                     cfg=apply("no_semantic", CFG))
    gate = trace.stage("gate")
    assert gate.skipped
    assert "NOT checked" in (gate.skip_reason or "")
