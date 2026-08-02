"""Determinism, config hashing, and the guards that keep diffs meaningful."""

from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from autopsy.ablations import ABLATIONS, apply, compose, normalise
from autopsy.config import GenerationConfig, PipelineConfig, default_config
from autopsy.determinism import (
    DeterminismError,
    assert_comparable,
    assert_deterministic,
    canonical_json,
    config_hash,
    content_id,
    resolved_config,
)
from autopsy.stages.fuse import rrf_score


def test_config_hash_is_stable_across_equal_configs():
    assert config_hash(default_config()) == config_hash(default_config())


def test_every_ablation_changes_the_config_hash():
    """If an ablation did not change the hash, cache keys would collide across
    variants and every diff would come back IDENTICAL while proving nothing."""
    base = default_config()
    base_hash = config_hash(base)
    for name in ABLATIONS:
        if name == "baseline":
            continue
        assert config_hash(apply(name, base)) != base_hash, name


def test_ablated_stage_is_none_not_missing():
    cfg = apply("no_rerank", default_config())
    assert cfg.rerank is None
    assert "rerank" in resolved_config(cfg)
    assert resolved_config(cfg)["rerank"] is None


def test_resolved_config_is_full_not_a_diff():
    """A trace must be interpretable years later without the code that produced it."""
    resolved = resolved_config(default_config())
    for key in ("lexical", "semantic", "fusion", "gate", "rerank", "expansion",
                "generation", "runtime", "rewrite_enabled"):
        assert key in resolved
    assert resolved["generation"]["temperature"] == 0.0


def test_temperature_assertion_fires():
    cfg = replace(default_config(), generation=GenerationConfig(temperature=0.7))
    with pytest.raises(DeterminismError, match="temperature"):
        assert_deterministic(cfg)


def test_refuses_to_compare_across_provenance():
    a = {"corpus": "seed@aaa", "code": "git@1", "provider": "offline",
         "embed_model": "m", "gen_model": "g", "rerank_model": "r"}
    b = dict(a, corpus="seed@bbb")
    with pytest.raises(DeterminismError, match="refusing to diff"):
        assert_comparable(a, b)


def test_ablating_a_leg_is_not_provenance_drift():
    """no_semantic legitimately sets embed_model to 'none'; that is the experiment,
    not a corpus change, and raising on it would make the ablation unrunnable."""
    a = {"corpus": "c", "code": "x", "provider": "offline",
         "embed_model": "m", "gen_model": "g", "rerank_model": "r"}
    assert_comparable(a, dict(a, embed_model="none"))
    assert_comparable(a, dict(a, rerank_model="none"))


def test_both_legs_ablated_is_rejected():
    with pytest.raises(ValueError, match="both retrieval legs"):
        PipelineConfig(lexical=None, semantic=None)


def test_ablation_order_does_not_change_the_trace_id():
    assert normalise(["no_gate", "no_lexical"]) == normalise(["no_lexical", "no_gate"])
    base = default_config()
    assert config_hash(compose(["no_gate", "no_lexical"], base)) == config_hash(
        compose(["no_lexical", "no_gate"], base)
    )


def test_content_id_is_derived_not_random():
    assert content_id("a", 1, ["x"]) == content_id("a", 1, ["x"])
    assert content_id("a", 1, ["x"]) != content_id("a", 1, ["y"])
    assert len(content_id("a")) == 26


def test_offline_traces_never_claim_a_real_model_ran():
    """A trace saying `gen_model: claude-sonnet-4-6` next to `provider: offline` gets
    quoted as "Sonnet produced this". The prefix makes that misreading impossible."""
    from autopsy.determinism import versions_block

    versions = versions_block(default_config(), "seed@abc")
    assert versions["gen_model"].startswith("sim:")
    assert versions["embed_model"].startswith("sim:")
    assert "claude-sonnet-4-6" in versions["gen_model"]  # still reproducible


def test_live_traces_are_not_prefixed():
    from autopsy.config import RuntimeConfig
    from autopsy.determinism import versions_block

    cfg = replace(default_config(), runtime=RuntimeConfig(provider="live"))
    assert versions_block(cfg, "seed@abc")["gen_model"] == "claude-sonnet-4-6"


def test_offline_and_live_traces_are_never_diffed():
    from autopsy.config import RuntimeConfig
    from autopsy.determinism import versions_block

    offline = versions_block(default_config(), "seed@abc")
    live = versions_block(
        replace(default_config(), runtime=RuntimeConfig(provider="live")), "seed@abc"
    )
    with pytest.raises(DeterminismError):
        assert_comparable(offline, live)


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


# --------------------------------------------------------------------------------------
# Property tests on the fusion maths
# --------------------------------------------------------------------------------------


@given(
    ranks=st.lists(st.integers(min_value=1, max_value=500), min_size=1, max_size=2),
    k=st.integers(min_value=1, max_value=200),
)
def test_rrf_is_positive_and_bounded(ranks, k):
    score = rrf_score(ranks, k)
    assert 0 < score <= len(ranks) / (k + 1)


@given(
    a=st.integers(min_value=1, max_value=500),
    b=st.integers(min_value=1, max_value=500),
    k=st.integers(min_value=1, max_value=200),
)
def test_rrf_is_monotonically_decreasing_in_rank(a, b, k):
    """Better rank must never score worse. This is the only property the fused
    ordering actually depends on."""
    if a < b:
        assert rrf_score([a], k) > rrf_score([b], k)
    elif a == b:
        assert rrf_score([a], k) == rrf_score([b], k)


@settings(max_examples=50)
@given(
    rank_a=st.integers(min_value=1, max_value=50),
    rank_b=st.integers(min_value=1, max_value=50),
    k=st.integers(min_value=1, max_value=100),
)
def test_appearing_in_both_legs_beats_appearing_in_one(rank_a, rank_b, k):
    """The whole point of fusion: a document both legs found should outrank one that
    only a single leg found at the same position."""
    assert rrf_score([rank_a, rank_b], k) > rrf_score([rank_a], k)
