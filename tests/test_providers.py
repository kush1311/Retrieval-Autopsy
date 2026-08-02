"""Provider plumbing that must work without a network call.

The Groq path was exercised for real, but the parts most likely to break quietly —
unwrapping reasoning-model output and deciding what to retry — are pure functions, and
pure functions should be pinned by tests rather than by a live run someone has to
remember to repeat.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from autopsy.config import PROVIDER_DEFAULTS, RuntimeConfig, default_config
from autopsy.determinism import config_hash, versions_block
from autopsy.providers.base import ProviderError, accepts_temperature


# --------------------------------------------------------------------------------------
# Reasoning-model output
# --------------------------------------------------------------------------------------


class _Msg:
    def __init__(self, content=None, reasoning_content=None, reasoning=None):
        self.content = content
        if reasoning_content is not None:
            self.reasoning_content = reasoning_content
        if reasoning is not None:
            self.reasoning = reasoning


def test_think_blocks_are_stripped():
    """Qwen prefixes its answer with <think>…</think>. Left in place, the judge's
    first-line verdict parse reads the reasoning instead and scores UNSTABLE."""
    from autopsy.providers.oaicompat import unwrap as _unwrap

    got = _unwrap(_Msg(content="<think>Let me consider the options.</think>\nEQUIVALENT\nsame claims"))
    assert got.startswith("EQUIVALENT")
    assert "consider" not in got


def test_reasoning_field_is_used_when_content_is_empty():
    """GPT-OSS returns empty `content` and puts the answer in a reasoning field.
    Reading only `content` yields "" — which parses as an unparseable verdict and
    silently drops the case."""
    from autopsy.providers.oaicompat import unwrap as _unwrap

    assert _unwrap(_Msg(content="", reasoning_content="DEGRADED\nthinner")) .startswith("DEGRADED")
    assert _unwrap(_Msg(content=None, reasoning="CONTRADICTORY\nno")).startswith("CONTRADICTORY")


def test_ordinary_content_passes_through():
    from autopsy.providers.oaicompat import unwrap as _unwrap

    assert _unwrap(_Msg(content="  KLV-4021 means X.  ")) == "KLV-4021 means X."


# --------------------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------------------


class _StatusErr(Exception):
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


class APIConnectionError(Exception):
    """Same class name the OpenAI SDK uses; the policy matches on name."""


def test_rate_limits_and_5xx_are_retried():
    from autopsy.providers.oaicompat import is_transient as _is_transient

    for status in (408, 429, 500, 502, 503, 504, 529):
        assert _is_transient(_StatusErr(status)), status


def test_connection_errors_are_retried():
    """A dropped TLS handshake carries no status code at all. A retry policy written
    only around HTTP statuses lets it through, and a sweep twenty minutes in dies on
    one flaky socket with nothing to show — which is how this was found."""
    from autopsy.providers.oaicompat import is_transient as _is_transient

    assert _is_transient(APIConnectionError("WinError 10054"))


def test_client_errors_are_not_retried():
    from autopsy.providers.oaicompat import is_transient as _is_transient

    for status in (400, 401, 403, 404, 422):
        assert not _is_transient(_StatusErr(status)), status


def test_a_daily_quota_is_not_retried():
    """Per-minute limits clear in seconds; a daily one clears in hours. Backing off six
    times to fail anyway turns a clear error into a two-minute hang — and the generic
    "rate limited" message it produced sent me checking per-minute headers that showed
    plenty of headroom while the real limit was a separate daily counter."""
    from autopsy.providers.oaicompat import is_daily_limit, is_transient

    class DailyErr(Exception):
        status_code = 429
        body = {"message": "Rate limit reached ... on tokens per day (TPD): Limit 100000, "
                           "Used 99847, Requested 2343"}

    assert is_daily_limit(DailyErr())
    assert not is_transient(DailyErr()), "a daily quota must fail fast, not back off"

    class MinuteErr(Exception):
        status_code = 429
        body = {"message": "Rate limit reached on tokens per minute (TPM)"}

    assert not is_daily_limit(MinuteErr())
    assert is_transient(MinuteErr()), "a per-minute limit clears in seconds; retry it"


def test_retry_delay_prefers_the_servers_own_advice():
    from autopsy.providers.oaicompat import retry_after as _retry_after

    class WithHeader(Exception):
        class response:  # noqa: N801
            headers = {"retry-after": "7"}

    assert 7.0 <= _retry_after(WithHeader(), 0) <= 8.0
    # No header: capped exponential backoff, never unbounded.
    assert _retry_after(Exception(), 0) <= 60.0
    assert _retry_after(Exception(), 10) == 60.0


# --------------------------------------------------------------------------------------
# Provider / embedder as independent axes
# --------------------------------------------------------------------------------------


def test_every_provider_declares_a_complete_default_set():
    required = {"embedder", "embed_model", "generation_model", "rerank_model",
                "gate_reads", "gate_threshold"}
    for provider, defaults in PROVIDER_DEFAULTS.items():
        assert required <= set(defaults), f"{provider} is missing {required - set(defaults)}"


def test_groq_pairs_a_remote_chat_model_with_a_local_embedder():
    """Groq serves no embeddings endpoint, which is the whole reason `provider` and
    `embedder` are separate fields."""
    from autopsy.providers.groq import GROQ_PRICES

    groq = PROVIDER_DEFAULTS["groq"]
    assert groq["embedder"] == "fastembed"
    # A local ONNX model, not a hosted endpoint — that is the point of the pairing.
    assert "/" in str(groq["embed_model"]) and "bge" in str(groq["embed_model"])
    # Generation is whatever Groq serves; asserting a specific family here would break
    # every time a daily budget forces a swap, which is an operational choice rather
    # than a design one.
    assert str(groq["generation_model"]) in GROQ_PRICES


def test_the_judge_is_a_different_family_from_the_generator():
    """Self-preference has no prompt-level fix — models favour their own family's
    output. The only control is choosing a different one."""
    from autopsy.providers.groq import GROQ_GENERATION_MODEL, GROQ_JUDGE_MODEL

    generator_family = GROQ_GENERATION_MODEL.split("-")[0].split("/")[0]
    judge_family = GROQ_JUDGE_MODEL.split("-")[0].split("/")[0]
    assert generator_family != judge_family, (GROQ_GENERATION_MODEL, GROQ_JUDGE_MODEL)


def test_changing_the_embedder_changes_the_config_hash():
    """The cache is keyed on the config hash. If the embedder were outside it, a
    concept-embedded answer would be served to a fastembed run."""
    base = default_config()
    swapped = replace(base, runtime=replace(base.runtime, embedder="fastembed"))
    assert config_hash(base) != config_hash(swapped)


def test_a_simulated_embedder_is_labelled_even_on_a_real_provider():
    """Per-component stamping. A Groq run with the concept embedder has a real chat
    model and a fake dense leg; one flag for the whole trace would mislabel a half."""
    cfg = default_config()
    mixed = replace(cfg, runtime=RuntimeConfig(provider="groq", embedder="concept"))
    versions = versions_block(mixed, "seed@abc")
    assert versions["embed_model"].startswith("sim:")
    assert not versions["gen_model"].startswith("sim:")


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("AUTOPSY_PROVIDER", "definitely-not-a-provider")
    with pytest.raises(ValueError, match="AUTOPSY_PROVIDER"):
        default_config()


def test_unknown_embedder_is_rejected():
    from autopsy.providers.embedders import build_embedder

    with pytest.raises(ValueError, match="unknown embedder"):
        build_embedder("telepathy")


def test_missing_groq_key_is_a_clear_error(monkeypatch):
    """A missing key should name the variable and offer the keyless path, not surface as
    an auth error from inside the SDK."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from autopsy.providers.groq import GroqLLM

    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        GroqLLM()._make_client()


def test_temperature_guard_still_covers_the_anthropic_models():
    """Sampling params are removed on Opus 4.7+/Sonnet 5/Fable 5, so temperature=0 is
    a 400 there. Groq accepts them on everything it serves."""
    assert accepts_temperature("claude-sonnet-4-6")
    assert not accepts_temperature("claude-sonnet-5")
