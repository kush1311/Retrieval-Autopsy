"""The provider and embedder axes must be genuinely independent.

`RuntimeConfig` documents them as separate fields, and `providers/__init__.py` builds the
embedder and the chat model from separate branches. But `default_config()` used to take
`embed_model` from the *provider's* row unconditionally, so overriding only
AUTOPSY_EMBEDDER produced an impossible pair — `provider=offline` + `embedder=fastembed`
handed fastembed the model name "offline-concept" and raised
`Model offline-concept is not supported in TextEmbedding`.

Found while auditing the isolation suite under a non-default embedder, which is the only
reason it surfaced: every committed run happened to use a provider's own pairing.
"""

from __future__ import annotations

import pytest

from autopsy.config import EMBEDDER_MODELS, PROVIDER_DEFAULTS, default_config


@pytest.mark.parametrize("provider", sorted(PROVIDER_DEFAULTS))
@pytest.mark.parametrize("embedder", sorted(EMBEDDER_MODELS))
def test_every_provider_embedder_pair_resolves_coherently(provider, embedder, monkeypatch):
    monkeypatch.setenv("AUTOPSY_PROVIDER", provider)
    monkeypatch.setenv("AUTOPSY_EMBEDDER", embedder)
    monkeypatch.delenv("AUTOPSY_EMBED_MODEL", raising=False)

    cfg = default_config()
    assert cfg.runtime.embedder == embedder
    assert cfg.runtime.provider == provider
    assert cfg.semantic is not None
    # The model must be one this embedder can actually load.
    assert cfg.semantic.model_id == EMBEDDER_MODELS[embedder], (
        f"{provider}+{embedder} resolved to {cfg.semantic.model_id!r}, which "
        f"{embedder} cannot load"
    )


def test_a_providers_own_pairing_is_left_alone(monkeypatch):
    """Not overriding the embedder must keep the provider's declared model verbatim."""
    for provider, d in PROVIDER_DEFAULTS.items():
        monkeypatch.setenv("AUTOPSY_PROVIDER", provider)
        monkeypatch.delenv("AUTOPSY_EMBEDDER", raising=False)
        monkeypatch.delenv("AUTOPSY_EMBED_MODEL", raising=False)
        cfg = default_config()
        assert cfg.semantic.model_id == d["embed_model"], provider


def test_an_explicit_model_override_still_wins(monkeypatch):
    monkeypatch.setenv("AUTOPSY_PROVIDER", "offline")
    monkeypatch.setenv("AUTOPSY_EMBEDDER", "fastembed")
    monkeypatch.setenv("AUTOPSY_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    assert default_config().semantic.model_id == "BAAI/bge-base-en-v1.5"


def test_every_embedder_in_provider_defaults_has_a_model(monkeypatch):
    """Guard against a new provider naming an embedder the table does not know."""
    for provider, d in PROVIDER_DEFAULTS.items():
        assert d["embedder"] in EMBEDDER_MODELS, (
            f"{provider} declares embedder {d['embedder']!r} with no EMBEDDER_MODELS entry"
        )


def test_the_offline_fastembed_pair_no_longer_crashes(monkeypatch):
    """The exact combination that raised during the audit."""
    monkeypatch.setenv("AUTOPSY_PROVIDER", "offline")
    monkeypatch.setenv("AUTOPSY_EMBEDDER", "fastembed")
    monkeypatch.delenv("AUTOPSY_EMBED_MODEL", raising=False)
    cfg = default_config()
    assert "offline-concept" not in cfg.semantic.model_id
    assert cfg.semantic.model_id.startswith("BAAI/")
