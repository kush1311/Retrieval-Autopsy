"""An ephemeral index must never reach a shared vector store.

This is a data-integrity boundary, not a convenience. The isolation suite builds its own
planted competing-documents corpus — a couple of dozen synthetic chunks with per-run
canaries. With ``AUTOPSY_VECTOR_BACKEND=qdrant`` set globally, that corpus was about to
be upserted into the same persistent collection the real 449-chunk index lives in,
destroying it. The symptom would have been an inspector quietly answering every question
out of twenty synthetic probe documents.

It surfaced as an embedded-Qdrant file lock, which was luck: on a Qdrant *server* there
is no lock and the overwrite would have succeeded silently.
"""

from __future__ import annotations

from dataclasses import replace

from autopsy.config import default_config
from autopsy.ingest import Document, build_index
from autopsy.store.chunks import Index
from autopsy.store.vectors import LocalVectorStore, build_vector_store


def _tiny_index() -> Index:
    docs = [
        Document(
            tenant_id="tenant_a", doc_id="a.md",
            markdown="# A\n\nThe alpha widget threshold is 41 on every supported release.\n",
        ),
        Document(
            tenant_id="tenant_b", doc_id="b.md",
            markdown="# B\n\nThe beta widget threshold is 77 on every supported release.\n",
        ),
    ]
    index, _stats = build_index(docs, cfg=default_config())
    return index


def test_a_built_index_is_not_marked_persisted():
    assert _tiny_index().persisted is False


def test_the_index_read_from_disk_is_marked_persisted(tmp_path):
    ix = _tiny_index()
    ix.write(tmp_path)
    assert Index.read(tmp_path).persisted is True


def test_qdrant_backend_is_ignored_for_an_ephemeral_index(monkeypatch):
    """The backend is chosen per index, not per process."""
    monkeypatch.setenv("AUTOPSY_VECTOR_BACKEND", "qdrant")
    ix = _tiny_index()
    store = build_vector_store(ix, ix.stats)
    assert isinstance(store, LocalVectorStore), (
        "an ephemeral corpus was routed to the shared vector store; running the "
        "isolation suite would overwrite the real collection"
    )


def test_a_provider_mismatch_is_caught_at_startup(monkeypatch):
    """A dense index searched by the concept simulator must fail at boot.

    Left to fail lazily this surfaces mid-query as "chunk c_dd2aa… was indexed with
    dense vectors but the query is a concept bag", which reads as a corrupt corpus
    rather than the truth: the process is running a different provider than the one
    that built the index. That is exactly how it presented in the browser — the
    launcher was not loading .env, so the app silently fell back to offline.
    """
    import pytest

    from autopsy.config import RuntimeConfig

    ix = _tiny_index()
    ix.meta["vector_kind"] = "dense"
    ix.meta["embed_model"] = "BAAI/bge-small-en-v1.5"

    concept_cfg = replace(
        default_config(), runtime=RuntimeConfig(provider="offline", embedder="concept")
    )
    with pytest.raises(ValueError, match="index/config mismatch"):
        ix.assert_matches(concept_cfg)

    # And the matching config is accepted.
    dense_cfg = replace(
        default_config(), runtime=RuntimeConfig(provider="groq", embedder="fastembed")
    )
    ix.assert_matches(dense_cfg)


def test_an_unknown_backend_is_rejected_rather_than_defaulted(monkeypatch):
    import pytest

    monkeypatch.setenv("AUTOPSY_VECTOR_BACKEND", "pinecone")
    ix = _tiny_index()
    with pytest.raises(ValueError, match="AUTOPSY_VECTOR_BACKEND"):
        build_vector_store(ix, ix.stats)
