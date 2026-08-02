"""A Qdrant collection must never answer for an index it does not hold.

`_sync()` used to consider a collection current when its point count and vector dimension
matched the index. Neither field can see the change that matters: swap `bge-small-en-v1.5`
for any other 384-dimensional model, or regenerate the corpus to a different 449 chunks,
and count and dimension both still agree. The store keeps serving the old vectors while
every report names the new model.

This is not hypothetical. A collection built on 26 July held 449 points at 384-d. Six days
and several re-ingests later the index still had 449 chunks at 384-d, so sync reported "up
to date" and wrote nothing. It was harmless by luck — same seed, same model — but nothing
in the process established that.

The fix is structural rather than a better comparison: the collection name carries a
fingerprint of corpus version, model and dimension, so a different index resolves to a
different collection and the stale one is unreachable.
"""

from __future__ import annotations

import dataclasses

import pytest

from autopsy.store.vectors import QdrantVectorStore

pytest.importorskip("qdrant_client", reason="pip install qdrant-client")


class _FakeIndex:
    """Only what `_fingerprint` reads. A real Index needs a corpus on disk."""

    def __init__(self, *, corpus_version: str, embed_model: str, dim: int, n: int) -> None:
        self.meta = {"corpus_version": corpus_version, "embed_model": embed_model, "dim": dim}
        self.chunks = [object()] * n


BASE = dict(corpus_version="seed@cab5c9bd", embed_model="BAAI/bge-small-en-v1.5", dim=384, n=449)


def _fp(**overrides) -> str:
    return QdrantVectorStore._fingerprint(_FakeIndex(**{**BASE, **overrides}))


def test_identical_indexes_share_a_collection() -> None:
    """Otherwise every process rebuilds, and the sync is worse than useless."""
    assert _fp() == _fp()


def test_a_same_dimension_model_swap_changes_the_collection() -> None:
    """THE regression. Both models are 384-d over the same 449 chunks, so the old
    count-and-dimension check saw no difference at all."""
    before = _fp(embed_model="BAAI/bge-small-en-v1.5")
    after = _fp(embed_model="sentence-transformers/all-MiniLM-L6-v2")
    assert before != after, (
        "two different 384-d models over the same corpus resolve to the same collection; "
        "the second one would answer from the first one's vectors"
    )


def test_a_regenerated_corpus_changes_the_collection() -> None:
    """Same chunk count, different content — the other blind spot."""
    assert _fp(corpus_version="seed@cab5c9bd") != _fp(corpus_version="seed@deadbeef")


def test_a_changed_chunk_count_changes_the_collection() -> None:
    assert _fp(n=449) != _fp(n=450)


def test_a_changed_dimension_changes_the_collection() -> None:
    assert _fp(dim=384) != _fp(dim=1536)


def test_the_fingerprint_is_short_and_filesystem_safe() -> None:
    """It becomes part of a collection name, which becomes a directory in embedded mode."""
    fp = _fp()
    assert len(fp) == 8 and fp.isalnum() and fp.islower()


def test_a_missing_meta_field_does_not_crash() -> None:
    """An index written by an older version has no `dim`. Degrade to a different
    fingerprint — which forces a rebuild — rather than raising at startup."""
    index = _FakeIndex(**BASE)
    del index.meta["dim"]
    assert QdrantVectorStore._fingerprint(index) != _fp()
