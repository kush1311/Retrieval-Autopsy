"""Dense retrieval with an in-query tenant filter.

Two backends behind one interface:

* :class:`LocalVectorStore` — the default. Numpy cosine for real embeddings, concept
  coverage for the offline simulator. Handles a corpus of this size instantly and has
  no operational surface at all.
* :class:`QdrantVectorStore` — the production shape, filter pushed into the query.

Both filter *before* scoring. Post-filtering a top-20 down to a tenant is the bug this
module exists to prevent: it silently shrinks the result set, changes recall, and
presents as a ranking problem rather than an access-control one.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from autopsy.providers.base import Embedding, ProviderError
from autopsy.store.chunks import GLOBAL_TENANT, INDEX_DIR, Chunk, Index
from autopsy.textutil import DILUTION_HALFPOINT, DILUTION_WEIGHT, ConceptStats


class VectorStore(Protocol):
    def search(
        self, *, query: Embedding, tenant_id: str, top_k: int
    ) -> list[tuple[Chunk, float]]: ...


# --------------------------------------------------------------------------------------
# Local
# --------------------------------------------------------------------------------------


class LocalVectorStore:
    def __init__(self, index: Index, stats: ConceptStats) -> None:
        self._index = index
        self._stats = stats
        self._dense_cache: dict[str, tuple[np.ndarray, list[Chunk]]] = {}

    # -- dense --------------------------------------------------------------------

    def _dense_matrix(self, tenant_id: str) -> tuple[np.ndarray, list[Chunk]]:
        hit = self._dense_cache.get(tenant_id)
        if hit is not None:
            return hit
        chunks = self._index.scope(tenant_id)
        rows = []
        for c in chunks:
            emb = self._index.vectors[c.chunk_id]
            if emb.dense is None:
                raise ProviderError(
                    f"chunk {c.chunk_id} was indexed with the concept simulator but the "
                    "query is a dense vector. The index and the provider disagree — "
                    "re-run ingest under the same AUTOPSY_PROVIDER."
                )
            rows.append(emb.dense)
        matrix = np.asarray(rows, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        self._dense_cache[tenant_id] = (matrix, chunks)
        return matrix, chunks

    def _search_dense(
        self, query: Embedding, tenant_id: str, top_k: int
    ) -> list[tuple[Chunk, float]]:
        matrix, chunks = self._dense_matrix(tenant_id)
        q = np.asarray(query.dense, dtype=np.float64)
        if q.shape[0] != matrix.shape[1]:
            # Fail loud. A dimension mismatch that silently broadcasts or truncates
            # returns plausible-looking garbage, which is far more expensive to debug
            # than a crash at query time.
            raise ProviderError(
                f"embedding dimension mismatch: query is {q.shape[0]}-d, index is "
                f"{matrix.shape[1]}-d. The index was built with a different model."
            )
        n = np.linalg.norm(q) or 1.0
        scores = matrix @ (q / n)
        order = np.argsort(-scores, kind="stable")[: top_k * 2]
        ranked = [(chunks[i], float(scores[i])) for i in order if scores[i] > 0]
        ranked.sort(key=lambda r: (-r[1], r[0].chunk_id))
        return ranked[:top_k]

    # -- concept (offline simulator) ----------------------------------------------

    def _search_concept(
        self, query: Embedding, tenant_id: str, top_k: int
    ) -> list[tuple[Chunk, float]]:
        q_concepts = list(query.concepts or ())
        if not q_concepts:
            return []
        denom = sum(self._stats.idf(c) for c in q_concepts)
        if denom <= 0:
            return []
        ranked: list[tuple[Chunk, float]] = []
        for chunk in self._index.scope(tenant_id):
            emb = self._index.vectors[chunk.chunk_id]
            if emb.concepts is None:
                raise ProviderError(
                    f"chunk {chunk.chunk_id} was indexed with dense vectors but the query "
                    "is a concept bag. Re-run ingest under the same AUTOPSY_PROVIDER."
                )
            doc = set(emb.concepts)
            num = sum(self._stats.idf(c) for c in q_concepts if c in doc)
            if num <= 0:
                continue
            # Discount for dilution, so a focused chunk outranks a long one with the
            # same coverage. Without this the score is step-valued and dozens of
            # chunks tie, leaving the ordering to an arbitrary tiebreaker.
            dilution = len(doc) / (len(doc) + DILUTION_HALFPOINT)
            ranked.append((chunk, round((num / denom) * (1.0 - DILUTION_WEIGHT * dilution), 6)))
        ranked.sort(key=lambda r: (-r[1], r[0].chunk_id))
        return ranked[:top_k]

    def search(
        self, *, query: Embedding, tenant_id: str, top_k: int
    ) -> list[tuple[Chunk, float]]:
        if query.kind == "dense":
            return self._search_dense(query, tenant_id, top_k)
        return self._search_concept(query, tenant_id, top_k)


# --------------------------------------------------------------------------------------
# Qdrant
# --------------------------------------------------------------------------------------


#: One Qdrant client per storage target, per process, reference-counted.
#:
#: Embedded Qdrant takes an exclusive file lock, so a *second* client on the same path
#: fails even inside the same process. That is not hypothetical: the silent-failure suite
#: builds its own ``Pipeline`` over the persisted index, and running it from inside the
#: API server — which already holds the lock — raised ``AlreadyLocked`` and the whole
#: suite reported as one failed probe.
#:
#: The isolation suite was unaffected only by luck: it builds an *ephemeral* corpus, and
#: ephemeral indexes are routed to the in-process store. So the bug was invisible in one
#: suite and fatal in the other.
_CLIENTS: dict[str, tuple[Any, int]] = {}
_CLIENT_LOCK = threading.Lock()


def _acquire_client(target: str, factory) -> Any:
    with _CLIENT_LOCK:
        client, refs = _CLIENTS.get(target, (None, 0))
        if client is None:
            client = factory()
        _CLIENTS[target] = (client, refs + 1)
        return client


def _release_client(target: str) -> None:
    """Drop a reference; close only when the last holder lets go.

    Without the count, one store closing would pull the connection out from under every
    other holder — which is worse than the leak it was trying to avoid.
    """
    with _CLIENT_LOCK:
        client, refs = _CLIENTS.get(target, (None, 0))
        if client is None:
            return
        if refs <= 1:
            _CLIENTS.pop(target, None)
            try:
                client.close()
            except Exception:  # noqa: BLE001 - closing twice is harmless
                pass
        else:
            _CLIENTS[target] = (client, refs - 1)


class QdrantVectorStore:
    """Dense search against Qdrant with the tenant filter inside the query.

    Two deployment shapes, same code path:

    * **embedded** — ``QDRANT_PATH=./corpus/index/qdrant``. A real on-disk collection
      driven by the real client, with no daemon. This is what runs here, because
      Docker Desktop is a 500MB install and a reboot to gain nothing at 449 chunks.
      Caveat worth stating: embedded mode is ``qdrant-client``'s Python implementation,
      not the Rust server. The client API, the payload filters and the collection
      semantics are the same; the performance envelope and the server-only features
      (sharding, on-disk quantization) are not.
    * **server** — ``QDRANT_URL=http://localhost:6333``. Set it and nothing else
      changes.

    Dense vectors only. The offline simulator's concept bags are not a vector space, and
    this store refuses them rather than indexing something meaningless.

    The collection is **synced on construction** rather than by a separate command. A
    store silently serving a stale collection is the kind of bug that reads as a ranking
    regression, so the invariant is enforced where it cannot be forgotten.
    """

    def __init__(
        self, index: Index, collection: str = "autopsy",
        url: str | None = None, path: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "qdrant-client is not installed. `pip install qdrant-client`, or set "
                "AUTOPSY_VECTOR_BACKEND=local."
            ) from exc

        self._index = index
        self._collection = collection
        resolved_url = url or os.environ.get("QDRANT_URL")
        resolved_path = path or os.environ.get("QDRANT_PATH")

        if resolved_url:
            self._target = f"url:{resolved_url}"
            self._client = _acquire_client(self._target, lambda: QdrantClient(url=resolved_url))
            self.mode = f"server {resolved_url}"
        else:
            target = resolved_path or str(INDEX_DIR / "qdrant")
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            self._target = f"path:{target}"
            try:
                self._client = _acquire_client(
                    self._target, lambda: QdrantClient(path=target)
                )
            except RuntimeError as exc:
                if "already accessed" not in str(exc):
                    raise
                # Embedded mode holds an exclusive file lock. One process at a time —
                # so the API server and a CLI query cannot both be open. The raw
                # message mentions portalocker and a storage folder, which does not
                # tell you the actual thing to do.
                raise ProviderError(
                    f"embedded Qdrant at {target} is locked by another process.\n"
                    "  Embedded mode allows one process at a time. Most likely the API "
                    "server is running.\n"
                    "\n"
                    "  Either stop it (Ctrl+C) and re-run this command, or avoid the "
                    "lock entirely:\n"
                    "    AUTOPSY_VECTOR_BACKEND=local   — numpy in-process, same results\n"
                    "    QDRANT_URL=http://…            — a real server, concurrent access"
                ) from exc
            self.mode = f"embedded {target}"

        self._sync()

    # -- collection management ----------------------------------------------------

    def _points(self):
        from qdrant_client.models import PointStruct

        for i, chunk in enumerate(self._index.chunks):
            emb = self._index.vectors[chunk.chunk_id]
            if emb.dense is None:
                raise ProviderError(
                    "Qdrant needs real embeddings; this index holds offline concept "
                    "bags. Re-run ingest with AUTOPSY_EMBEDDER=fastembed (free) or "
                    "=openai."
                )
            yield PointStruct(
                id=i,
                vector=list(emb.dense),
                payload={"chunk_id": chunk.chunk_id, "tenant_id": chunk.tenant_id},
            )

    def _sync(self) -> None:
        """Create and populate the collection if it is missing or out of date."""
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        expected = len(self._index.chunks)
        if not expected:
            raise ProviderError("cannot sync an empty index into Qdrant")

        first = self._index.vectors[self._index.chunks[0].chunk_id]
        if first.dense is None:
            raise ProviderError(
                "Qdrant needs real embeddings; this index holds offline concept bags. "
                "Re-run ingest with AUTOPSY_EMBEDDER=fastembed (free) or =openai."
            )
        dim = len(first.dense)

        try:
            info = self._client.get_collection(self._collection)
            same_dim = info.config.params.vectors.size == dim  # type: ignore[union-attr]
            if (info.points_count or 0) == expected and same_dim:
                self.synced = False
                return
        except Exception:  # noqa: BLE001 - "missing" surfaces as several exception types
            pass

        self._client.recreate_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        # Index the payload field the filter uses. On a server this matters: without a
        # cardinality estimate Qdrant cannot decide when to fall back to exhaustive
        # search, and an unindexed filter field is exactly where filtered-recall
        # problems hide. Embedded mode filters exhaustively anyway and may not
        # implement the call, so its absence is not an error here.
        try:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="tenant_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:  # noqa: BLE001 - unsupported in embedded mode
            pass
        batch: list = []
        for point in self._points():
            batch.append(point)
            if len(batch) >= 256:
                self._client.upsert(collection_name=self._collection, points=batch)
                batch = []
        if batch:
            self._client.upsert(collection_name=self._collection, points=batch)
        self.synced = True

    # -- search -------------------------------------------------------------------

    def search(
        self, *, query: Embedding, tenant_id: str, top_k: int
    ) -> list[tuple[Chunk, float]]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        if query.dense is None:
            raise ProviderError(
                "Qdrant requires dense query vectors; got a concept bag. The index and "
                "the embedder disagree."
            )
        hits = self._client.query_points(
            collection_name=self._collection,
            query=list(query.dense),
            limit=top_k,
            # The filter goes in the query. Applying it to results afterwards would
            # silently shrink the result set and change recall, and would present as a
            # ranking problem rather than an access-control one.
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id", match=MatchAny(any=[tenant_id, GLOBAL_TENANT])
                    )
                ]
            ),
        ).points

        out: list[tuple[Chunk, float]] = []
        for h in hits:
            chunk = self._index.get((h.payload or {}).get("chunk_id", ""))
            if chunk is None:
                continue
            # Defence in depth: the filter should already guarantee this. If a foreign
            # tenant ever reaches here the right move is to fail loudly, not to drop the
            # row and serve a quietly-correct result that hides a broken boundary.
            if chunk.tenant_id not in (tenant_id, GLOBAL_TENANT):
                raise ProviderError(
                    f"tenant filter leaked: {chunk.chunk_id} belongs to "
                    f"{chunk.tenant_id}, query was for {tenant_id}"
                )
            out.append((chunk, float(h.score)))
        return out

    def close(self) -> None:
        """Release this store's reference to the shared client."""
        target = getattr(self, "_target", None)
        if target:
            _release_client(target)
            self._target = None  # type: ignore[assignment]


def build_vector_store(index: Index, stats: ConceptStats) -> VectorStore:
    """Pick a backend for this index.

    The backend is chosen per index, not per process. An ephemeral index — the isolation
    suite's planted corpus, a test fixture — always gets the in-process store, whatever
    ``AUTOPSY_VECTOR_BACKEND`` says. Honouring the env var for those would upsert a
    twenty-chunk synthetic corpus over the real collection and leave the persistent
    index quietly wrong.
    """
    backend = os.environ.get("AUTOPSY_VECTOR_BACKEND", "local").strip().lower()
    if backend not in ("local", "qdrant"):
        raise ValueError(f"AUTOPSY_VECTOR_BACKEND must be 'local' or 'qdrant', got {backend!r}")
    if backend == "qdrant" and index.persisted:
        return QdrantVectorStore(index)
    return LocalVectorStore(index, stats)


__all__ = ["LocalVectorStore", "QdrantVectorStore", "VectorStore", "build_vector_store"]
