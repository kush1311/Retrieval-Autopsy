"""Endpoints that expose the embedding space itself.

The trace records what each retrieval leg *scored*, never the vectors behind those
scores. That is the right default — a 384-float array per candidate would bloat every
trace by two orders of magnitude for something almost nobody reads. But it left the one
stage in the pipeline you could not inspect, in a project whose whole claim is that
every stage is observable.

Two read-only endpoints close that:

* ``GET /api/embedding/{chunk_id}`` — the stored document vector.
* ``GET /api/embedding?q=…``       — embed a query live and show its nearest neighbours
  with the cosine to each, so "why did this chunk match?" has an answer you can see
  rather than infer from a rank.

Both are bounded by default: a truncated vector plus its statistics, because the useful
information is the norm, the distribution, and the neighbours — not 384 raw floats.
Pass ``full=true`` when you actually want the array.
"""

from __future__ import annotations

import math
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from api.state import get_state

router = APIRouter()

#: Chunk IDs are content-addressed: ``c_`` plus 16 hex characters. Validated before the
#: value reaches a dict lookup — same reasoning as the trace-ID guard.
_CHUNK_ID = re.compile(r"^c_[0-9a-f]{16}$")

PREVIEW = 12


def _stats(values: tuple[float, ...]) -> dict[str, Any]:
    n = len(values)
    norm = math.sqrt(sum(v * v for v in values))
    mean = sum(values) / n if n else 0.0
    return {
        "dim": n,
        "l2_norm": round(norm, 6),
        "mean": round(mean, 6),
        "min": round(min(values), 6) if n else 0.0,
        "max": round(max(values), 6) if n else 0.0,
        # Sparsity is the tell that separates a real embedding from the offline
        # simulator's concept bag: a dense model fills essentially every dimension.
        "nonzero": sum(1 for v in values if v != 0.0),
    }


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return round(sum(x * y for x, y in zip(a, b)) / (na * nb), 6)


def _describe(emb, *, full: bool) -> dict[str, Any]:
    """Render an Embedding for the wire, dense or concept-bag."""
    if emb.dense is not None:
        body: dict[str, Any] = {"kind": "dense", "model_id": emb.model_id, **_stats(emb.dense)}
        body["values"] = list(emb.dense) if full else [round(v, 6) for v in emb.dense[:PREVIEW]]
        if not full:
            body["truncated"] = f"showing {PREVIEW} of {len(emb.dense)}; pass full=true for all"
        return body
    concepts = list(emb.concepts or ())
    return {
        "kind": "concept",
        "model_id": emb.model_id,
        "note": "offline simulator — a bag of concepts, not a vector space",
        "count": len(concepts),
        "concepts": concepts if full else concepts[:40],
    }


#: Cached PCA basis, keyed by corpus version. Recomputing a 449x384 SVD per request is
#: wasteful, and the basis must be *identical* across requests or the query point would
#: land in a different space than the chunks it is being compared against.
_PCA: dict[str, Any] = {}


def _pca_basis(state) -> dict[str, Any]:
    key = state.index.meta.get("corpus_version", "?") + "|" + str(state.index.meta.get("embed_model"))
    hit = _PCA.get(key)
    if hit is not None:
        return hit

    import numpy as np

    chunks = [c for c in state.index.chunks if state.index.vectors.get(c.chunk_id)]
    rows = []
    kept = []
    for c in chunks:
        emb = state.index.vectors[c.chunk_id]
        if emb.dense is None:
            continue
        rows.append(emb.dense)
        kept.append(c)
    if not rows:
        raise HTTPException(
            status_code=409,
            detail="the 3D view needs dense vectors; this index holds the offline "
                   "simulator's concept bags. Re-ingest with AUTOPSY_EMBEDDER=fastembed.",
        )

    matrix = np.asarray(rows, dtype=np.float64)
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    # Economy SVD: only the top 3 right-singular vectors are needed.
    _u, s, vt = np.linalg.svd(centred, full_matrices=False)
    components = vt[:3]
    coords = centred @ components.T

    total_var = float((s**2).sum()) or 1.0
    explained = [float(v) / total_var for v in (s[:3] ** 2)]

    # Normalise to a unit cube so the client never has to guess a scale.
    span = float(np.abs(coords).max()) or 1.0
    coords = coords / span

    basis = {
        "mean": mean,
        "components": components,
        "span": span,
        "explained": explained,
        "chunks": kept,
        "coords": coords,
    }
    _PCA[key] = basis
    return basis


@router.get("/embedding/projection")
def projection(state_dep: None = None) -> dict[str, Any]:
    """A 3D PCA projection of every chunk in the index.

    **This is a cartoon of the embedding space, and the response says how much of one.**
    384 dimensions squashed into 3 discards most of the variance, so two points sitting
    close together on screen are not necessarily close in the space the retriever
    actually searches. ``explained_variance`` is returned alongside the points so the
    view can label itself rather than inviting the reader to over-read it.
    """
    state = get_state()
    basis = _pca_basis(state)
    points = [
        {
            "chunk_id": c.chunk_id,
            "tenant_id": c.tenant_id,
            "doc_id": c.doc_id,
            "heading": c.heading_path[-1] if c.heading_path else "",
            "x": round(float(v[0]), 4),
            "y": round(float(v[1]), 4),
            "z": round(float(v[2]), 4),
        }
        for c, v in zip(basis["chunks"], basis["coords"])
    ]
    return {
        "model_id": state.index.meta.get("embed_model"),
        "dim": len(next(iter(state.index.vectors.values())).dense or ()),
        "count": len(points),
        "explained_variance": [round(e, 4) for e in basis["explained"]],
        "explained_total": round(sum(basis["explained"]), 4),
        "tenants": sorted({p["tenant_id"] for p in points}),
        "points": points,
    }


def _project_query(state, emb) -> dict[str, float] | None:
    """Put a query vector into the same 3D basis as the chunks."""
    if emb.dense is None:
        return None
    import numpy as np

    basis = _pca_basis(state)
    v = (np.asarray(emb.dense, dtype=np.float64) - basis["mean"]) @ basis["components"].T
    v = v / (basis["span"] or 1.0)
    return {"x": round(float(v[0]), 4), "y": round(float(v[1]), 4), "z": round(float(v[2]), 4)}


@router.get("/embedding/{chunk_id}")
def chunk_embedding(chunk_id: str, full: bool = False) -> dict[str, Any]:
    """The stored vector for one indexed chunk, with its text for context."""
    if not _CHUNK_ID.match(chunk_id):
        raise HTTPException(status_code=400, detail="chunk_id must match c_[0-9a-f]{16}")
    state = get_state()
    chunk = state.index.get(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail=f"no chunk {chunk_id}")
    emb = state.index.vectors.get(chunk_id)
    if emb is None:
        raise HTTPException(status_code=500, detail=f"chunk {chunk_id} has no vector; re-run ingest")
    return {
        "chunk": {
            "chunk_id": chunk.chunk_id,
            "tenant_id": chunk.tenant_id,
            "doc_id": chunk.doc_id,
            "ordinal": chunk.ordinal,
            "heading_path": list(chunk.heading_path),
            "text": chunk.text,
        },
        "embedding": _describe(emb, full=full),
    }


@router.get("/embedding")
def query_embedding(
    q: str = Query(min_length=1, max_length=2000),
    tenant_id: str = Query(default="", max_length=100),
    k: int = Query(default=8, ge=1, le=50),
    full: bool = False,
) -> dict[str, Any]:
    """Embed a query live and show its nearest neighbours.

    This is the teachable one. It answers "why did that chunk match?" with the cosine to
    every neighbour, rather than leaving you to infer it from a rank. Tenant-scoped, so
    it cannot be used to read across the boundary the isolation suite defends.
    """
    state = get_state()
    from autopsy.providers import build_providers

    providers = build_providers(state.config, state.index.stats)
    emb, usage = providers.embedder.embed_query(q)

    tenant = tenant_id or next(
        (t for t in state.index.tenants() if t != "tenant_global"), "tenant_global"
    )
    scope = state.index.scope(tenant)

    neighbours: list[dict[str, Any]] = []
    for chunk in scope:
        other = state.index.vectors.get(chunk.chunk_id)
        if other is None:
            continue
        if emb.dense is not None and other.dense is not None:
            score = _cosine(emb.dense, other.dense)
        elif emb.concepts is not None and other.concepts is not None:
            shared = set(emb.concepts) & set(other.concepts)
            score = round(len(shared) / max(1, len(set(emb.concepts))), 6)
        else:
            raise HTTPException(
                status_code=500,
                detail="index and embedder disagree on vector kind; re-run ingest",
            )
        neighbours.append({
            "chunk_id": chunk.chunk_id,
            "cosine": score,
            "tenant_id": chunk.tenant_id,
            "doc_id": chunk.doc_id,
            "heading_path": list(chunk.heading_path),
            "preview": chunk.text[:160],
        })

    neighbours.sort(key=lambda r: (-r["cosine"], r["chunk_id"]))
    top = neighbours[:k]

    # The floor matters as much as the top hit. bge-small scores *unrelated* English at
    # ~0.55, so a 0.6 cosine means almost nothing — which is precisely why the gate
    # threshold could not be inherited and had to be derived from the corpus.
    scores = [r["cosine"] for r in neighbours]
    try:
        projected = _project_query(state, emb)
    except HTTPException:
        projected = None  # concept bags have no basis to project into
    return {
        "query": q,
        "tenant_id": tenant,
        "scoped_chunks": len(scope),
        "projection": projected,
        "embedding": _describe(emb, full=full),
        "usage": {"tokens_in": usage.tokens_in, "cost_usd": usage.cost_usd},
        "distribution": {
            "max": max(scores) if scores else None,
            "median": sorted(scores)[len(scores) // 2] if scores else None,
            "min": min(scores) if scores else None,
            "note": "compare max against median — the gap is the real signal, not the "
                    "absolute value",
        },
        "neighbours": top,
    }


__all__ = ["router"]
