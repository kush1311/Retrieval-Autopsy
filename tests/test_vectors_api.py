"""The embedding-inspection endpoints.

These exist because the trace records what each retrieval leg *scored* but never the
vectors behind those scores — leaving the one stage in a self-described observability
project that you could not actually observe.

The distribution block is the part worth testing. `max` alone is meaningless: bge-small
scores unrelated English at ~0.55, so a 0.7 cosine sounds strong and may be noise. The
gap between `max` and `median` is the real signal, and it is what makes the gate's
choice of `lexical_top1` legible rather than arbitrary.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture(scope="module")
def client(tiny_index, tiny_config):
    with TestClient(create_app(index=tiny_index, config=tiny_config)) as c:
        yield c


def test_query_embedding_returns_vector_and_neighbours(client, tiny_index):
    tenant = next(t for t in tiny_index.tenants() if t != "tenant_global")
    r = client.get("/api/embedding", params={"q": "widget threshold", "tenant_id": tenant, "k": 3})
    assert r.status_code == 200
    body = r.json()

    assert body["embedding"]["model_id"]
    assert body["scoped_chunks"] > 0
    assert 1 <= len(body["neighbours"]) <= 3
    # Descending by similarity, so "nearest" means what it says.
    scores = [n["cosine"] for n in body["neighbours"]]
    assert scores == sorted(scores, reverse=True)


def test_neighbours_never_cross_the_tenant_boundary(client, tiny_index):
    """The endpoint reads the index directly, so it needs its own scope check.

    A debug endpoint that bypasses the boundary the isolation suite defends would be a
    leak with a friendly URL.
    """
    tenants = [t for t in tiny_index.tenants() if t != "tenant_global"]
    if len(tenants) < 2:
        pytest.skip("needs at least two tenants")
    me = tenants[0]
    r = client.get("/api/embedding", params={"q": "widget threshold", "tenant_id": me, "k": 50})
    seen = {n["tenant_id"] for n in r.json()["neighbours"]}
    assert seen <= {me, "tenant_global"}, f"leaked {seen - {me, 'tenant_global'}}"


def test_distribution_reports_the_floor_not_just_the_top(client, tiny_index):
    """Reporting only the top score is how a meaningless similarity looks convincing."""
    tenant = next(t for t in tiny_index.tenants() if t != "tenant_global")
    d = client.get("/api/embedding", params={"q": "widget threshold", "tenant_id": tenant}).json()
    dist = d["distribution"]
    assert dist["max"] is not None and dist["median"] is not None and dist["min"] is not None
    assert dist["min"] <= dist["median"] <= dist["max"]


def test_chunk_embedding_round_trips(client, tiny_index):
    chunk_id = tiny_index.chunks[0].chunk_id
    body = client.get(f"/api/embedding/{chunk_id}").json()
    assert body["chunk"]["chunk_id"] == chunk_id
    assert body["chunk"]["text"]
    assert body["embedding"]["model_id"]


def test_truncated_by_default_and_full_on_request(client, tiny_index):
    """A 384-float array in every response is noise; the stats are the useful part."""
    chunk_id = tiny_index.chunks[0].chunk_id
    short = client.get(f"/api/embedding/{chunk_id}").json()["embedding"]
    full = client.get(f"/api/embedding/{chunk_id}", params={"full": "true"}).json()["embedding"]
    if short["kind"] != "dense":
        pytest.skip("concept-bag embedder; truncation applies to concepts instead")
    assert len(short["values"]) < len(full["values"])
    assert len(full["values"]) == full["dim"]


@pytest.mark.parametrize(
    "chunk_id,expected",
    [
        ("not-a-chunk-id", 400),          # fails the format guard
        ("../../etc/passwd", 404),        # path traversal never reaches a lookup
        ("c_0000000000000000", 404),      # well-formed, absent
    ],
)
def test_malformed_ids_are_rejected_before_lookup(client, chunk_id, expected):
    assert client.get(f"/api/embedding/{chunk_id}").status_code == expected
