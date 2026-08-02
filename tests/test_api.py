"""API surface: routes, validation, and the websocket stream."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """The app, against an index built here rather than whatever is on disk.

    Reading ``corpus/index`` made these tests depend on the provider the developer last
    ran ``ingest`` with: a dense index plus a concept-embedder test run is a vector-kind
    mismatch, and the whole file starts failing for reasons unrelated to the API.
    """
    from autopsy.config import default_config
    from autopsy.ingest import Document, build_index

    docs = [
        Document(
            tenant_id="tenant_kelvin",
            doc_id="errors.md",
            markdown=(
                "# Kelvin errors\n\n"
                "## KLV-4021\n\nKLV-4021 means the pellshale budget was exhausted while "
                "accepting incoming batches. The operation is abandoned and retried at "
                "the next scheduled attempt; nothing is lost, but the backlog grows.\n\n"
                "## KLV-4022\n\nKLV-4022 means the murrvale budget was exhausted while "
                "merging segments. This is distinct from KLV-4021 and has a different "
                "remedy; raising the budget resolves it and restarting does not.\n"
            ),
        ),
        Document(
            tenant_id="tenant_global",
            doc_id="policy.md",
            markdown=(
                "# Shared policy\n\n## Support\n\nEvery tenant may read this shared "
                "policy document, which describes the escalation path and applies "
                "uniformly across every product in the catalogue.\n"
            ),
        ),
    ]
    cfg = default_config()
    index, _ = build_index(docs, cfg=cfg, label="apitest")
    with TestClient(create_app(index=index, config=cfg)) as c:
        yield c


def test_health_reports_the_corpus_it_loaded(client: TestClient):
    body = client.get("/health").json()
    assert body["ok"] and body["chunks"] > 0
    assert "tenant_kelvin" in body["tenants"]


def test_meta_lists_ablations_with_their_predicted_failure(client: TestClient):
    body = client.get("/api/meta").json()
    names = {a["name"] for a in body["ablations"]}
    assert {"no_lexical", "no_gate", "no_rerank"} <= names
    assert all(a["expected"] for a in body["ablations"])
    assert "tenant_global" not in body["tenants"]


def test_query_returns_a_complete_trace(client: TestClient):
    body = client.post(
        "/api/query", json={"query": "what does KLV-4021 mean", "tenant_id": "tenant_kelvin"}
    ).json()
    assert body["answer"]["status"] in ("grounded", "refused", "ungrounded")
    assert len(body["stages"]) == 9
    assert body["config"]["generation"]["temperature"] == 0.0


def test_unknown_ablation_is_rejected_rather_than_ignored(client: TestClient):
    response = client.post(
        "/api/query",
        json={"query": "q", "tenant_id": "tenant_kelvin", "ablations": ["no_such_thing"]},
    )
    assert response.status_code == 400
    assert "no_such_thing" in str(response.json()["detail"])


def test_counterfactual_returns_both_traces_and_a_computed_explanation(client: TestClient):
    body = client.post(
        "/api/counterfactual",
        json={
            "query": "what does KLV-4021 mean",
            "tenant_id": "tenant_kelvin",
            "ablations": ["no_lexical"],
        },
    ).json()
    assert body["baseline"]["ablations"] == []
    assert body["variant"]["ablations"] == ["no_lexical"]
    assert body["explanation"]
    assert body["outcome"]


def test_counterfactual_requires_an_ablation(client: TestClient):
    response = client.post(
        "/api/counterfactual", json={"query": "q", "tenant_id": "tenant_kelvin"}
    )
    assert response.status_code == 400


def test_trace_id_is_validated_before_touching_the_filesystem(client: TestClient):
    """The id goes straight into a path join, so a malformed one must be refused
    rather than resolved."""
    for bad in ("../../etc/passwd", "..", "short", "a" * 27):
        assert client.get(f"/api/trace/{bad}").status_code in (400, 404)


def test_trace_round_trips_through_the_api(client: TestClient, tmp_path, monkeypatch):
    import api.routes_trace as routes_trace

    trace = client.post(
        "/api/query", json={"query": "what does KLV-4021 mean", "tenant_id": "tenant_kelvin"}
    ).json()
    monkeypatch.setattr(routes_trace, "TRACES_DIR", tmp_path)
    from autopsy.trace import Trace

    Trace.model_validate(trace).write(tmp_path)
    fetched = client.get(f"/api/trace/{trace['trace_id']}").json()
    assert fetched["trace_id"] == trace["trace_id"]


def test_websocket_streams_stages_then_done(client: TestClient):
    with client.websocket_connect("/stream") as ws:
        ws.send_json({"query": "what does KLV-4021 mean", "tenant_id": "tenant_kelvin"})
        types: list[str] = []
        while True:
            event = ws.receive_json()
            types.append(event["type"])
            # Break on `error` as well as `done`. Looping only until `done` means that
            # when the server correctly reports a failure, the client blocks forever on
            # the next receive — the test hangs instead of failing, which is strictly
            # worse than a red test. Any real consumer needs the same terminal set.
            if event["type"] == "error":
                pytest.fail(f"pipeline errored: {event['message']}")
            if event["type"] == "done":
                assert event["trace"]["answer"]["text"]
                break
    assert types.count("stage") == 9, "skipped stages must stream too, never be omitted"
    assert "fused" in types
    assert types[-1] == "done"


def test_websocket_rejects_an_unknown_ablation_without_dropping_the_socket(client: TestClient):
    with client.websocket_connect("/stream") as ws:
        ws.send_json({"query": "q", "tenant_id": "tenant_kelvin", "ablations": ["nope"]})
        event = ws.receive_json()
        assert event["type"] == "error"
        # The socket must survive a bad request so the UI does not have to reconnect.
        ws.send_json({"query": "what does KLV-4021 mean", "tenant_id": "tenant_kelvin"})
        assert ws.receive_json()["type"] in ("stage", "candidates")
