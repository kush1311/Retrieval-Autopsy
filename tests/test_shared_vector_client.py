"""Two pipelines over one persisted index must not fight over the vector store.

Embedded Qdrant takes an exclusive file lock, and a second client on the same path fails
even inside the same process. The silent-failure suite built its own ``Pipeline`` over
the persisted index, so running it from the API server — which already held the lock —
raised ``AlreadyLocked`` and the entire suite reported as a single failed probe.

The isolation suite was unaffected purely by luck: it builds an *ephemeral* corpus, and
ephemeral indexes are routed to the in-process store. Same latent bug, invisible in one
suite and fatal in the other, which is why the fix is at the store rather than in the
one suite that happened to expose it.
"""

from __future__ import annotations

import pytest

from autopsy.pipeline import Pipeline


class _FakeClient:
    """Stands in for QdrantClient so the registry is testable without a store.

    The suite runs on the local backend, so a Pipeline-level test would not exercise the
    Qdrant path at all — it would pass whether or not the registry worked. This tests the
    mechanism itself.
    """

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_registry_hands_out_one_client_per_target():
    from autopsy.store import vectors

    made: list[_FakeClient] = []

    def factory() -> _FakeClient:
        c = _FakeClient()
        made.append(c)
        return c

    a = vectors._acquire_client("path:/tmp/probe", factory)
    b = vectors._acquire_client("path:/tmp/probe", factory)
    try:
        assert a is b, "second acquire opened a competing client"
        assert len(made) == 1
    finally:
        vectors._release_client("path:/tmp/probe")
        vectors._release_client("path:/tmp/probe")


def test_registry_closes_only_on_the_last_release():
    from autopsy.store import vectors

    client = _FakeClient()
    vectors._acquire_client("path:/tmp/probe2", lambda: client)
    vectors._acquire_client("path:/tmp/probe2", lambda: client)

    vectors._release_client("path:/tmp/probe2")
    assert not client.closed, "closed while another holder was still using it"

    vectors._release_client("path:/tmp/probe2")
    assert client.closed, "leaked — never closed after the last release"
    assert "path:/tmp/probe2" not in vectors._CLIENTS


def test_registry_separates_distinct_targets():
    from autopsy.store import vectors

    a = vectors._acquire_client("path:/tmp/one", _FakeClient)
    b = vectors._acquire_client("path:/tmp/two", _FakeClient)
    try:
        assert a is not b
    finally:
        vectors._release_client("path:/tmp/one")
        vectors._release_client("path:/tmp/two")


def test_releasing_an_unknown_target_is_harmless():
    from autopsy.store import vectors

    vectors._release_client("path:/tmp/never-acquired")


def test_two_pipelines_over_one_index_coexist(tiny_index, tiny_config):
    """The case that used to raise AlreadyLocked."""
    a = Pipeline(tiny_index)
    b = Pipeline(tiny_index)
    try:
        for pipe in (a, b):
            trace = pipe.run("widget threshold", tenant_id="tenant_kelvin", cfg=tiny_config)
            assert trace.answer.text
    finally:
        a.close()
        b.close()


def test_closing_one_pipeline_does_not_break_the_other(tiny_index, tiny_config):
    """Refcounting, not just caching.

    A shared client that the first closer tears down is worse than no sharing at all —
    it converts a startup failure into an intermittent mid-run one.
    """
    a = Pipeline(tiny_index)
    b = Pipeline(tiny_index)
    a.close()
    try:
        trace = b.run("widget threshold", tenant_id="tenant_kelvin", cfg=tiny_config)
        assert trace.answer.text
    finally:
        b.close()


def test_close_is_idempotent(tiny_index):
    pipe = Pipeline(tiny_index)
    pipe.close()
    pipe.close()  # must not raise


def test_silent_failure_suite_accepts_a_shared_pipeline(tiny_index, tiny_config):
    """The API passes its own pipeline in; the suite must use it rather than build one."""
    from evals.suites.silent_failure import SilentFailureSuite

    pipe = Pipeline(tiny_index)
    try:
        suite = SilentFailureSuite(index=tiny_index, cfg=tiny_config, pipeline=pipe, traps=[])
        findings = suite.run()
        # No traps supplied, so only the summary finding — the point is that constructing
        # and running it against a shared pipeline does not raise.
        assert findings
    finally:
        pipe.close()


def test_eval_stream_runs_while_the_app_holds_the_store(tiny_index, tiny_config):
    """End to end: the exact path that failed in the browser.

    The app builds a pipeline at startup and holds it. Starting the silent-failure suite
    over the same index previously opened a competing client and the run died before the
    first probe.
    """
    from fastapi.testclient import TestClient

    from api.main import create_app

    with TestClient(create_app(index=tiny_index, config=tiny_config)) as c:
        with c.websocket_connect("/eval/stream") as ws:
            ws.send_json({"suite": "silent_failure", "confirm": False})
            assert ws.receive_json()["type"] == "estimate"
            ws.send_json({"suite": "silent_failure", "confirm": True})

            saw_suite_error = False
            while True:
                m = ws.receive_json()
                if m["type"] == "finding" and m["case_id"] == "__suite__":
                    saw_suite_error = True
                    pytest.fail(f"suite raised instead of running: {m['detail'][:200]}")
                if m["type"] in ("done", "error"):
                    break
            assert not saw_suite_error
