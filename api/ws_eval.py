"""WS /eval/stream — eval suites reporting probe by probe as they run.

Design principle #5 says eval is headless and the inspector is live, and that stands for
the *batch* study: 200 cases across 14 ablations is a progress bar, not a visualisation.
What this endpoint streams is the other thing — a single suite of 12 isolation probes or
10 traps, which takes tens of seconds and where each individual result is legible.

Two guards, because the suites make real model calls:

* **Nothing runs until asked.** No suite fires on page load or on connect.
* **The estimated token cost is sent first** and the run only proceeds after the client
  confirms. Free-tier daily budgets are small enough that an accidental double-click
  matters, and a UI that spends a user's quota without telling them first is the kind
  of thing this whole project is about not doing.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError

from evals.runner import Finding, base_versions, run_suites
from api.state import get_state

router = APIRouter()

#: Rough per-probe token cost, measured on this corpus: ~1.7k of retrieved context in,
#: ~150 out. Used only to warn before spending; the run reports what it actually used.
TOKENS_PER_PROBE = 1_900

def _trap_count() -> int:
    """Count the traps on disk rather than hardcoding it.

    This said 10 while `load_traps()` returned 9, so the UI quoted a token estimate for a
    trap that does not exist and drew a progress bar out of 10 that could never fill. The
    number also leaked outward: an external reviewer described the suite as "10 traps"
    because that is what the interface claimed.
    """
    from evals.suites.silent_failure import load_traps

    return len(load_traps())


SUITES = {
    "isolation": {
        "label": "tenant isolation",
        # Mirrors the probe list in IsolationSuite.run(); test_evals asserts they agree.
        "probes": 12,
        "about": "12 probes on a planted competing-documents corpus. Every tenant holds a "
                 "document on the same topic with different values plus a per-run random "
                 "canary, so a foreign document is a strong retrieval candidate and the "
                 "boundary is the only thing holding. Includes 2 positive controls.",
    },
    "silent_failure": {
        "label": "silent failure",
        "probes": _trap_count(),
        "about": "10 traps whose answers do not exist in the corpus. Scores "
                 "wrong-and-confident, not accuracy — a system that is 70% right and "
                 "flags its uncertainty is deployable; one that is 85% right and never "
                 "flags anything is not.",
    },
}


class EvalRequest(BaseModel):
    suite: str = Field(pattern="^(isolation|silent_failure)$")
    confirm: bool = False


@router.websocket("/eval/stream")
async def eval_stream(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_json()
            try:
                req = EvalRequest.model_validate(raw)
            except ValidationError as exc:
                await ws.send_json({"type": "error", "message": str(exc)})
                continue

            spec = SUITES[req.suite]
            state = get_state()
            if not req.confirm:
                # Quote the cost, then stop. The client re-sends with confirm=true.
                await ws.send_json({
                    "type": "estimate",
                    "suite": req.suite,
                    "label": spec["label"],
                    "about": spec["about"],
                    "probes": spec["probes"],
                    "est_tokens": spec["probes"] * TOKENS_PER_PROBE,
                    "provider": state.config.runtime.provider,
                    "free": state.config.runtime.provider in ("offline", "groq"),
                })
                continue

            await _run(ws, req.suite, spec)
    except WebSocketDisconnect:
        return


async def _run(ws: WebSocket, name: str, spec: dict[str, Any]) -> None:
    from evals.suites.isolation import IsolationSuite
    from evals.suites.silent_failure import SilentFailureSuite

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    state = get_state()

    # The isolation suite builds its own ephemeral corpus by design, so it gets no
    # pipeline. The silent-failure suite runs against the real index and must share the
    # server's pipeline rather than opening a second vector-store client.
    suite = (
        IsolationSuite(cfg=state.config)
        if name == "isolation"
        else SilentFailureSuite(
            index=state.index, cfg=state.config, pipeline=state.pipeline
        )
    )

    await ws.send_json({
        "type": "started", "suite": name, "label": spec["label"],
        "probes": spec["probes"],
    })

    def push(payload: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    def on_finding(f: Finding) -> None:
        payload = f.to_dict()
        # Project the probe's query into the same 3D basis the corpus is drawn in, so the
        # visualiser can place it without re-embedding. Best effort: a suite that runs on
        # its own ephemeral corpus (isolation) has no basis here, and a failure to draw a
        # picture must never fail the probe that produced it.
        query = (payload.get("meta") or {}).get("query")
        if query:
            try:
                from api.routes_vectors import _project_query
                from autopsy.providers import build_providers

                emb, _usage = build_providers(
                    state.config, state.index.stats
                ).embedder.embed_query(query)
                payload["meta"]["projection"] = _project_query(state, emb)
            except Exception:  # noqa: BLE001 - decoration, never load-bearing
                payload["meta"]["projection"] = None
        push({"type": "finding", **payload})

    def on_progress(msg: str) -> None:
        push({"type": "progress", "message": msg})

    def work() -> None:
        try:
            report = run_suites(
                [suite],
                versions=base_versions(state.index.meta, state.config.runtime.provider),
                provider=state.config.runtime.provider,
                on_progress=on_progress,
                on_finding=on_finding,
            )
            passed = sum(1 for f in report.findings if f.passed)
            push({
                "type": "summary",
                "suite": name,
                "total": len(report.findings),
                "passed": passed,
                "failed": len(report.failures),
                "blocking": len(report.blocking),
                "versions": report.versions,
            })
        except Exception as exc:  # noqa: BLE001 - report it, do not kill the socket
            push({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            await ws.send_json(event)
        await ws.send_json({"type": "done", "suite": name})
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            await task


__all__ = ["SUITES", "TOKENS_PER_PROBE", "router"]
