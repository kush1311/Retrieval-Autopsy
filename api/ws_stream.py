"""WS /stream — stage events as they complete.

The progressive fill is a large part of why the inspector reads as an instrument
rather than a report: panels populate in pipeline order, and the reranker's decision
appears *before* the answer does. Blocking on the full run and rendering once throws
that away and takes four seconds to show nothing.

The pipeline is synchronous, so it runs in a worker thread and pushes events onto an
asyncio queue that the socket drains. That keeps the event loop free to actually send
them — running it inline would buffer every event and deliver them in one burst at the
end, which looks identical to not streaming at all.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from autopsy.ablations import ABLATIONS, compose
from autopsy.pipeline import PipelineError
from autopsy.trace import ErrorEvent
from api.state import get_state

router = APIRouter()


class StreamRequest(BaseModel):
    query: str
    tenant_id: str
    ablations: list[str] = []
    history: list[str] = []


@router.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_json()
            try:
                req = StreamRequest.model_validate(raw)
            except ValidationError as exc:
                await ws.send_json(ErrorEvent(message=str(exc)).model_dump(mode="json"))
                continue

            unknown = [n for a in req.ablations for n in a.split("+") if n not in ABLATIONS]
            if unknown:
                await ws.send_json(
                    ErrorEvent(message=f"unknown ablation(s): {unknown}").model_dump(mode="json")
                )
                continue

            await _run_streaming(ws, req)
    except WebSocketDisconnect:
        return


async def _run_streaming(ws: WebSocket, req: StreamRequest) -> None:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    state = get_state()
    cfg = compose(req.ablations, state.config) if req.ablations else state.config

    def emit(event: Any) -> None:
        # Called from the worker thread; hop back onto the loop to enqueue.
        loop.call_soon_threadsafe(queue.put_nowait, event.model_dump(mode="json"))

    def work() -> None:
        try:
            state.pipeline.run(
                req.query, tenant_id=req.tenant_id, cfg=cfg,
                history=req.history, ablations=req.ablations, emit=emit,
            )
        except PipelineError:
            pass  # the pipeline already emitted an ErrorEvent through `emit`
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(
                queue.put_nowait, ErrorEvent(message=str(exc)).model_dump(mode="json")
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            await ws.send_json(event)
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            await task
