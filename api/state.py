"""Application state, in a module that imports nothing from the rest of ``api``.

This exists to break an import cycle. Every route module needs ``get_state``, and it used
to live in ``api/main.py`` — so ``api.routes_query``, ``api.routes_vectors``,
``api.ws_stream`` and ``api.ws_eval`` each did ``from api.main import get_state``, while
``api.main`` imported all four to register them on the app.

That is a genuine cycle. It never fired only because nothing imported a route module
first: importing ``api.main`` gets to ``get_state`` before it reaches the routers, so by
the time a router asks for it, it is there. Import ``api.ws_eval`` on its own — a test, a
script, an editor's autoreload — and Python starts with a half-initialised ``api.main``
and raises ``ImportError: cannot import name 'get_state'``, from a file nobody touched.

A leaf module with no ``api`` imports of its own makes the order irrelevant.
"""

from __future__ import annotations

from dataclasses import dataclass

from autopsy.config import PipelineConfig
from autopsy.pipeline import Pipeline
from autopsy.store.chunks import Index


@dataclass
class AppState:
    index: Index
    pipeline: Pipeline
    config: PipelineConfig


_STATE: AppState | None = None


def get_state() -> AppState:
    if _STATE is None:  # pragma: no cover - only reachable outside the lifespan
        raise RuntimeError("app state is not initialised")
    return _STATE


def set_state(state: AppState | None) -> None:
    """Install (or clear) the process-wide state. Called only from the app lifespan.

    A setter rather than letting the lifespan assign the global directly: `global _STATE`
    in ``api.main`` would bind a name in *that* module, leaving every route module reading
    the original ``None`` here. Rebinding across modules is exactly the bug this shape
    prevents.
    """
    global _STATE
    _STATE = state


def close_state() -> None:
    """Release the current state's vector store, then clear it.

    Embedded Qdrant holds an exclusive file lock. Leaving it to garbage collection means a
    restart races the outgoing worker for it — which is how `--reload` killed this server
    once: an unrelated file edit triggered a reload, the new worker could not take the lock
    the old one still held, and it exited 255 with a clean log and no traceback.
    """
    close = getattr(getattr(_STATE, "pipeline", None), "close", None)
    if callable(close):
        close()
    set_state(None)
