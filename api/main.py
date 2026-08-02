"""FastAPI app for the inspector.

Serves one request completely. There is no cross-request aggregation here, no user
management, and no saved sessions — that is a dashboard, and a dashboard is a different
product that would eat the timeline.

The API is a thin shell over ``Pipeline``: it holds the index open, converts HTTP into
a ``pipeline.run`` call, and streams the trace out. Any behaviour worth testing lives
in ``autopsy/``, where it can be tested without a server.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from autopsy.config import PipelineConfig, default_config
from autopsy.pipeline import Pipeline
from autopsy.store.chunks import Index

# State lives in `api.state`, a leaf module, so the route modules can reach it without
# importing this one — see that module's docstring for the cycle it removes. Re-exported
# here because `from api.main import get_state` is the documented spelling.
from api.state import AppState, close_state, get_state, set_state

__all__ = ["AppState", "create_app", "get_state"]


def _make_lifespan(index: Index | None, config: PipelineConfig | None):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved = index if index is not None else Index.read()
        cfg = config or default_config()
        # Refuse to serve an index this config cannot search. Left to fail lazily, the
        # mismatch appears as a per-chunk error inside a query and reads like a corrupt
        # corpus instead of a provider misconfiguration.
        resolved.assert_matches(cfg)
        set_state(AppState(index=resolved, pipeline=Pipeline(resolved), config=cfg))
        try:
            yield
        finally:
            # Releases the embedded-Qdrant file lock before the process exits; see
            # `api.state.close_state` for why that matters under `--reload`.
            close_state()

    return lifespan


def create_app(index: Index | None = None, config: PipelineConfig | None = None) -> FastAPI:
    """Build the app, optionally against a supplied index and config.

    Both are injectable so the tests can hand in a small in-memory corpus. Reading
    ``corpus/index`` unconditionally made the API tests depend on whichever provider
    the developer last ran ``ingest`` with — a dense index and a concept-embedder test
    run produce a vector-kind mismatch, and the suite starts failing for reasons that
    have nothing to do with the code under test.
    """
    app = FastAPI(
        title="Retrieval Autopsy",
        version="0.1.0",
        description="A RAG pipeline built to be observed.",
        lifespan=_make_lifespan(index, config),
    )
    # The web app is served from a different origin in development. Locked to explicit
    # origins rather than "*" so a wildcard does not survive into a deployment.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get(
            "AUTOPSY_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(","),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    from api.routes_query import router as query_router
    from api.routes_trace import router as trace_router
    from api.routes_vectors import router as vector_router
    from api.ws_eval import router as eval_router
    from api.ws_stream import router as ws_router

    app.include_router(query_router, prefix="/api")
    app.include_router(trace_router, prefix="/api")
    app.include_router(vector_router, prefix="/api")
    app.include_router(ws_router)
    app.include_router(eval_router)

    # The no-build inspector, served from the same origin as the API so it needs no
    # CORS entry and no bundler. `web/` holds the React version for anyone with Node;
    # this one is what runs on a machine that has only Python.
    _INSPECTOR = Path(__file__).parent / "static" / "inspector.html"

    @app.get("/", response_class=HTMLResponse)
    def inspector() -> str:
        return _INSPECTOR.read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, Any]:
        state = get_state()
        return {
            "ok": True,
            "chunks": len(state.index),
            "tenants": state.index.tenants(),
            "corpus": state.index.meta.get("corpus_version"),
            "provider": state.config.runtime.provider,
        }

    return app


app = create_app()
