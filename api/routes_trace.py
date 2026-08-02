"""GET /trace/{id} — read a trace back off disk."""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException

from autopsy.cli import DEMO_TRACES_DIR, TRACES_DIR
from autopsy.trace import Trace

router = APIRouter()

#: Trace IDs are 26 Crockford-base32 characters. Validating the shape before touching
#: the filesystem is what stops `GET /trace/..%2f..%2fetc%2fpasswd` from being a path
#: traversal — the ID goes straight into a path join, so it must never be free text.
_TRACE_ID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


@router.get("/trace/{trace_id}", response_model=Trace)
def get_trace(trace_id: str) -> Trace:
    if not _TRACE_ID_RE.match(trace_id):
        raise HTTPException(status_code=400, detail="malformed trace id")
    for directory in (TRACES_DIR, DEMO_TRACES_DIR):
        path = directory / f"{trace_id}.json"
        if path.exists():
            return Trace.read(path)
    raise HTTPException(status_code=404, detail="no such trace")
