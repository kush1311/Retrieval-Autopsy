"""Every api module must import on its own, in any order.

`get_state` used to live in `api/main.py`, which imports all four route modules in order to
register them. Each route module imported it back. A cycle — but a silent one: importing
`api.main` reaches `get_state` before it reaches the routers, so the common path always
worked and nothing ever complained.

It only bites when a route module is imported *first* — a focused test, a script, an
editor's autoreload, a `python -c "import api.ws_eval"`. Then Python starts with a
half-initialised `api.main` and raises `ImportError: cannot import name 'get_state'`
pointing at a file nobody edited.

A subprocess per module, because import cycles are order-dependent and pytest has already
imported `api.main` by the time any in-process check would run — the very ordering that
hides the bug.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

MODULES = [
    "api.state",
    "api.main",
    "api.routes_query",
    "api.routes_vectors",
    "api.ws_stream",
    "api.ws_eval",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_first_in_a_cold_interpreter(module: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"`import {module}` fails when it is the first api module imported.\n"
        f"This is an import cycle, even if the app still starts normally.\n\n"
        f"{proc.stderr[-2000:]}"
    )


def test_state_module_does_not_import_the_app() -> None:
    """`api.state` must stay a leaf. One `from api.<x> import` in it restores the cycle,
    and the app would keep working, so nothing else would notice."""
    source = (REPO_ROOT / "api" / "state.py").read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("from api", "import api"))
        and not line.strip().startswith("#")
    ]
    assert not offenders, f"api/state.py must not import from api/: {offenders}"
