"""The inspector's inline JavaScript must parse.

There is no Node in this environment, so nothing else checks it. A syntax error in a
700-line inline script does not break one handler — it takes the *entire page* with it,
silently, with a working-looking server behind it. That is precisely the failure mode
this project exists to complain about, so it should not ship one.

QuickJS is a real ES2020 engine with a Python binding: no Node, no build step, runs in
CI. `esprima` was tried first and rejected — it predates ES2020 and chokes on `??`, and
a checker that forces the source to avoid modern syntax is worse than no checker.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

quickjs = pytest.importorskip("quickjs", reason="pip install quickjs")

INSPECTOR = Path(__file__).resolve().parent.parent / "api" / "static" / "inspector.html"


@pytest.fixture(scope="module")
def script() -> str:
    html = INSPECTOR.read_text(encoding="utf-8")
    blocks = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    assert blocks, "no <script> block in inspector.html"
    return "\n".join(blocks)


def test_javascript_parses(script):
    """Compile the script in a real engine.

    Wrapped in a function body that is never called, so compilation checks syntax
    without executing anything — the script's top level touches `document`, `fetch` and
    `WebSocket`, none of which exist here.
    """
    ctx = quickjs.Context()
    try:
        ctx.eval("(function(){" + script + "\n})")
    except Exception as exc:  # noqa: BLE001 - quickjs raises its own error type
        pytest.fail(f"inspector.html JavaScript is not parseable: {exc}")


def test_the_page_loads_nothing_from_the_network(script):
    """A demo that needs a CDN is a demo that breaks on a plane — and on a locked-down
    corporate network, which is where it is most likely to be opened."""
    html = INSPECTOR.read_text(encoding="utf-8")
    external = re.findall(r'(?:src|href)\s*=\s*["\'](https?://[^"\']+)', html)
    allowed = ("http://www.w3.org",)  # SVG namespace in the inlined favicon
    bad = [u for u in external if not u.startswith(allowed)]
    assert not bad, f"external resources would break offline use: {bad}"


@pytest.mark.parametrize(
    "name",
    ["svLoad", "svTick", "svProject", "svOnEvent", "svReset", "svPlaceQuery",
     "svClearProbes", "svBind", "ask", "replay", "renderAll", "startEval"],
)
def test_expected_entry_points_exist(script, name):
    """Cheap guard against a rename landing in one place and not the other."""
    assert re.search(rf"function\s+{name}\b", script), f"{name} is gone"


def test_the_3d_view_declares_its_own_lossiness(script):
    """36.8% of the variance is on screen. Without saying so, the picture claims a
    precision it does not have, and someone will read two adjacent dots as similar."""
    html = INSPECTOR.read_text(encoding="utf-8")
    assert "of the variance" in html
    assert "not the metric" in html


def test_point_radius_is_clamped(script):
    """A 335px radius once turned 449 points into one purple disc. The clamp means the
    next arithmetic slip degrades to 'dots slightly too big' instead of no picture."""
    assert re.search(r"Math\.min\(\s*\d+\s*,\s*Math\.max\(", script), \
        "the point-radius clamp is gone"


# ── eval probes must not outlive the run that produced them ─────────────────────────
# One renderer serves two canvases and one shared SV object. That is the whole bug class:
# an eval run left its markers in SV.marks, so opening SPACE showed the last suite's
# failures overlapping a new query's retrieval beams — a reading of the corpus that no
# run produced. Three independent guards, tested independently.

def _extract(script: str, name: str) -> str:
    """Pull one top-level `function name(...){...}` out by brace matching."""
    m = re.search(rf"function\s+{name}\s*\(", script)
    assert m, f"{name} not found"
    i = script.index("{", m.end() - 1)
    depth = 0
    for j in range(i, len(script)):
        if script[j] == "{":
            depth += 1
        elif script[j] == "}":
            depth -= 1
            if depth == 0:
                return script[m.start():j + 1]
    raise AssertionError(f"unbalanced braces in {name}")


@pytest.fixture(scope="module")
def probe_ctx(script):
    """svReset and svClearProbes, in a real engine, with the DOM stubbed just enough."""
    ctx = quickjs.Context()
    ctx.eval("""
      var CLASSES = {};
      function stubEl(id){ return {
        classList: { remove: function(c){ CLASSES[id] = false; },
                     add:    function(c){ CLASSES[id] = true;  },
                     contains: function(c){ return !!CLASSES[id]; } },
        style: {}, innerHTML: "", className: "", textContent: "" }; }
      var document = { getElementById: stubEl };
      function $(sel){ return stubEl(sel.replace("#","")); }
      var SV = { pts: [], marks: [], links: [], ctx: [], cid: "space",
                 llmT: 0, ansT: 0, thinking: false, active: null, ready: false };
      function svResize(){}
    """)
    for fn in ("svReset", "svClearProbes", "svBind"):
        ctx.eval(_extract(script, fn))
    return ctx


def _load_a_finished_eval_run(ctx):
    ctx.eval("""
      CLASSES["evalmap"] = true;
      SV.cid = "evalspace";
      SV.marks = [{x:0,y:0,z:0,outcome:"wrong_confident",case_id:"trap_1",t:1},
                  {x:1,y:0,z:0,outcome:"correct",case_id:"trap_2",t:1}];
      SV.pts = [{lit:1},{lit:0.45},{lit:0}];
    """)


def test_a_new_question_discards_the_previous_eval_run(probe_ctx):
    """svReset runs at the top of ask(). If it leaves marks behind, they render under the
    new query's beams and look like part of this answer."""
    _load_a_finished_eval_run(probe_ctx)
    probe_ctx.eval("svReset()")
    assert probe_ctx.eval("SV.marks.length") == 0, "eval markers survived a new question"
    assert probe_ctx.eval("SV.pts.filter(function(p){return p.lit;}).length") == 0


def test_clearing_probes_also_puts_the_map_away(probe_ctx):
    """A visible-but-empty map box reads as 'the run found nothing', not 'no run'."""
    _load_a_finished_eval_run(probe_ctx)
    probe_ctx.eval("svClearProbes()")
    assert probe_ctx.eval("SV.marks.length") == 0
    assert probe_ctx.eval("SV.pts.filter(function(p){return p.lit;}).length") == 0
    assert probe_ctx.eval('CLASSES["evalmap"]') is False, "the eval map stayed on"


def test_marks_are_drawn_only_on_the_eval_canvas(script):
    """The last-resort guard. Even if every clear above is forgotten, the renderer must
    refuse to draw eval markers into SPACE — the two views share one SV."""
    tick = _extract(script, "svTick")
    m = re.search(r"for\s*\(\s*const\s+mk\s+of\s*\(([^)]*)\)\s*\)", tick)
    assert m, "the eval-marker loop is no longer guarded by a canvas check"
    guard = m.group(1)
    assert "evalspace" in guard and "SV.cid" in guard, \
        f"marker loop is not scoped to the eval canvas: {guard!r}"


def test_every_suite_clears_the_previous_map(script):
    """Isolation plots nothing in this projection. It must still clear: leaving the
    previous suite's markers up attributes that run's failures to this one."""
    started = script[script.index('m.type==="started"'):]
    started = started[:started.index('m.type==="finding"')]
    assert started.count("svClearProbes()") >= 2, (
        "only one branch of the suite-start handler clears probes; the other leaves the "
        "previous run's map on screen"
    )
