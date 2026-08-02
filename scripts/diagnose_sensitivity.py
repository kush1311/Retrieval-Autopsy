"""Why does the ablation table read all zeros?

An ablation can only change the answer if it changes what reaches the generator. If the
evidence is identical, the answer is identical by construction — no model call needed to
establish that. So retrieval sensitivity is measurable for free, at full corpus scale.

That is also the fix for the token budget: spend generation calls only on cases whose
context actually moved, instead of burning a capped daily quota on cases that cannot
move.

The gold signal needs no judge. Every generated case carries an ``answer_key`` — a
globally unique coined token that appears in exactly the chunks that answer it. If no
chunk containing that token reaches the generator, the answer *cannot* be correct. That
is ground truth by construction.

Run:  python scripts/diagnose_sensitivity.py [n_cases]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("AUTOPSY_PROVIDER", "groq")
os.environ.setdefault("AUTOPSY_EMBEDDER", "fastembed")
# Retrieval sensitivity is independent of the chat provider; only the embedder matters,
# and fastembed is local. Forcing the numpy store also avoids fighting a running API
# server for the embedded-Qdrant lock.
os.environ["AUTOPSY_VECTOR_BACKEND"] = "local"

from autopsy.ablations import compose  # noqa: E402
from autopsy.config import default_config  # noqa: E402
from autopsy.pipeline import Pipeline, PipelineError  # noqa: E402
from autopsy.store import Index  # noqa: E402
from corpus.synthetic import load_testset, sample  # noqa: E402

ABLATIONS = ["no_lexical", "no_semantic", "no_gate", "no_rerank", "no_expansion", "top_k_1"]


def gold_chunks(index: Index, answer_key: str) -> set[str]:
    """Chunks that contain the case's unique answer token."""
    if not answer_key:
        return set()
    needle = answer_key.lower()
    return {c.chunk_id for c in index.chunks if needle in c.text.lower()}


def context_of(pipe: Pipeline, case: dict, cfg) -> list[str] | None:
    try:
        trace = pipe.run(
            case["query"], tenant_id=case["tenant_id"], cfg=cfg, generate=False
        )
    except PipelineError:
        return None
    return [c.chunk_id for c in trace.context_chunks()]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    index = Index.read()
    pipe = Pipeline(index)
    base = default_config()
    cases = sample(load_testset(), n)

    print(f"corpus {len(index)} chunks · {len(cases)} cases · retrieval only, 0 tokens")
    print(f"max_context_chunks={base.generation.max_context_chunks} "
          f"lexical.top_k={base.lexical.top_k} semantic.top_k={base.semantic.top_k}\n")

    # Baseline: does the gold chunk even reach the generator normally?
    baseline: dict[str, list[str]] = {}
    gold: dict[str, set[str]] = {}
    base_has_gold = 0
    scored = []
    for case in cases:
        ctx = context_of(pipe, case, base)
        if ctx is None:
            continue
        g = gold_chunks(index, case.get("answer_key", ""))
        if not g:
            continue
        baseline[case["case_id"]] = ctx
        gold[case["case_id"]] = g
        scored.append(case)
        if g & set(ctx):
            base_has_gold += 1

    total = len(scored)
    print(f"baseline: gold chunk reached the generator in {base_has_gold}/{total} cases "
          f"({base_has_gold/max(1,total):.0%})\n")

    print(f"{'ablation':<16} {'ctx same':>9} {'reordered':>10} {'set changed':>12} "
          f"{'GOLD LOST':>10} {'gold gained':>12}")
    print("-" * 74)

    rows = {}
    for name in ABLATIONS:
        cfg = compose([name], base)
        same = reordered = set_changed = lost = gained = 0
        for case in scored:
            cid = case["case_id"]
            b = baseline[cid]
            ctx = context_of(pipe, case, cfg)
            if ctx is None:
                continue
            g = gold[cid]
            had, has = bool(g & set(b)), bool(g & set(ctx))
            if had and not has:
                lost += 1
            elif has and not had:
                gained += 1
            if ctx == b:
                same += 1
            elif set(ctx) == set(b):
                reordered += 1
            else:
                set_changed += 1
        rows[name] = (same, reordered, set_changed, lost, gained)
        print(f"{name:<16} {same:>9} {reordered:>10} {set_changed:>12} {lost:>10} {gained:>12}")

    print("\n" + "=" * 74)
    print("DIAGNOSIS")
    print("=" * 74)
    worst = max(rows.items(), key=lambda kv: kv[1][3])
    if all(v[3] == 0 for v in rows.values()):
        print("No ablation ever removes the gold chunk from the context.")
        print("The zeros are correct, not a bug: identical evidence must give an")
        print("identical answer.")
        print()
        print(f"Cause: context breadth. {base.generation.max_context_chunks} chunks reach")
        print(f"the generator out of up to {base.lexical.top_k + base.semantic.top_k}")
        print("candidates, so an ablation reshuffles ranking without changing evidence.")
        print("Ranking damage is invisible when the window is wide enough to absorb it.")
        print()
        print("Two honest routes:")
        print("  1. Tighten max_context_chunks until ranking is load-bearing, and report")
        print("     the sensitivity curve. `top_k_1` above is the extreme of this.")
        print("  2. Report it as the finding: hybrid-retrieval ablations do not change")
        print("     answers at 12-chunk context on this corpus. That is a real result,")
        print("     and more useful than a rigged one.")
    else:
        print(f"`{worst[0]}` removes the gold chunk in {worst[1][3]}/{total} cases.")
        print("Those are exactly the cases worth spending generation tokens on —")
        print(f"{worst[1][3]} calls instead of {total}, a {total/max(1,worst[1][3]):.0f}x saving.")


if __name__ == "__main__":
    main()
