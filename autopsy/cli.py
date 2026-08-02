"""Command line entry point.

    python -m autopsy.cli ingest       # build the index (idempotent)
    python -m autopsy.cli query "..."  # run one query, print the trace
    python -m autopsy.cli eval         # isolation + silent-failure, exit non-zero on regression
    python -m autopsy.cli ablate       # the counterfactual sweep -> reports/ablation.md
    python -m autopsy.cli calibrate    # judge calibration -> reports/judge-calibration.md
    python -m autopsy.cli schema       # export the JSON Schema the TS types come from
    python -m autopsy.cli demo         # freeze pre-recorded traces for the keyless demo
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from autopsy.ablations import CORE_THREE, all_ablations
from autopsy.config import default_config
from autopsy.counterfactual import CounterfactualEngine, render_report, tabulate
from autopsy.determinism import REPO_ROOT
from autopsy.ingest import default_sources, ingest as run_ingest
from autopsy.pipeline import Pipeline
from autopsy.store.chunks import INDEX_DIR, Index
from autopsy.trace import Trace, export_json_schema

TRACES_DIR = REPO_ROOT / "reports" / "traces"
DEMO_TRACES_DIR = REPO_ROOT / "web" / "src" / "demo" / "traces"


def _load_index() -> Index:
    try:
        return Index.read()
    except FileNotFoundError as exc:
        print(f"{exc}\n", file=sys.stderr)
        raise SystemExit(2) from exc


# Every Pipeline the command builds is registered here and closed by main() before the
# interpreter starts tearing down.
#
# `Pipeline` has been a context manager since embedded Qdrant arrived, but the commands
# each built one and dropped it, leaving the exclusive file lock to whenever the garbage
# collector got round to it. That surfaced as a traceback printed *after* a successful
# answer:
#
#     Exception ignored in: <function QdrantClient.__del__ ...>
#     ImportError: sys.meta_path is None, Python is likely shutting down
#
# Cosmetic on its own — the command had already done its work — but it reads as a crash,
# and the lock outliving the process by an unpredictable margin is the thing that makes
# "run a query, then start the server" fail intermittently.
#
# An ExitStack rather than four `with` blocks: the commands have many early returns, and
# re-indenting each body to wrap it would be a much larger change than the bug warrants.
_CLEANUP = contextlib.ExitStack()


def _pipeline(index: Index) -> Pipeline:
    """Build a Pipeline whose resources are released deterministically by main()."""
    return _CLEANUP.enter_context(Pipeline(index))


def _provider() -> str:
    return default_config().runtime.provider


# --------------------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    from corpus.synthetic import write as write_generated

    if not args.no_generate:
        n_docs, n_cases = write_generated()
        print(f"  generated {n_docs} documents and {n_cases} test cases")
    index, stats = run_ingest(default_sources(), out=Path(args.out))
    print(f"  {stats['chunks']} chunks ({stats['embedded']} embedded, {stats['reused']} reused)")
    print(f"  corpus version: {index.meta['corpus_version']}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from autopsy.ablations import compose

    index = _load_index()
    pipe = _pipeline(index)
    cfg = default_config()
    if args.ablation:
        cfg = compose(args.ablation, cfg)

    trace = pipe.run(
        args.query, tenant_id=args.tenant, cfg=cfg,
        history=args.history or [], ablations=args.ablation or [],
    )
    if args.json:
        print(trace.to_json())
        return 0

    print(f"\n  query      {trace.query}")
    if trace.rewritten_query:
        print(f"  rewritten  {trace.rewritten_query}")
    print(f"  tenant     {trace.tenant_id}")
    print(f"  ablations  {', '.join(trace.ablations) or 'none (baseline)'}")
    print(f"  trace      {trace.trace_id}\n")

    print("  stages")
    for stage in trace.stages:
        mark = "skip" if stage.skipped else "run "
        cache = f" cache={stage.cache}" if stage.cache else ""
        print(f"    [{mark}] {stage.name:16s} {stage.ms:7.2f}ms{cache}")
        if stage.skip_reason:
            print(f"             ↳ {stage.skip_reason}")

    print("\n  retrieval competition (top 8)")
    for c in trace.candidates[:8]:
        flag = "*" if c.in_context else " "
        lex = f"L{c.lexical_rank}" if c.lexical_rank else "L-"
        sem = f"S{c.semantic_rank}" if c.semantic_rank else "S-"
        rr = f" rr={c.rerank_score}" if c.rerank_score is not None else ""
        why = c.inclusion_reason or c.rejected_by or ""
        head = c.heading_path[-1] if c.heading_path else c.doc_id
        print(f"   {flag} #{c.fused_rank or '-':<3} {lex:<4} {sem:<4}{rr:<10} {why:<19} {head[:44]}")

    print(f"\n  answer [{trace.answer.status}]")
    for line in _wrap(trace.answer.text, 76):
        print(f"    {line}")
    print(
        f"\n  {trace.totals.ms:.0f}ms · ${trace.totals.cost_usd:.5f} · "
        f"{trace.totals.llm_calls} llm calls\n"
    )
    if args.save:
        path = trace.write(TRACES_DIR)
        print(f"  written to {path.relative_to(REPO_ROOT)}\n")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from evals.judge import GOLDEN_DIR, build_judge, calibrate, load_golden
    from evals.runner import (
        REPORTS_DIR, base_versions, exit_code, load_baseline,
        render_report as render_eval, run_suites, write_baseline, write_report,
    )
    from evals.suites.isolation import IsolationSuite
    from evals.suites.silent_failure import SilentFailureSuite

    index = _load_index()
    provider = _provider()
    versions = base_versions(index.meta, provider)

    suites: list[Any] = [IsolationSuite(), SilentFailureSuite(index=index)]
    report = run_suites(suites, versions=versions, provider=provider,
                        on_progress=lambda m: print(f"  {m}"))

    golden = load_golden()
    if golden:
        cal = calibrate(build_judge(provider), golden)
        report.notes.append(
            f"judge agreement with {cal.label_source} labels: {cal.agreement:.2f} "
            f"(kappa {cal.kappa:.2f}, n={cal.n}) — see reports/judge-calibration.md"
        )
    else:
        # `load_golden()` reads every *.jsonl under evals/golden/. Finding none is a
        # correct result, not a discovery bug — but the old advice was
        # "Run `python -m autopsy.cli calibrate`", which without --derive also finds
        # nothing and prints its own "run with --derive" hint. That sent a reader in a
        # loop, and it made reports/judge-calibration.md look like it contradicted this
        # line when in fact that report is a stale artifact of a run whose derived set is
        # no longer on disk.
        stale = (REPORTS_DIR / "judge-calibration.md").exists()
        note = (
            f"judge calibration: no labels in {GOLDEN_DIR.relative_to(REPO_ROOT).as_posix()}/ "
            f"(*.jsonl), so every judge-derived number below is UNCALIBRATED. "
            f"`calibrate --derive` writes a synthetic set from the corpus; hand labels go "
            f"in evals/golden/human.jsonl and are the ones that count."
        )
        if stale:
            note += (
                " NOTE: reports/judge-calibration.md exists but its input set is absent — "
                "treat that report as stale, not as evidence this run was calibrated."
            )
        report.notes.append(note)

    baseline = load_baseline(provider=provider)
    path = write_report(render_eval(report, baseline), "eval.md")
    print(f"\n  report: {path.relative_to(REPO_ROOT)}")

    # A standalone isolation report, because it is the one people ask to see on its
    # own — a tenant-isolation result buried in a combined eval is a result nobody
    # forwards.
    from evals.runner import Report as EvalReport

    for suite_name, filename in (("isolation", "isolation.md"),
                                 ("silent_failure", "silent-failure.md")):
        subset = [f for f in report.findings if f.suite == suite_name]
        if not subset:
            continue
        sub = EvalReport(findings=subset, versions=versions, provider=provider,
                         notes=list(report.notes))
        title = suite_name.replace("_", " ").title() + " report"
        sub_path = write_report(render_eval(sub, baseline, title=title), filename)
        print(f"  report: {sub_path.relative_to(REPO_ROOT)}")

    if args.update_baseline:
        write_baseline(report)
        print("  baseline updated")
        return 0

    code = exit_code(report, baseline)
    passed = sum(1 for f in report.findings if f.passed)
    print(f"  {passed}/{len(report.findings)} findings passed · exit {code}")
    for finding in report.blocking:
        print(f"  BLOCKING {finding.severity.name} {finding.key}: {finding.detail[:110]}")
    return code


def cmd_ablate(args: argparse.Namespace) -> int:
    from corpus.synthetic import load_testset, sample
    from evals.judge import build_judge
    from evals.runner import base_versions, write_report
    from evals.suites.ablation import write_snapshot

    index = _load_index()
    provider = _provider()
    pipe = _pipeline(index)
    cases = sample(load_testset(), args.n)
    ablations = list(CORE_THREE) if args.core else all_ablations()

    engine = CounterfactualEngine(
        pipeline=pipe,
        base_config=default_config(),
        judge=build_judge(provider),
        max_workers=args.workers,
        trace_sink=(lambda t: t.write(TRACES_DIR)) if args.save_traces else None,
    )
    print(f"  {len(cases)} cases × {len(ablations)} ablations, provider={provider}")

    def progress(done: int, total: int) -> None:
        if done % 20 == 0 or done == total:
            print(f"    {done}/{total} cases")

    diffs = engine.sweep(cases, ablations, on_progress=progress)
    versions = base_versions(index.meta, provider)
    text = render_report(diffs, versions=versions, n_cases=len(cases), provider=provider)

    # The filename carries the provider. Simulator and real-model sweeps answer different
    # questions and must never overwrite each other: an offline sweep can afford 3,000
    # runs and a free-tier sweep cannot, so the big impressive table is always the
    # simulated one. Sharing `ablation.md` between them is how a rule-based number ends
    # up quoted as a model result — which is exactly what happened here before this
    # change, with a 220x14 simulator table sitting under a README heading that said
    # "The finding" and named no provider.
    stem = "ablation" if provider != "offline" else "ablation-simulated"
    path = write_report(text, f"{stem}.md")
    print(f"\n  report: {path.relative_to(REPO_ROOT)}")

    table = tabulate(diffs)
    if args.snapshot:
        write_snapshot(table, versions)
        print("  ablation snapshot updated")

    Path(REPO_ROOT / "reports" / f"{stem}-diffs.jsonl").write_text(
        "\n".join(json.dumps(d.to_dict()) for d in diffs) + "\n", encoding="utf-8"
    )
    for ablation in sorted(table, key=lambda a: -table[a].get("now_confident_wrong", 0)):
        row = table[ablation]
        total = sum(row.values()) or 1
        cw = row.get("now_confident_wrong", 0)
        print(f"    {ablation:28s} confident-wrong {cw:4d}/{total} ({100 * cw / total:.1f}%)")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from evals.judge import GOLDEN_DIR, GoldenCase, build_judge, calibrate, load_golden, render_calibration
    from evals.runner import write_report

    index = _load_index()
    provider = _provider()

    if args.derive:
        cases = _derive_golden(index, n=args.n)
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        (GOLDEN_DIR / "derived.jsonl").write_text(
            "\n".join(json.dumps(c.to_dict()) for c in cases) + "\n", encoding="utf-8"
        )
        print(f"  derived {len(cases)} labelled pairs -> evals/golden/derived.jsonl")

    golden = load_golden()
    if not golden:
        print("  no golden cases; run with --derive or add evals/golden/human.jsonl",
              file=sys.stderr)
        return 2
    cal = calibrate(build_judge(provider), golden)
    path = write_report(render_calibration(cal), "judge-calibration.md")
    print(f"  agreement {cal.agreement:.2f} · kappa {cal.kappa:.2f} · n={cal.n}")
    print(f"  order instability {cal.instability_rate:.2f} · verbosity bias {cal.verbosity_bias:+.3f}")
    print(f"  report: {path.relative_to(REPO_ROOT)}")
    return 0


def _derive_golden(index: Index, n: int = 100):
    """Build labelled answer pairs from the corpus's own ground truth.

    Objective, reproducible, and *not* a substitute for human labels — the report says
    so at the top. A human also judges whether differently-worded answers mean the same
    thing, which no amount of substring checking captures.
    """
    from corpus.synthetic import load_testset, sample
    from evals.judge import GoldenCase, Verdict

    pipe = _pipeline(index)
    cfg = default_config()
    from autopsy.ablations import apply

    variants = [
        "no_lexical", "no_gate", "no_rerank", "top_k_1", "no_discriminator_guard",
        "no_semantic", "no_rewrite", "no_lexical+no_discriminator_guard",
    ]
    # Oversample: byte-equal pairs are skipped below because there is nothing for a
    # judge to have an opinion about, and on this corpus most ablations leave most
    # answers untouched. Drawing exactly `n` cases yields a fraction of that.
    cases = sample(load_testset(), min(n * 6, 486))
    out = []
    for i, case in enumerate(cases):
        if len(out) >= n:
            break
        ablation = variants[i % len(variants)]
        base = pipe.run(case["query"], tenant_id=case["tenant_id"], cfg=cfg,
                        history=case.get("history") or [])
        var = pipe.run(case["query"], tenant_id=case["tenant_id"],
                       cfg=apply(ablation, cfg), history=case.get("history") or [])
        if base.answer.text == var.answer.text:
            continue  # nothing for a judge to disagree about
        key = (case.get("answer_key") or "").lower()
        expects_refusal = case.get("expect") == "refuse_or_hedge"

        def ok(trace) -> bool:
            if expects_refusal:
                return trace.answer.status == "refused" or trace.answer.hedged
            return bool(key) and key in trace.answer.text.lower()

        b_ok, v_ok = ok(base), ok(var)

        def declined(trace) -> bool:
            return trace.answer.status == "refused" or trace.answer.hedged

        # Correctness alone is too blunt to be a reference label. "One is right and
        # the other is wrong" collapses two very different relationships: an answer
        # that *declines* has not contradicted anything, while an answer that asserts
        # a different fact has. Labelling both CONTRADICTORY makes the reference
        # disagree with any sane judge and produces a meaningless kappa.
        if b_ok and v_ok:
            label = Verdict.EQUIVALENT
        elif b_ok != v_ok:
            loser = var if b_ok else base
            label = Verdict.DEGRADED if declined(loser) else Verdict.CONTRADICTORY
        else:
            label = Verdict.DEGRADED
        out.append(
            GoldenCase(
                case_id=f"{case['case_id']}|{ablation}",
                question=case["query"],
                answer_a=base.answer.text,
                answer_b=var.answer.text,
                label=label,
                source="derived",
            )
        )
    return out


def cmd_calibrate_gate(args: argparse.Namespace) -> int:
    from autopsy.gatecal import calibrate_all, render_comparison
    from corpus.synthetic import load_testset, sample
    from evals.runner import write_report

    index = _load_index()
    cfg = default_config()
    results = calibrate_all(index, cfg, sample(load_testset(), args.n))
    path = write_report(render_comparison(results), "gate-calibration.md")

    print(f"  embedder {results[0].embed_model}\n")
    print(f"  {'signal':<14} {'thresh':>8} {'false-refuse':>13} {'false-admit':>12} {'separation':>11}")
    for cal in results:
        print(
            f"  {cal.reads:<14} {cal.threshold:>8g} {cal.false_refusal_rate:>12.1%} "
            f"{cal.false_admit_rate:>11.1%} {cal.separation:>+11.3f}"
        )

    best = results[0]
    print(f"\n  best: gate.reads={best.reads} threshold={best.threshold}")
    if not best.usable:
        print("  WARNING: even the best signal overlaps; no threshold gates this corpus")
    print(f"  report: {path.relative_to(REPO_ROOT)}")
    print(f"\n  AUTOPSY_GATE_READS={best.reads} AUTOPSY_GATE_THRESHOLD={best.threshold}")
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    from autopsy.tsgen import generate as generate_ts

    # newline="\n" on both, because CI enforces `make schema && git diff --exit-code`.
    # Without it, Python's text mode translates \n to \r\n on Windows while .gitattributes
    # stores LF, so a Windows contributor sees the entire file as modified after every
    # regeneration and the sync check becomes noise they learn to ignore.
    json_path = Path(args.out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(export_json_schema(), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"  wrote {json_path.relative_to(REPO_ROOT)}")

    ts_path = Path(args.ts)
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    ts_path.write_text(generate_ts(), encoding="utf-8", newline="\n")
    print(f"  wrote {ts_path.relative_to(REPO_ROOT)}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Freeze a handful of traces so the deployed demo needs no key and no backend."""
    index = _load_index()
    pipe = _pipeline(index)
    cfg = default_config()
    from autopsy.ablations import apply

    scripted = [
        ("tenant_kelvin", "what does KLV-4021 mean", ["no_lexical"]),
        ("tenant_kelvin", "how do I disable append log rewrite throttling in Kelvin", ["no_gate"]),
        ("tenant_atlas", "what is the default checkpoint_timeout_seconds in Atlas", ["top_k_1"]),
        ("tenant_vela", "which Vela distance metric is the fastest", ["no_rerank"]),
    ]
    DEMO_TRACES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for tenant, query, ablations in scripted:
        base = pipe.run(query, tenant_id=tenant, cfg=cfg)
        base.write(DEMO_TRACES_DIR)
        entry = {"query": query, "tenant": tenant, "baseline": base.trace_id, "variants": {}}
        for ablation in ablations:
            var = pipe.run(query, tenant_id=tenant, cfg=apply(ablation, cfg),
                           ablations=[ablation])
            var.write(DEMO_TRACES_DIR)
            entry["variants"][ablation] = var.trace_id
        manifest.append(entry)
    (DEMO_TRACES_DIR / "index.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  {len(manifest)} scripted queries frozen to "
          f"{DEMO_TRACES_DIR.relative_to(REPO_ROOT)}")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autopsy", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest")
    p.add_argument("--out", default=str(INDEX_DIR))
    p.add_argument("--no-generate", action="store_true")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("query")
    p.add_argument("query")
    p.add_argument("--tenant", default="tenant_kelvin")
    p.add_argument("--ablation", action="append", help="repeatable")
    p.add_argument("--history", action="append", help="repeatable, oldest first")
    p.add_argument("--json", action="store_true")
    p.add_argument("--save", action="store_true")
    p.set_defaults(fn=cmd_query)

    p = sub.add_parser("eval")
    p.add_argument("--update-baseline", action="store_true")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("ablate")
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--core", action="store_true", help="only no_lexical/no_gate/no_rerank")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--snapshot", action="store_true", help="pin this run as the regression reference")
    p.add_argument("--save-traces", action="store_true")
    p.set_defaults(fn=cmd_ablate)

    p = sub.add_parser("calibrate")
    p.add_argument("--derive", action="store_true", help="rebuild derived labels first")
    p.add_argument("-n", type=int, default=100)
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("calibrate-gate", help="derive gate.threshold from the corpus")
    p.add_argument("-n", type=int, default=240)
    p.set_defaults(fn=cmd_calibrate_gate)

    p = sub.add_parser("schema")
    p.add_argument("--out", default=str(REPO_ROOT / "web" / "src" / "lib" / "trace.schema.json"))
    p.add_argument("--ts", default=str(REPO_ROOT / "web" / "src" / "lib" / "trace.ts"))
    p.set_defaults(fn=cmd_schema)

    p = sub.add_parser("demo")
    p.set_defaults(fn=cmd_demo)

    args = parser.parse_args(argv)
    # `with _CLEANUP` closes every pipeline the command opened, on the success path and on
    # the way out of an exception alike, while the import system is still alive.
    with _CLEANUP:
        return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
