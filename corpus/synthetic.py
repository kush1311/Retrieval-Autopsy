"""A generated corpus with ground truth, and the test set that goes with it.

The handwritten seed corpus is readable and good for the demo, but 66 chunks is too
small to measure anything: with ``top_k`` at 20, retrieval returns most of the corpus
and every ablation comes back identical because ranking never mattered. This module
generates a controlled corpus large enough that ranking is the deciding factor.

**The point of generating it is the ground truth.** Every fact carries a globally
unique token — a value like ``4271``, or a coined marker word like ``sporewood`` — that
appears in exactly one chunk. So "did this answer come from the right chunk?" reduces
to a substring check, with no LLM judge in the loop. That matters more than it sounds:
the headline ablation numbers are then arithmetic over string matches, and the judge
is needed only for the narrower question of whether two *differently worded* answers
say the same thing.

Four query shapes, each designed so a different ablation breaks it:

* ``identifier`` — "what does KLV-4213 mean". The dense leg smears digit runs, so it
  retrieves the whole ``KLV-42xx`` family and cannot rank within it. Breaks under
  ``no_lexical``.
* ``value`` — "what is the default value of `klv_...`". Exact key lookup.
* ``paraphrase`` — worded with synonyms of the terms the document uses, so BM25 has
  nothing to match on. Breaks under ``no_semantic``.
* ``absent`` — asks about an identifier in a real family that was never assigned. The
  correct behaviour is to refuse or hedge; answering about a neighbour is the
  near-miss failure the whole project is about.

Everything is seeded and deterministic: same seed, byte-identical corpus.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from autopsy.determinism import REPO_ROOT
from autopsy.ingest import Document

GENERATED_DIR = REPO_ROOT / "corpus" / "generated"
TESTSET_PATH = REPO_ROOT / "evals" / "testsets" / "generated.jsonl"

PRODUCTS = {
    "tenant_kelvin": ("Kelvin", "KLV"),
    "tenant_atlas": ("Atlas", "ATS"),
    "tenant_vela": ("Vela", "VLA"),
}

#: (module slug, plain-language activity, the noun the module operates on)
SUBSYSTEMS: list[tuple[str, str, str]] = [
    ("compaction", "merging segments", "segment"),
    ("checkpointing", "flushing state to disk", "checkpoint"),
    ("ingestion", "accepting incoming batches", "batch"),
    ("eviction", "releasing memory", "entry"),
    ("replication", "streaming changes to followers", "change"),
    ("hydration", "warming caches after a restart", "cache"),
    ("sharding", "distributing keys across nodes", "shard"),
    ("throttling", "pacing client requests", "request"),
    ("indexing", "building lookup structures", "index"),
    ("snapshotting", "capturing point-in-time images", "snapshot"),
    ("reconciliation", "repairing divergent replicas", "divergence"),
    ("scrubbing", "verifying stored checksums", "block"),
    ("prefetching", "reading ahead of demand", "page"),
    ("quiescing", "draining in-flight work", "operation"),
    ("federation", "forwarding queries to peers", "query"),
    ("archival", "moving cold data to object storage", "object"),
    ("tiering", "promoting hot data to fast storage", "tier"),
    ("coalescing", "batching small writes together", "write"),
]

#: ``(term used in the document, synonym used in the paraphrase query)``.
#:
#: Both members of each pair collapse to the same concept in ``textutil.SYNONYMS``, so
#: the dense leg matches and BM25 does not. This is what makes the ``paraphrase``
#: query shape a genuine test of the semantic leg rather than a reworded keyword
#: search — if a pair ever stopped being a synonym in that map, these queries would
#: silently become unanswerable, which is why the generator asserts on it.
SYNONYM_PAIRS: list[tuple[str, str]] = [
    ("threshold", "cap"),
    ("configuration", "config"),
    ("parameter", "param"),
    ("remove", "delete"),
    ("disable", "turn-off"),
    ("performance", "latency"),
    ("persistence", "durability"),
    ("replica", "standby"),
    ("limit", "maximum"),
    ("expiry", "ttl"),
]

_SYL_A = ["spore", "quill", "brack", "fen", "gild", "murr", "sable", "thorn", "wick", "yarrow",
          "cinder", "hollow", "marrow", "pell", "roan", "tarn", "vetch", "welk"]
_SYL_B = ["wood", "marsh", "hollow", "ridge", "fall", "gate", "loom", "reef", "spire", "vale",
          "drift", "shale", "coil", "mire", "crest", "bank"]

ASPECTS = ["interval_seconds", "threshold_bytes", "max_concurrency", "retry_budget"]

#: Subsystems that exist in no product's documentation, used for the ``out_of_scope``
#: query shape.
#:
#: These are the only queries where retrieval confidence is genuinely low, and they are
#: what makes the gate measurable. Every other shape — including ``absent``, which asks
#: about an unassigned code inside a *real* family — retrieves strongly, so the gate
#: passes and some later stage catches the problem. Without these, ``no_gate`` scores
#: "no measurable effect" and the reader would reasonably conclude the gate is useless,
#: when in fact the test set simply never asked it a question it could answer.
PHANTOM_SUBSYSTEMS = [
    "wavefront pinning", "lattice folding", "cascade damping", "harmonic pruning",
    "isotope balancing", "spindle collation", "vector annealing", "drift stitching",
]


@dataclass
class Fact:
    """One checkable claim: a query, and the unique token a correct answer contains."""

    case_id: str
    tenant_id: str
    kind: str
    query: str
    expect: str  # "answer_contains" | "refuse_or_hedge"
    answer_key: str | None
    doc_id: str
    heading: str
    followup: str | None = None
    history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "tenant_id": self.tenant_id,
            "kind": self.kind,
            "query": self.query,
            "expect": self.expect,
            "answer_key": self.answer_key,
            "doc_id": self.doc_id,
            "heading": self.heading,
            "history": self.history,
        }


class _Unique:
    """Hands out globally unique tokens so a substring check is unambiguous."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.numbers: set[int] = set()
        self.markers: set[str] = set()

    def number(self, low: int = 1000, high: int = 99999) -> int:
        for _ in range(10_000):
            n = self.rng.randint(low, high)
            if n not in self.numbers:
                self.numbers.add(n)
                return n
        raise RuntimeError("exhausted the unique-number space")

    def marker(self) -> str:
        for _ in range(10_000):
            word = self.rng.choice(_SYL_A) + self.rng.choice(_SYL_B)
            if word not in self.markers:
                self.markers.add(word)
                return word
        raise RuntimeError("exhausted the unique-marker space")


def _config_section(product: str, key: str, value: int, unit: str, activity: str) -> str:
    return (
        f"### `{key}`\n\n"
        f"The default value of `{key}` is {value}. It bounds {activity} on a single "
        f"node, expressed in {unit}. {product} reads this value once at startup and "
        f"does not reload it on a configuration change, so a value edited in place "
        f"takes effect only after a restart. Setting it below the documented minimum "
        f"is accepted at startup and clamped silently, which is a common source of "
        f"configurations that appear to have no effect.\n"
    )


def _error_section(prefix: str, code: int, marker: str, noun: str, activity: str) -> str:
    ident = f"{prefix}-{code}"
    return (
        f"### {ident} — {noun} operation aborted\n\n"
        f"{ident} means the {marker} budget was exhausted while {activity}. The "
        f"operation is abandoned and retried at the next scheduled attempt; nothing "
        f"is lost, but the backlog grows until a retry succeeds.\n\n"
        f"The usual cause is a node that has been running with the budget set below "
        f"steady-state demand, so every attempt starts already behind. Raising the "
        f"budget resolves it; restarting the node does not.\n"
    )


def _overview_section(product: str, module: str, activity: str, noun: str, doc_term: str) -> str:
    return (
        f"## Overview\n\n"
        f"{module.capitalize()} is the {product} subsystem responsible for {activity}. "
        f"It runs continuously in the background and is governed by a {doc_term} that "
        f"determines how much work it may do per cycle. Each cycle selects the oldest "
        f"eligible {noun} and processes it to completion before considering the next, "
        f"so a single oversized {noun} can delay everything queued behind it.\n\n"
        f"{module.capitalize()} never blocks foreground traffic. If it cannot keep up, "
        f"the backlog grows and read amplification rises, but writes continue to be "
        f"accepted at full rate.\n"
    )


def generate(
    *, seed: int = 1729, modules_per_tenant: int = 18
) -> tuple[list[Document], list[Fact]]:
    rng = random.Random(seed)
    unique = _Unique(rng)
    docs: list[Document] = []
    facts: list[Fact] = []

    for tenant_id, (product, prefix) in PRODUCTS.items():
        for m_index in range(modules_per_tenant):
            module, activity, noun = SUBSYSTEMS[m_index % len(SUBSYSTEMS)]
            variant = m_index // len(SUBSYSTEMS)
            slug = module if variant == 0 else f"{module}_v{variant + 1}"
            doc_id = f"reference/{slug}.md"
            doc_term, query_term = SYNONYM_PAIRS[m_index % len(SYNONYM_PAIRS)]

            body = [f"# {product} {slug.replace('_', ' ')}\n"]
            body.append(_overview_section(product, module, activity, noun, doc_term))

            body.append("## Configuration\n")
            keys: list[tuple[str, int, str]] = []
            for aspect in ASPECTS[:3]:
                key = f"{prefix.lower()}_{slug}_{aspect}"
                value = unique.number()
                unit = aspect.rsplit("_", 1)[-1]
                keys.append((key, value, unit))
                body.append(_config_section(product, key, value, unit, activity))

            body.append("## Errors\n")
            base_code = 4000 + m_index * 10
            codes: list[tuple[int, str]] = []
            for offset in (1, 2, 3):
                code = base_code + offset
                marker = unique.marker()
                codes.append((code, marker))
                body.append(_error_section(prefix, code, marker, noun, activity))

            docs.append(
                Document(tenant_id=tenant_id, doc_id=doc_id, markdown="\n".join(body))
            )

            # -- ground truth --------------------------------------------------

            for code, marker in codes:
                facts.append(
                    Fact(
                        case_id=f"{tenant_id}:{slug}:id:{code}",
                        tenant_id=tenant_id,
                        kind="identifier",
                        query=f"what does {prefix}-{code} mean",
                        expect="answer_contains",
                        answer_key=marker,
                        doc_id=doc_id,
                        heading=f"{prefix}-{code}",
                    )
                )

            for key, value, _unit in keys:
                facts.append(
                    Fact(
                        case_id=f"{tenant_id}:{slug}:value:{key}",
                        tenant_id=tenant_id,
                        kind="value",
                        query=f"what is the default value of {key}",
                        expect="answer_contains",
                        answer_key=str(value),
                        doc_id=doc_id,
                        heading=key,
                    )
                )

            # Paraphrase: the document says `doc_term`, the query says `query_term`.
            # BM25 sees no shared rare token; the concept mapping sees one.
            facts.append(
                Fact(
                    case_id=f"{tenant_id}:{slug}:para",
                    tenant_id=tenant_id,
                    kind="paraphrase",
                    query=(
                        f"which {query_term} controls how much work {product} does per "
                        f"cycle when {activity}"
                    ),
                    expect="answer_contains",
                    answer_key=doc_term,
                    doc_id=doc_id,
                    heading="Overview",
                )
            )

            # Absent: a plausible neighbour in a real family that was never assigned.
            facts.append(
                Fact(
                    case_id=f"{tenant_id}:{slug}:absent",
                    tenant_id=tenant_id,
                    kind="absent",
                    query=f"what does {prefix}-{base_code + 7} mean",
                    expect="refuse_or_hedge",
                    answer_key=None,
                    doc_id=doc_id,
                    heading="(none)",
                )
            )

            # Out of scope: no document anywhere mentions this subsystem, so retrieval
            # confidence is genuinely low and the gate is the stage that should catch
            # it. This is the only query shape that exercises the gate.
            phantom = PHANTOM_SUBSYSTEMS[m_index % len(PHANTOM_SUBSYSTEMS)]
            facts.append(
                Fact(
                    case_id=f"{tenant_id}:{slug}:oos",
                    tenant_id=tenant_id,
                    kind="out_of_scope",
                    query=f"how do I configure {phantom} in {product}",
                    expect="refuse_or_hedge",
                    answer_key=None,
                    doc_id="(none)",
                    heading="(none)",
                )
            )

            # Follow-up: exercises the rewrite path, which is a second entry into
            # retrieval and therefore a place isolation can break.
            facts.append(
                Fact(
                    case_id=f"{tenant_id}:{slug}:followup",
                    tenant_id=tenant_id,
                    kind="followup",
                    query="what is its default value",
                    expect="answer_contains",
                    answer_key=str(keys[0][1]),
                    doc_id=doc_id,
                    heading=keys[0][0],
                    history=[f"tell me about {keys[0][0]}"],
                )
            )

    _assert_unique_keys(facts)
    return docs, facts


def _assert_unique_keys(facts: Iterable[Fact]) -> None:
    """Ground truth is only ground truth if each key identifies exactly one fact.

    A duplicated answer key would let a wrong answer pass the substring check, which
    would inflate every accuracy number in the report without failing anything.
    """
    seen: dict[str, tuple[str, str, str]] = {}
    for fact in facts:
        if fact.answer_key is None or fact.kind == "paraphrase":
            continue  # paraphrase keys are shared vocabulary by design
        source = (fact.doc_id, fact.heading, fact.tenant_id)
        prior = seen.get(fact.answer_key)
        if prior is None:
            seen[fact.answer_key] = source
            continue
        if prior != source:
            # Two different facts sharing a token means a wrong answer could satisfy
            # the substring check for the other one, silently inflating accuracy.
            # Two *questions* about the same fact sharing a token is fine and
            # intended — the follow-up case re-asks the value case.
            raise AssertionError(
                f"answer key {fact.answer_key!r} is claimed by two different facts: "
                f"{prior} and {source}; the substring check would be ambiguous"
            )


def write(
    *, seed: int = 1729, modules_per_tenant: int = 18,
    docs_dir: Path = GENERATED_DIR, testset: Path = TESTSET_PATH,
) -> tuple[int, int]:
    docs, facts = generate(seed=seed, modules_per_tenant=modules_per_tenant)
    for doc in docs:
        path = docs_dir / doc.tenant_id / doc.doc_id
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc.markdown, encoding="utf-8")
    testset.parent.mkdir(parents=True, exist_ok=True)
    with testset.open("w", encoding="utf-8") as fh:
        for fact in facts:
            fh.write(json.dumps(fact.to_dict(), ensure_ascii=False) + "\n")
    return len(docs), len(facts)


def load_testset(path: Path = TESTSET_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m corpus.synthetic` to generate it."
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sample(cases: list[dict[str, Any]], n: int, seed: int = 4242) -> list[dict[str, Any]]:
    """A deterministic, kind-stratified sample.

    Stratified rather than uniform so the ablation table cannot be moved by a lucky
    draw that happens to over-sample the query shape a given ablation is weakest on.
    """
    if n >= len(cases):
        return list(cases)
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_kind.setdefault(case["kind"], []).append(case)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    kinds = sorted(by_kind)
    per_kind = max(1, n // len(kinds))
    for kind in kinds:
        pool = sorted(by_kind[kind], key=lambda c: c["case_id"])
        out.extend(rng.sample(pool, min(per_kind, len(pool))))
    remaining = [c for c in cases if c not in out]
    rng.shuffle(remaining)
    out.extend(remaining[: max(0, n - len(out))])
    return sorted(out, key=lambda c: c["case_id"])


if __name__ == "__main__":
    n_docs, n_cases = write()
    print(f"  wrote {n_docs} documents to {GENERATED_DIR.relative_to(REPO_ROOT)}")
    print(f"  wrote {n_cases} test cases to {TESTSET_PATH.relative_to(REPO_ROOT)}")


__all__ = [
    "Fact",
    "GENERATED_DIR",
    "PRODUCTS",
    "SUBSYSTEMS",
    "SYNONYM_PAIRS",
    "TESTSET_PATH",
    "generate",
    "load_testset",
    "sample",
    "write",
]
