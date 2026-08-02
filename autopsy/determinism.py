"""Seed pinning, config hashing, version stamping, and the runtime assertions that
keep counterfactual diffs meaningful.

Design principle #4: a diff computed across two different model versions is *worse*
than no diff, because it looks meaningful. Everything here exists to make that
impossible rather than merely unlikely.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from autopsy.config import PipelineConfig

REPO_ROOT = Path(__file__).resolve().parent.parent

# Crockford base32, ULID's alphabet: no I, L, O, U.
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class DeterminismError(RuntimeError):
    """Raised when a run would produce results that cannot be honestly compared."""


# --------------------------------------------------------------------------------------
# Canonical serialisation
# --------------------------------------------------------------------------------------


def to_plain(obj: Any) -> Any:
    """Dataclasses -> dicts, recursively, with ``None`` preserved.

    ``None`` is load-bearing here: it is how an ablated stage is represented, so it
    must survive into the hash and into the trace.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    """Stable JSON: sorted keys, no incidental whitespace, no NaN."""
    return json.dumps(
        to_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def config_hash(cfg: PipelineConfig) -> str:
    return "sha256:" + sha256_of(cfg)


def resolved_config(cfg: PipelineConfig) -> dict[str, Any]:
    """The fully resolved config, as it goes into the trace.

    Stored in full rather than as a diff from defaults: a trace has to be
    interpretable in isolation, years later, without the code that produced it.
    """
    return to_plain(cfg)


# --------------------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------------------


def content_id(*parts: Any) -> str:
    """A 26-character Crockford-base32 identifier derived from content.

    Deliberately *not* random. The whole system is built on comparing runs, so
    re-running an identical (query, tenant, config) should land on the same trace ID
    and overwrite the same file. Reports then diff cleanly instead of churning on
    fresh UUIDs every run.
    """
    digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).digest()
    n = int.from_bytes(digest[:16], "big")
    out = []
    for _ in range(26):
        out.append(_CROCKFORD[n & 31])
        n >>= 5
    return "".join(reversed(out))


# --------------------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------------------


def set_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except ImportError:  # pragma: no cover - numpy is a hard dep, this is belt and braces
        pass


# --------------------------------------------------------------------------------------
# Versions
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def code_version() -> str:
    """Best available identifier for the code that produced a trace.

    Prefers a git SHA. Falls back to a hash of the Python sources under ``autopsy/``
    and ``evals/`` so that a checkout without git still produces *something* that
    changes when the code changes — an unversioned trace is a trace you cannot trust
    a diff against.
    """
    override = os.environ.get("AUTOPSY_CODE_VERSION")
    if override:
        return override
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if sha.returncode == 0 and sha.stdout.strip():
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            suffix = "-dirty" if dirty.stdout.strip() else ""
            return f"git@{sha.stdout.strip()}{suffix}"
    except (OSError, subprocess.SubprocessError):
        pass

    h = hashlib.sha256()
    for pkg in ("autopsy", "evals", "api"):
        root = REPO_ROOT / pkg
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            h.update(path.relative_to(REPO_ROOT).as_posix().encode())
            h.update(path.read_bytes())
    return f"src@{h.hexdigest()[:8]}"


def versions_block(
    cfg: PipelineConfig, corpus_version: str, embed_model: str | None = None
) -> dict[str, str]:
    """Provenance stamped into every trace.

    Model IDs carry a ``sim:`` prefix under the offline provider. Without it a trace
    reads ``"gen_model": "claude-sonnet-4-6"`` next to ``"provider": "offline"``, and
    anyone quoting a single field — a screenshot, a grep, a spreadsheet column — walks
    away believing Sonnet produced that answer. The configured ID is still there,
    because it is part of the config hash and the run is not reproducible without it;
    it is just no longer possible to read it as a claim that the model ran.

    It also makes ``assert_comparable`` do the right thing for free: an offline trace
    and a live trace now differ on ``gen_model`` as well as ``provider``, and they are
    genuinely not comparable.

    Prefixing is per-component, not per-run, because the two halves can differ: a Groq
    run pairs a real chat model with a real local embedder, but a run could equally
    pair a real chat model with the concept simulator. Stamping the whole trace off a
    single flag would mislabel one half of it.

    ``embed_model`` should be the ID the *index* was actually built with — passed in by
    the pipeline — not the one the config asked for. Provenance records what produced
    the vectors being searched.
    """
    sim_chat = cfg.runtime.provider == "offline"
    sim_embed = cfg.runtime.embedder == "concept"

    def stamp(model_id: str | None, simulated: bool) -> str:
        if model_id is None:
            return "none"
        return f"sim:{model_id}" if simulated else model_id

    resolved_embed = embed_model or (cfg.semantic.model_id if cfg.semantic else None)
    return {
        "corpus": corpus_version,
        "embed_model": stamp(resolved_embed if cfg.semantic else None, sim_embed),
        "gen_model": stamp(cfg.generation.model_id, sim_chat),
        "rerank_model": stamp(cfg.rerank.model_id if cfg.rerank else None, sim_chat),
        "provider": cfg.runtime.provider,
        "embedder": cfg.runtime.embedder,
        "code": code_version(),
    }


# --------------------------------------------------------------------------------------
# Runtime assertions
# --------------------------------------------------------------------------------------


def assert_deterministic(cfg: PipelineConfig) -> None:
    """Raise if this config cannot produce comparable results.

    Called on every ablation run, not just in tests. Temperature drift is silent,
    survives review, and poisons every number downstream.
    """
    if cfg.generation.temperature != 0.0:
        raise DeterminismError(
            f"generation.temperature is {cfg.generation.temperature}, must be 0.0 for "
            "any run whose output will be compared against another run"
        )


COMPARABLE_KEYS = ("corpus", "embed_model", "gen_model", "rerank_model", "provider", "code")


def assert_comparable(a: dict[str, str], b: dict[str, str]) -> None:
    """Refuse to diff two traces whose provenance differs.

    Raises rather than warns. A warning in a log is a warning nobody reads, and the
    resulting table looks exactly as authoritative as a correct one.

    ``rerank_model`` is exempt when one side ablated the reranker away — that is the
    ablation itself, not provenance drift.
    """
    mismatched = []
    for key in COMPARABLE_KEYS:
        av, bv = a.get(key), b.get(key)
        if av == bv:
            continue
        if key == "rerank_model" and "none" in (av, bv):
            continue  # no_rerank ablation, expected
        if key == "embed_model" and "none" in (av, bv):
            continue  # no_semantic ablation, expected
        mismatched.append(f"{key}: {av!r} vs {bv!r}")
    if mismatched:
        raise DeterminismError(
            "refusing to diff traces from different provenance — "
            + "; ".join(mismatched)
            + ". Re-run both sides against the same corpus, models and code."
        )


__all__ = [
    "DeterminismError",
    "REPO_ROOT",
    "assert_comparable",
    "assert_deterministic",
    "canonical_json",
    "code_version",
    "config_hash",
    "content_id",
    "resolved_config",
    "set_seeds",
    "sha256_of",
    "to_plain",
    "versions_block",
]
