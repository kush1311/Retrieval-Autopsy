"""Chunks and the on-disk index bundle.

The index is three plain files under ``corpus/index/``:

* ``chunks.jsonl``  — one chunk per line
* ``vectors.jsonl`` — one ``{chunk_id, ...}`` per line, dense or concept
* ``meta.json``     — corpus version, embedding model, dimensions, concept stats

No database. A directory of newline-delimited JSON is diffable, greppable, and
survives every tool you own; swap it for something else when it hurts, not before.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from autopsy.determinism import REPO_ROOT, sha256_of
from autopsy.providers.base import Embedding
from autopsy.textutil import ConceptStats

INDEX_DIR = REPO_ROOT / "corpus" / "index"

#: Documents every tenant may see. Shared reference material, and the second of the
#: isolation suite's two positive controls — if this stops being reachable, a system
#: that returns nothing to anybody would pass every leak probe.
GLOBAL_TENANT = "tenant_global"


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    tenant_id: str
    doc_id: str
    ordinal: int
    heading_path: tuple[str, ...]
    text: str

    @staticmethod
    def make_id(tenant_id: str, doc_id: str, ordinal: int, text: str) -> str:
        """Content-addressed, so ingest is idempotent: re-running over an unchanged
        corpus produces byte-identical IDs and therefore a no-op."""
        return "c_" + sha256_of([tenant_id, doc_id, ordinal, text])[:16]

    @classmethod
    def create(
        cls, *, tenant_id: str, doc_id: str, ordinal: int, heading_path: list[str], text: str
    ) -> "Chunk":
        return cls(
            chunk_id=cls.make_id(tenant_id, doc_id, ordinal, text),
            tenant_id=tenant_id,
            doc_id=doc_id,
            ordinal=ordinal,
            heading_path=tuple(heading_path),
            text=text,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "tenant_id": self.tenant_id,
            "doc_id": self.doc_id,
            "ordinal": self.ordinal,
            "heading_path": list(self.heading_path),
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=d["chunk_id"],
            tenant_id=d["tenant_id"],
            doc_id=d["doc_id"],
            ordinal=int(d["ordinal"]),
            heading_path=tuple(d.get("heading_path", [])),
            text=d["text"],
        )


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: dict[str, Embedding]
    meta: dict[str, Any]
    stats: ConceptStats
    #: True only for the index loaded from ``corpus/index/``. Everything else — the
    #: isolation suite's planted competing-documents corpus, test fixtures — is
    #: ephemeral and must never reach a shared vector store.
    #:
    #: This is a correctness boundary, not a convenience flag. Without it, running the
    #: isolation suite would upsert a twenty-chunk synthetic corpus into the persistent
    #: Qdrant collection and silently destroy the real one. It also happens to avoid
    #: embedded Qdrant's single-client-per-folder lock, but the data-integrity reason is
    #: the one that matters.
    persisted: bool = False
    _by_id: dict[str, Chunk] = field(default_factory=dict, repr=False)
    _by_tenant: dict[str, list[Chunk]] = field(default_factory=dict, repr=False)
    _by_doc: dict[tuple[str, str], dict[int, Chunk]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for c in self.chunks:
            self._by_id[c.chunk_id] = c
            self._by_tenant.setdefault(c.tenant_id, []).append(c)
            self._by_doc.setdefault((c.tenant_id, c.doc_id), {})[c.ordinal] = c

    # ---- lookups ----

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def tenants(self) -> list[str]:
        return sorted(self._by_tenant)

    def scope(self, tenant_id: str) -> list[Chunk]:
        """Every chunk a tenant is allowed to see: its own plus global.

        This is *the* authorisation boundary. Both retrieval legs and neighbour
        expansion resolve their candidate set through this one function, so there is a
        single place to audit and a single place a leak could originate.
        """
        own = self._by_tenant.get(tenant_id, [])
        if tenant_id == GLOBAL_TENANT:
            return list(own)
        return own + self._by_tenant.get(GLOBAL_TENANT, [])

    def scope_ids(self, tenant_id: str) -> frozenset[str]:
        return frozenset(c.chunk_id for c in self.scope(tenant_id))

    def neighbours(self, chunk: Chunk, span: int) -> list[Chunk]:
        """Chunks adjacent by ``ordinal`` within the same document.

        Note the ``(tenant_id, doc_id)`` key. Expansion looks documents up by ID, which
        is exactly the code path that bypasses a filtered query — keying on the tenant
        as well makes a cross-tenant neighbour unrepresentable rather than merely
        unlikely. This is the second of the two high-yield leak vectors.
        """
        doc = self._by_doc.get((chunk.tenant_id, chunk.doc_id), {})
        out = []
        for delta in range(-span, span + 1):
            if delta == 0:
                continue
            n = doc.get(chunk.ordinal + delta)
            if n is not None:
                out.append(n)
        return out

    def __len__(self) -> int:
        return len(self.chunks)

    # ---- persistence ----

    def write(self, directory: Path = INDEX_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        with (directory / "vectors.jsonl").open("w", encoding="utf-8") as fh:
            for cid, emb in self.vectors.items():
                row: dict[str, Any] = {"chunk_id": cid, "model_id": emb.model_id}
                if emb.dense is not None:
                    row["dense"] = list(emb.dense)
                else:
                    row["concepts"] = list(emb.concepts or ())
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        meta = dict(self.meta)
        meta["concept_stats"] = self.stats.to_dict()
        (directory / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def assert_matches(self, cfg: Any) -> None:
        """Fail at startup if this index cannot answer queries from this config.

        An index built with 384-d BGE vectors cannot be searched with the concept
        simulator's bags, or vice versa. Detected lazily, that mismatch surfaces as a
        per-chunk error in the middle of a query — the pipeline reports something like
        "chunk c_dd2aa… was indexed with dense vectors but the query is a concept bag",
        which reads as a corrupt corpus rather than the truth: the process is running a
        different provider than the one that built the index.

        Checked once, up front, where the fix is obvious and nothing has been rendered
        yet.
        """
        want_concept = getattr(getattr(cfg, "runtime", None), "embedder", None) == "concept"
        have = self.meta.get("vector_kind")
        if have is None:
            return
        have_concept = have == "concept"
        if want_concept == have_concept:
            return

        provider = getattr(getattr(cfg, "runtime", None), "provider", "?")
        embedder = getattr(getattr(cfg, "runtime", None), "embedder", "?")
        raise ValueError(
            "index/config mismatch — the index was built by a different provider.\n"
            f"  index holds : {have} vectors from {self.meta.get('embed_model')!r}\n"
            f"  this process: provider={provider!r} embedder={embedder!r}\n"
            "\n"
            "Either point the process at the provider that built the index, or rebuild:\n"
            "  .\\run.ps1 ingest      (loads .env)\n"
            "  make ingest\n"
            "\n"
            "If you launched without loading .env, the process fell back to the offline "
            "simulator and this is the first place it became visible."
        )

    @classmethod
    def read(cls, directory: Path = INDEX_DIR) -> "Index":
        directory = Path(directory)
        if not (directory / "chunks.jsonl").exists():
            raise FileNotFoundError(
                f"no index at {directory}. Run `python -m autopsy.cli ingest` first."
            )
        chunks = [Chunk.from_dict(json.loads(line)) for line in _lines(directory / "chunks.jsonl")]
        vectors: dict[str, Embedding] = {}
        for line in _lines(directory / "vectors.jsonl"):
            row = json.loads(line)
            vectors[row["chunk_id"]] = Embedding(
                model_id=row["model_id"],
                dense=tuple(row["dense"]) if "dense" in row else None,
                concepts=tuple(row["concepts"]) if "concepts" in row else None,
            )
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        stats = ConceptStats.from_dict(meta.pop("concept_stats"))

        missing = [c.chunk_id for c in chunks if c.chunk_id not in vectors]
        if missing:
            raise ValueError(
                f"{len(missing)} chunks have no vector (first: {missing[0]}). The index is "
                "half-written — re-run ingest rather than querying a partial corpus."
            )
        return cls(chunks=chunks, vectors=vectors, meta=meta, stats=stats, persisted=True)


def _lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


__all__ = ["Chunk", "GLOBAL_TENANT", "INDEX_DIR", "Index"]
