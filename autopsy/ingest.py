"""Ingest: documents in, searchable index out.

Idempotent and content-addressed. A chunk's ID is the hash of
``(tenant_id, doc_id, ordinal, text)``, so re-running ingest over an unchanged corpus
produces identical IDs, reuses every existing embedding, and writes the same bytes.
That is what makes ``make ingest`` safe to put in a Makefile prerequisite instead of
a comment telling people to run it manually.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autopsy.chunking import chunk_markdown
from autopsy.config import PipelineConfig, default_config
from autopsy.determinism import sha256_of
from autopsy.providers import build_providers
from autopsy.providers.base import Embedding
from autopsy.store.chunks import INDEX_DIR, Chunk, Index
from autopsy.textutil import ConceptStats


@dataclass(slots=True)
class Document:
    tenant_id: str
    doc_id: str
    markdown: str


def corpus_version(docs: list[Document], label: str) -> str:
    """A version string that changes when, and only when, the corpus content changes."""
    digest = sha256_of(
        sorted((d.tenant_id, d.doc_id, sha256_of(d.markdown)) for d in docs)
    )
    return f"{label}@{digest[:8]}"


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in sorted(docs, key=lambda d: (d.tenant_id, d.doc_id)):
        for ordinal, piece in enumerate(chunk_markdown(doc.markdown)):
            chunks.append(
                Chunk.create(
                    tenant_id=doc.tenant_id,
                    doc_id=doc.doc_id,
                    ordinal=ordinal,
                    heading_path=piece.heading_path,
                    text=piece.text,
                )
            )
    return chunks


def build_index(
    docs: list[Document],
    *,
    cfg: PipelineConfig | None = None,
    label: str = "seed",
    reuse_from: Index | None = None,
) -> tuple[Index, dict[str, int]]:
    """Chunk, embed, and index. Returns the index and a small stats dict."""
    cfg = cfg or default_config()
    chunks = chunk_documents(docs)
    if not chunks:
        raise ValueError("no chunks produced — is the corpus directory empty?")

    stats = ConceptStats.from_texts([c.text for c in chunks])
    providers = build_providers(cfg, stats)
    embed_model = providers.embedder.model_id

    reusable: dict[str, Embedding] = {}
    if reuse_from is not None and reuse_from.meta.get("embed_model") == embed_model:
        reusable = reuse_from.vectors

    todo = [c for c in chunks if c.chunk_id not in reusable]
    vectors: dict[str, Embedding] = {
        c.chunk_id: reusable[c.chunk_id] for c in chunks if c.chunk_id in reusable
    }
    if todo:
        embedded = providers.embedder.embed_documents([c.text for c in todo])
        vectors.update({c.chunk_id: e for c, e in zip(todo, embedded)})

    sample = next(iter(vectors.values()))
    meta = {
        "corpus_version": corpus_version(docs, label),
        "embed_model": embed_model,
        "provider": providers.label,
        "vector_kind": sample.kind,
        "dim": len(sample.dense) if sample.dense else None,
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "tenants": sorted({c.tenant_id for c in chunks}),
    }
    index = Index(chunks=chunks, vectors=vectors, meta=meta, stats=stats)
    return index, {"chunks": len(chunks), "embedded": len(todo), "reused": len(vectors) - len(todo)}


def load_markdown_tree(root: Path) -> list[Document]:
    """Read ``<root>/<tenant_id>/**/*.md`` into documents.

    The tenant is the first path component. Filesystem layout as the source of truth
    for tenancy keeps the assignment visible in ``git status`` rather than buried in a
    field inside a file.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"corpus root {root} does not exist")
    docs: list[Document] = []
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root)
        if len(rel.parts) < 2:
            raise ValueError(
                f"{rel} sits at the corpus root; every document must live under a "
                "tenant directory so its tenancy is unambiguous"
            )
        docs.append(
            Document(
                tenant_id=rel.parts[0],
                doc_id="/".join(rel.parts[1:]),
                markdown=path.read_text(encoding="utf-8"),
            )
        )
    return docs


def ingest(
    sources: Path | list[Path],
    *,
    out: Path = INDEX_DIR,
    cfg: PipelineConfig | None = None,
    label: str = "seed",
) -> tuple[Index, dict[str, int]]:
    """Index one or more corpus roots into a single index.

    Multiple roots because the handwritten and generated corpora are the same tenants
    seen from different angles: the handwritten docs are what a human reads in the
    demo, the generated ones give retrieval enough material that ranking decides the
    outcome. Splitting them into two indexes would mean the thing being demonstrated
    and the thing being measured were different systems.
    """
    roots = [sources] if isinstance(sources, Path) else list(sources)
    docs: list[Document] = []
    seen: set[tuple[str, str]] = set()
    for root in roots:
        for doc in load_markdown_tree(root):
            key = (doc.tenant_id, doc.doc_id)
            if key in seen:
                raise ValueError(
                    f"{doc.tenant_id}/{doc.doc_id} appears in more than one corpus root; "
                    "duplicate documents would double-count in the IDF statistics"
                )
            seen.add(key)
            docs.append(doc)

    existing: Index | None = None
    try:
        existing = Index.read(out)
    except (FileNotFoundError, ValueError, KeyError):
        existing = None
    index, stats = build_index(docs, cfg=cfg, label=label, reuse_from=existing)
    index.write(out)
    return index, stats


def default_sources() -> list[Path]:
    """The corpus roots ``make ingest`` uses: handwritten, plus generated if present."""
    from autopsy.determinism import REPO_ROOT

    roots = [REPO_ROOT / "corpus" / "seed"]
    generated = REPO_ROOT / "corpus" / "generated"
    if generated.exists() and any(generated.rglob("*.md")):
        roots.append(generated)
    return roots


__all__ = [
    "Document",
    "build_index",
    "chunk_documents",
    "corpus_version",
    "default_sources",
    "ingest",
    "load_markdown_tree",
]
