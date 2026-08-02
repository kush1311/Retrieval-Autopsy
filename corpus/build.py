"""Corpus acquisition: fetch, convert, and hand off to ingest.

Two entry points:

    python -m corpus.build ingest            # seed corpus -> corpus/index (no network)
    python -m corpus.build fetch             # download the real sources
    python -m corpus.build ingest --real     # index what fetch downloaded

The fetcher pins provenance *on fetch* rather than shipping hand-written commit SHAs
in the manifest. A SHA typed into a config file by a human is a claim nobody checked;
a SHA recorded by the code that actually downloaded the bytes is evidence. The
resolved commit, the licence file it saw, and a content digest all land in
``corpus/manifest.lock.yaml``.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from autopsy.determinism import REPO_ROOT, sha256_of
from autopsy.ingest import default_sources, ingest
from autopsy.store.chunks import INDEX_DIR

MANIFEST = REPO_ROOT / "corpus" / "manifest.yaml"
LOCK = REPO_ROOT / "corpus" / "manifest.lock.yaml"


class LicenceError(RuntimeError):
    """Raised when a source may not be redistributed, or hasn't been checked."""


@dataclass(slots=True)
class Source:
    tenant_id: str
    repo: str
    ref: str
    subdir: str
    include: list[str]
    licence: str
    licence_verified: bool


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sources(manifest: dict[str, Any]) -> list[Source]:
    return [
        Source(
            tenant_id=s["tenant_id"],
            repo=s["repo"],
            ref=s.get("ref", "main"),
            subdir=s.get("subdir", ""),
            include=s.get("include", ["**/*.md"]),
            licence=s.get("licence", "UNKNOWN"),
            licence_verified=bool(s.get("licence_verified", False)),
        )
        for s in manifest.get("real", {}).get("sources", [])
    ]


def check_licence(source: Source, allowed: list[str]) -> None:
    if not source.licence_verified:
        raise LicenceError(
            f"{source.repo}: licence_verified is false.\n"
            f"  Declared licence: {source.licence}\n"
            f"  Open https://github.com/{source.repo} , read the LICENSE file, and set\n"
            f"  licence_verified: true in corpus/manifest.yaml once you have confirmed\n"
            f"  it permits redistributing the fetched text. This gate is the whole\n"
            f"  reason the manifest exists; do not flip it without reading."
        )
    if source.licence not in allowed:
        raise LicenceError(
            f"{source.repo}: licence {source.licence!r} is not in the allowed list "
            f"({', '.join(allowed)}). Add it deliberately or drop the source."
        )


def fetch_source(source: Source, dest_root: Path) -> dict[str, Any]:
    """Download a repo tarball and extract the matching files."""
    import httpx

    url = f"https://codeload.github.com/{source.repo}/tar.gz/{source.ref}"
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        response = client.get(url)
        response.raise_for_status()
        blob = response.content

    dest = dest_root / source.tenant_id
    dest.mkdir(parents=True, exist_ok=True)

    written = 0
    licence_head = ""
    resolved = source.ref
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        members = tar.getmembers()
        root = members[0].name.split("/")[0] if members else ""
        # codeload names the top directory <repo>-<ref>; when ref is a branch this is
        # the branch name, so it is not itself a pin. The content digest below is.
        for member in members:
            if not member.isfile():
                continue
            rel = member.name.split("/", 1)[1] if "/" in member.name else member.name
            if rel.upper().startswith(("LICENSE", "LICENCE", "COPYING")) and not licence_head:
                fh = tar.extractfile(member)
                if fh:
                    licence_head = fh.read(400).decode("utf-8", "replace").strip()
            prefix = f"{source.subdir}/" if source.subdir else ""
            if not rel.startswith(prefix):
                continue
            inner = rel[len(prefix) :]
            if not any(fnmatch.fnmatch(inner, pat) for pat in source.include):
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            data = fh.read().decode("utf-8", "replace")
            out = dest / inner.replace("/", "__")
            if out.suffix != ".md":
                out = out.with_suffix(".md")
            out.write_text(data, encoding="utf-8")
            written += 1

    if written == 0:
        raise RuntimeError(
            f"{source.repo}: fetched the archive but matched zero files under "
            f"{source.subdir!r} with {source.include}. The repository layout has "
            "probably changed — fix the manifest rather than indexing an empty tenant."
        )

    return {
        "repo": source.repo,
        "ref": source.ref,
        "root_dir": root,
        "resolved": resolved,
        "tenant_id": source.tenant_id,
        "files": written,
        "declared_licence": source.licence,
        "licence_file_head": licence_head[:200],
        "content_digest": sha256_of(sorted(p.name for p in dest.glob("*.md"))),
    }


def cmd_fetch(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    allowed = manifest.get("real", {}).get("allowed_licences", [])
    dest_root = REPO_ROOT / manifest.get("real", {}).get("path", "corpus/raw")
    records = []
    failures = 0
    for source in _sources(manifest):
        if args.tenant and source.tenant_id != args.tenant:
            continue
        try:
            check_licence(source, allowed)
            record = fetch_source(source, dest_root)
            records.append(record)
            print(f"  fetched {source.tenant_id}: {record['files']} files")
        except (LicenceError, RuntimeError) as exc:
            failures += 1
            print(f"  SKIPPED {source.tenant_id}\n{exc}", file=sys.stderr)

    if records:
        LOCK.write_text(
            yaml.safe_dump({"sources": records}, sort_keys=True), encoding="utf-8"
        )
        print(f"  provenance written to {LOCK.relative_to(REPO_ROOT)}")
    if not records:
        print(
            "\nNothing was fetched. Every source starts with licence_verified: false — "
            "that is deliberate. Read the licences, then flip the flags.",
            file=sys.stderr,
        )
    return 1 if failures and not records else 0


def cmd_ingest(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    if args.real:
        roots = [REPO_ROOT / manifest["real"]["path"]]
        label = manifest["real"]["label"]
    else:
        # Regenerate first so the corpus and the test set can never drift apart. A
        # test set asserting facts the corpus no longer contains fails in a way that
        # looks like a retrieval regression.
        from corpus.synthetic import write as write_generated

        n_docs, n_cases = write_generated()
        print(f"  generated {n_docs} documents and {n_cases} test cases")
        roots = default_sources()
        label = manifest["seed"]["label"]

    if not any(r.exists() and any(r.rglob("*.md")) for r in roots):
        print(
            f"no markdown under {', '.join(str(r) for r in roots)}. "
            + ("Run `python -m corpus.build fetch` first." if args.real else ""),
            file=sys.stderr,
        )
        return 1
    index, stats = ingest(roots, out=Path(args.out), label=label)
    print(
        f"  {stats['chunks']} chunks across {index.meta['n_docs']} docs "
        f"({stats['embedded']} embedded, {stats['reused']} reused)"
    )
    print(f"  corpus version: {index.meta['corpus_version']}")
    print(f"  tenants: {', '.join(index.meta['tenants'])}")
    print(f"  index: {Path(args.out).relative_to(REPO_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="corpus.build", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download the real sources named in the manifest")
    f.add_argument("--tenant", help="fetch only this tenant")
    f.set_defaults(fn=cmd_fetch)

    i = sub.add_parser("ingest", help="chunk, embed, and index")
    i.add_argument("--real", action="store_true", help="index corpus/raw instead of corpus/seed")
    i.add_argument("--out", default=str(INDEX_DIR))
    i.set_defaults(fn=cmd_ingest)

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
