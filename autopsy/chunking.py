"""Markdown chunking: structural first, then size-capped with overlap.

Three rules, in priority order:

1. **Never split a code block.** A truncated code block is worse than an oversized
   chunk. Exact identifiers inside fenced code are the entire reason the lexical leg
   beats the dense one on this corpus; cutting a fence in half destroys the property
   the whole hybrid demo rests on.
2. **Split on headings before splitting on size.** A chunk that begins mid-argument
   retrieves badly and reads worse.
3. **Record ``heading_path`` and ``ordinal`` on every chunk.** ``ordinal`` is what
   neighbour expansion walks; ``heading_path`` gives readable citations for free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_TOKENS = 800
MIN_TOKENS = 30
OVERLAP_RATIO = 0.15

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass(slots=True)
class Section:
    heading_path: list[str]
    body: str


@dataclass(slots=True)
class Piece:
    heading_path: list[str]
    text: str


def split_sections(markdown: str) -> list[Section]:
    """Split into heading-delimited sections, ignoring headings inside code fences.

    A ``#`` at the start of a line inside a shell block is a comment, not a heading.
    Tracking the fence state is the difference between chunking a document and
    shredding it.
    """
    stack: list[tuple[int, str]] = []
    sections: list[Section] = []
    buf: list[str] = []
    path: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body or path:
            sections.append(Section(heading_path=list(path), body=body))
        buf.clear()

    for line in markdown.splitlines():
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence = False
            buf.append(line)
            continue

        heading = None if in_fence else _HEADING_RE.match(line)
        if heading:
            flush()
            level, title = len(heading.group(1)), heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            path = [t for _, t in stack]
        else:
            buf.append(line)

    flush()
    return [s for s in sections if s.body]


def _fence_safe_split(body: str, target: int, overlap: int) -> list[str]:
    """Size-cap a section without ever cutting a fenced block.

    Splits on blank-line-delimited blocks, treating a whole fence as one atomic block.
    A single fence larger than the target is emitted oversized on purpose — rule 1
    beats rule 3.
    """
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in body.splitlines():
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence = False
            current.append(line)
            continue
        if not in_fence and not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))

    out: list[str] = []
    buf: list[str] = []
    size = 0
    for block in blocks:
        btoks = estimate_tokens(block)
        if buf and size + btoks > target:
            out.append("\n\n".join(buf))
            # Carry the tail of the previous chunk forward so a claim split across the
            # boundary is still retrievable from at least one side.
            carry: list[str] = []
            carried = 0
            for prev in reversed(buf):
                ptoks = estimate_tokens(prev)
                if carried + ptoks > overlap:
                    break
                carry.insert(0, prev)
                carried += ptoks
            buf, size = carry, carried
        buf.append(block)
        size += btoks
    if buf:
        out.append("\n\n".join(buf))
    return out or [body]


def chunk_markdown(
    markdown: str, *, target_tokens: int = TARGET_TOKENS, min_tokens: int = MIN_TOKENS
) -> list[Piece]:
    """Chunk a document into ordered pieces."""
    sections = split_sections(markdown)
    overlap = int(target_tokens * OVERLAP_RATIO)

    # Merge a section too small to stand alone into the next one. A bare
    # "## Persistence faults" with one line under it is a retrieval distractor, not a
    # chunk: it matches the topic and carries no answer.
    merged: list[Section] = []
    carry: Section | None = None
    for sec in sections:
        if carry is not None:
            sec = Section(heading_path=sec.heading_path, body=carry.body + "\n\n" + sec.body)
            carry = None
        if estimate_tokens(sec.body) < min_tokens:
            carry = sec
            continue
        merged.append(sec)
    if carry is not None:
        if merged:
            merged[-1] = Section(
                heading_path=merged[-1].heading_path, body=merged[-1].body + "\n\n" + carry.body
            )
        else:
            merged.append(carry)

    pieces: list[Piece] = []
    for sec in merged:
        for part in _fence_safe_split(sec.body, target_tokens, overlap):
            text = part.strip()
            if text:
                pieces.append(Piece(heading_path=list(sec.heading_path), text=text))
    return pieces


__all__ = [
    "MIN_TOKENS",
    "OVERLAP_RATIO",
    "Piece",
    "Section",
    "TARGET_TOKENS",
    "chunk_markdown",
    "estimate_tokens",
    "split_sections",
]
