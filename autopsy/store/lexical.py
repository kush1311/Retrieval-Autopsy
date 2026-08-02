"""BM25 over tenant-scoped chunk text.

**One index per tenant scope, not one index with a filter applied afterwards.**

Post-filtering BM25 is the classic version of the pre/post-filter bug and it is worse
here than in vector search, because BM25 has a second, invisible failure: IDF is
computed over whatever document set the index was built on. Score the whole corpus and
then drop foreign hits and you get two errors at once — a top-20 that collapses to
three results, *and* term weights derived from documents the tenant cannot see. The
second one is untraceable from the outside; it just looks like the ranking is bad.

Building a small index per scope costs a few milliseconds on a corpus this size and
makes both problems structurally impossible.
"""

from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache

from rank_bm25 import BM25Okapi

from autopsy.store.chunks import Chunk, Index
from autopsy.textutil import content_tokens


def _smoothed_idf(corpus: list[list[str]]) -> dict[str, float]:
    """Lucene-style IDF: ``log(1 + (N - df + 0.5) / (df + 0.5))``.

    The classic Robertson–Sparck-Jones form that ``rank_bm25`` implements,
    ``log(N - df + 0.5) - log(df + 0.5)``, goes to **exactly zero** when a term appears
    in half the documents, and negative above that. A term with zero IDF contributes
    nothing, so on a small tenant scope a query for the one identifier that matters can
    score 0.0 against every chunk and return an empty result set — and the failure
    presents as "retrieval found nothing", not as "the weighting collapsed".

    This never fired on the 449-chunk corpus and fired immediately on a four-chunk test
    fixture, which is a good argument for keeping small fixtures around. The ``+1``
    inside the log keeps IDF strictly positive at every document frequency.
    """
    n = len(corpus)
    df: Counter[str] = Counter()
    for doc in corpus:
        df.update(set(doc))
    return {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}


class LexicalIndex:
    def __init__(self, index: Index) -> None:
        self._index = index
        self._cache: dict[tuple[str, float, float], tuple[BM25Okapi, list[Chunk]]] = {}

    def _build(self, tenant_id: str, k1: float, b: float) -> tuple[BM25Okapi, list[Chunk]]:
        key = (tenant_id, k1, b)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        chunks = self._index.scope(tenant_id)
        if not chunks:
            raise ValueError(f"tenant {tenant_id!r} has no chunks in scope")
        corpus = [_tokens(c.text) for c in chunks]
        bm25 = BM25Okapi(corpus, k1=k1, b=b)
        # rank_bm25 handles term frequency and length normalisation correctly; only its
        # IDF needs replacing. Swapping the dict after construction keeps the library
        # doing the part it does well.
        bm25.idf = _smoothed_idf(corpus)
        self._cache[key] = (bm25, chunks)
        return bm25, chunks

    def search(
        self, *, query: str, tenant_id: str, top_k: int, k1: float, b: float
    ) -> list[tuple[Chunk, float]]:
        bm25, chunks = self._build(tenant_id, k1, b)
        tokens = _tokens(query)
        if not tokens:
            return []
        scores = bm25.get_scores(tokens)
        ranked = sorted(
            ((c, float(s)) for c, s in zip(chunks, scores) if s > 0.0),
            # chunk_id as the tiebreaker so an equal-score pair orders identically on
            # every run — otherwise a stable-sort artefact becomes a phantom diff.
            key=lambda r: (-r[1], r[0].chunk_id),
        )
        return ranked[:top_k]


@lru_cache(maxsize=200_000)
def _tokens_cached(text: str) -> tuple[str, ...]:
    return tuple(content_tokens(text))


def _tokens(text: str) -> list[str]:
    """Surface tokens, identifiers intact, no stemming.

    Not stemming is the point of this leg. ``appendfsync`` must not collapse into
    ``appendfsyn``, and ``error_code_4021`` must stay distinct from
    ``error_code_4022`` — the dense leg is the one that generalises, and the whole
    value of hybrid retrieval is that the two legs fail differently.
    """
    return list(_tokens_cached(text))


__all__ = ["LexicalIndex"]
