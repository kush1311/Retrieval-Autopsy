"""The offline provider — a deterministic simulator of a RAG stack's model layer.

**Read this before quoting any number produced under it.** This is not a small
language model and it is not a cheap embedding API. It is a rule-based stand-in whose
purpose is to make the pipeline, the eval suites, the counterfactual engine, and the
inspector fully runnable and testable with no API key, so that ``make eval`` can be a
CI gate. Every report generated in this mode is stamped ``provider=offline``, and the
report writer refuses to omit that stamp.

What it *does* faithfully reproduce is the **mechanism** each ablation is meant to
expose. Those mechanisms are real, documented failure modes, not inventions:

* Dense retrieval smears exact numeric identifiers (subword tokenisation), so it
  retrieves the right *family* of documents and the wrong *member*. This is what
  makes ``no_lexical`` produce a confidently wrong answer instead of an obviously
  empty one.
* A generator answers from whatever context it is handed, far past the point where
  a calibrated retrieval gate would have refused. That gap is the whole reason a gate
  exists, and it is why ``no_gate`` is not a no-op.
* Merging BM25 and cosine scores by naive addition lets the unbounded scale dominate
  the bounded one. That is what ``no_fusion`` demonstrates, and it is why RRF works
  on ranks.

The thresholds below are the simulator's parameters. They are collected in one place,
named, and commented, rather than scattered as magic numbers — if you disagree with
one, you can see exactly what it changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from autopsy.providers.base import (
    Completion,
    Embedding,
    GeneratedAnswer,
    SourceChunk,
    Usage,
    estimate_tokens,
)
from autopsy.textutil import (
    ConceptStats,
    clean_sentence,
    concept_set,
    concepts,
    content_tokens,
    coverage,
    discriminators,
    is_identifier,
    is_prose,
    missing_concepts,
    raw_tokens,
    sentences,
    strip_code_fences,
)

def _lexicon_fingerprint() -> str:
    """Fold the concept-mapping rules into the model ID.

    Ingest reuses a stored vector whenever the chunk text and the embedding model ID
    both match. For a hosted embedding API the model ID is a real version; for this
    simulator the "model" is the synonym lexicon, the stopword list, and the stemmer,
    and all three live in the repo and change under you. Editing a synonym while the
    ID stayed `offline-concept-v1` meant the index silently kept vectors built by the
    *previous* rules — retrieval scored against one concept space while queries were
    embedded in another, with no error anywhere.

    Hashing the rules into the ID makes that impossible: change a rule and every vector
    is correctly treated as stale.
    """
    from autopsy.determinism import sha256_of
    from autopsy.textutil import _SUFFIXES, POLARITY, STOPWORDS, SYNONYMS

    return sha256_of(
        [sorted(SYNONYMS.items()), list(_SUFFIXES), sorted(STOPWORDS), sorted(POLARITY)]
    )[:8]


OFFLINE_EMBED_MODEL = f"offline-concept-{_lexicon_fingerprint()}"

# --------------------------------------------------------------------------------------
# Simulator parameters
# --------------------------------------------------------------------------------------

#: Below this whole-context coverage the generator gives up. Deliberately *far* below
#: the default gate threshold of 0.42. A real generator is much more willing to answer
#: from thin context than a calibrated gate is to allow it, and that asymmetry is
#: precisely the value the gate adds. Raising this to meet the gate would make the
#: ``no_gate`` ablation return "identical" and quietly prove nothing.
REFUSE_BELOW_CONTEXT_COVERAGE = 0.18

#: Above this single-sentence coverage the generator answers from one source and is
#: fully confident.
CONFIDENT_ABOVE = 0.55

#: Below this it stitches the two best sentences from different sources together —
#: the confident synthesis of two unrelated settings that the near-miss traps target.
SYNTHESISE_BELOW = 0.55

#: The discriminator guard fires whenever a query discriminator — an identifier,
#: version number, or polarity word — appears in none of the retrieved sources.
#: Unconditionally: if the question is *about* `KLV-4213` and no source mentions
#: `KLV-4213`, no amount of topical coverage makes an answer about `KLV-4214`
#: acceptable.
#:
#: It is not free. A query saying "version 7" against a source saying "7.x" trips it
#: and produces an unnecessary hedge, so the guard trades some over-refusal for a
#: large reduction in confident wrongness. That trade is exactly what the
#: ``no_discriminator_guard`` ablation measures, which is why this is a config field
#: and not a constant.
GUARD_FIRES_ON_ANY_UNMET_DISCRIMINATOR = True

#: Reranker score = this much joint query/chunk coverage, the rest exact-identifier
#: overlap. The identifier term is what lets the reranker repair a dense-only
#: ordering, which is why ``no_rerank`` costs something on identifier queries.
RERANK_COVERAGE_WEIGHT = 0.6

_DEICTIC = frozenset(
    {"it", "that", "this", "they", "them", "those", "these", "there", "its", "one"}
)


# --------------------------------------------------------------------------------------
# Embedder
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class OfflineEmbedder:
    model_id: str = OFFLINE_EMBED_MODEL

    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        return [
            Embedding(model_id=self.model_id, concepts=tuple(sorted(concept_set(t))))
            for t in texts
        ]

    def embed_query(self, text: str) -> tuple[Embedding, Usage]:
        # Query side keeps duplicates and order — the store weights each occurrence by
        # IDF, so collapsing to a set here would silently change the similarity.
        return (
            Embedding(model_id=self.model_id, concepts=tuple(concepts(text))),
            Usage(tokens_in=estimate_tokens(text), calls=0),
        )


# --------------------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------------------


class OfflineLLM:
    """Rule-based stand-ins for rewrite, rerank, and generate."""

    def __init__(self, stats: ConceptStats) -> None:
        self.stats = stats

    # -- rewrite ------------------------------------------------------------------

    def rewrite(self, *, query: str, history: list[str], model_id: str) -> Completion:
        """Resolve a dangling reference using the previous turn.

        Substitutes the first deictic token with the highest-IDF content phrase from
        the most recent history entry. Crude, deterministic, and enough to exercise
        the rewrite path — which matters because rewrite is a *second entry into
        retrieval* and therefore one of the two places tenant isolation actually
        breaks.
        """
        if not history:
            return Completion(text=query, model_id=model_id)

        tokens = raw_tokens(query)
        if not any(t in _DEICTIC for t in tokens):
            return Completion(text=query, model_id=model_id)

        prev = history[-1]
        already = set(concepts(query))

        def concept_of(token: str) -> str:
            got = concepts(token)
            return got[0] if got else token

        # Only tokens that actually occur in the corpus are candidates. IDF alone is
        # the wrong ranking here and fails in a way that looks like a retrieval bug:
        # a word appearing in *zero* documents scores the maximum IDF, so "tell me
        # about `foo_bar`" resolves the pronoun to "tell" and the follow-up retrieves
        # nothing. Filtering on df > 0 is the fix; preferring identifiers is the
        # refinement, since a follow-up almost always refers back to a named thing.
        salient = [
            t
            for t in content_tokens(prev)
            if len(t) > 2
            and concept_of(t) not in already
            and self.stats.df.get(concept_of(t), 0) > 0
        ]
        salient.sort(
            key=lambda t: (not is_identifier(t), -self.stats.idf(concept_of(t)), t)
        )
        phrase = " ".join(dict.fromkeys(salient[:2]))
        if not phrase:
            return Completion(text=query, model_id=model_id)

        out_words: list[str] = []
        replaced = False
        for word in query.split():
            bare = word.strip(".,?!:;").lower()
            if not replaced and bare in _DEICTIC:
                out_words.append(phrase)
                replaced = True
            else:
                out_words.append(word)
        text = " ".join(out_words)
        return Completion(
            text=text,
            model_id=model_id,
            usage=Usage(
                tokens_in=estimate_tokens(query + prev), tokens_out=estimate_tokens(text), calls=1
            ),
        )

    # -- rerank -------------------------------------------------------------------

    def rerank(
        self, *, query: str, candidates: list[SourceChunk], model_id: str
    ) -> tuple[dict[str, float], Usage]:
        """Score each candidate 0–100 by joint query/chunk inspection.

        The identifier term is the point. Dense retrieval collapsed ``foo_4021`` and
        ``foo_4022`` into one concept; the reranker sees both strings side by side and
        can tell them apart. That is a real capability difference between a bi-encoder
        and a cross-encoder, and it is what the ``no_rerank`` ablation removes.
        """
        q_ids = {t for t in raw_tokens(query) if is_identifier(t)}
        scores: dict[str, float] = {}
        tokens_in = estimate_tokens(query)
        for c in candidates:
            cov = coverage(query, c.text, self.stats)
            if q_ids:
                c_ids = {t for t in raw_tokens(c.text) if is_identifier(t)}
                exact = len(q_ids & c_ids) / len(q_ids)
            else:
                exact = cov
            score = 100.0 * (
                RERANK_COVERAGE_WEIGHT * cov + (1 - RERANK_COVERAGE_WEIGHT) * exact
            )
            scores[c.chunk_id] = round(score, 2)
            tokens_in += estimate_tokens(c.text)
        return scores, Usage(
            tokens_in=tokens_in, tokens_out=len(candidates) * 4, cost_usd=0.0, calls=1
        )

    # -- generate -----------------------------------------------------------------

    def generate(
        self,
        *,
        query: str,
        sources: list[SourceChunk],
        model_id: str,
        temperature: float,
        max_tokens: int,
        discriminator_guard: bool = True,
    ) -> GeneratedAnswer:
        usage = Usage(
            tokens_in=estimate_tokens(query) + sum(estimate_tokens(s.text) for s in sources),
            calls=1,
        )

        if not sources:
            return self._refuse(query, "no sources were retrieved", usage)

        joined = "\n".join(s.text for s in sources)
        context_cov = coverage(query, joined, self.stats)

        if context_cov < REFUSE_BELOW_CONTEXT_COVERAGE:
            return self._refuse(query, "the retrieved sources are unrelated", usage)

        ranked = self._best_sentences(query, sources)
        if not ranked:
            return self._refuse(query, "the retrieved sources are unrelated", usage)

        best_cov, best_src, best_sent = ranked[0]

        # Discriminator guard. Without it the generator happily answers a question
        # about KLV-4213 using the chunk that documents KLV-4214, at full confidence,
        # because topical coverage is near-perfect — the two chunks are near-identical
        # apart from the four digits that are the entire question.
        if discriminator_guard:
            unmet = sorted(discriminators(query) - discriminators(joined))
            if unmet:
                return self._hedge(unmet, best_src, best_sent, usage)

        if best_cov >= CONFIDENT_ABOVE or len(ranked) == 1:
            return self._confident([(best_src, best_sent)], usage)

        # Synthesis: two sentences from *different* documents, asserted together with
        # no acknowledgement that they came from unrelated settings.
        second = next(
            (r for r in ranked[1:] if r[1].chunk_id != best_src.chunk_id),
            None,
        )
        pair = [(best_src, best_sent)]
        if second is not None:
            pair.append((second[1], second[2]))
        return self._confident(pair, usage)

    # -- answer construction ------------------------------------------------------

    def _best_sentences(
        self, query: str, sources: list[SourceChunk]
    ) -> list[tuple[float, SourceChunk, str]]:
        scored: list[tuple[float, SourceChunk, str]] = []
        for src in sources:
            # Code fences are stripped before sentence splitting: quoting a YAML line
            # back at a user is unreadable and still reads as an assertion.
            for _s, _e, raw in sentences(strip_code_fences(src.text)):
                sent = clean_sentence(raw)
                if len(sent) < 25 or not is_prose(sent):
                    continue
                scored.append((coverage(query, sent, self.stats), src, sent))
        # Sort by coverage, then by source order, then by the sentence text, so ties
        # break identically on every run.
        scored.sort(key=lambda r: (-r[0], r[1].n, r[2]))
        return scored

    @staticmethod
    def _assemble(parts: list[tuple[str, list[str]]]) -> GeneratedAnswer:
        text = ""
        spans: list[tuple[int, int, list[str]]] = []
        citations: list[str] = []
        for piece, chunk_ids in parts:
            if text:
                text += " "
            start = len(text)
            text += piece
            spans.append((start, len(text), list(chunk_ids)))
            for cid in chunk_ids:
                if cid not in citations:
                    citations.append(cid)
        return GeneratedAnswer(text=text, refused=False, hedged=False, spans=spans,
                               citations=citations)

    def _confident(
        self, pairs: list[tuple[SourceChunk, str]], usage: Usage
    ) -> GeneratedAnswer:
        parts = [(f"{sent} [{src.n}]", [src.chunk_id]) for src, sent in pairs]
        out = self._assemble(parts)
        out.usage = usage
        out.usage.tokens_out = estimate_tokens(out.text)
        return out

    def _hedge(
        self, unmet: list[str], src: SourceChunk, sent: str, usage: Usage
    ) -> GeneratedAnswer:
        named = ", ".join(f"`{u.removeprefix('pol:')}`" for u in unmet[:3])
        lead = f"The sources do not document {named}."
        parts = [
            (lead, []),
            (f"The closest documented behaviour is: {sent} [{src.n}]", [src.chunk_id]),
        ]
        out = self._assemble(parts)
        out.hedged = True
        out.usage = usage
        out.usage.tokens_out = estimate_tokens(out.text)
        return out

    def _refuse(self, query: str, why: str, usage: Usage) -> GeneratedAnswer:
        missing = [c.removeprefix("id:") for c in missing_concepts(query, "")][:4]
        subject = ", ".join(f"`{m}`" for m in missing) if missing else "this topic"
        text = (
            f"The provided sources do not document {subject}; I could not find "
            f"supporting evidence because {why}. Answering would require information "
            "outside the retrieved documents."
        )
        usage.tokens_out = estimate_tokens(text)
        return GeneratedAnswer(
            text=text, refused=True, hedged=True, spans=[], citations=[], usage=usage
        )


__all__ = [
    "CONFIDENT_ABOVE",
    "GUARD_FIRES_ON_ANY_UNMET_DISCRIMINATOR",
    "OFFLINE_EMBED_MODEL",
    "OfflineEmbedder",
    "OfflineLLM",
    "REFUSE_BELOW_CONTEXT_COVERAGE",
    "SYNTHESISE_BELOW",
]
