"""Chunking, ingest idempotence, and the ground truth the study rests on."""

from __future__ import annotations

import pytest

from autopsy.chunking import chunk_markdown, split_sections
from autopsy.config import default_config
from autopsy.ingest import Document, build_index, chunk_documents, corpus_version
from autopsy.store.chunks import Chunk
from autopsy.textutil import (
    ConceptStats, clean_sentence, concepts, coverage, discriminators, is_identifier,
    is_prose, sentences, to_concept,
)

CFG = default_config()

FENCED = """# Title

## Config

Set the value like this:

```yaml
# this heading-looking line is a comment, not a heading
threshold: 4096
retries: 3
```

That block must survive chunking intact because the exact identifiers inside it are
what the lexical retrieval leg matches on.

## Other

Some other prose that is long enough to stand on its own as a separate chunk of the
document rather than being folded into its neighbour.
"""


def test_headings_inside_code_fences_are_not_headings():
    sections = split_sections(FENCED)
    titles = [s.heading_path[-1] for s in sections]
    assert "this heading-looking line is a comment, not a heading" not in titles
    assert titles == ["Config", "Other"]


def test_code_blocks_are_never_split():
    for piece in chunk_markdown(FENCED, target_tokens=20):
        assert piece.text.count("```") % 2 == 0, piece.text


def test_heading_path_is_recorded():
    pieces = chunk_markdown(FENCED)
    assert pieces[0].heading_path == ["Title", "Config"]


def test_tiny_sections_merge_forward_instead_of_becoming_distractors():
    markdown = "# T\n\n## Bare\n\nShort.\n\n## Real\n\n" + ("word " * 80) + "\n"
    pieces = chunk_markdown(markdown)
    assert len(pieces) == 1
    assert "Short." in pieces[0].text


def test_sentences_span_soft_line_wraps():
    """Markdown prose is hard-wrapped. Splitting on line boundaries truncates almost
    every sentence, and a half sentence is still a plausible-looking string."""
    text = "The default value is\n4096 bytes per cycle. A second sentence follows here."
    found = [s for _a, _b, s in sentences(text)]
    assert found[0] == "The default value is 4096 bytes per cycle."
    assert len(found) == 2


def test_sentence_offsets_index_into_the_original_text():
    text = "First sentence here. Second sentence here.\n\nThird one in a new paragraph."
    for start, end, piece in sentences(text):
        assert text[start:end].strip() == piece


def test_is_prose_rejects_config_fragments():
    assert not is_prose("distance: cosine        # cosine | dot | euclid")
    assert not is_prose("| a | b |")
    assert is_prose("The default value of the threshold is four thousand bytes.")


def test_clean_sentence_strips_markdown_markers():
    assert clean_sentence("- `binary` is one bit.") == "`binary` is one bit."
    assert clean_sentence("1. Then do this.") == "Then do this."


# --------------------------------------------------------------------------------------
# Concept mapping — the mechanism behind no_lexical
# --------------------------------------------------------------------------------------


def test_digit_runs_in_identifiers_are_smeared():
    """This is *the* mechanism the whole hybrid demo rests on: the dense leg cannot
    tell two neighbouring error codes apart, and the lexical leg can."""
    assert to_concept("klv-4021") == to_concept("klv-4022")
    assert to_concept("klv-4021").startswith("id:")


def test_the_lexical_leg_keeps_them_distinct():
    from autopsy.store.lexical import _tokens

    assert _tokens("KLV-4021") != _tokens("KLV-4022")


def test_synonyms_collapse_so_paraphrases_match():
    assert to_concept("cap") == to_concept("threshold")
    assert to_concept("config") == to_concept("configuration")


def test_polarity_words_survive_stopword_removal():
    assert "not" in concepts("which settings are not replicated")


def test_discriminators_pick_up_identifiers_versions_and_polarity():
    found = discriminators("is klv-4021 not supported in version 7")
    assert "klv-4021" in found
    assert "pol:not" in found
    assert "7" in found


def test_coverage_is_asymmetric_and_bounded():
    stats = ConceptStats.from_texts(["alpha beta gamma", "delta epsilon"])
    assert coverage("alpha beta", "alpha beta gamma delta", stats) == pytest.approx(1.0)
    assert 0.0 <= coverage("alpha zeta", "alpha beta", stats) <= 1.0
    assert coverage("", "anything", stats) == 0.0


# --------------------------------------------------------------------------------------
# Refusal detection — the predicate the headline metric is computed from
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # The one that caused a false positive against a real model: substring
        # matching on "does not state" missed it over one adverb and a plural verb.
        "The sources do not explicitly state which metric is the fastest.",
        "The documentation does not specify a default for this key.",
        "This setting is not documented anywhere in the provided sources.",
        "There is no such configuration option in Kelvin.",
        "I could not find any evidence for that in the retrieved documents.",
        "I cannot determine the answer from these sources.",
        "The sources do not mention a minimum dimension.",
        "That value is not specified in the documentation.",
        "The provided sources don't contain this information.",
        "I do not have enough information to answer that.",
        "Answering would require information outside the retrieved documents.",
    ],
)
def test_real_refusal_phrasings_are_detected(text):
    from autopsy.textutil import has_refusal_marker

    assert has_refusal_marker(text), text


@pytest.mark.parametrize(
    "text",
    [
        "KLV-4021 means the pellshale budget was exhausted while accepting batches.",
        "The default value of the threshold is 4096 bytes per cycle.",
        "Set append_fsync to always to flush before acknowledging each write.",
        # Documentation prose that merely contains hedge-ish words. Counting these as
        # hedges would mark confident answers as flagged and understate the
        # wrong-and-confident rate — an error in the flattering direction.
        "Replicas may themselves have replicas; chained replication reduces load.",
        "A snapshot is written by a forked child process, so peak memory may double.",
    ],
)
def test_confident_answers_are_not_mistaken_for_refusals(text):
    from autopsy.textutil import has_refusal_marker

    assert not has_refusal_marker(text), text


def test_hedging_is_narrower_than_refusal():
    """Hedging and refusal are different outcomes and the taxonomy has to hold.

    "The sources are not clear on whether this is enforced" *answered* while signalling
    uncertainty — it did not decline. Both count as flagged for the wrong-and-confident
    metric, but collapsing them would lose the distinction between a system that
    refused and one that answered nervously.
    """
    from autopsy.textutil import has_refusal_marker, is_hedged

    hedges = [
        "It appears to be governed by the compaction interval.",
        "The sources are not clear on whether this is enforced.",
        "This is possibly related to the eviction policy.",
    ]
    for text in hedges:
        assert is_hedged(text), text
        assert not has_refusal_marker(text), f"{text} is a hedge, not a refusal"

    assert not is_hedged("The default is 4096 bytes.")
    # A refusal is also a hedge: it is the strongest form of flagging uncertainty.
    assert is_hedged("The sources do not document this setting.")


def test_trap_supplied_markers_still_match_literally():
    from autopsy.textutil import has_refusal_marker

    assert has_refusal_marker("no benchmark exists", extra=["no benchmark"])
    assert not has_refusal_marker("the default is 900", extra=["no benchmark"])


def test_is_identifier():
    assert is_identifier("klv_compaction_interval_seconds")
    assert is_identifier("KLV-4021")
    assert is_identifier("1.2.3")
    assert not is_identifier("compaction")


# --------------------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------------------


def _docs() -> list[Document]:
    return [
        Document(tenant_id="t1", doc_id="a.md",
                 markdown="# A\n\n## S\n\n" + ("alpha beta gamma delta " * 20)),
        Document(tenant_id="t2", doc_id="a.md",
                 markdown="# B\n\n## S\n\n" + ("epsilon zeta eta theta " * 20)),
    ]


def test_chunk_ids_are_content_addressed():
    first = chunk_documents(_docs())
    second = chunk_documents(_docs())
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_the_same_text_under_a_different_tenant_gets_a_different_id():
    a = Chunk.make_id("t1", "a.md", 0, "same text")
    b = Chunk.make_id("t2", "a.md", 0, "same text")
    assert a != b


def test_ingest_reuses_embeddings_on_an_unchanged_corpus():
    index, first = build_index(_docs(), cfg=CFG, label="test")
    _index2, second = build_index(_docs(), cfg=CFG, label="test", reuse_from=index)
    assert first["embedded"] == first["chunks"]
    assert second["embedded"] == 0


def test_corpus_version_changes_only_when_content_changes():
    docs = _docs()
    assert corpus_version(docs, "x") == corpus_version(list(reversed(docs)), "x")
    changed = [Document(d.tenant_id, d.doc_id, d.markdown + " more") for d in docs]
    assert corpus_version(docs, "x") != corpus_version(changed, "x")


def test_duplicate_documents_across_roots_are_rejected():
    from autopsy.ingest import ingest

    with pytest.raises(ValueError, match="more than one corpus root"):
        from pathlib import Path

        from autopsy.determinism import REPO_ROOT

        seed = REPO_ROOT / "corpus" / "seed"
        ingest([seed, seed], out=Path(REPO_ROOT / "corpus" / "index-test-dup"))


# --------------------------------------------------------------------------------------
# Generated ground truth
# --------------------------------------------------------------------------------------


def test_generated_answer_keys_are_unambiguous():
    """If two different facts shared an answer token, a wrong answer would pass the
    substring check and every accuracy number in the report would be inflated."""
    from corpus.synthetic import generate

    _docs_, facts = generate(seed=7, modules_per_tenant=6)
    seen: dict[str, tuple] = {}
    for fact in facts:
        if fact.answer_key is None or fact.kind == "paraphrase":
            continue
        source = (fact.doc_id, fact.heading, fact.tenant_id)
        assert seen.setdefault(fact.answer_key, source) == source


def test_generation_is_deterministic():
    from corpus.synthetic import generate

    a_docs, a_facts = generate(seed=11, modules_per_tenant=4)
    b_docs, b_facts = generate(seed=11, modules_per_tenant=4)
    assert [d.markdown for d in a_docs] == [d.markdown for d in b_docs]
    assert [f.answer_key for f in a_facts] == [f.answer_key for f in b_facts]


def test_paraphrase_queries_use_a_real_synonym_of_the_document_term():
    """These queries only test the semantic leg if the pair genuinely collapses to one
    concept. If a synonym were dropped from the lexicon they would silently become
    unanswerable and `no_semantic` would look harmless."""
    from corpus.synthetic import SYNONYM_PAIRS

    for doc_term, query_term in SYNONYM_PAIRS:
        assert to_concept(doc_term) == to_concept(query_term), (doc_term, query_term)


def test_every_generated_kind_is_represented():
    from corpus.synthetic import generate

    _d, facts = generate(seed=3, modules_per_tenant=3)
    kinds = {f.kind for f in facts}
    assert kinds == {"identifier", "value", "paraphrase", "absent", "out_of_scope", "followup"}
