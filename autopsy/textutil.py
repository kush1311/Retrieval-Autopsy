"""Deterministic text utilities shared by the offline provider, the grounding check,
the judge, and the counterfactual explanation generator.

One module rather than four copies, because "is this claim supported by this chunk"
has to mean the same thing in the eval as it does in the UI. Two subtly different
implementations of that predicate would make the grounding numbers unfalsifiable.

Nothing here is stochastic and nothing here calls a model.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

# --------------------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------------------

#: Identifier-shaped tokens are kept whole: ``error_code_4021``, ``maxmemory-policy``,
#: ``pg_stat_activity``, ``appendfsync``, ``0.9.2``. Splitting these on punctuation
#: destroys the exact-identifier property the whole hybrid demo depends on.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*[A-Za-z0-9_]|[A-Za-z]|\d+(?:\.\d+)+|\d+")

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does doing don for from had has
    have having he her here hers him his how i if in into is it its just me my no nor not
    of on or our ours out over own same she should so some such than that the their them
    then there these they this those to too under until up very was we were what when
    where which while who whom why will with would you your yours am been being
    """.split()
)

#: Words that must never be dropped as stopwords: they flip the meaning of a query and
#: they are the difference between a correct answer and a confidently inverted one.
POLARITY = frozenset({"not", "no", "never", "without", "excluding", "except", "unless", "non"})

#: Light, predictable suffix stripping. Not Porter — Porter's edge cases are noise here,
#: and a stemmer that is wrong in a *stable* way is fine for a similarity signal.
_SUFFIXES = (
    ("ization", "ize"),
    ("izations", "ize"),
    ("iness", "y"),
    ("ingly", ""),
    ("edly", ""),
    ("tions", "t"),
    ("tion", "t"),
    ("sions", "s"),
    ("sion", "s"),
    ("ments", "ment"),
    ("ness", ""),
    ("ing", ""),
    ("ers", "er"),
    ("ies", "y"),
    ("ied", "y"),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
)

#: Domain synonyms, so a paraphrase reaches the same concept. Deliberately small and
#: auditable — a large opaque lexicon would make the offline scores unexplainable.
SYNONYMS: dict[str, str] = {
    "delete": "remove",
    "erase": "remove",
    "drop": "remove",
    "purge": "remove",
    "evict": "remove",
    "turn-off": "disable",
    "switch-off": "disable",
    "deactivate": "disable",
    "turn-on": "enable",
    "activate": "enable",
    "config": "configuration",
    "conf": "configuration",
    "param": "parameter",
    "parameters": "parameter",
    "arg": "parameter",
    "opt": "option",
    "setting": "option",
    "flag": "option",
    "doc": "documentation",
    "docs": "documentation",
    "db": "database",
    "mem": "memory",
    "ram": "memory",
    "err": "error",
    "fail": "failure",
    "failing": "failure",
    "broken": "failure",
    "speed": "performance",
    "fast": "performance",
    "faster": "performance",
    "slow": "performance",
    "latency": "performance",
    "persist": "persistence",
    "durable": "persistence",
    "durability": "persistence",
    "replica": "replication",
    "secondary": "replica",
    "standby": "replica",
    "index": "indexing",
    "indexes": "indexing",
    "indices": "indexing",
    "vector": "vector",
    "embedding": "vector",
    "shard": "sharding",
    "partition": "sharding",
    "backup": "snapshot",
    "dump": "snapshot",
    "restore": "recovery",
    "recover": "recovery",
    "limit": "limit",
    "cap": "limit",
    "threshold": "limit",
    "max": "limit",
    "maximum": "limit",
    "timeout": "timeout",
    "expire": "expiry",
    "ttl": "expiry",
    "auth": "authentication",
    "login": "authentication",
    "credential": "authentication",
    "tune": "tuning",
    "throttle": "throttling",
    "throttled": "throttling",
    "rate-limit": "throttling",
}

_IDENTIFIER_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*[\d_.\-])[A-Za-z0-9_.\-]+$|^\d+(?:\.\d+)+$")
_DIGIT_RUN_RE = re.compile(r"\d+")


def strip_code_fences(text: str) -> str:
    return _CODE_FENCE_RE.sub(" ", text)


def raw_tokens(text: str) -> list[str]:
    """Lowercased surface tokens, identifiers preserved. This is what BM25 indexes."""
    return [t.lower().strip(".-_") for t in _TOKEN_RE.findall(text) if t.strip(".-_")]


def is_identifier(token: str) -> bool:
    """Does this token look like a machine identifier rather than an English word?"""
    return bool(_IDENTIFIER_RE.match(token))


def stem(token: str) -> str:
    if len(token) <= 3 or is_identifier(token):
        return token
    for suf, rep in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)] + rep
    return token


@lru_cache(maxsize=100_000)
def to_concept(token: str) -> str:
    """Map a surface token to the concept the *dense* leg sees.

    Two deliberate behaviours, both modelling how real embedding models actually
    behave rather than how we might wish they did:

    1. Synonyms and inflections collapse — that is why the dense leg survives
       paraphrase and the lexical leg does not.
    2. **Digit runs inside identifiers are smeared.** ``error_code_4021`` and
       ``error_code_4022`` both become ``id:error_code_#``. Subword tokenisers
       genuinely do lose exact numeric identity this way, and it is the precise
       reason a dense-only pipeline confidently returns the neighbouring error
       code. This is the mechanism the ``no_lexical`` ablation exposes.
    """
    # The synonym lexicon wins over the identifier heuristic. `turn-off`, `rate-limit`,
    # and `read-only` are hyphenated English, not machine identifiers, and routing them
    # down the identifier path would put them in their own concept bucket — silently
    # breaking every paraphrase query built on them.
    if token not in SYNONYMS and is_identifier(token):
        return "id:" + _DIGIT_RUN_RE.sub("#", token)

    # Iterate synonym-then-stem to a fixed point rather than applying each once.
    # The lexicon contains chains (`standby` -> `replica` -> `replication`), and a
    # single pass lands on a different concept depending on where in the chain a token
    # enters: `replica` stemmed to `replicat` while `standby` resolved to
    # `replication`. Two words that are synonyms by construction ended up in different
    # buckets, which quietly breaks exactly the paraphrase matching the lexicon exists
    # to provide.
    t = token
    for _ in range(4):
        nxt = SYNONYMS.get(t, t)
        nxt = stem(nxt)
        nxt = SYNONYMS.get(nxt, nxt)
        if nxt == t:
            break
        t = nxt
    return t


def content_tokens(text: str) -> list[str]:
    """Surface tokens with stopwords removed, polarity words kept."""
    return [t for t in raw_tokens(text) if t not in STOPWORDS or t in POLARITY]


def concepts(text: str) -> list[str]:
    """The dense leg's view of a piece of text."""
    return [to_concept(t) for t in content_tokens(text)]


def concept_set(text: str) -> frozenset[str]:
    return frozenset(concepts(text))


# --------------------------------------------------------------------------------------
# IDF over the corpus
# --------------------------------------------------------------------------------------


class ConceptStats:
    """Document frequencies over the concept vocabulary, built once at ingest.

    Kept explicit rather than global so a test can build a tiny corpus without
    polluting the real one.
    """

    __slots__ = ("df", "n_docs", "_idf_cache")

    def __init__(self, df: dict[str, int], n_docs: int) -> None:
        self.df = df
        self.n_docs = max(1, n_docs)
        self._idf_cache: dict[str, float] = {}

    @classmethod
    def from_texts(cls, texts: list[str]) -> "ConceptStats":
        df: dict[str, int] = {}
        for t in texts:
            for c in concept_set(t):
                df[c] = df.get(c, 0) + 1
        return cls(df, len(texts))

    def idf(self, concept: str) -> float:
        cached = self._idf_cache.get(concept)
        if cached is not None:
            return cached
        # Smoothed IDF, floored at 0.25 so a very common concept still carries a little
        # weight — zeroing it makes queries made entirely of common words score 0/0.
        v = max(0.25, math.log((self.n_docs + 1) / (self.df.get(concept, 0) + 1)) + 0.5)
        self._idf_cache[concept] = v
        return v

    def to_dict(self) -> dict:
        return {"df": self.df, "n_docs": self.n_docs}

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptStats":
        return cls(dict(d["df"]), int(d["n_docs"]))


# --------------------------------------------------------------------------------------
# Similarity
# --------------------------------------------------------------------------------------


def coverage(query: str, document: str, stats: ConceptStats | None = None) -> float:
    """How much of the *query's* meaning the document accounts for, in ``[0, 1]``.

    Asymmetric on purpose. Symmetric cosine between a six-token query and a
    600-token chunk is dominated by the chunk's length and lands around 0.2 for
    obviously relevant pairs, which makes every interpretable threshold impossible
    to set. Query-coverage is the quantity a gate actually wants to threshold on,
    and it lands in the same 0.3–0.9 band that a real asymmetric embedding model
    reports for query/passage pairs.
    """
    q = concepts(query)
    if not q:
        return 0.0
    d = concept_set(document)
    num = 0.0
    den = 0.0
    for c in q:
        w = stats.idf(c) if stats else 1.0
        den += w
        if c in d:
            num += w
    if den == 0:
        return 0.0
    return round(num / den, 6)


#: How strongly a long chunk is discounted for diluting its match. Retrieval coverage
#: alone is a coarse, step-valued quantity — a three-concept query can only produce a
#: handful of distinct values, so dozens of chunks tie and the ordering falls through
#: to the tiebreaker, which is arbitrary. A saturating length discount restores a
#: continuous ordering and encodes something true: a short chunk that is *about* the
#: query beats a long one that merely mentions it.
DILUTION_WEIGHT = 0.30
DILUTION_HALFPOINT = 80


def retrieval_similarity(
    query: str, document: str, stats: ConceptStats | None = None
) -> float:
    """The offline simulator's dense-retrieval score, in ``[0, 1]``.

    Query coverage, discounted for how much unrelated material the chunk carries.
    Distinct from :func:`coverage`, which answers "is this claim supported" and must
    stay unpenalised — a long chunk supports a claim exactly as well as a short one.
    Retrieval and grounding are different questions and get different functions.
    """
    base = coverage(query, document, stats)
    if base <= 0.0:
        return 0.0
    n = len(concept_set(document))
    dilution = n / (n + DILUTION_HALFPOINT)
    return round(base * (1.0 - DILUTION_WEIGHT * dilution), 6)


def missing_concepts(query: str, document: str) -> list[str]:
    """Query concepts the document does not account for — the raw material for a
    hedge, and for the ``partial_evidence`` trap."""
    d = concept_set(document)
    out: list[str] = []
    for c in concepts(query):
        if c not in d and c not in out:
            out.append(c)
    return out


DISCRIMINATOR_RE = re.compile(r"^(id:|\d)")


def discriminators(text: str) -> frozenset[str]:
    """Tokens whose exact value changes which answer is correct.

    Version numbers, identifiers, and polarity words. Two queries that differ only
    in a discriminator are *different questions*, however close their embeddings
    sit — "config value in Redis 6" and "config value in Redis 7" are about 0.98
    cosine apart and have different answers. Any semantic cache needs this guard;
    see the note in the spec's gotchas.
    """
    out = set()
    for t in raw_tokens(text):
        if t in POLARITY:
            out.add("pol:" + t)
        elif is_identifier(t) or t.isdigit():
            out.add(t)
    return frozenset(out)


# --------------------------------------------------------------------------------------
# Sentences
# --------------------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(`\[])")
_PARA_SPLIT_RE = re.compile(r"\n[ \t]*\n")


def sentences(text: str) -> list[tuple[int, int, str]]:
    """Split into sentences as ``(start, end, text)`` offsets into the original string.

    Offsets rather than substrings because ``Span`` in the trace carries character
    offsets, and recomputing them by searching for the substring breaks the moment a
    sentence repeats.

    Paragraphs, not lines. Markdown prose is hard-wrapped at 88 columns, so splitting
    on line boundaries truncates almost every sentence mid-clause — and the truncation
    is invisible downstream, because a half sentence is still a plausible-looking
    string. Newlines inside a paragraph are flattened to spaces with a
    length-preserving replace, which keeps every offset valid.
    """
    out: list[tuple[int, int, str]] = []
    cursor = 0
    for para in _PARA_SPLIT_RE.split(text):
        if not para:
            continue
        base = text.index(para, cursor)
        cursor = base + len(para)
        flat = para.replace("\n", " ")  # same length, so offsets survive
        pos = 0
        for piece in _SENT_SPLIT_RE.split(flat):
            if not piece.strip():
                continue
            idx = flat.index(piece, pos)
            out.append((base + idx, base + idx + len(piece), piece.strip()))
            pos = idx + len(piece)
    return out


_MARKDOWN_LEAD_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s+|>\s+|\|)")
_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z]{2,}")


def is_prose(sentence: str) -> bool:
    """Is this a sentence a person wrote, or a fragment of a config block?

    Extractive answering that quotes ``distance: cosine        # cosine | dot | euclid``
    at a user is worse than useless: it is unreadable *and* it reads as an assertion.
    """
    stripped = sentence.strip()
    if stripped.startswith(("```", "~~~", "|")):
        return False
    if len(_ALPHA_TOKEN_RE.findall(stripped)) < 5:
        return False
    # A line with a config-ish "key: value" and no sentence-ending punctuation is a
    # snippet that escaped the fence detector.
    return not (":" in stripped and not stripped.rstrip().endswith((".", "!", "?")))


def clean_sentence(sentence: str) -> str:
    """Strip markdown list and heading markers so a quoted claim reads as a claim."""
    return _MARKDOWN_LEAD_RE.sub("", sentence).strip()


#: Refusal detection, as patterns rather than exact substrings.
#:
#: Substring matching looked adequate against the offline simulator, whose refusals are
#: templated, and broke immediately against a real model. Llama produced *"The sources
#: do not explicitly state which metric is the fastest"* — a textbook correct refusal —
#: and the literal marker ``"does not state"`` missed it because of one adverb and a
#: plural verb. The trap was scored as wrong-and-confident, which inflated the single
#: headline number this project exists to report.
#:
#: The lesson generalises: any metric gated on a hand-written keyword list measures the
#: list as much as the system. These patterns absorb the modifiers a real model
#: actually inserts — adverbs, plurals, contractions.
REFUSAL_PATTERNS: tuple[str, ...] = (
    r"\b(?:do|does|did)(?:\s+not|n't)\s+(?:\w+\s+){0,2}"
    r"(?:state|say|specify|document|mention|rank|list|provide|indicate|contain|include|address|appear|exist)",
    r"\b(?:is|are|was|were)\s+(?:not|never)\s+(?:\w+\s+){0,2}"
    r"(?:documented|specified|stated|mentioned|listed|defined|described|available|covered|found|supported)",
    r"\bno\s+(?:\w+\s+){0,2}"
    r"(?:such|benchmark|evidence|documentation|mention|information|reference|indication|record|setting|option)",
    r"\b(?:cannot|can\s?not|could\s?not|couldn't|can't|unable\s+to|not\s+able\s+to)\s+"
    r"(?:\w+\s+){0,2}(?:find|determine|answer|verify|say|tell|confirm|establish)",
    r"\bnot\s+(?:explicitly\s+|directly\s+|clearly\s+)?"
    r"(?:documented|specified|stated|mentioned|available|covered|found|described|defined)",
    r"\b(?:i|we)\s+(?:do\s+not|don't)\s+have\s+(?:enough|sufficient|any)\b",
    r"\bnot\s+enough\s+(?:information|evidence|detail|context)\b",
    r"\boutside\s+the\s+(?:retrieved|provided|available)\s+(?:documents|sources|context)\b",
)

#: Hedging is weaker than refusal: the system answered, but signalled uncertainty.
#: Deliberately narrower than a bare word list — "Replicas may themselves have
#: replicas" is documentation prose, not a hedge, and counting it as one would mark
#: confident answers as flagged and *understate* the wrong-and-confident rate.
HEDGE_PATTERNS: tuple[str, ...] = (
    r"\b(?:may|might|could)\s+not\b",
    r"\bit\s+(?:may|might|could)\s+be\b",
    r"\b(?:appears|seems)\s+to\b",
    r"\b(?:is|are)\s+unclear\b",
    r"\bnot\s+(?:certain|clear|conclusive)\b",
    r"\b(?:possibly|presumably|ostensibly)\b",
    r"\b(?:suggests|implies)\s+that\b",
    r"\blikely\s+(?:that|to)\b",
    r"\bhowever,?\s+the\s+sources\b",
    r"\bbased\s+only\s+on\b",
)

_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)
_HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS + REFUSAL_PATTERNS), re.IGNORECASE)


def is_hedged(text: str) -> bool:
    return bool(_HEDGE_RE.search(text))


def has_refusal_marker(text: str, extra: list[str] | None = None) -> bool:
    """Did the system decline, or flag that it could not answer?

    ``extra`` holds a trap's own ``acceptable_refusal_markers``, matched as plain
    substrings — those are author-supplied and deliberately literal.
    """
    if _REFUSAL_RE.search(text):
        return True
    low = text.lower()
    return any(m.lower() in low for m in (extra or []))


__all__ = [
    "ConceptStats",
    "DILUTION_HALFPOINT",
    "DILUTION_WEIGHT",
    "DISCRIMINATOR_RE",
    "clean_sentence",
    "is_prose",
    "retrieval_similarity",
    "HEDGE_PATTERNS",
    "POLARITY",
    "REFUSAL_PATTERNS",
    "STOPWORDS",
    "SYNONYMS",
    "concept_set",
    "concepts",
    "content_tokens",
    "coverage",
    "discriminators",
    "has_refusal_marker",
    "is_hedged",
    "is_identifier",
    "missing_concepts",
    "raw_tokens",
    "sentences",
    "stem",
    "strip_code_fences",
    "to_concept",
]
