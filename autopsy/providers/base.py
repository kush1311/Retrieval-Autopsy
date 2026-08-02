"""Provider protocols, embeddings, and cost accounting.

Two backends implement these:

* ``offline`` — a deterministic simulator. No network, no keys, no cost. Everything
  in this repo runs end to end with it, which is what makes ``make eval`` a real CI
  gate rather than a thing you run once with a credit card attached.
* ``live`` — the real SDKs.

The offline backend is a *simulator*, not a stand-in for measurement. Numbers
produced under it characterise the simulator; the report says so, loudly, and
refuses to omit it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

# --------------------------------------------------------------------------------------
# Model facts
# --------------------------------------------------------------------------------------

#: USD per 1M tokens, (input, output). Indicative, checked 2026-07-26 against the
#: Anthropic pricing table; treat as a config value, not a constant of nature. Cost
#: figures in traces are only as good as this table.
PRICES: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    # OpenAI — embeddings and the judge
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "gpt-4o-mini": (0.15, 0.60),
    # the simulator
    "offline-sim": (0.0, 0.0),
}

#: Models that still accept ``temperature`` / ``top_p`` / ``top_k``.
#:
#: This is the constraint that decides the model choice for this project, and it is
#: not obvious. Sampling parameters were **removed** on Claude Opus 4.7+, Sonnet 5,
#: and Fable 5 — sending ``temperature=0.0`` to those returns a 400. Since design
#: principle #4 requires temperature 0 on every run whose output gets diffed, the
#: generation and rerank models have to come from this set. Upgrading the generator
#: to a newer model is not a one-line change; it means giving up the explicit
#: temperature pin and re-deriving what "deterministic" means for the study.
ACCEPTS_SAMPLING_PARAMS: frozenset[str] = frozenset(
    {
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-sonnet-4-5",
    }
)


def accepts_temperature(model_id: str) -> bool:
    return model_id in ACCEPTS_SAMPLING_PARAMS


def price_of(model_id: str, tokens_in: int, tokens_out: int) -> float:
    inp, out = PRICES.get(model_id, (0.0, 0.0))
    return round((tokens_in / 1_000_000) * inp + (tokens_out / 1_000_000) * out, 8)


def estimate_tokens(text: str) -> int:
    """Rough token count for the offline backend's cost accounting.

    ~4 characters per token. Deliberately crude and clearly labelled: the live
    backend reports real usage from the API, and mixing a real number with an
    estimate under the same field name without saying so would be worse than either.
    """
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Embedding:
    """A query or document vector.

    Two shapes, because the two backends are genuinely different objects:

    * ``dense`` — a real float vector from an embedding API, compared by cosine.
    * ``concepts`` — the offline simulator's concept bag, compared by query coverage.

    The vector store dispatches on which one is populated and raises if a backend is
    handed the wrong kind, rather than silently producing a similarity that means
    nothing.
    """

    model_id: str
    dense: tuple[float, ...] | None = None
    concepts: tuple[str, ...] | None = None

    @property
    def kind(self) -> Literal["dense", "concept"]:
        return "dense" if self.dense is not None else "concept"

    def __post_init__(self) -> None:
        if (self.dense is None) == (self.concepts is None):
            raise ValueError("an Embedding is exactly one of dense or concepts")


@dataclass(slots=True)
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.tokens_in += other.tokens_in
        self.tokens_out += other.tokens_out
        self.cost_usd = round(self.cost_usd + other.cost_usd, 8)
        self.calls += other.calls


@dataclass(slots=True)
class Completion:
    text: str
    usage: Usage = field(default_factory=Usage)
    model_id: str = ""


# --------------------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    model_id: str

    def embed_documents(self, texts: list[str]) -> list[Embedding]: ...

    def embed_query(self, text: str) -> tuple[Embedding, Usage]: ...


@runtime_checkable
class Chat(Protocol):
    """Free-form completion. Used by the judge; the pipeline uses the task methods."""

    def complete(
        self,
        *,
        system: str,
        user: str,
        model_id: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> Completion: ...


@dataclass(slots=True)
class SourceChunk:
    """One numbered source as the generator sees it."""

    n: int
    chunk_id: str
    heading_path: list[str]
    text: str


@dataclass(slots=True)
class GeneratedAnswer:
    text: str
    refused: bool
    hedged: bool
    #: ``(start, end, [chunk_id, ...])`` — character offsets into ``text``.
    spans: list[tuple[int, int, list[str]]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


@runtime_checkable
class LLM(Protocol):
    """The task-level surface the pipeline actually uses.

    Task methods rather than one ``complete(prompt)`` because the offline backend is
    a simulator, not a model — asking it to parse a natural-language prompt and infer
    which job it was given would be a pile of brittle string matching. Both backends
    implement the same three jobs; only the live one builds prompts.
    """

    def rewrite(self, *, query: str, history: list[str], model_id: str) -> Completion: ...

    def rerank(
        self, *, query: str, candidates: list[SourceChunk], model_id: str
    ) -> tuple[dict[str, float], Usage]:
        """Return ``{chunk_id: score 0-100}``."""

    def generate(
        self,
        *,
        query: str,
        sources: list[SourceChunk],
        model_id: str,
        temperature: float,
        max_tokens: int,
        discriminator_guard: bool = True,
    ) -> GeneratedAnswer: ...


class ProviderError(RuntimeError):
    pass


__all__ = [
    "ACCEPTS_SAMPLING_PARAMS",
    "PRICES",
    "Chat",
    "Completion",
    "Embedder",
    "Embedding",
    "ProviderError",
    "Usage",
    "accepts_temperature",
    "estimate_tokens",
    "price_of",
]
