"""Caches, and the one bug that would make this whole project prove nothing.

**Every cache key includes the config hash.** Without it, an ablated run hits the
baseline's cached answer, every diff comes back ``IDENTICAL``, and the counterfactual
engine appears to work perfectly while measuring nothing at all. It is a one-line
omission that produces a confident, wrong, fully green result — the worst failure mode
a measurement tool has.

Two cache families with deliberately different key shapes:

* **Embeddings** are keyed on ``(model_id, text)`` and shared across tenants. An
  embedding is a function of the text and the model; nothing tenant-specific enters
  it, so sharing is safe and saves most of the cost of an ablation sweep.
* **Everything else** is keyed on ``(config_hash, tenant_id, stage, input_hash)``.
  Answer caches keyed on query text alone leak across tenants — that is precisely
  what the ``cache_namespace`` isolation probe exists to catch, and it catches it
  because this module makes the tenant part of the key structurally rather than by
  convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from autopsy.determinism import sha256_of
from autopsy.textutil import discriminators


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses


@dataclass
class StageCache:
    """Per-stage memo, namespaced by config and tenant."""

    enabled: bool = True
    _data: dict[str, Any] = field(default_factory=dict, repr=False)
    stats: CacheStats = field(default_factory=CacheStats)

    @staticmethod
    def key(*, config_hash: str, tenant_id: str, stage: str, payload: Any) -> str:
        return sha256_of([config_hash, tenant_id, stage, payload])

    def get_or_compute(
        self, key: str, compute: Callable[[], Any]
    ) -> tuple[Any, str | None]:
        """Returns ``(value, "hit" | "miss" | None)``. ``None`` means caching is off."""
        if not self.enabled:
            return compute(), None
        if key in self._data:
            self.stats.hits += 1
            return self._data[key], "hit"
        self.stats.misses += 1
        value = compute()
        self._data[key] = value
        return value, "miss"

    def clear(self) -> None:
        self._data.clear()


@dataclass
class EmbeddingCache:
    """Shared across tenants and across ablations, on purpose.

    The query embedding is identical for every variant except ``no_semantic``, so
    embedding once and reusing it across a sweep is most of the cost saving available
    in an ablation study. The key deliberately excludes the config hash: including it
    would be *safe* but would throw away that saving for no benefit, since no config
    field can change what an embedding of a given text under a given model is.
    """

    enabled: bool = True
    _data: dict[tuple[str, str], Any] = field(default_factory=dict, repr=False)
    stats: CacheStats = field(default_factory=CacheStats)

    def get_or_compute(self, model_id: str, text: str, compute: Callable[[], Any]):
        if not self.enabled:
            return compute(), None
        key = (model_id, text)
        if key in self._data:
            self.stats.hits += 1
            return self._data[key], "hit"
        self.stats.misses += 1
        value = compute()
        self._data[key] = value
        return value, "miss"


# --------------------------------------------------------------------------------------
# Semantic cache
# --------------------------------------------------------------------------------------


@dataclass
class SemanticAnswerCache:
    """A near-duplicate query cache with a discriminator guard.

    The motivating problem: *"config value in Kelvin 6"* and *"config value in Kelvin
    7"* sit at roughly 0.98 cosine similarity and have different answers. Any
    threshold high enough to separate them also rejects the paraphrases the cache
    exists to catch, so similarity alone cannot work at any setting.

    The fix is not a better threshold — it is a second, exact condition. A hit
    requires **both** high similarity **and** an exact match on the extracted
    discriminators: version numbers, identifiers, and polarity words like ``not`` or
    ``excluding``. Those are exactly the tokens that flip an answer while barely
    moving an embedding.

    Tenant is part of the key, not part of the similarity computation. A cache that
    matches on query text across tenants is the ``cache_namespace`` leak.
    """

    threshold: float = 0.93
    enabled: bool = True
    _entries: list[tuple[str, str, frozenset[str], frozenset[str], Any]] = field(
        default_factory=list, repr=False
    )
    stats: CacheStats = field(default_factory=CacheStats)
    rejected_by_discriminator: int = 0

    def lookup(
        self, *, config_hash: str, tenant_id: str, query: str, concepts: frozenset[str]
    ) -> tuple[Any, str | None]:
        if not self.enabled:
            return None, None
        ns = f"{config_hash}|{tenant_id}"
        q_disc = discriminators(query)
        for entry_ns, _q, entry_concepts, entry_disc, value in self._entries:
            if entry_ns != ns:
                continue
            sim = _jaccard(concepts, entry_concepts)
            if sim < self.threshold:
                continue
            if entry_disc != q_disc:
                # Near-identical wording, different question. This counter is worth
                # watching: if it is always zero, the guard is not doing anything and
                # the threshold is probably too high to ever match.
                self.rejected_by_discriminator += 1
                continue
            self.stats.hits += 1
            return value, "hit"
        self.stats.misses += 1
        return None, "miss"

    def store(
        self, *, config_hash: str, tenant_id: str, query: str, concepts: frozenset[str], value: Any
    ) -> None:
        if not self.enabled:
            return
        self._entries.append(
            (f"{config_hash}|{tenant_id}", query, concepts, discriminators(query), value)
        )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


__all__ = ["CacheStats", "EmbeddingCache", "SemanticAnswerCache", "StageCache"]
