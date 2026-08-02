"""Provider selection.

One factory, chosen by ``PipelineConfig.runtime.provider``, which is part of the
config hash — so an offline answer can never be served from cache to a live run.
"""

from __future__ import annotations

from dataclasses import dataclass

from autopsy.config import PipelineConfig
from autopsy.providers.base import (
    Completion,
    Embedder,
    Embedding,
    GeneratedAnswer,
    LLM,
    ProviderError,
    SourceChunk,
    Usage,
    accepts_temperature,
    price_of,
)
from autopsy.textutil import ConceptStats


@dataclass(slots=True)
class Providers:
    embedder: Embedder
    llm: LLM
    label: str


def build_providers(cfg: PipelineConfig, stats: ConceptStats) -> Providers:
    """Assemble the embedder and the chat model, which are chosen independently.

    They have to be independent: Groq serves no embeddings endpoint, so a Groq run
    pairs a remote chat model with a local ONNX embedder. Collapsing that into one
    "provider" field would make the trace unable to say which model did which job.
    """
    from autopsy.providers.embedders import build_embedder

    runtime = cfg.runtime
    embed_model = cfg.semantic.model_id if cfg.semantic else None

    if runtime.embedder == "concept":
        from autopsy.providers.offline import OfflineEmbedder

        embedder = OfflineEmbedder()
    else:
        embedder = build_embedder(runtime.embedder, embed_model)

    if runtime.provider == "offline":
        from autopsy.providers.offline import OfflineLLM

        llm = OfflineLLM(stats)
    elif runtime.provider == "groq":
        from autopsy.providers.groq import GroqLLM

        llm = GroqLLM()
    elif runtime.provider == "openai":
        from autopsy.providers.openai_chat import OpenAILLM

        llm = OpenAILLM()
    else:
        from autopsy.providers.live import AnthropicLLM

        llm = AnthropicLLM()

    return Providers(embedder=embedder, llm=llm, label=runtime.provider)


__all__ = [
    "Completion",
    "Embedder",
    "Embedding",
    "GeneratedAnswer",
    "LLM",
    "Providers",
    "ProviderError",
    "SourceChunk",
    "Usage",
    "accepts_temperature",
    "build_providers",
    "price_of",
]
