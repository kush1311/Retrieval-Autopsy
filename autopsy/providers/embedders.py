"""Embedding backends, chosen independently of the chat provider.

Separate from ``provider`` because they are separate decisions. Groq has no embeddings
endpoint, so a Groq run needs a dense leg from somewhere else; folding both into one
"provider" field would mean a trace could not say which model produced which half.

``fastembed`` is the interesting one: BAAI/bge-small-en-v1.5 through ONNX Runtime,
locally, no API key, no GPU, no PyTorch. ~130MB downloaded once. It makes a genuinely
real hybrid pipeline runnable for free — which matters, because the offline concept
embedder is a simulator and its numbers describe the simulator.

**One thing that does not carry across embedders: the gate threshold.** BGE-small puts
*unrelated* text at roughly 0.55 cosine and relevant matches at 0.72–0.85, so a
threshold tuned for a different model — or for the concept simulator, whose scale is
query-coverage in [0,1] with a real floor at 0 — will either pass everything or refuse
everything. Derive it with ``python -m autopsy.cli calibrate-gate``; do not inherit it.
"""

from __future__ import annotations

from functools import lru_cache

from autopsy.providers.base import Embedding, ProviderError, Usage, estimate_tokens

FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=2)
def _fastembed(model_name: str):
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover
        raise ProviderError(
            "fastembed is not installed. `pip install fastembed`, or set "
            "AUTOPSY_EMBEDDER=concept to use the offline simulator."
        ) from exc
    return TextEmbedding(model_name=model_name)


class FastEmbedEmbedder:
    """Local ONNX embeddings. Free, offline after the first download, and real."""

    def __init__(self, model_id: str = FASTEMBED_MODEL, batch: int = 64) -> None:
        self.model_id = model_id
        self.batch = batch

    def _embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        model = _fastembed(self.model_id)
        # fastembed yields in input order; materialise per batch so a large ingest
        # does not hold every vector from every batch alive at once.
        out: list[tuple[float, ...]] = []
        for i in range(0, len(texts), self.batch):
            out.extend(
                tuple(float(x) for x in vec)
                for vec in model.embed(texts[i : i + self.batch])
            )
        return out

    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        return [Embedding(model_id=self.model_id, dense=v) for v in self._embed(texts)]

    def embed_query(self, text: str) -> tuple[Embedding, Usage]:
        vector = self._embed([text])[0]
        return (
            Embedding(model_id=self.model_id, dense=vector),
            # Local inference: real tokens, no cost. Recording an estimate rather than
            # zero keeps the token column meaningful across providers.
            Usage(tokens_in=estimate_tokens(text), cost_usd=0.0, calls=1),
        )


def build_embedder(kind: str, model_id: str | None = None):
    if kind == "concept":
        from autopsy.providers.offline import OfflineEmbedder

        return OfflineEmbedder()
    if kind == "fastembed":
        return FastEmbedEmbedder(model_id or FASTEMBED_MODEL)
    if kind == "openai":
        from autopsy.providers.live import OpenAIEmbedder

        return OpenAIEmbedder(model_id or "text-embedding-3-small")
    raise ValueError(f"unknown embedder {kind!r}; expected concept | fastembed | openai")


__all__ = ["FASTEMBED_MODEL", "FastEmbedEmbedder", "build_embedder"]
