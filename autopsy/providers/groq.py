"""Groq provider — open-weight chat models on the free tier.

Everything about the wire format is shared with OpenAI and lives in ``oaicompat``. What
is specific to Groq:

**No embeddings endpoint.** ``POST /embeddings`` returns 404 for every model, so the
dense leg comes from elsewhere — see ``embedders.py``. ``RuntimeConfig`` keeps
``provider`` and ``embedder`` as separate fields precisely so a trace can say which
model did which job instead of implying Groq did both.

**Limits bind on tokens per minute, not requests.** 1000 requests/day but only ~12k
tokens/minute on the 70b, against ~1.7k tokens of retrieved context per generation
call. That is roughly seven calls a minute — fine for a person asking questions,
hopeless for a 3000-run sweep.

**Three model families are available**, which is what lets the judge stay independent
of the generator after moving off Anthropic/OpenAI. Llama generates, Qwen judges.
"""

from __future__ import annotations

import os

from autopsy.providers.base import ProviderError
from autopsy.providers.oaicompat import OpenAICompatibleLLM

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

#: Verified against ``GET /models`` rather than recalled.
#:
#: Generation is ``gpt-oss-120b``, not the larger ``llama-3.3-70b-versatile``, for a
#: purely operational reason: **each model carries its own daily token budget**, and the
#: 70b's is 100k/day — about 43 questions at ~2.3k tokens each. Exhaust it and every
#: query fails for the rest of the day while a five-token health check still succeeds,
#: because the per-minute budget is a separate counter that looks fine. Spreading the
#: roles across models means the reranker and judge cannot starve generation either.
GROQ_GENERATION_MODEL = "openai/gpt-oss-120b"
GROQ_RERANK_MODEL = "llama-3.1-8b-instant"
#: A different family from the generator, deliberately. Models favour their own
#: family's output, so a Llama judge grading Llama answers inflates every number.
GROQ_JUDGE_MODEL = "qwen/qwen3.6-27b"

#: Free tier. Kept explicit so the cost column stays structurally correct — moving to a
#: paid tier changes only this table.
GROQ_PRICES: dict[str, tuple[float, float]] = {
    GROQ_GENERATION_MODEL: (0.0, 0.0),
    GROQ_RERANK_MODEL: (0.0, 0.0),
    GROQ_JUDGE_MODEL: (0.0, 0.0),
    "openai/gpt-oss-120b": (0.0, 0.0),
    "openai/gpt-oss-20b": (0.0, 0.0),
}


class GroqLLM(OpenAICompatibleLLM):
    service = "groq"
    #: Groq accepts `response_format` on most models but not uniformly across the
    #: catalogue, and a 400 here would take down the reranker. The lenient score parser
    #: handles plain text fine, so this stays off.
    supports_json_mode = False
    judge_model = GROQ_JUDGE_MODEL

    def _make_client(self):
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise ProviderError(
                "GROQ_API_KEY is not set. Put it in .env and export it, or run with "
                "AUTOPSY_PROVIDER=offline for the keyless simulator."
            )
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("pip install 'retrieval-autopsy[llm]'") from exc
        # Groq speaks the OpenAI wire format, so the official SDK works with a base_url
        # swap. max_retries=0 because the retry policy lives in OpenAICompatibleLLM,
        # shared with the OpenAI provider so both back off identically.
        return openai.OpenAI(
            api_key=key, base_url=GROQ_BASE_URL, max_retries=0, timeout=120.0
        )


__all__ = [
    "GROQ_BASE_URL",
    "GROQ_GENERATION_MODEL",
    "GROQ_JUDGE_MODEL",
    "GROQ_PRICES",
    "GROQ_RERANK_MODEL",
    "GroqLLM",
]
