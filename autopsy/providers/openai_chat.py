"""OpenAI provider — ``gpt-4o-mini`` for every chat role.

One model doing generation, reranking, rewriting, *and* judging. That last one is a
known methodological compromise and it is recorded here rather than buried:

**The judge shares a family with the generator.** Models systematically prefer their
own outputs, so a ``gpt-4o-mini`` judge grading ``gpt-4o-mini`` answers will score them
higher than an independent judge would. Every judge-derived number under this provider
is therefore an upper bound, and ``reports/judge-calibration.md`` says so on its face.

What survives the compromise: the ablation table's headline column
(``now_confident_wrong``) is decided by unique-token substring checks against known
ground truth, not by the judge. The most important number in the project does not depend
on the judge at all — which is exactly why it was built that way.

To restore independence, set ``AUTOPSY_JUDGE_MODEL`` to something outside the family —
a Groq-hosted Llama or Qwen works and is free.
"""

from __future__ import annotations

import os

from autopsy.providers.base import ProviderError
from autopsy.providers.oaicompat import OpenAICompatibleLLM

OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

#: USD per 1M tokens, (input, output). Checked 2026-07-26.
OPENAI_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


class OpenAILLM(OpenAICompatibleLLM):
    service = "openai"
    #: gpt-4o-mini honours JSON mode, which makes the reranker's output reliably
    #: parseable instead of dependent on the model not adding a markdown fence.
    supports_json_mode = True

    def __init__(self, judge_model: str | None = None) -> None:
        self.judge_model = judge_model or os.environ.get(
            "AUTOPSY_JUDGE_MODEL", OPENAI_CHAT_MODEL
        )

    def _make_client(self):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ProviderError(
                "OPENAI_API_KEY is not set. Put it in .env and export it, or run with "
                "AUTOPSY_PROVIDER=offline for the keyless simulator."
            )
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("pip install 'retrieval-autopsy[llm]'") from exc
        # max_retries=0: retry policy lives in OpenAICompatibleLLM so that Groq and
        # OpenAI back off identically and the trace records one attempt per call.
        return openai.OpenAI(api_key=key, max_retries=0, timeout=120.0)

    @property
    def judge_is_independent(self) -> bool:
        """False when the judge shares a family with the generator.

        Read by the calibration report, which refuses to present an agreement rate
        without saying whether the judge was independent of what it graded.
        """
        return not self.judge_model.startswith(("gpt-", "o1", "o3", "o4"))


__all__ = ["OPENAI_CHAT_MODEL", "OPENAI_EMBED_MODEL", "OPENAI_PRICES", "OpenAILLM"]
