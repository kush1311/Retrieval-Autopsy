"""Shared implementation for every OpenAI-wire-format chat provider.

OpenAI, Groq, and anything else speaking ``/v1/chat/completions`` differ only in base
URL, model names, and which optional features they honour. One implementation with
those as parameters; the alternative was two copies of the retry logic drifting apart.

Subclasses supply a client and a set of defaults. Everything that is easy to get subtly
wrong — retry classification, reasoning-field unwrapping, score parsing, span
attribution — lives here once.
"""

from __future__ import annotations

import re
import time

from autopsy.providers.base import (
    Completion,
    GeneratedAnswer,
    ProviderError,
    SourceChunk,
    Usage,
    price_of,
)
from autopsy.providers.live import (
    _DISCRIMINATOR_CLAUSE,
    _GENERATE_SYSTEM,
    _RERANK_SYSTEM,
    _REWRITE_SYSTEM,
    _attribute,
    _parse_scores,
    _render_sources,
)

MAX_RETRIES = 6

#: Reasoning models prefix their answer with a thinking block. Stripped before the text
#: reaches the reranker's parser or the judge's classifier, both of which read line one.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

#: Statuses worth another attempt: the rate limiter plus the load-shedding family.
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}

#: Exception type names that mean "the network failed", which carry no HTTP status at
#: all. A retry policy written only around status codes lets these through, and a long
#: unattended run then dies on one dropped TLS handshake with nothing to show for it.
_RETRY_TYPES = {
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "RateLimitError",
    "ConnectError",
    "ReadTimeout",
    "RemoteProtocolError",
}


#: A 429 that names a *daily* budget is not worth retrying. Per-minute limits clear in
#: seconds; a daily one clears in hours, and backing off six times just to fail anyway
#: turns a clear error into a two-minute hang followed by a misleading message.
_DAILY_MARKERS = ("per day", "(tpd)", "tokens per day", "requests per day", "(rpd)")


def is_daily_limit(exc: Exception) -> bool:
    body = str(getattr(exc, "body", "") or "") + " " + str(exc)
    return any(m in body.lower() for m in _DAILY_MARKERS)


def is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status == 429 and is_daily_limit(exc):
        return False
    if status in _RETRY_STATUS:
        return True
    if status is not None:
        return False
    return type(exc).__name__ in _RETRY_TYPES


def retry_after(exc: Exception, attempt: int) -> float:
    """Prefer the server's own advice; fall back to capped exponential backoff."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        raw = headers.get(key)
        if not raw:
            continue
        try:
            return min(60.0, float(str(raw).rstrip("s")) + 0.5)
        except ValueError:
            continue
    return min(60.0, 2.0 * (2**attempt))


def unwrap(message) -> str:
    """Extract the answer whatever shape the model returned it in.

    Reasoning models put their output in unexpected places: Qwen prefixes
    ``<think>…</think>``, GPT-OSS leaves ``content`` empty and fills a ``reasoning``
    field. Both otherwise parse as an empty response — which the reranker rejects
    outright and the judge scores as unstable.
    """
    text = (getattr(message, "content", None) or "").strip()
    if not text:
        for attr in ("reasoning_content", "reasoning"):
            alt = getattr(message, attr, None)
            if alt:
                text = str(alt).strip()
                break
    return _THINK_RE.sub("", text).strip()


class OpenAICompatibleLLM:
    """Base class. Subclasses set the class attributes and implement ``_make_client``."""

    #: Shown in error messages so a failure names the right service.
    service: str = "openai-compatible"
    #: Does this endpoint honour ``response_format={"type": "json_object"}``? Used only
    #: for the reranker, whose output is machine-parsed.
    supports_json_mode: bool = False
    judge_model: str = ""

    def _make_client(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def _client(self):
        cached = getattr(self, "_cached_client", None)
        if cached is None:
            cached = self._make_client()
            self._cached_client = cached
        return cached

    def _call(
        self, *, system: str, user: str, model_id: str, max_tokens: int,
        temperature: float | None = 0.0, json_mode: bool = False,
    ) -> tuple[str, Usage]:
        kwargs: dict = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode and self.supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        client = self._client()
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - narrowed by is_transient
                if is_daily_limit(exc):
                    # Surface the provider's own words. A generic "rate limited" here
                    # sent me looking at per-minute headers that showed plenty of
                    # headroom, while the real limit was a separate daily counter.
                    raise ProviderError(
                        f"{self.service}: daily quota exhausted for {model_id}.\n"
                        f"  {getattr(exc, 'body', None) or exc}\n"
                        f"  Each model has its own daily budget — switching "
                        f"AUTOPSY_PROVIDER's generation model to one you have not used "
                        f"today is usually enough. Or use AUTOPSY_PROVIDER=offline (free) "
                        f"or =openai (paid, ~$0.0003/query)."
                    ) from exc
                if not is_transient(exc):
                    raise
                last = exc
                time.sleep(retry_after(exc, attempt))
                continue

            usage = getattr(response, "usage", None)
            tin = getattr(usage, "prompt_tokens", 0) or 0
            tout = getattr(usage, "completion_tokens", 0) or 0
            return (
                unwrap(response.choices[0].message),
                Usage(
                    tokens_in=tin, tokens_out=tout,
                    cost_usd=price_of(model_id, tin, tout), calls=1,
                ),
            )

        raise ProviderError(
            f"{self.service}: {model_id} did not clear after {MAX_RETRIES} attempts.\n"
            f"  last error: {type(last).__name__ if last else 'unknown'}: "
            f"{getattr(last, 'body', None) or last}"
        ) from last

    # -- pipeline stages ----------------------------------------------------------

    def rewrite(self, *, query: str, history: list[str], model_id: str) -> Completion:
        if not history:
            return Completion(text=query, model_id=model_id)
        convo = "\n".join(f"- {h}" for h in history[-4:])
        text, usage = self._call(
            system=_REWRITE_SYSTEM,
            user=f"Conversation so far:\n{convo}\n\nLatest question: {query}",
            model_id=model_id, max_tokens=200,
        )
        return Completion(text=text.strip() or query, usage=usage, model_id=model_id)

    def rerank(
        self, *, query: str, candidates: list[SourceChunk], model_id: str
    ) -> tuple[dict[str, float], Usage]:
        text, usage = self._call(
            system=_RERANK_SYSTEM,
            user=f"Question: {query}\n\n" + _render_sources(candidates),
            model_id=model_id, max_tokens=80 + 20 * len(candidates), json_mode=True,
        )
        by_n = {c.n: c.chunk_id for c in candidates}
        scores = {
            by_n[n]: max(0.0, min(100.0, float(score)))
            for n, score in _parse_scores(text)
            if n in by_n
        }
        if not scores:
            raise ProviderError(f"reranker returned no parsable scores: {text[:200]!r}")
        return scores, usage

    def generate(
        self, *, query: str, sources: list[SourceChunk], model_id: str,
        temperature: float, max_tokens: int, discriminator_guard: bool = True,
    ) -> GeneratedAnswer:
        if not sources:
            return GeneratedAnswer(
                text="The provided sources do not document this; I could not find any.",
                refused=True, hedged=True,
            )
        system = _GENERATE_SYSTEM + (_DISCRIMINATOR_CLAUSE if discriminator_guard else "")
        text, usage = self._call(
            system=system,
            user=f"{_render_sources(sources)}\n\nQuestion: {query}",
            model_id=model_id, max_tokens=max_tokens, temperature=temperature,
        )
        return _attribute(text, sources, usage)

    # -- judge --------------------------------------------------------------------

    def complete(
        self, *, system: str, user: str, model_id: str = "",
        temperature: float = 0.0, max_tokens: int = 700,
    ) -> Completion:
        model = model_id or self.judge_model
        text, usage = self._call(
            system=system, user=user, model_id=model,
            max_tokens=max_tokens, temperature=temperature,
        )
        return Completion(text=text, usage=usage, model_id=model)


__all__ = [
    "MAX_RETRIES",
    "OpenAICompatibleLLM",
    "is_transient",
    "retry_after",
    "unwrap",
]
