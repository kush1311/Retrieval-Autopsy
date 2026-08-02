"""Embed the (possibly rewritten) query.

Cached on ``(model_id, text)`` and shared across tenants and across ablations. An
embedding is a pure function of the text and the model — no tenant-specific
information enters it — so this is the one cache in the system that safely omits the
tenant and the config hash. Answer caches are the opposite case; see ``cache.py``.

Reusing this across an ablation sweep is most of the cost saving available: the query
embedding is identical for every variant except ``no_semantic``.
"""

from __future__ import annotations

from autopsy.stages.base import Context, State


class EmbedStage:
    name = "embed"

    def skip(self, state: State, ctx: Context) -> str | None:
        if ctx.cfg.semantic is None:
            return "semantic leg ablated, nothing to embed"
        return None

    def run(self, state: State, ctx: Context) -> State:
        record = ctx.current
        text = state.effective_query
        model = ctx.providers.embedder.model_id

        def compute():
            return ctx.providers.embedder.embed_query(text)

        (embedding, usage), cache_state = ctx.embed_cache.get_or_compute(model, text, compute)
        state.embedding = embedding
        if record is not None:
            record.cache = cache_state
            record.detail = {"model": model, "kind": embedding.kind}
            if cache_state != "hit":
                record.tokens_in = usage.tokens_in
                record.cost_usd = usage.cost_usd
        return state


__all__ = ["EmbedStage"]
