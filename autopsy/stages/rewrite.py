"""Rewrite a follow-up into a standalone query.

**This is a second entry into retrieval.** The rewritten query goes on to drive both
retrieval legs, and in most real systems the rewrite path is added after the primary
path is already correct and reviewed — which is exactly why it is the highest-yield
cross-tenant leak vector. The tenant is threaded through explicitly here, and the
``followup_rewrite`` isolation probe exists to prove it stays that way.
"""

from __future__ import annotations

from autopsy.stages.base import Context, State


class RewriteStage:
    name = "rewrite"

    def skip(self, state: State, ctx: Context) -> str | None:
        if not ctx.cfg.rewrite_enabled:
            return "rewrite ablated by config"
        if not state.history:
            return "no conversation history, query is already standalone"
        return None

    def run(self, state: State, ctx: Context) -> State:
        record = ctx.current
        key = ctx.stage_cache.key(
            config_hash=ctx.config_hash,
            tenant_id=ctx.tenant_id,  # tenant in the key, always
            stage=self.name,
            payload=[state.query, state.history],
        )
        completion, cache_state = ctx.stage_cache.get_or_compute(
            key,
            lambda: ctx.providers.llm.rewrite(
                query=state.query,
                history=state.history,
                model_id=ctx.cfg.generation.model_id,
            ),
        )
        if record is not None:
            record.cache = cache_state
            record.tokens_in = completion.usage.tokens_in
            record.tokens_out = completion.usage.tokens_out
            record.cost_usd = completion.usage.cost_usd

        rewritten = (completion.text or "").strip()
        if rewritten and rewritten != state.query:
            state.rewritten_query = rewritten
            if record is not None:
                record.detail = {"from": state.query, "to": rewritten}
        elif record is not None:
            record.detail = {"note": "nothing to resolve; query unchanged"}
        return state


__all__ = ["RewriteStage"]
