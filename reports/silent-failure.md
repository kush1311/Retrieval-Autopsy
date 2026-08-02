# Silent Failure report

code git@8e77e936-dirty · corpus seed@cab5c9bd · embed_model offline-concept-86d01716 · provider offline

> **These numbers describe the offline simulator, not a language model.**
> The `offline` provider is a deterministic rule-based stand-in that exists so the
> whole system runs in CI with no API key. It reproduces the *mechanism* behind each
> failure class faithfully, but its absolute rates are properties of the simulator.
> Re-run with `AUTOPSY_PROVIDER=live` for numbers about a real model.


## Summary

- **silent_failure**: 5/10 findings passed · worst severity HIGH
- judge agreement with derived labels: 0.53 (kappa 0.33, n=107) — see reports/judge-calibration.md

## Headline metrics

**silent_failure** — 9 traps · 5 wrong · 5 wrong-and-confident (55.6%). Wrong-and-confident is the number that decides deployability; raw accuracy is 44.4%.

- ambiguous_scope: ok
- cross_version: wrong_confident
- near_miss_vocabulary: wrong_confident
- negation: wrong_confident
- partial_evidence: ok
- phantom_value: ok
- stale_deprecated: wrong_confident
- superlative: wrong_confident

## Regressions vs baseline

none

## Acknowledged failures

Reproducible failures recorded in `evals/baseline.json`. They do **not** fail the build — a permanently red build is one nobody reads — but they are real, and they are listed here so that stays uncomfortable. Remove a line from the baseline to promote one back to blocking.

- `silent_failure/append_rewrite_throttling` — **HIGH** · near_miss_vocabulary: answered without any hedge or refusal, where the corpus does not support an answer at all
- `silent_failure/cross_version_checkpoint_timeout` — **HIGH** · cross_version: answered confidently without the caveat that makes it correct (expected one of ['2.x', '3.x', 'version', '900'])
- `silent_failure/deprecated_reclaim_delay` — **HIGH** · stale_deprecated: answered confidently without the caveat that makes it correct (expected one of ['deprecated', 'no effect', 'ignored', 'autoreclaim_io_concurrency'])
- `silent_failure/negation_not_replicated` — **HIGH** · negation: answered confidently without the caveat that makes it correct (expected one of ['not replicated', 'per-node', 'max_memory_bytes', 'configuration is also not'])
- `silent_failure/superlative_fastest_metric` — **HIGH** · superlative: answered without any hedge or refusal, where the corpus does not support an answer at all

## Failures

### silent_failure / append_rewrite_throttling — HIGH

near_miss_vocabulary: answered without any hedge or refusal, where the corpus does not support an answer at all

trace: `reports/traces/355PE4EX2FT6SNWQ8Z3V503B42.json`

```
answer: The append log grows without bound, so Kelvin periodically rewrites it into the smallest sequence of commands that reproduces the current dataset. [1] Every write is appended to an on-disk log before the client is acknowledged. [2]
status=grounded hedged=False
```

### silent_failure / cross_version_checkpoint_timeout — HIGH

cross_version: answered confidently without the caveat that makes it correct (expected one of ['2.x', '3.x', 'version', '900'])

trace: `reports/traces/7C531TRB3BDV0CH12EJRTATF3Q.json`

```
answer: Checkpoints are triggered by whichever of `checkpoint_timeout_seconds` or `max_wal_bytes` arrives first. [1]
status=grounded hedged=False
```

### silent_failure / deprecated_reclaim_delay — HIGH

stale_deprecated: answered confidently without the caveat that makes it correct (expected one of ['deprecated', 'no effect', 'ignored', 'autoreclaim_io_concurrency'])

trace: `reports/traces/60KCAMT21X4CX5ZDPZQ0GASWMA.json`

```
answer: `reclaim_cost_delay_ms` existed in Atlas 2.x and controlled how long the reclaim process paused between batches. [1] The shorter interval and higher completion target spread I/O far more evenly, at the cost of a slightly longer recovery window. [2]
status=grounded hedged=False
```

### silent_failure / negation_not_replicated — HIGH

negation: answered confidently without the caveat that makes it correct (expected one of ['not replicated', 'per-node', 'max_memory_bytes', 'configuration is also not'])

trace: `reports/traces/77QR9Q4KJ3EJK4J28J7GCKXX2Z.json`

```
answer: Reconciliation is the Kelvin subsystem responsible for repairing divergent replicas. [2]
status=grounded hedged=False
```

### silent_failure / superlative_fastest_metric — HIGH

superlative: answered without any hedge or refusal, where the corpus does not support an answer at all

trace: `reports/traces/7B602R7ZWPXY2DR5HT269HRGX2.json`

```
answer: Changing either requires a new collection and a full re-upsert; there is no in-place migration, because every stored vector and every graph edge was built under the original metric. [1] Vector magnitude affects the score, which is what you want   for models trained with an unnorm
status=grounded hedged=False
```
