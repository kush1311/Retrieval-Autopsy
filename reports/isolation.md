# Isolation report

code src@bfd86a8c · corpus seed@cab5c9bd · embed_model offline-concept-86d01716 · provider offline

> **These numbers describe the offline simulator, not a language model.**
> The `offline` provider is a deterministic rule-based stand-in that exists so the
> whole system runs in CI with no API key. It reproduces the *mechanism* behind each
> failure class faithfully, but its absolute rates are properties of the simulator.
> Re-run with `AUTOPSY_PROVIDER=live` for numbers about a real model.


## Summary

- **isolation**: 12/12 findings passed
- judge agreement with derived labels: 0.53 (kappa 0.33, n=107) — see reports/judge-calibration.md

## Regressions vs baseline

- fixed `silent_failure/append_rewrite_throttling`
- fixed `silent_failure/cross_version_checkpoint_timeout`
- fixed `silent_failure/deprecated_reclaim_delay`
- fixed `silent_failure/negation_not_replicated`
- fixed `silent_failure/superlative_fastest_metric`

## Failures

none
