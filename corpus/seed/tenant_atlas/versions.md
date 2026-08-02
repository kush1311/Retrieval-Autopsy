# Atlas version differences

This page records behaviour that changed between major versions. Where a value differs
between versions, both are listed; there is no single "current" value.

## Checkpoint defaults

In **Atlas 2.x**, `checkpoint_timeout_seconds` defaulted to 900 and
`checkpoint_completion_target` defaulted to 0.5. Checkpoints were consequently
infrequent and I/O-spiky, and most production deployments overrode both.

In **Atlas 3.x**, the defaults changed to 300 and 0.9 respectively. The shorter
interval and higher completion target spread I/O far more evenly, at the cost of a
slightly longer recovery window.

An upgrade does **not** rewrite an existing configuration file. A cluster upgraded from
2.x keeps 900 and 0.5 unless someone changes them, so two clusters on the same version
can behave completely differently depending on where they came from.

## Reclaim settings

`reclaim_cost_delay_ms` existed in Atlas 2.x and controlled how long the reclaim
process paused between batches. It is deprecated in 3.0 and has **no effect** — the
value is parsed, accepted, and ignored. `autoreclaim_io_concurrency` replaces it.

Because the old key is still accepted at startup, a configuration carried forward from
2.x looks like it is throttling reclaim I/O and is not. The only signal is a
deprecation warning in the startup log.

## Connection defaults

`max_connections` defaulted to 100 in Atlas 2.x and 200 in Atlas 3.x.
`superuser_reserved_connections` was 3 in 2.x and is 5 in 3.x.

## What did not change

The write-ahead log format is unchanged across 2.x and 3.x, so a 3.x server can replay
a log written by 2.x. The reverse is not true: a 2.x server refuses to start against a
directory a 3.x server has opened.
