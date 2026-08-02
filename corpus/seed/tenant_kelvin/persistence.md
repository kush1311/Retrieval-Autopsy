# Kelvin persistence

Kelvin offers two independent durability mechanisms. They can be enabled together,
and most production deployments do exactly that: the append log bounds how much data
a crash can lose, and snapshots bound how long recovery takes.

## Append log

Every write is appended to an on-disk log before the client is acknowledged. The
`append_fsync` setting controls how often that log is flushed:

```yaml
append_log: true
append_fsync: every_second   # off | every_second | always
```

- `always` — flush before acknowledging each write. Durable to the last write, and
  roughly an order of magnitude slower on spinning disks.
- `every_second` — the default. A crash can lose up to one second of writes.
- `off` — leave flushing to the operating system. Kelvin does not recommend this and
  logs a warning at startup.

### Rewriting the log

The append log grows without bound, so Kelvin periodically rewrites it into the
smallest sequence of commands that reproduces the current dataset. A rewrite is
triggered when the log exceeds `append_rewrite_percent` of its size after the previous
rewrite, and never below `append_rewrite_min_size`.

A rewrite is a background operation and does not block writes. If it fails, Kelvin
reports `KV-4021` and retries at the next trigger point.

## Snapshots

A snapshot is a compact binary image of the whole dataset at a point in time.

```yaml
snapshot_interval_seconds: 900
snapshot_dir: /var/lib/kelvin
snapshot_compression: lz4    # none | lz4 | zstd
```

Snapshots are written by a forked child process, so peak memory during a snapshot can
approach twice the resident set on a write-heavy instance. Budget for that; the most
common Kelvin outage is an out-of-memory kill during snapshotting on a host sized for
steady-state usage.

## Recovery precedence

On start, Kelvin loads the append log if `append_log` is enabled, because it is the
more complete of the two sources. The snapshot is used only when the append log is
absent or disabled. There is no merge step and no setting that changes this order.
