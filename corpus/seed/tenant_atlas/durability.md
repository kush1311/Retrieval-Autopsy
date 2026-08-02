# Atlas durability and the write-ahead log

## The write-ahead log

Every change is written to the write-ahead log before the corresponding data page is
modified. On crash, Atlas replays the log from the last checkpoint to restore a
consistent state.

```yaml
wal_dir: /var/lib/atlas/wal
wal_segment_bytes: 16777216
wal_sync_method: fdatasync   # fsync | fdatasync | open_datasync | off
synchronous_commit: on       # on | remote_write | local | off
```

`synchronous_commit: off` acknowledges a commit before its log record reaches disk.
It does **not** risk corruption — the database stays consistent — but a crash can lose
the most recent transactions, bounded by `wal_writer_delay`. This is the single
highest-leverage durability/throughput trade-off Atlas exposes.

## Checkpoints

A checkpoint flushes dirty pages and records a point in the log from which recovery
may begin.

```yaml
checkpoint_timeout_seconds: 300
checkpoint_completion_target: 0.9
max_wal_bytes: 1073741824
```

Checkpoints are triggered by whichever of `checkpoint_timeout_seconds` or
`max_wal_bytes` arrives first. A checkpoint driven by `max_wal_bytes` under a write
burst is the usual cause of a sudden I/O spike; if you see periodic latency cliffs,
check which trigger is firing before tuning either.

`checkpoint_completion_target` spreads the flush across that fraction of the interval.
The default of 0.9 smooths I/O at the cost of a slightly longer recovery window.

## Archival and point-in-time recovery

```yaml
wal_archive_mode: on
wal_archive_command: "aws s3 cp %p s3://atlas-wal/%f"
```

The command receives the segment path as `%p` and its file name as `%f`, and must
return zero only on success. An archive command that reports success without durably
storing the segment produces an archive with holes in it, which will be discovered at
the worst possible time. Atlas cannot detect this; test your restore path.

Archival failure is reported as `ATL-4022`.

## What durability does not cover

Neither the write-ahead log nor archival protects against a logical error — a bad
migration or an erroneous bulk update is faithfully replicated and faithfully
archived. Point-in-time recovery to just before the mistake is the only remedy, which
is why the archive is worth the operational cost.
