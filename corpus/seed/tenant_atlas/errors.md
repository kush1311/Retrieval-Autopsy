# Atlas error reference

Atlas error codes use the `ATL-` prefix. The `ATL-40xx` range covers write-ahead log
and checkpoint faults, `ATL-41xx` covers connection and session management, and
`ATL-42xx` covers query planning.

## Write-ahead log faults

### ATL-4021 — checkpoint could not complete

A checkpoint failed to flush all dirty pages within `checkpoint_timeout_seconds`.
Atlas remains available and retries at the next checkpoint interval, but write-ahead
log segments accumulate in the meantime, and crash recovery will have more to replay.

The usual cause is a checkpoint interval that is short relative to
`checkpoint_completion_target`, which spreads the flush over a fraction of the
interval. If checkpoints routinely overrun, raise `checkpoint_timeout_seconds` before
touching anything else — increasing the shared buffer pool usually makes it worse, not
better, because there are then more dirty pages to flush.

```
ATL-4021 checkpoint could not complete: 41200 dirty pages remaining after 300s
```

### ATL-4022 — WAL segment archival failed

The `wal_archive_command` returned a non-zero exit status. Atlas will not recycle a
segment it has not successfully archived, so segments accumulate in `wal_dir` until
either archival succeeds or the disk fills.

This is deliberate: recycling an unarchived segment would leave a hole in the archive
and silently break point-in-time recovery at some unknown future date. If you would
rather lose archive continuity than fill the disk, set
`wal_archive_failure_action: recycle` — Atlas logs a `WARN` for every segment it
discards under that setting.

### ATL-4023 — WAL record checksum mismatch

A write-ahead log record failed its checksum during replay. Atlas stops recovery at
that record and refuses to open the database, because the records after a corrupt one
cannot be trusted to apply cleanly.

Recovery from this state requires either restoring from a base backup and replaying
the archive, or accepting data loss with `wal_recovery_stop_at_corruption: true`,
which opens the database as of the last good record.

## Connection faults

### ATL-4101 — connection limit reached

The server has `max_connections` sessions open and refused a new one. Reserved slots
governed by `superuser_reserved_connections` remain available for administrative
access, which is why an operator can usually still connect while applications cannot.
