# Kelvin error reference

Kelvin returns a machine-readable error code on every failed operation. Codes in the
`KV-40xx` range are durability and persistence faults; `KV-41xx` covers cluster
membership; `KV-42xx` covers client protocol violations.

## Persistence faults

### KV-4021 — append log rewrite failed

The background rewrite of the append log could not complete. Kelvin keeps serving
reads and writes, but the append log continues to grow until the rewrite succeeds, so
recovery time after a restart grows with it.

The usual cause is insufficient free disk space: the rewrite writes a fresh log
alongside the existing one and needs headroom equal to the live dataset before it can
swap them. Check `kelvin-cli info persistence` for `last_rewrite_status`. Raising
`append_rewrite_min_free_bytes` does not fix a full disk; it only makes Kelvin decline
the rewrite earlier and more clearly.

```
KV-4021 append log rewrite failed: needed 4.2GiB free, found 900MiB
```

### KV-4022 — snapshot write failed

A point-in-time snapshot could not be written to disk. Unlike `KV-4021`, this is
**fatal for the snapshot only** — the append log is unaffected and remains a complete
recovery source. Kelvin will retry at the next `snapshot_interval_seconds` boundary.

The most common cause is a permissions problem on `snapshot_dir` after a container
restart under a different UID. Kelvin does not fall back to a temporary directory,
because a snapshot written somewhere unexpected is worse than no snapshot at all.

```
KV-4022 snapshot write failed: open /var/lib/kelvin/snap.tmp: permission denied
```

### KV-4023 — append log corrupted at offset

Kelvin found a truncated or malformed record while replaying the append log at
startup. By default it refuses to start, because silently discarding the tail of a
durability log turns a loud failure into quiet data loss.

Set `append_truncate_on_corruption: true` to discard everything from the bad offset
onward and start anyway. This is a deliberate data-loss switch and Kelvin logs the
number of discarded records at `WARN` on every start while it is enabled.

## Cluster faults

### KV-4101 — replica handshake rejected

A replica presented a `replication_id` that does not match the primary's. This
happens after a primary is restored from a snapshot: the restored node has a new
replication history and existing replicas cannot continue streaming from it. Replicas
must perform a full resynchronisation.
