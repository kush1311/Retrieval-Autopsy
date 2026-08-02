# Kelvin replication

## Topology

Kelvin replication is asynchronous and single-primary. A replica connects to the
primary, receives a full dataset transfer, then streams subsequent writes. Replicas
may themselves have replicas; chained replication reduces load on the primary at the
cost of additional lag on the leaf nodes.

```yaml
role: replica
primary_endpoint: kelvin-0.internal:7711
replica_read_only: true
```

`replica_read_only` defaults to `true`. Writes accepted directly by a replica are not
propagated anywhere and are silently overwritten on the next resynchronisation, so
turning it off is almost always a mistake.

## Replication lag

Lag is reported in bytes of unacknowledged stream, not in seconds, because Kelvin has
no way to know how long the primary will take to produce the next write.

```
kelvin-cli info replication
# replica_0: endpoint=10.0.4.7:7711 lag_bytes=8192 state=streaming
```

`replica_lag_max_bytes` bounds how far behind a replica may fall before the primary
drops the connection and forces a full resynchronisation. Setting it too low causes
resynchronisation storms under write bursts; too high, and a stalled replica holds
replication buffer memory on the primary indefinitely.

## Failover

Kelvin does not perform automatic failover. Promotion is an explicit operation:

```
kelvin-cli replica promote --confirm
```

Promotion assigns the node a new `replication_id`. Every other replica of the old
primary must fully resynchronise against the new one; they cannot continue streaming,
and attempting to do so produces `KV-4101`.

## What is not replicated

Expiry is driven by the primary. A replica does not independently expire keys; it
waits for the primary's deletion to arrive in the stream. A read served by a replica
can therefore return a key that has logically expired but has not yet been collected
on the primary. Applications that need strict expiry semantics must read from the
primary.

Configuration is also not replicated. `max_memory_bytes`, `max_memory_policy`, and
the persistence settings are per-node and must be managed by your deployment tooling.
