# Atlas storage maintenance

## Row versions and reclamation

Atlas keeps old row versions so that concurrent readers see a consistent snapshot. A
row version becomes reclaimable once no open transaction can still see it. The
`reclaim` process removes them.

```yaml
autoreclaim: on
autoreclaim_scale_factor: 0.2
autoreclaim_threshold_rows: 50
autoreclaim_max_workers: 3
```

A table is eligible for reclamation when its dead-row count exceeds
`autoreclaim_threshold_rows` plus `autoreclaim_scale_factor` times its live row count.
On a large table the scale factor dominates: at the default 0.2, a table of 100
million rows is not touched until roughly 20 million rows are dead. Large tables
generally want a lower per-table factor.

## What blocks reclamation

Reclamation cannot remove a row version that any open transaction might still see.
One long-running transaction anywhere on the server therefore blocks reclamation
**everywhere**, and the symptom is table bloat on tables that transaction never
touched. This is the most common cause of unexplained disk growth in Atlas.

Check `atlas_stat_activity` for the oldest transaction before investigating the
reclaim settings.

## Full rewrite

`autoreclaim` marks space reusable but does not return it to the filesystem. To
shrink a table on disk you need a full rewrite:

```sql
RECLAIM FULL orders;
```

`RECLAIM FULL` takes an exclusive lock for the duration and needs free disk space
equal to the size of the table being rewritten. It is not an online operation and
should not be scheduled casually.

## Statistics

The planner depends on table statistics that the reclaim process refreshes as a side
effect. A table that is never reclaimed also has stale statistics, so bloat and bad
query plans tend to arrive together. `ANALYZE` refreshes statistics without touching
storage and is cheap enough to run on demand.

## Deprecated settings

`reclaim_cost_delay_ms` is deprecated and has no effect from Atlas 3.0 onward. It was
replaced by `autoreclaim_io_concurrency`, which expresses the same intent as an I/O
budget rather than a sleep. Configurations that still set the old key are accepted at
startup with a deprecation warning, which is easy to miss in a container log.
