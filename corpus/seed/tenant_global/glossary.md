# Shared glossary

Terms used consistently across all product documentation. Where a product uses a term
differently, its own documentation says so explicitly.

## Durability

**Acknowledged write** — a write the server has confirmed to the client. What an
acknowledgement guarantees is product-specific and is stated in each product's
durability documentation.

**Checkpoint** — a recorded position from which recovery may begin, taken after
flushing the state accumulated up to that point.

**Recovery point objective** — the maximum data loss an operator is willing to accept,
expressed as a time window. Configuration should be derived from this number, not the
other way around.

## Replication

**Primary** — the node accepting writes. Exactly one per replication group.

**Replica** — a node receiving a stream of changes from a primary. Read-only unless
explicitly configured otherwise.

**Lag** — how far a replica trails its primary. Reported in bytes where the product
cannot know the time equivalent, and in seconds where it can.

**Full resynchronisation** — a replica discarding its state and receiving a complete
copy. Expensive, and triggered by any break in the change stream that cannot be
repaired incrementally.

## Storage

**Segment** — a unit of on-disk storage that can be written, merged, and searched
independently.

**Compaction** — merging segments to reduce their number and reclaim space.

**Bloat** — space occupied by data that is no longer live but has not been reclaimed.

## Search

**Recall** — the fraction of true results a search actually returns. Distinct from
precision, and the metric that approximate indexes trade away for speed.

**Rescoring** — re-ranking an oversampled candidate set with a more expensive and more
accurate scoring function before returning the final result.

## Support

**Workaround** — a documented way to avoid an issue's impact without a code change.
The presence of a workaround affects severity, not priority.
