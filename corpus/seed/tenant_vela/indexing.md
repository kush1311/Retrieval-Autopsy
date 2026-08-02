# Vela indexing

## Graph index parameters

Vela builds a hierarchical navigable small-world graph per segment.

```yaml
hnsw_m: 16                  # edges per node
hnsw_ef_construct: 128      # candidate list size during build
hnsw_ef_search: 64          # candidate list size at query time
full_scan_threshold_vectors: 10000
```

`hnsw_m` sets graph connectivity. Higher values improve recall and increase both
memory and build time; the memory cost is roughly linear in `hnsw_m`. Values between
16 and 32 cover most workloads, and above 64 the returns are very small.

`hnsw_ef_construct` controls build-time search breadth. It affects graph quality
permanently — you cannot recover a poorly built graph by raising `hnsw_ef_search`
later.

`hnsw_ef_search` is a per-query knob and can be raised at request time. It trades
latency for recall linearly and is the right dial to move first when recall is
disappointing.

## Small segments

Below `full_scan_threshold_vectors` Vela does not build a graph at all; it scans the
segment exhaustively. Exhaustive search on ten thousand vectors is fast and exact, and
a graph over that few points has poor recall anyway. This is why recall measured on a
small development collection does not predict production behaviour.

## Filtering and recall

A filtered search restricts the graph traversal to points matching the filter. When
the filter is highly selective, the reachable subgraph can become disconnected and
recall collapses — the traversal simply cannot reach candidates that are only
connected through excluded points.

Vela mitigates this by falling back to exhaustive scan when the estimated match count
drops below `full_scan_threshold_vectors`. The estimate comes from payload index
cardinality, so a filter on an unindexed payload field has no estimate, gets no
fallback, and is exactly where filtered recall problems show up.

Index the payload fields you filter on:

```yaml
payload_index:
  - field: tenant_id
    type: keyword
  - field: published_at
    type: datetime
```

## Rebuilds

Changing `hnsw_m` or `hnsw_ef_construct` affects only segments built afterwards.
Existing segments keep the parameters they were built with, so a collection can hold a
mix. Use an explicit reindex to make a change uniform.
