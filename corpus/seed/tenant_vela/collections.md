# Vela collections

## Creating a collection

```yaml
name: documents
vectors:
  size: 1536
  distance: cosine        # cosine | dot | euclid
segment_max_vectors: 200000
replication_factor: 2
write_consistency_factor: 1
```

`size` and `distance` are fixed at creation. Changing either requires a new collection
and a full re-upsert; there is no in-place migration, because every stored vector and
every graph edge was built under the original metric.

## Distance metrics

- `cosine` — Vela normalises vectors on upsert and uses the dot product internally.
  Normalisation is applied to what you send, so upserting an already-normalised vector
  is harmless.
- `dot` — no normalisation. Vector magnitude affects the score, which is what you want
  for models trained with an unnormalised objective and a mistake otherwise.
- `euclid` — squared euclidean distance. Lower is better, which inverts the ordering
  relative to the other two; client code that assumes higher-is-better silently ranks
  backwards.

## Segments

A collection is stored as multiple segments. Search runs against every segment and
merges the results, so a collection with many small segments spends most of its query
budget on merge overhead. `segment_max_vectors` bounds segment size; Vela merges small
segments in the background.

The optimizer will not merge a segment that is currently being searched, so under
sustained query load a collection can accumulate more segments than the configuration
suggests. This is normal and resolves during quieter periods.

## Consistency

`write_consistency_factor` is the number of replicas that must acknowledge a write
before it is reported successful. With `replication_factor: 2` and
`write_consistency_factor: 1`, a write is acknowledged by one replica and propagated
asynchronously — a read served by the other replica can miss it briefly.

Setting `write_consistency_factor` equal to `replication_factor` makes writes fully
synchronous and makes any single replica outage a write outage. There is no quorum
mode between the two.

## Aliases

An alias is a stable name pointing at a collection. The intended use is a rebuild:
build `documents_v2` alongside `documents_v1`, verify it, then repoint the alias
atomically. Alias switches take effect for new requests only; in-flight searches
complete against the collection they started on.
