# Kelvin memory limits and eviction

## Setting a limit

`max_memory_bytes` caps the memory Kelvin uses for stored data. It excludes
replication buffers, client output buffers, and allocator fragmentation, so the
process resident set will always be somewhat larger than the configured value. A
common sizing mistake is to set `max_memory_bytes` to the container limit; leave at
least 25% headroom.

```yaml
max_memory_bytes: 6442450944   # 6 GiB
max_memory_policy: evict_lru
```

## Eviction policies

When the limit is reached, `max_memory_policy` decides what happens next:

| Policy | Behaviour |
|---|---|
| `reject_writes` | Writes fail with `KV-4210`; reads continue. The default. |
| `evict_lru` | Evict the least recently used key, across all keys. |
| `evict_lru_expiring` | Evict least recently used, but only among keys that carry an expiry. |
| `evict_random` | Evict an arbitrary key. Cheapest, and appropriate only for pure caches. |

`evict_lru` uses sampled approximate LRU rather than a strict global ordering. The
sample size is `eviction_sample_size`, default 5. Raising it improves eviction quality
and costs CPU on every eviction; values above 10 rarely help.

## Throttling

Kelvin does not throttle writes as it approaches `max_memory_bytes`. There is no
gradual back-pressure mode and no setting that introduces one. Behaviour at the limit
is a step change governed entirely by `max_memory_policy`.

## Expiry

Keys with a TTL are removed by two mechanisms working together: a lazy check when the
key is next accessed, and a background sampler that runs `expiry_scan_per_second`
times a second. Neither guarantees prompt removal, so a key that has logically expired
may still occupy memory. Do not rely on TTL alone to keep the dataset under
`max_memory_bytes`.
