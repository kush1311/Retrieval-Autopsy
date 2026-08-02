# Vela error reference

Vela uses the `VEC-` prefix. `VEC-40xx` covers index build and search faults,
`VEC-41xx` covers collection lifecycle, and `VEC-42xx` covers payload and filtering.

## Index faults

### VEC-4021 — index build exceeded memory budget

Building the graph index for a segment required more memory than
`index_build_memory_bytes` allows. The segment stays searchable through brute-force
scan, which is correct but orders of magnitude slower, and Vela retries the build when
memory frees up.

Build memory scales with `hnsw_m` and the segment's vector count. Halving `hnsw_m`
roughly halves build memory and costs some recall at high filter selectivity. Reducing
`segment_max_vectors` is usually the better lever, because it bounds the peak without
changing search quality.

```
VEC-4021 index build exceeded memory budget: needed 3.1GiB, budget 2.0GiB
```

### VEC-4022 — vector dimension mismatch

An upserted vector's dimension does not match the collection's configured `size`. The
point is rejected; the rest of the batch is unaffected.

Vela never pads or truncates a mismatched vector. A silently reshaped vector produces
a searchable point whose neighbours are meaningless, and the resulting quality
regression is untraceable to its cause. Failing the point is the cheaper outcome.

```
VEC-4022 vector dimension mismatch: collection size=1536, got 768
```

### VEC-4023 — quantized index missing original vectors

A search requested rescoring, but the collection was created with
`quantization.keep_original: false`, so the full-precision vectors needed for the
rescore pass are not on disk. The search returns quantized-only results.

This setting cannot be changed in place. Recovering full-precision rescoring requires
recreating the collection and re-upserting.

## Collection faults

### VEC-4101 — collection already exists

A create call named an existing collection. Vela does not implicitly replace
collections, because an accidental replace destroys an index that may have taken hours
to build. Use an explicit recreate call if that is what you meant.
