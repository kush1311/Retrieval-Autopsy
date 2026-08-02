# Vela quantization

Quantization shrinks vectors so more of the index fits in memory, trading a small
amount of accuracy for a large amount of capacity.

## Modes

```yaml
quantization:
  mode: scalar_int8       # none | scalar_int8 | binary
  keep_original: true
  rescore: true
  oversampling: 2.0
```

- `scalar_int8` — each dimension becomes one byte. Roughly 4× smaller than float32,
  with recall loss usually under one point when rescoring is enabled. The default
  recommendation.
- `binary` — each dimension becomes one bit. Roughly 32× smaller. Only viable for
  high-dimensional embeddings (1024 and above) and only with rescoring; below that,
  recall degrades sharply.
- `none` — full precision.

## Rescoring

With `rescore: true`, Vela retrieves `limit × oversampling` candidates using the
quantized vectors, then reranks them using the full-precision vectors before returning
the final `limit`. This recovers most of the recall lost to quantization, at the cost
of reading original vectors from disk for the oversampled set.

Rescoring requires `keep_original: true`. If original vectors were not kept, a
rescoring search silently returns quantized-only results and reports `VEC-4023`.
`keep_original` cannot be changed after the collection is created.

## Choosing oversampling

`oversampling` is a multiplier on the candidate set, not a quality score. At 1.0 there
is nothing to rerank and rescoring does nothing useful. Values between 2.0 and 4.0
cover most cases; beyond that the disk reads dominate the latency budget and you would
be better off not quantizing.

## What quantization does not do

Quantization reduces the memory held by *vectors*. It does not shrink the graph index,
the payload store, or payload indexes. On collections with large payloads the vectors
are not the dominant cost and quantizing them buys less than expected — measure the
breakdown before assuming quantization is the answer.

There is no setting that quantizes the graph structure itself.
