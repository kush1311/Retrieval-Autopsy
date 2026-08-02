# Gate calibration

embedding model: `BAAI/bge-small-en-v1.5`

## Which signal should the gate read?

The spec left this open and made `gate.reads` a config field so it could be
measured instead of argued about. This is the measurement, on this corpus with
this embedding model — it is not a universal answer, which is the point.

| `gate.reads` | threshold | false refusals | false admits | separation |
|---|---|---|---|---|
| `lexical_top1` **←** | 1.738 | 0.0% | 4.0% | +1.675 |
| `fused_top1` | 0.032 | 27.5% | 20.0% | -0.002 |
| `dense_top1` | 0.729 | 25.5% | 26.0% | -0.037 |

*Separation is answerable-p10 minus out-of-scope-p90: how much daylight sits
between the two populations. Negative means they overlap and no threshold
separates them.*

---
## Winning signal: `lexical_top1`

embedding model: `BAAI/bge-small-en-v1.5`

### Recommended threshold

    gate.reads     = lexical_top1
    gate.threshold = 1.738

| | answerable | out of scope |
|---|---|---|
| n | 200 | 50 |
| min | 2.634 | 0.842 |
| p10 | 2.703 | 0.842 |
| median | 7.123 | 0.856 |
| p90 | 16.968 | 1.028 |
| max | 22.592 | 6.491 |

### What this threshold costs

- **false refusals**: 0.0% of answerable queries score below it and would be refused despite the answer being in the corpus
- **false admits**: 4.0% of out-of-scope queries score above it and reach the generator anyway
- **separation** (answerable p10 − out-of-scope p90): +1.675

### Why this is not a constant

Every embedding model puts similarity on its own scale. The offline concept
simulator scores query-coverage in [0, 1] with a genuine floor at 0;
`bge-small-en-v1.5` scores *unrelated* text around 0.55, because a normalised
cosine between two English sentences is never near zero. A threshold moved
between them without re-deriving it either passes everything or refuses
everything — and in both cases the trace still reports `gate: passed` or
`gate: refused` exactly as if it were working.
