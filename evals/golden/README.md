# Golden labels for judge calibration

Every `*.jsonl` file in this directory is loaded and used to calibrate the judge. One
JSON object per line:

```json
{
  "case_id": "unique-id",
  "question": "what does KLV-4021 mean",
  "answer_a": "…",
  "answer_b": "…",
  "label": "equivalent | degraded | contradictory",
  "source": "human | derived"
}
```

## `derived.jsonl` — generated, objective, and not the number the spec asks for

Produced by `python -m autopsy.cli calibrate --derive`. Labels come from the synthetic
corpus's ground truth: every generated fact carries a globally unique token, so whether
an answer is correct is a substring check, and the reference label for a pair follows
from the two correctness values plus whether the losing answer *declined* or *asserted*
something different.

This is reproducible and free of opinion, which makes it a good regression signal. It
is **not** a human agreement rate. A human also judges whether two differently-worded
answers mean the same thing, and no amount of substring checking captures that. The
calibration report prints a warning saying exactly this whenever no human labels are
present.

## `human.jsonl` — the one that counts

Hand-label 100 pairs and drop them here. The report picks them up automatically, the
warning disappears, and `label_source` changes to include `human`.

Suggested procedure, so the labels are worth having:

1. `python -m autopsy.cli calibrate --derive -n 200` to generate candidate pairs.
2. Label them yourself **before** looking at what the judge said. Seeing the judge's
   answer first turns labelling into agreeing.
3. Label the *claims*, not the prose. Length is not quality.
4. Keep the pairs where you found it genuinely hard to decide. Those are the ones that
   discriminate between judges; the easy ones inflate agreement and tell you nothing.
5. Re-run `python -m autopsy.cli calibrate`.

Publishing "my judge disagrees with me 23% of the time, and here is the confusion
matrix" is worth more than any passing test suite, precisely because almost nobody
does it.
