# Judge calibration

judge: `two-way(rule-judge)` · n=107 · labels: **derived**

> **These are derived labels, not human labels.** They come from the synthetic
> corpus's ground truth: each generated fact carries a globally unique token, so
> whether an answer is correct is a substring check, and the reference label for
> a pair follows from the two correctness values. That makes the number below
> objective and reproducible, but it is *not* the number the spec asks for.
> Agreement with a human is a different and harder question, because a human
> also judges whether a differently-worded answer means the same thing.
> Drop human-labelled rows into `evals/golden/human.jsonl` and re-run; the
> harness will use them and this warning will disappear.

## Headline

- agreement: **0.53**
- Cohen's kappa: **0.33** — read this one, not the raw agreement; one class dominates and inflates the raw figure
- order instability: **0.00** of comparisons changed verdict when the two answers were swapped
- verbosity bias: **+0.257** correlation between 'B is longer' and 'judge said equivalent' (0 is unbiased)
- unparseable or unstable verdicts: 0

## Confusion matrix

| reference \ judge | contradictory | degraded | equivalent |
|---|---|---|---|
| **contradictory** | 3 | 39 | 0 |
| **degraded** | 1 | 30 | 0 |
| **equivalent** | 0 | 10 | 24 |

## The systematic error

The single largest disagreement is **reference `contradictory` → judge `degraded`**, 39 cases (36% of the set). That is not noise; it is a direction. Read it as a standing correction to apply to every judge-derived number: this judge under-reports `contradictory` and over-reports `degraded`.

Concretely: when two answers assert *different facts* with similar vocabulary — the same sentence shape with one identifier swapped — a judge scoring lexical overlap sees a near-match and calls it a mild degradation. Distinguishing them needs domain knowledge the rule judge does not have. This is the main reason the ablation table's headline column is computed from ground truth rather than from the judge wherever ground truth exists.

## Where the judge and the reference disagree

### `tenant_atlas:eviction:absent|no_discriminator_guard` — reference `contradictory`, judge `degraded`

> what does ATS-4037 mean

- **A**: The sources do not document `ats-4037`. The closest documented behaviour is: ATS-4123 means the roanfall budget was exhausted while reading ahead of demand. [1]
- **B**: ATS-4123 means the roanfall budget was exhausted while reading ahead of demand. [1]
- judge said: the first answer hedges and the other does not (symmetric judge; direction not claimed)

### `tenant_atlas:federation:para|no_discriminator_guard` — reference `equivalent`, judge `degraded`

> which turn-off controls how much work Atlas does per cycle when forwarding queries to peers

- **A**: The sources do not document `turn-off`. The closest documented behaviour is: It runs continuously in the background and is governed by a disable that determines how much work it may do per cycle. [1]
- **B**: It runs continuously in the background and is governed by a disable that determines how much work it may do per cycle. [1] It bounds forwarding queries to peers on a single node, expressed in seconds. [2]
- judge said: the first answer hedges and the other does not (symmetric judge; direction not claimed)

### `tenant_atlas:hydration:absent|no_lexical+no_discriminator_guard` — reference `contradictory`, judge `degraded`

> what does ATS-4057 mean

- **A**: The sources do not document `ats-4057`. The closest documented behaviour is: ATS-4123 means the roanfall budget was exhausted while reading ahead of demand. [1]
- **B**: ATS-4123 means the roanfall budget was exhausted while reading ahead of demand. [1]
- judge said: the first answer hedges and the other does not (symmetric judge; direction not claimed)

### `tenant_atlas:hydration:id:4053|no_rerank` — reference `contradictory`, judge `degraded`

> what does ATS-4053 mean

- **A**: ATS-4053 means the fenbank budget was exhausted while warming caches after a restart. [1]
- **B**: ATS-4123 means the roanfall budget was exhausted while reading ahead of demand. [1]
- judge said: partial overlap 0.36 (symmetric judge; direction not claimed)

### `tenant_atlas:indexing:absent|no_lexical+no_discriminator_guard` — reference `contradictory`, judge `degraded`

> what does ATS-4087 mean

- **A**: The sources do not document `ats-4087`. The closest documented behaviour is: ATS-4123 means the roanfall budget was exhausted while reading ahead of demand. [1]
- **B**: ATS-4123 means the roanfall budget was exhausted while reading ahead of demand. [1]
- judge said: the first answer hedges and the other does not (symmetric judge; direction not claimed)

### `tenant_atlas:indexing:id:4082|no_rerank` — reference `contradictory`, judge `degraded`

> what does ATS-4082 mean

- **A**: ATS-4082 means the cinderhollow budget was exhausted while building lookup structures. [1]
- **B**: ATS-4123 means the roanfall budget was exhausted while reading ahead of demand. [1]
- judge said: partial overlap 0.38 (symmetric judge; direction not claimed)

### `tenant_atlas:quiescing:absent|no_discriminator_guard` — reference `contradictory`, judge `degraded`

> what does ATS-4137 mean

- **A**: The sources do not document `ats-4137`. The closest documented behaviour is: ATS-4123 means the roanfall budget was exhausted while reading ahead of demand. [1]
- **B**: ATS-4123 means the roanfall budget was exhausted while reading ahead of demand. [1]
- judge said: the first answer hedges and the other does not (symmetric judge; direction not claimed)

### `tenant_atlas:quiescing:id:4132|no_lexical+no_discriminator_guard` — reference `contradictory`, judge `degraded`

> what does ATS-4132 mean

- **A**: ATS-4132 means the yarrowvale budget was exhausted while draining in-flight work. [1]
- **B**: ATS-4123 means the roanfall budget was exhausted while reading ahead of demand. [1]
- judge said: partial overlap 0.38 (symmetric judge; direction not claimed)

### `tenant_atlas:reconciliation:para|top_k_1` — reference `equivalent`, judge `degraded`

> which cap controls how much work Atlas does per cycle when repairing divergent replicas

- **A**: It runs continuously in the background and is governed by a threshold that determines how much work it may do per cycle. [1] It bounds repairing divergent replicas on a single node, expressed in seconds. [2]
- **B**: It runs continuously in the background and is governed by a threshold that determines how much work it may do per cycle. [1]
- judge said: partial overlap 0.57 (symmetric judge; direction not claimed)

### `tenant_atlas:replication:id:4043|no_rerank` — reference `contradictory`, judge `degraded`

> what does ATS-4043 mean

- **A**: ATS-4043 means the wickdrift budget was exhausted while streaming changes to followers. [1]
- **B**: ATS-4123 means the roanfall budget was exhausted while reading ahead of demand. [1]
- judge said: partial overlap 0.38 (symmetric judge; direction not claimed)

### `tenant_atlas:replication:para|no_discriminator_guard` — reference `equivalent`, judge `degraded`

> which turn-off controls how much work Atlas does per cycle when streaming changes to followers

- **A**: The sources do not document `turn-off`. The closest documented behaviour is: It runs continuously in the background and is governed by a disable that determines how much work it may do per cycle. [1]
- **B**: It runs continuously in the background and is governed by a disable that determines how much work it may do per cycle. [1] It bounds streaming changes to followers on a single node, expressed in seconds. [2]
- judge said: the first answer hedges and the other does not (symmetric judge; direction not claimed)

### `tenant_atlas:scrubbing:id:4112|no_rerank` — reference `contradictory`, judge `degraded`

> what does ATS-4112 mean

- **A**: ATS-4112 means the tarndrift budget was exhausted while verifying stored checksums. [1]
- **B**: ATS-4123 means the roanfall budget was exhausted while reading ahead of demand. [1]
- judge said: partial overlap 0.38 (symmetric judge; direction not claimed)

### `tenant_atlas:sharding:id:4062|no_rerank` — reference `contradictory`, judge `degraded`

> what does ATS-4062 mean

- **A**: ATS-4062 means the yarrowfall budget was exhausted while distributing keys across nodes. [1]
- **B**: ATS-4123 means the roanfall budget was exhausted while reading ahead of demand. [1]
- judge said: partial overlap 0.36 (symmetric judge; direction not claimed)

### `tenant_kelvin:archival:absent|no_discriminator_guard` — reference `contradictory`, judge `degraded`

> what does KLV-4157 mean

- **A**: The sources do not document `klv-4157`. The closest documented behaviour is: KLV-4003 means the murrvale budget was exhausted while merging segments. [1]
- **B**: KLV-4003 means the murrvale budget was exhausted while merging segments. [1]
- judge said: the first answer hedges and the other does not (symmetric judge; direction not claimed)

### `tenant_kelvin:archival:id:4152|no_lexical+no_discriminator_guard` — reference `contradictory`, judge `degraded`

> what does KLV-4152 mean

- **A**: KLV-4152 means the welkmarsh budget was exhausted while moving cold data to object storage. [1]
- **B**: KLV-4003 means the murrvale budget was exhausted while merging segments. [1]
- judge said: partial overlap 0.36 (symmetric judge; direction not claimed)

### `tenant_kelvin:checkpointing:followup|no_rewrite` — reference `contradictory`, judge `degraded`

> what is its default value

- **A**: The default value of `klv_checkpointing_interval_seconds` is 69144. [1]
- **B**: The default value of `klv_eviction_max_concurrency` is 93619. [1]
- judge said: partial overlap 0.43 (symmetric judge; direction not claimed)

### `tenant_kelvin:checkpointing:id:4011|no_lexical+no_discriminator_guard` — reference `contradictory`, judge `degraded`

> what does KLV-4011 mean

- **A**: KLV-4011 means the sablefall budget was exhausted while flushing state to disk. [1]
- **B**: KLV-4003 means the murrvale budget was exhausted while merging segments. [1]
- judge said: partial overlap 0.42 (symmetric judge; direction not claimed)

### `tenant_kelvin:compaction:absent|no_lexical+no_discriminator_guard` — reference `contradictory`, judge `degraded`

> what does KLV-4007 mean

- **A**: The sources do not document `klv-4007`. The closest documented behaviour is: KLV-4003 means the murrvale budget was exhausted while merging segments. [1]
- **B**: KLV-4003 means the murrvale budget was exhausted while merging segments. [1]
- judge said: the first answer hedges and the other does not (symmetric judge; direction not claimed)

### `tenant_kelvin:hydration:id:4053|no_lexical+no_discriminator_guard` — reference `contradictory`, judge `degraded`

> what does KLV-4053 mean

- **A**: KLV-4053 means the welkcrest budget was exhausted while warming caches after a restart. [1]
- **B**: KLV-4003 means the murrvale budget was exhausted while merging segments. [1]
- judge said: partial overlap 0.38 (symmetric judge; direction not claimed)

### `tenant_kelvin:indexing:followup|no_rewrite` — reference `contradictory`, judge `degraded`

> what is its default value

- **A**: The default value of `klv_indexing_interval_seconds` is 23818. [1]
- **B**: The default value of `klv_eviction_max_concurrency` is 93619. [1]
- judge said: partial overlap 0.43 (symmetric judge; direction not claimed)

### `tenant_kelvin:indexing:id:4081|no_lexical+no_discriminator_guard` — reference `contradictory`, judge `degraded`

> what does KLV-4081 mean

- **A**: KLV-4081 means the roangate budget was exhausted while building lookup structures. [1]
- **B**: KLV-4003 means the murrvale budget was exhausted while merging segments. [1]
- judge said: partial overlap 0.42 (symmetric judge; direction not claimed)

### `tenant_kelvin:indexing:para|top_k_1` — reference `equivalent`, judge `degraded`

> which maximum controls how much work Kelvin does per cycle when building lookup structures

- **A**: It runs continuously in the background and is governed by a limit that determines how much work it may do per cycle. [1] It bounds building lookup structures on a single node, expressed in seconds. [2]
- **B**: It runs continuously in the background and is governed by a limit that determines how much work it may do per cycle. [1]
- judge said: partial overlap 0.57 (symmetric judge; direction not claimed)

### `tenant_kelvin:ingestion:id:4023|no_rerank` — reference `contradictory`, judge `degraded`

> what does KLV-4023 mean

- **A**: KLV-4023 means the marrowmarsh budget was exhausted while accepting incoming batches. [1]
- **B**: KLV-4003 means the murrvale budget was exhausted while merging segments. [1]
- judge said: partial overlap 0.42 (symmetric judge; direction not claimed)

### `tenant_kelvin:prefetching:absent|no_lexical+no_discriminator_guard` — reference `contradictory`, judge `degraded`

> what does KLV-4127 mean

- **A**: The sources do not document `klv-4127`. The closest documented behaviour is: KLV-4003 means the murrvale budget was exhausted while merging segments. [1]
- **B**: KLV-4003 means the murrvale budget was exhausted while merging segments. [1]
- judge said: the first answer hedges and the other does not (symmetric judge; direction not claimed)

### `tenant_kelvin:quiescing:id:4131|no_rerank` — reference `contradictory`, judge `degraded`

> what does KLV-4131 mean

- **A**: KLV-4131 means the roanhollow budget was exhausted while draining in-flight work. [1]
- **B**: KLV-4003 means the murrvale budget was exhausted while merging segments. [1]
- judge said: partial overlap 0.42 (symmetric judge; direction not claimed)

_...and 25 more._
