# Support and release policy

This policy applies to every product in the catalogue and is visible to all tenants.

## Release channels

- **stable** — supported for 24 months from release. Security and correctness fixes
  only after the first 12 months.
- **preview** — supported until the next stable release supersedes it. Preview
  releases may change behaviour without a deprecation period.
- **nightly** — unsupported. Not covered by any compatibility commitment.

## Version numbering

Releases are `MAJOR.MINOR.PATCH`. A major release may remove deprecated settings; a
minor release may deprecate them but never removes them. A patch release changes
behaviour only when the previous behaviour was a defect.

## Deprecation

A setting marked deprecated continues to work for at least one full major version. It
emits a warning at startup naming the replacement. Deprecated settings are listed in
each product's own documentation, not here — this document describes the policy, not
the inventory.

## Severity definitions

| Severity | Definition | First response |
|---|---|---|
| S1 | Production down, no workaround | 1 hour |
| S2 | Production impaired, workaround exists | 4 business hours |
| S3 | Non-production, or cosmetic | 2 business days |
| S4 | Question or documentation issue | 5 business days |

Severity is set by the reporter and may be adjusted by support with an explanation.
Response time is time to first human response, not time to resolution; no resolution
time is committed to for any severity.

## Data handling during support

Diagnostic bundles may contain configuration, logs, and schema. They do not contain
stored records. Attaching a bundle from one environment to a ticket raised for another
is the most common way customer data ends up somewhere it should not be; check the
environment before attaching.
