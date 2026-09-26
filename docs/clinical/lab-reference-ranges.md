# Typed laboratory reference-range provenance

Laboratory values are unsafe to compare when the unit, population, precision,
or originating instrument is implicit. OpenMed provides a small offline record
for synthetic reference ranges and resolves ranges only when their provenance
is explicit and unambiguous.

This is assistive metadata for review. It is not a clinical decision, a device
claim, or a substitute for the originating laboratory's report.

## Build a local range

Use a stable local source descriptor. The descriptor is hashed immediately and
is not retained in the range or its serialized provenance.

```python
from openmed.clinical import build_reference_range

adult_sodium = build_reference_range(
    "sodium",
    135,
    145,
    unit="mmol/L",
    population="adult",
    precision=0,
    source={"instrument": "synthetic-analyzer-a", "version": 1},
    locale="en-US",
)

print(adult_sodium.to_dict()["provenance"])
```

The resulting provenance contains `unit`, `population`, `precision`,
`source_fingerprint`, and the optional normalized `locale`. It contains no raw
instrument identifier or clinical note text.

## Resolve without guessing

Pass the target provenance when selecting from multiple local ranges. Matching
is exact: units are not converted, populations are not substituted, and an
instrument or locale is never inferred.

```python
from openmed.clinical import ReferenceRangeStatus, resolve_reference_range

resolution = resolve_reference_range(
    [adult_sodium],
    analyte="SODIUM",
    provenance=adult_sodium.provenance,
)
assert resolution.status is ReferenceRangeStatus.KNOWN

missing_locale = resolve_reference_range(
    [adult_sodium],
    analyte="sodium",
    unit="mmol/L",
    population="adult",
    precision=0,
    source_fingerprint=adult_sodium.provenance.source_fingerprint,
    locale="fr-FR",
)
assert missing_locale.status is ReferenceRangeStatus.UNKNOWN
```

When two explicit candidates match the requested provenance but disagree on
their bounds, the result is `CONFLICT`. When no exact candidate exists, the
result is `UNKNOWN`; callers should request the missing context or keep the
result un-compared. `compare_reference_ranges()` provides the same fail-closed
behavior for a pair of typed ranges.

All functions are deterministic and local-only. Keep committed fixtures
synthetic and use hashes, offsets, or counts rather than patient values.
