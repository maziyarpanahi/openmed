# Evidence recency labels

Evidence recency is a disclosure signal for guarded clinical claims. It makes
the age of supporting evidence visible without turning age into a clinical
decision or removing the requirement for qualified human review.

The local-only classifier assigns one of four controlled labels:

| Label | Meaning | Additional recency review |
| --- | --- | --- |
| `current` | The evidence is within the configured current window. | No additional recency flag; guarded clinical review still applies. |
| `stale` | The evidence is older than the current window. | Yes. |
| `future-dated` | The evidence timestamp is later than the fixed reference time. | Yes. |
| `unknown` | No trustworthy evidence or reference timestamp is available. | Yes. |

## Classify one timestamp

Always provide a fixed `as_of` value when a reproducible result is needed.
OpenMed does not substitute the machine clock, call a network service, or
guess when a timestamp is missing.

```python
from datetime import timedelta

from openmed.clinical.evidence_recency import (
    EvidenceRecencyPolicy,
    classify_evidence_recency,
)

policy = EvidenceRecencyPolicy(
    current_window=timedelta(days=30),
    future_tolerance=timedelta(0),
)

label = classify_evidence_recency(
    "2026-01-15T00:00:00Z",
    as_of="2026-02-01T00:00:00Z",
    policy=policy,
)
assert label.value == "current"
```

ISO-8601 strings, `datetime`, and `date` values are accepted. Invalid,
missing, or unsupported evidence timestamps produce the explicit `unknown`
label. Invalid reference timestamps also fail closed to `unknown`; they never
fall back to the current time.

The current window and allowed future tolerance can also be supplied as a
value-free mapping. Numeric `*_days` and `*_seconds` fields are normalized
locally:

```python
policy = EvidenceRecencyPolicy.from_value(
    {
        "stale_after_days": 7,
        "future_tolerance_seconds": 3600,
    }
)
```

Evidence exactly on the current-window boundary is `current`. Evidence older
than that boundary is `stale`. A future timestamp is `future-dated` unless it
falls within the configured tolerance.

## Build a value-free report

Reports accept timestamps directly or mappings/objects with a supported field
such as `evidence_timestamp`, `timestamp`, `observed_at`, or `occurred_at`.
Other fields—including claim text, identifiers, and source metadata—are
ignored and never serialized.

```python
from openmed.clinical.evidence_recency import build_evidence_recency_report

report = build_evidence_recency_report(
    [
        {"evidence_timestamp": "2026-01-15T00:00:00Z", "claim": "synthetic"},
        {"claim": "timestamp unavailable"},
    ],
    as_of="2026-02-01T00:00:00Z",
)

print(report.to_json())
```

The report contains only controlled labels, aggregate counts, thresholds, and
fixed review flags. It does not contain timestamps, source text, identifiers,
or arbitrary input metadata. `stale`, `future-dated`, and `unknown` records
request additional recency review; a `current` label does not waive the
broader guarded-clinical human-review requirement.

The implementation uses only the Python standard library and performs no
mandatory network call. The labels are an assistive disclosure aid, not a
compliance certification, freshness guarantee, diagnosis, treatment decision,
or autonomous clinical action.
