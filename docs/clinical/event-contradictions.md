# Clinical event contradiction reports

Clinical event contradiction reports are deterministic review signals for
synthetic or caller-supplied event timelines. They compare typed day-level
event intervals and status assertions, but never select a clinical truth or
trigger a clinical decision.

## Compare intervals and statuses

Use `EventInterval` for a typed interval and `EventStatusAssertion` for a
status tied to a logical entity. Source offsets are inclusive/exclusive
character offsets. Pass a SHA-256 or HMAC-SHA256 fingerprint for the source
span when one is available; an omitted fingerprint receives a deterministic
metadata fingerprint.

```python
from openmed.clinical import (
    EventInterval,
    EventStatusAssertion,
    report_event_contradictions,
)

report = report_event_contradictions(
    [
        EventInterval(
            event_id="event-a",
            event_type="medication_change",
            entity_id="synthetic-entity",
            interval_start="2026-06-01",
            interval_end="2026-06-03",
            source_start=10,
            source_end=25,
            fingerprint="sha256:" + "0" * 64,
        )
    ],
    [
        EventStatusAssertion(
            entity_id="synthetic-entity",
            status="active",
            source_start=30,
            source_end=38,
            fingerprint="sha256:" + "1" * 64,
        )
    ],
)

safe_payload = report.to_dict()
```

The report classifies three review signals:

- `overlap`: two ranges for the same entity share a day;
- `impossible_order`: an interval runs backward, or an explicit sequence,
  start/end type, or `precedes` relation is reversed;
- `conflicting_status`: overlapping or unbounded assertions for one entity
  disagree on an incompatible status such as active versus resolved or
  active versus refuted.

An active assertion followed by a resolved assertion in non-overlapping
intervals is not reported as a contradiction. The comparator preserves all
findings and does not resolve which assertion is correct.

## Privacy and determinism

`report.to_dict()` contains counts, controlled categories, source offsets, and
fingerprints only. It does not contain source text, event values, or interval
dates. Source text supplied in compatible input mappings is used only in
memory to derive a fingerprint and is never copied into the report or an
exception. Keep any caller-owned raw input under the repository's normal PHI
handling policy.

The implementation is rules-only, uses no network service or environment
state, and sorts input records before comparison. Reordering equivalent input
records therefore produces the same serialized report.
