# Conservative temporal intervals

`openmed.clinical.temporal_intervals` normalizes caller-supplied synthetic or
clinical temporal spans locally. It handles date, clock-time, duration, and
finite or open-ended interval forms while retaining the exact half-open source
offsets.

## Basic use

```python
from openmed.clinical.temporal_intervals import normalize_temporal_intervals

text = "Symptoms since 2024-01-01; review from 2024-02 to present."
spans = [
    (text.index("since"), text.index("since") + len("since 2024-01-01")),
    (text.index("from"), text.index("present") + len("present")),
]

records = normalize_temporal_intervals(text, spans)
records[0].value  # "2024-01-01/.."
records[1].value  # "2024-02/.."
```

`normalize_temporal_interval(text, span)` is the one-span form. Omitting
`spans` normalizes the complete input string. Spans are inclusive at `start`
and exclusive at `end`, and the returned `source_span` is always the original
pair of offsets.

## Conservative values

Each `TemporalInterval` exposes `kind`, `value`, `precision`,
`timezone_state`, `unknown_components`, and `conflicts`. A scalar date, time,
or duration uses its `start` component; an interval uses `start` and `end`.
Open bounds are `None` and are represented canonically by `..` in `value`.
The normalizer never substitutes the current date/time for `present`,
`ongoing`, or a missing timezone.

Examples of intentional outcomes:

| Input | Result |
| --- | --- |
| `2026-04` | month precision, `2026-04` |
| `03/04/2026` | `value=None`, conflict `date_order` |
| `14:30` | minute precision, timezone state `unknown` |
| `14:30+02:00` | explicit timezone state |
| `for 2 hours 30 minutes` | `PT2H30M`, minute precision |
| `since 2024-01-01` | `2024-01-01/..`, open end |
| `2024-03-01/2024-02-01` | `value=None`, conflict `interval_order` |

Naive clock times are retained as wall-clock values but carry an explicit
`timezone_state="unknown"`; no machine-local timezone is applied. Partial
dates retain their source precision. A numeric date whose month/day ordering
cannot be established is marked conflicting rather than assigned a locale.

## Privacy and determinism

Normalization is rules-based, offline, and deterministic. It does not read
environment state, consult a timezone database, use the wall clock, emit logs,
or make network calls. `to_dict()` includes offsets and structured metadata but
does not include the raw source surface. Callers can therefore attach the
result to an audit record without duplicating note text; any source text shown
to a reviewer should still be handled under the caller's PHI policy.

This is assistive normalization metadata, not a diagnosis, treatment
recommendation, or clinical decision.
