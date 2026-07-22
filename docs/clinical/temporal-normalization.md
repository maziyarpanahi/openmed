# Temporal Normalization

Clinical notes mix absolute dates with expressions such as `3 weeks ago`,
`since last March`, `POD 2`, and `q6h x5 days`. The temporal normalizer converts
caller-supplied spans into deterministic TIMEX3-style `DATE`, `TIME`,
`DURATION`, and `SET` records with ISO 8601 values.

## Normalize temporal spans

```python
from openmed.clinical import normalize_temporal

text = "Symptoms began 3 weeks ago; continue q6h x5 days."
spans = [
    {"start": 15, "end": 26},
    {"start": 37, "end": 48},
]

records = normalize_temporal(
    text,
    spans,
    reference_time="2026-06-15T10:30:00Z",
)

records[0].to_dict()
# {
#   "text": "3 weeks ago",
#   "span": [15, 26],
#   "start": 15,
#   "end": 26,
#   "type": "DATE",
#   "value": "2026-05-25",
#   "anchor": "2026-06-15T10:30:00+00:00",
#   "granularity_flags": ["day"],
# }

records[1].value
# "R20/PT6H"
```

Spans can also be two-item `(start, end)` sequences. Results preserve input
order and use inclusive start/exclusive end offsets, so
`text[result.start:result.end] == result.text` always holds.

## Values and anchors

| Type | Examples | Value form |
|---|---|---|
| `DATE` | `2026-06-01`, `3 weeks ago`, `last March`, `POD 2` | ISO date, month, year, or datetime |
| `TIME` | `14:30`, `2:30 PM` | ISO datetime when anchored; ISO time when unanchored |
| `DURATION` | `for 5 days`, `2 weeks` | ISO duration such as `P5D` or `P2W` |
| `SET` | `daily`, `q6h`, `q6h x5 days` | ISO repeating interval such as `R/P1D` or `R20/PT6H` |

For relative expressions, `anchor` records the normalized reference time used
for date arithmetic. `reference_time` accepts an ISO string, `date`,
`datetime`, or `None`. Passing `None` never substitutes the current time.

`since X` stores the normalized start in `value`, the supplied document time
as `anchor`, and adds the `since` flag. This keeps the value layer separate
from interval and event ordering, which belong to the timeline resolver.

## Granularity and ambiguity

`granularity_flags` identifies the supported precision (`second`, `minute`,
`hour`, `day`, `week`, `month`, `year`, or `part_of_day`) and adds qualifiers
when needed:

- `unanchored`: a relative date has no supplied reference time, or a time has
  no document date.
- `ambiguous`: the source permits multiple interpretations, such as
  `03/04/2026`; the normalizer returns `value=None` instead of choosing a
  locale convention.
- `approximate`: the source explicitly says `about`, `around`, `roughly`, or
  `approximately`.
- `bounded`: a recurring set includes a duration, such as `q6h x5 days`.
- `since`: the value is the beginning of a since-interval whose end is the
  recorded anchor.

Month-only expressions without `last`, `next`, or an explicit year remain
ambiguous. Numeric dates with two-digit years, impossible dates, zero-length
recurrence periods, inexact bounded schedules, and vague quantities are
likewise flagged rather than coerced into precise timestamps.

## Privacy and determinism

Normalization is pure, offline, and rules-based. The module performs no
network access, reads no environment state, emits no logs, and never defaults
to the wall clock. Temporal spans can contain sensitive dates, so callers
should apply the same PHI handling policy to returned values as to the source
document.
