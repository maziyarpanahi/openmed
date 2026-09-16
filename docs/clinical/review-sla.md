# Human-review queue SLA reports

`openmed.clinical.review_sla` provides a local, deterministic summary of a
human-review queue. It is an operational review aid, not a compliance
certification, medical device, or clinical decision mechanism.

## Deterministic classification

Always inject `now` (or a callable `clock`). The module never consults the
system clock and performs no network calls. Queue entries need a synthetic
`case_key` and `queued_at` timestamp. Priority defaults to `normal` and uses
these local target durations when an entry does not provide `expires_at` or
`sla`:

| Priority | Target |
| --- | ---: |
| `urgent` | 4 hours |
| `high` | 8 hours |
| `normal` | 24 hours |
| `low` | 72 hours |

Age buckets are `0-1h`, `1-4h`, `4-24h`, and `24h+`. Expiry is classified as
`expired`, `due-within-4h`, or `due-after-4h`; overdue time is classified as
`on-time`, `0-24h-overdue`, or `24h+-overdue`. An explicit `expires_at` takes
precedence over the priority target.

```python
from datetime import datetime, timezone

from openmed.clinical import build_review_sla_report

as_of = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
report = build_review_sla_report(
    [
        {
            "case_key": "synthetic-case-a",
            "queued_at": "2026-08-11T11:30:00+00:00",
            "priority": "urgent",
        },
        {
            "case_key": "synthetic-case-b",
            "queued_at": "2026-08-10T08:00:00+00:00",
            "priority": "normal",
        },
    ],
    now=as_of,
)

print(report.to_json())
```

The rendered report contains only the injected timestamp, total count, and
bucket counts. It contains no case keys, case contents, or reviewer
identities. When a detailed classification is needed for local routing,
`compute_review_sla` returns records whose `case_key` is a stable SHA-256
value. The raw key is never serialized by this module.

```python
from openmed.clinical import compute_review_sla

records = compute_review_sla(
    [{
        "case_key": "synthetic-case-a",
        "queued_at": as_of,
    }],
    now=as_of,
)
assert records[0].case_key.startswith("sha256:")
```

Use only synthetic or already-authorized opaque keys in committed fixtures and
avoid putting case data in logs or exception messages. The report is intended
for capacity and queue-management review; it does not determine a clinical
action or guarantee a regulatory SLA.
