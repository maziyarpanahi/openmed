# Federated round scheduling

OpenMed represents a federated round schedule as five strictly ordered UTC
boundaries. The boundaries define contiguous enrollment, update-submission,
aggregation, and evaluation windows without selecting a lifecycle state or
reading the system clock.

## Window semantics

Each phase is a half-open interval: its start is included and the next boundary
is excluded. At an exact boundary, the new phase is active. At `finishes_at`,
no phase remains active.

| Phase | Window |
| --- | --- |
| `enrollment` | `enrollment_starts_at` to `update_submission_starts_at` |
| `update_submission` | `update_submission_starts_at` to `aggregation_starts_at` |
| `aggregation` | `aggregation_starts_at` to `evaluation_starts_at` |
| `evaluation` | `evaluation_starts_at` to `finishes_at` |

`active_phase_at(timestamp)` returns the phase containing the supplied UTC
timestamp. `next_phase_at(timestamp)` returns the first phase with a future
start boundary. Before the schedule starts, enrollment is next; during
evaluation and after finish, there is no next phase.

## Usage

```python
from datetime import datetime, timedelta, timezone

from openmed.training import FederatedRoundSchedule

start = datetime(2026, 9, 1, tzinfo=timezone.utc)
schedule = FederatedRoundSchedule(
    enrollment_starts_at=start,
    update_submission_starts_at=start + timedelta(hours=12),
    aggregation_starts_at=start + timedelta(hours=24),
    evaluation_starts_at=start + timedelta(hours=30),
    finishes_at=start + timedelta(hours=36),
    max_enrollment_duration_seconds=12 * 60 * 60,
)

active = schedule.active_phase_at(start)
payload = schedule.to_json()
restored = FederatedRoundSchedule.from_json(payload)
assert restored == schedule
```

All timestamps must be timezone-aware with a zero UTC offset. Serialization
uses canonical `Z` timestamps and preserves microseconds when present. Naive or
non-UTC timestamps and equal or reversed boundaries are rejected.

## Duration limits

Each phase can have an optional positive integer maximum duration in seconds.
The configured maximum is inclusive, and each phase also has a hard upper bound
of 365 days. These limits reject accidental year-scale windows while allowing a
caller to apply a tighter scheduling policy.

The schema contains only boundaries, duration limits, phase names, and its
version. It has no fields for site or client identities, patient counts, local
metrics, network endpoints, notifications, or retry policy.
