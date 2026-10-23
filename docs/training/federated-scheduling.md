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

## Clock skew tolerance

Exact boundaries are deterministic for one clock, but sites and the coordinator
do not share one. Schema `openmed.training.federated_schedule.v2` adds a
required `max_clock_skew_seconds` field: the largest tolerated difference
between any site clock and the coordinator clock. The coordinator clock is
authoritative; site clocks are advisory.

The bound is an integer from 0 to 900 seconds (15 minutes). Negative, larger,
fractional, non-finite, boolean, and missing values are rejected. A larger
bound would hide an unsynchronized site rather than tolerate drift, so fix the
site's time source instead. Every phase must also last longer than twice the
bound, so each phase keeps a non-empty site window.

Two checks consume the schedule. Neither reads the system clock; callers pass
readings from their injected clock sources.

| Check | Clock | Admits when | Refusal reason codes |
| --- | --- | --- | --- |
| `client_admission_at(phase, local_time)` | site, advisory | `start + skew <= local_time < end - skew` | `local_clock_before_window`, `local_clock_after_window` |
| `coordinator_admission_at(phase, received_at, coordinator_sequence=n)` | coordinator receipt, authoritative | `start <= received_at < end` | `received_before_window`, `received_after_window` |

A site whose clock is within the bound and reads inside the shrunk window
therefore starts only when the phase is open on the coordinator clock. A site
must check its decision before training and stop on any refusal. The
coordinator ignores site-reported times and the skew bound; it refuses a late
arrival by receipt time only, even if the site started in time. The client
check requires a v2 schedule, while the coordinator check works for v1 and v2.

```python
from openmed.training import (
    FEDERATED_SCHEDULE_V2_SCHEMA_VERSION,
    FederatedRoundSchedule,
    FederatedSchedulePhase,
    order_admission_evidence,
)

schedule = FederatedRoundSchedule(
    enrollment_starts_at=start,
    update_submission_starts_at=start + timedelta(hours=12),
    aggregation_starts_at=start + timedelta(hours=24),
    evaluation_starts_at=start + timedelta(hours=30),
    finishes_at=start + timedelta(hours=36),
    schema_version=FEDERATED_SCHEDULE_V2_SCHEMA_VERSION,
    max_clock_skew_seconds=120,
)

phase = FederatedSchedulePhase.UPDATE_SUBMISSION
site_clock = lambda: start + timedelta(hours=12, minutes=1)
decision = schedule.client_admission_at(phase, site_clock())
assert decision.reason_code == "local_clock_before_window"
assert not decision.admitted  # refuse before training

coordinator_clock = lambda: start + timedelta(hours=12, minutes=5)
received = schedule.coordinator_admission_at(
    phase, coordinator_clock(), coordinator_sequence=1
)
evidence = order_admission_evidence([received])
```

Admission decisions are shareable status and evidence records. Their
`to_dict()` and `to_json()` output contains only `admitted`, `clock_basis`
(`site_advisory` or `coordinator_receipt`), `coordinator_sequence`, `phase`,
and `reason_code`. They never retain the local or receipt time, a clock
offset, or a site identity. `order_admission_evidence` orders coordinator
decisions by coordinator sequence and rejects site decisions and duplicate
sequences, because site timestamps cannot order events.

## Schema versions

Schema v1 is unchanged: it has no skew field, parses and serializes
byte-for-byte as before, and rejects `max_clock_skew_seconds`. Schema v2 adds
only the top-level `max_clock_skew_seconds` field. A payload whose fields do
not match its declared version is rejected.

The schema contains only boundaries, duration limits, the skew bound, phase
names, and its version. It has no fields for site or client identities,
patient counts, local metrics, network endpoints, notifications, or retry
policy.
