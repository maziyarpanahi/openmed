# No-PHI telemetry

`openmed.core.no_phi_telemetry.NoPHITelemetryExporter` provides a small,
local-first telemetry boundary for pipeline health and latency. It stores
aggregate counters and fixed-bucket latency histograms in memory. `export()`
returns a plain dictionary, and `export_json()` returns canonical JSON; both
are formatting operations only and make no network call.

## Safe recording

Use the typed pipeline method when possible:

```python
from openmed.core.no_phi_telemetry import NoPHITelemetryExporter

telemetry = NoPHITelemetryExporter()
telemetry.record_pipeline(
    stage="emit",
    status="success",
    method="mask",
    latency_ms=18.4,
    entity_count=2,
)

snapshot = telemetry.export()
```

The exporter exposes only fixed counter families for runs, failures,
rejections, and aggregate entity counts. Latency is exported as a fixed-bucket
histogram in seconds. The only dimensions are `stage`, `status`, `method`, and
`exception_category`. Each dimension has a finite allowlist; an unrecognized
value is recorded as `other` (or `unknown` for exception categories) rather
than becoming a new label.

The mapping-based `record()` method accepts only its documented schema. An
unapproved event or dimension key raises `UnapprovedTelemetryKeyError`, with a
generic message that does not echo the rejected key or value. Prompts, entity
text, request bodies, model identifiers, and arbitrary exception messages are
not accepted as telemetry fields.

Event mappings and pipeline-result stage timings are copied through bounded,
closed schemas before any counter changes. A rejected event therefore cannot
leave a partial update. Counter values use a signed 64-bit ceiling, entity
counts are capped at 10 million per call, pipeline-result timing maps at 64
stages, and individual latency observations at seven days. Public snapshot
types validate metric names, dimensions, bucket consistency, and canonical
ordering before serialization.

`record_pipeline_result()` interprets its documented `stage_durations_ms`
attribute as milliseconds and converts both stage and aggregate observations
to seconds. It reads no other result attributes beyond the length of `spans`.

## Exception handling

Pass an exception instance or class through `exception=`. Categories are
derived from the exception type only: the message is never read. Supported
categories include `validation`, `timeout`, `dependency`, `network`,
`capacity`, `cancelled`, and `internal`; unknown inputs map to `unknown`.

`render_prometheus()` formats the same allowlisted snapshot for a caller-owned
collector. It does not configure or contact a collector itself. This utility
provides an aggregate telemetry contract; it is not a compliance
certification or a clinical decision guarantee.

## Federated round telemetry

`openmed.core.federated_telemetry.FederatedRoundTelemetry` records federated
round phases on the same exporter. Every dimension value must come from a closed
vocabulary, so a client pseudonym, a filesystem path, a digest, or any other
free-form string is refused with `UnapprovedFederatedValueError` instead of
becoming a new label. The error names the field only and never echoes the
rejected value.

```python
from openmed.core.federated_telemetry import FederatedRoundTelemetry

telemetry = FederatedRoundTelemetry()
telemetry.record_phase_transition(phase="preflight")
telemetry.record_rejection(reason_code="quorum_not_met", phase="held")
telemetry.record_update_count(count=7, phase="aggregating")
telemetry.record_update_count(count=40, phase="aggregating", exact=True)
telemetry.observe_phase_latency(phase="aggregating", milliseconds=850.0)
```

Counters are `openmed_federated_phase_transitions_total{phase,status}`,
`openmed_federated_phase_rejections_total{phase,reason_code}`, and
`openmed_federated_phase_updates_total`. Phase and status values mirror
`FederatedRoundState` and the pipeline status vocabulary, and `reason_code`
mirrors `FederatedRoundReasonCode`. Phase latency is exported as
`openmed_federated_phase_latency_seconds{phase,status}` with the exporter's
fixed buckets.

Update counts stay banded. The default call advances the counter by one and
labels the sample with `update_band`, using the shared `suppressed`,
`minimum_to_under_double`, `double_to_under_fourfold`, and `fourfold_or_more`
bands relative to the minimum group size of 5. An exact aggregate is only
released with `exact=True`, which advances the counter by the count and omits
the band label; a count below the minimum group size raises
`UnapprovedFederatedValueError` rather than being rounded. A zero count records
nothing. `band_update_count()` exposes the same banding for callers that only
need the label.

The recorder owns no transport, endpoint, or background thread; its constructor
takes only an optional exporter and minimum group size. Exports remain local
formatting operations.

## Service-wide operational events

`openmed.service.operational.OperationalEvent` is the shared contract for
ingestion, model, store, job, queue, query, and export observations. Category,
operation, and state are closed enums; counts and durations are bounded
aggregates. The service Prometheus registry renders these as
`openmed_service_operational_total` and
`openmed_service_operational_duration_seconds`. The same validated event can
produce OpenTelemetry attributes with `operational_trace_attributes()`.

States preserve `partial`, `unknown`, `conflict`, `unsupported`, `denied`, and
`failure` instead of folding them into success. `OperationalAlert` carries only
the category, operation, state, observed count, threshold, and time window. It
has no field for source values, record identifiers, model input, or reviewer
identity.

The operational contract is versioned as `1.0.0` with `same_major`
compatibility. A caller cannot introduce arbitrary metric labels or trace
attributes by passing a free-form operation name. Committed JSON Schemas cover
operational events, alerts, and limit decisions, and are validated in the
offline test suite.

## Pre-work resource limits

`openmed.guard.operational_limits` provides one versioned policy for request
bytes, JSON depth and node count, string bytes, pagination, archive entry
count, declared uncompressed bytes, and compression ratio. ZIP inspection
reads central-directory metadata only and does not open or decompress members.
Encrypted, malformed, traversing, linked, duplicate, oversized, or
high-expansion archives fail closed with counts-only decisions.

The REST and GraphQL application installs a body-size middleware for `POST`,
`PUT`, and `PATCH`. `OPENMED_SERVICE_MAX_REQUEST_BYTES` may lower or raise the
default within the library's safe ceiling. Declared sizes are rejected before
the body is read; streamed bodies are counted and rejected before route parsing
or model execution.
