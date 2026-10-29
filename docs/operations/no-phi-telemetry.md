# No-PHI telemetry

## Opt-in agent OpenTelemetry bridge

`openmed.agent.telemetry.AgentTelemetry` maps validated `EventAttributes` and
existing `ActionPhase` values to caller-owned OpenTelemetry traces and metrics.
It is disabled by default, including when pipeline telemetry is enabled through
the environment. Disabled telemetry imports no OpenTelemetry modules, touches
no injected sinks or clock, and creates no spans. If the optional OpenTelemetry
API is absent, enabling the adapter without injected sinks remains a no-op.
The bridge creates no SDK provider, processor, reader, exporter, or network path.
Applications may install the existing `otel` extra and configure their own sinks.

```python
from openmed.agent.event_attributes import EventAttributes
from openmed.agent.telemetry import AgentTelemetry

telemetry = AgentTelemetry(enabled=True)  # explicit opt-in; uses global API sinks
event = EventAttributes.from_mapping({
    "sequence_number": 0,
    "execution_stage": "completed",
    "outcome_class": "success",
    "outcome_reason": "completed",
    "duration_ms": 12.5,
})
with telemetry.event_span("completed", event):
    pass  # or wrap the local operation, omitting duration_ms to measure its latency
```

This example emits `openmed.agent.completed` with the five validated event fields
prefixed by `openmed.agent.`. Span names use only the seven existing action
phases. Nested contexts inherit the caller's ambient parent; the bridge does not
invent a run lifecycle, enforce phase transitions, or decide approvals. Review
and denial observations use existing outcome classes and reasons. There is no
new breaker-trip or approval label: producers must use the existing validated
event vocabulary rather than attach decision objects, arguments, or outputs.

Span fields are limited to existing execution stages, outcomes/reasons, digests,
bounded counts/durations, and flags. All correlation and governance identifiers
are omitted by default. `run_id_mode="hash"` replaces the validated opaque run
ID with a `sha256:` digest under `openmed.agent.run_id`, using the domain prefix
`openmed.agent.telemetry.run_id.v1` plus a NUL separator. No raw run ID or other
identifier is emitted. This digest is stable for correlation and is not an
anonymization guarantee; leave the default `omit` mode for aggregate-only traces.
Attributes are revalidated before any span or counter changes. Arbitrary
mappings, unknown phases, and raw payload fields fail with value-free errors.

Each entered context increments `openmed.agent.events` once and records
`openmed.agent.event.duration` in milliseconds when it exits, including on
exceptions. An event's supplied duration is authoritative; otherwise the context
is measured with a monotonic clock (injectable for offline tests). Both metrics
use only the closed phase, execution stage, outcome class/reason, and flags as
dimensions. Counts, digests, identifiers, sequences, and durations never become
metric labels. Exceptions propagate without automatic exception events, stack
traces, or status descriptions. The context yields no raw span for arbitrary
attribute writes. Caller-owned providers still control resources, sampling,
ambient context, and export; the privacy boundary covers bridge-produced fields.

The bridge targets the Python agent event and OpenTelemetry API contracts named
in #3792; it adds no Swift/OpenMedKit execution surface.

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
