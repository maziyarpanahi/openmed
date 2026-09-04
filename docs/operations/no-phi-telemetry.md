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

## Exception handling

Pass an exception instance or class through `exception=`. Categories are
derived from the exception type only: the message is never read. Supported
categories include `validation`, `timeout`, `dependency`, `network`,
`capacity`, `cancelled`, and `internal`; unknown inputs map to `unknown`.

`render_prometheus()` formats the same allowlisted snapshot for a caller-owned
collector. It does not configure or contact a collector itself. This utility
provides an aggregate telemetry contract; it is not a compliance
certification or a clinical decision guarantee.
