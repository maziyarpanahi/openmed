# Core Pipeline Observability

OpenMed can expose the ten privacy-pipeline stages as OpenTelemetry spans and
aggregate metrics without placing clinical text or detected identifiers in the
observability stream. This capability is optional, local-first, and disabled
by default.

The core implementation never creates an SDK provider, processor, reader, or
exporter. Installing or enabling it therefore does not create a network
destination. Your application owns the OpenTelemetry providers and any export
policy.

## Install and opt in

Install the optional API and SDK support:

```bash
uv pip install -e ".[otel]"
```

Enable the pipeline explicitly in code:

```python
from openmed.core.pipeline import Pipeline

pipeline = Pipeline(telemetry_enabled=True)
```

Or enable newly constructed pipelines through the environment:

```bash
OPENMED_TELEMETRY_ENABLED=true python your_local_pipeline.py
```

An explicit `telemetry_enabled=False` constructor argument overrides the
environment flag. Unset, empty, `0`, `false`, `no`, `off`, and `disabled` values
keep telemetry off. An unrecognized non-empty value raises `ValueError` so a
deployment cannot silently select an ambiguous privacy posture.

Opting in without the `otel` extra is safe: OpenMed falls back to a no-op
backend. `PipelineResult.stage_durations_ms` remains available because it is an
in-process return value, not an exporter or background telemetry channel.

## Supply caller-owned providers

With no provider configured, the OpenTelemetry API returns non-recording global
objects. To collect signals, inject providers owned by your application. This
local example writes only to the process console:

```python
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openmed.core.pipeline import Pipeline
from openmed.core.telemetry import PipelineTelemetry

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
meter_provider = MeterProvider(metric_readers=[metric_reader])

telemetry = PipelineTelemetry(
    enabled=True,
    tracer=trace_provider.get_tracer("clinic.openmed"),
    meter=meter_provider.get_meter("clinic.openmed"),
)
pipeline = Pipeline(telemetry=telemetry)

# Run the pipeline with a locally available detector/model configuration.
# result = pipeline.run(note, method="mask")

# The application that created the providers also owns their shutdown.
trace_provider.shutdown()
meter_provider.shutdown()
```

If your application has already registered global providers, the environment
opt-in uses their tracer and meter. OpenMed does not register or replace global
providers.

## Signal contract

One span is created for each fixed stage:

1. `normalize`
2. `language_script`
3. `doc_type_section`
4. `deterministic_detectors`
5. `fast_pii_model`
6. `clinical_phi_model`
7. `span_arbitration`
8. `policy_actions`
9. `safety_sweep`
10. `emit`

Span names use `openmed.pipeline.<stage>`. Their attributes are restricted to:

- the fixed stage name and one-based index;
- non-negative span and entity counts;
- canonical OpenMed label sets, such as `PERSON`, `PHONE`, or `ID_NUM`;
- aggregate minimum/maximum character offsets;
- input or redacted-output character counts where relevant;
- the stage duration in milliseconds; and
- a boolean failure marker.

OpenMed records three histograms. Metric attributes contain only the fixed stage
name and index, which keeps labels bounded and PHI-free.

| Metric | Unit | Meaning |
| --- | --- | --- |
| `openmed.pipeline.stage.duration` | `ms` | Stage wall-clock duration |
| `openmed.pipeline.stage.span_count` | `1` | Canonical spans emitted by the stage |
| `openmed.pipeline.stage.entity_count` | `1` | Entities represented by the stage output |

The duration attribute, duration histogram, and
`PipelineResult.stage_durations_ms` share the same
`openmed.utils.profiling.Timer` measurement. Observability does not run a
second clock around the stage.

## No-PHI boundary

The telemetry API accepts no generic string attributes. Stage names are fixed,
and label values must exist in OpenMed's canonical taxonomy. The core path does
not emit:

- source, normalized, redacted, replacement, or detected entity text;
- document, patient, request, or surrogate identifiers;
- raw model output, prompts, mappings, or arbitrary metadata;
- exception messages, exception events, or stack traces; or
- exporter endpoints, headers, credentials, or user-defined resource data.

Failures add only `openmed.stage.failed=true` before the original exception is
re-raised. This avoids the common OpenTelemetry default of recording an
exception message and stack trace, either of which may contain clinical text.

The application remains responsible for any attributes it adds outside
OpenMed, its OpenTelemetry resource fields, retention, access control, and the
security of a separately configured exporter.

## Verify the contract

The regression suite uses synthetic identifiers with in-memory trace and
metric readers:

```bash
.venv/bin/python -m pytest \
  tests/unit/core/test_pipeline_telemetry.py \
  tests/unit/test_no_telemetry.py -q
```

For HTTP request spans, incoming trace context, and the separately configured
service exporter, see [REST Tracing](serving/tracing.md). Core pipeline
observability and REST tracing have independent opt-in flags.
