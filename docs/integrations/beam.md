# Apache Beam Redaction Transform

OpenMed provides a small Apache Beam-compatible redaction contract for bounded
batch pipelines. The Beam SDK remains optional; the direct synthetic harness
can validate serialization and retry behavior without installing Beam.

```bash
pip install "openmed[beam]"
```

## Contract

The transform accepts either a string element or a mapping with a configured
text field. Mapping records keep their outer shape and only the text field is
transformed. The default bounds are 10,000 records, 10 MiB of serialized input,
1 MiB per record, and three redaction attempts.

```python
import apache_beam as beam

from openmed.interop.beam import BeamRedactionSpec, BeamRedactionTransform

spec = BeamRedactionSpec(
    text_field="note",
    policy="hipaa_safe_harbor",
    max_records=10_000,
)

with beam.Pipeline() as pipeline:
    redacted = (
        pipeline
        | beam.Create([{"record_id": "synthetic-1", "note": "synthetic note"}])
        | BeamRedactionTransform(spec)
    )
```

Workers create their model loader locally. The default OpenMed deidentifier is
configured for cache-only loading, so constructing the transform does not
require network access. Pre-stage the model on each worker or inject a local
`deidentifier` for an air-gapped deployment and for tests.

## Direct synthetic harness

`run_synthetic_harness` applies the same schema validation, canonical JSON
serialization, bounds, and capped retry loop without starting a Beam runner:

```python
from openmed.interop.beam import BeamRedactionSpec, run_synthetic_harness

result = run_synthetic_harness(
    [{"note": "synthetic note"}],
    spec=BeamRedactionSpec(text_field="note"),
    deidentifier=my_local_deidentifier,
)
print(result.redacted_records)
print(result.report())
```

The report contains only schema metadata, SHA-256 fingerprints, and counters
for records, attempts, retries, bytes, and redacted spans. Input values,
output values, model exceptions, record identifiers, and deidentifier return
objects are never copied into reports, logs, or contract exceptions. Retry
backoff is bounded and disabled by default for deterministic direct runs.
