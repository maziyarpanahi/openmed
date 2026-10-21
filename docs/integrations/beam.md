# Apache Beam Redaction Transform

OpenMed provides a small Apache Beam-compatible redaction contract for bounded
batch pipelines. The Beam SDK remains optional; the direct synthetic harness
can validate serialization and retry behavior without installing Beam.

For v3.1, use Python 3.11+ and the validated Beam 2.75/Pillow 13 combination.
Beam currently declares `Pillow<13`, while OpenMed requires the patched Pillow
13 release. It also caps cryptography and httplib2 below the repository's
security floors. A plain `pip install "openmed[beam]"` cannot resolve the Pillow
requirements. Use the same three explicit overrides as the repository in a
dedicated environment; do not downgrade these packages or disable dependency
installation.

```bash
uv venv --python 3.11 .venv
printf '%s\n' 'Pillow==13.0.0' 'cryptography==50.0.0' 'httplib2==0.32.0' \
  > beam-security-overrides.txt
uv pip install --no-config --override beam-security-overrides.txt \
  "openmed[beam]==3.1.0" "apache-beam==2.75.0"
```

The overrides change only these three dependencies. The synthetic DirectRunner
integration tests pass with these versions, but this does not qualify other
Beam runners or image transforms. `pip check` reports Beam's declared conflicts
until upstream widens its bounds. Review any other dependency conflict
separately. See the
[dependency policy](../security/dependency-policy.md#image-dependencies-and-python-compatibility)
for the security floor and Python compatibility details.

## Contract

The transform accepts either a string element or a mapping with a configured
text field. Mapping records keep their outer shape and only the text field is
transformed. The default bounds are 10,000 records, 10 MiB each of serialized
input and output, 1 MiB per input or output record, and three redaction
attempts. Records must contain bounded, JSON-compatible values and string
mapping keys. Cyclic, hostile, or oversized inputs fail with stable value-free
errors before redaction.

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
configured for cache-only loading with credential discovery disabled, mapping
and audit retention off, and the safety sweep enabled, so constructing the
transform does not require network access. Pre-stage the model on each worker
or inject a local `deidentifier` for an air-gapped deployment and for tests.
Additional deidentifier options are snapshotted into a bounded, serializable
configuration; worker loader, policy, method, and safety controls cannot be
overridden through that mapping.

`extra_kwargs` are bounded and copied at specification construction. They may
not override `config`, `loader`, `method`, or `policy`, so they cannot replace
the worker-local loader or the cache-only default configuration. Reports expose
only the option count and a fingerprint of option keys, never raw keys or
values. Options must be data-only values: nulls, booleans, finite numbers,
strings, bytes, and bounded nested lists, tuples, or string-keyed dictionaries.
The snapshot accepts at most 128 top-level options, 4,096 nested values, and
4 MiB of aggregate key, string, and byte data.

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
Redacted records and batches also have bounded expansion budgets, preventing a
redactor from turning a valid input bound into unbounded worker output.
