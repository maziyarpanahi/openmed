# Dataflow bundle processor

OpenMed's Apache Beam transform de-identifies a configured field in bounded
batches and reuses one resident processor for each worker-side `DoFn` instance.
It uses `BatchElements` rather than grouping on record content, so PHI and
quasi-identifiers never become shuffle or partition keys.

Install Apache Beam alongside OpenMed, then run the same graph locally with
Beam's DirectRunner before submitting it to Dataflow:

```bash
pip install --upgrade "openmed" "apache-beam>=2.60,<3"
```

The integration is tested with Python 3.10-3.12. Apache Beam remains a separate
runtime dependency so it does not enlarge OpenMed's core dependency surface.

```python
import apache_beam as beam

from openmed.integrations.dataflow_processor import DataflowBatchProcessor

records = [
    {
        "record_id": "synthetic-001",
        "note": "Patient Jane Roe called 555-0101.",
        "facility": "example-clinic",
    }
]

with beam.Pipeline(runner="DirectRunner") as pipeline:
    outputs = (
        pipeline
        | "Create records" >> beam.Create(records)
        | "De-identify notes"
        >> DataflowBatchProcessor(
            text_field="note",
            policy="hipaa_safe_harbor",
            batch_size=32,
        )
    )
    cleaned = outputs.cleaned
    dead_letters = outputs.dead_letter
```

`cleaned` contains copies with only the configured field replaced.
`dead_letters` contains `DataflowDeadLetter` values with the original element
and a PHI-free reason code. Route that collection to a protected sink; do not
log or publish it. Empty strings pass through unchanged, while `None` elements,
missing fields, null fields, non-string fields, batch failures, and malformed
batch results go to the dead-letter output without failing the bundle.

The default processor uses OpenMed's batched de-identification path with one
resident `BatchProcessor` created during Beam `DoFn.setup()`. Dataflow may create
multiple `DoFn` instances for parallelism or worker recycling, so model setup is
once per instance rather than once for the entire job.
