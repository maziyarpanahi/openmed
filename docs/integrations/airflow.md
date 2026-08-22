# Airflow Redaction Operator

OpenMed provides an optional Airflow operator for bounded, local-first
redaction of one UTF-8 file or one in-memory record batch. Install the extra
when the Airflow runtime is available:

```bash
pip install "openmed[airflow]"
```

The operator does not make a network request itself. The default OpenMed
deidentifier is configured for cache-only model loading; pre-stage the model
on each worker or inject a local deidentifier/loader explicitly.

## File input

File tasks require an explicit output path. Plain files are treated as one
text value. Files ending in `.jsonl` or `.ndjson` are treated as record batches;
`.json` files may contain one record, one string, or a list of records.

```python
from openmed.interop.airflow import OpenMedRedactionOperator

redact_notes = OpenMedRedactionOperator(
    task_id="redact_notes",
    input_path="staged/notes.jsonl",
    output_path="staged/notes.redacted.jsonl",
    text_field="note",
    max_input_bytes=10 * 1024 * 1024,
    max_records=10_000,
)
```

Only the configured text field is transformed in structured records. Other
fields are copied as part of the requested output and should be selected with
the same privacy review as any other workflow output.

File and sidecar reads require stable regular files. Final symlinks, special
files, and files replaced or modified while they are being read fail closed.
Output size is limited to eight times the configured input-byte bound so a
faulty injected redactor cannot create an unbounded task artifact.

## Record batches

Record batches may contain strings or mappings. Mappings must contain the
configured text field. An output path is always required so Airflow's task
result and XCom contain only counts and fingerprints, never record content.
For production PHI, stage a bounded file on the worker rather than embedding
record values in a serialized DAG definition or other control-plane metadata.

```python
redact_batch = OpenMedRedactionOperator(
    task_id="redact_batch",
    records=[{"text": "synthetic note"}, {"text": "another note"}],
    output_path="staged/batch.redacted.jsonl",
    text_field="text",
)
```

The default bounds are 10 MiB per file and 10,000 records. Lower them for
smaller task contracts. Oversized or malformed inputs fail before any output
is written.

## Retry and privacy behavior

Outputs are written atomically. A file output receives a companion
`<output>.openmed-fingerprint.json` sidecar unless `fingerprint_path` is set.
The sidecar contains only schema metadata, SHA-256 fingerprints, counts, and
byte sizes. On a retry, a matching input/configuration fingerprint and a
verified output fingerprint produce a `status="skipped"` result without
running the redactor again. A mismatched existing sidecar fails closed rather
than overwriting an output belonging to another run.

Additional deidentifier options are snapshotted when the operator is created
and restored afresh for every value. The retry fingerprint covers that exact
snapshot plus the callback code and serializable captured state, so changing
a custom redactor or its configuration cannot silently reuse an older output.

Task results, XCom, logs, and failure messages contain only operation metadata
and fingerprints; they do not include paths, input values, output text, or the
original exception text from a deidentifier.
