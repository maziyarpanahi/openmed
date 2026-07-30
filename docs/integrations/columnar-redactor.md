# Columnar Redactor

OpenMed can redact selected free-text columns from Parquet and ORC datasets
while profiling structured quasi-identifiers for residual linkage risk. The
columnar redactor streams Parquet row groups and ORC stripes, so it does not
need to load the full file before writing redacted output.

Install the optional columnar dependency when working with Parquet or ORC:

```bash
uv pip install -e ".[columnar]"
```

## Basic Use

```python
from openmed.integrations.columnar_redactor import redact_columnar_dataset

result = redact_columnar_dataset(
    "patients.parquet",
    text_columns=["clinical_note"],
    quasi_identifier_columns=["age", "zip", "diagnosis_group"],
    output_path="patients.redacted.parquet",
    manifest_path="patients.redaction-manifest.json",
    qi_report_path="patients.qi-report.json",
    progress_path="patients.progress.json",
    batch_size=512,
    low_k_threshold=5,
)
```

The returned `ColumnarRedactionResult` includes:

- `output_path`: the redacted Parquet or ORC file.
- `manifest`: PHI-free redaction counts by text column processing batch, label,
  and row group or stripe.
- `qi_report`: aggregate k-anonymity style equivalence-class counts for the
  configured quasi-identifier columns.
- `progress_path`: an optional checkpoint file updated after each row group or
  stripe.

## Privacy Contract

The manifest, progress file, and QI report are designed for audit and workflow
coordination. They contain counts, labels, row-group indexes, and aggregate
equivalence-class sizes only. They do not include raw text cells, redacted text
cells, or raw quasi-identifier values.

The QI report uses the same quasi-identifier normalization semantics as the
OpenMed risk package, then reports only aggregate class sizes:

```json
{
  "record_count": 100000,
  "quasi_identifier_columns": ["age", "zip", "diagnosis_group"],
  "k_min": 1,
  "low_k_threshold": 5,
  "low_k_class_count": 42,
  "risk_flags": [
    {
      "type": "low_equivalence_class_size",
      "threshold": 5,
      "class_count": 42,
      "record_count": 71,
      "min_class_size": 1
    }
  ],
  "raw_cell_values_included": false
}
```

## Resume Behavior

When `progress_path` is set, the redactor writes a PHI-free checkpoint after
each completed Parquet row group or ORC stripe. If a run is interrupted, call
the function again with `resume=True` and the same `output_path` and
`progress_path`; completed group parts are reused and missing groups are
processed.

```python
result = redact_columnar_dataset(
    "patients.parquet",
    text_columns=["clinical_note"],
    output_path="patients.redacted.parquet",
    progress_path="patients.progress.json",
    resume=True,
)
```

## Notes

- Text columns must be string-typed.
- If `quasi_identifier_columns` is omitted, every non-text column is profiled.
- Quasi-identifiers are reported only; the redactor does not generalize,
  suppress, or transform structured QI values.
- The redacted output preserves the original columnar schema.
