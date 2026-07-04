# Lakehouse Table Redaction

OpenMed can redact selected text columns across a local lakehouse-style table
without mutating the source data files. The lakehouse redactor scans Parquet
data files under a table root, preserves the relative partition layout, and
writes a new OpenMed-managed snapshot under `_openmed_lakehouse/snapshots/`.

Install the optional columnar dependency before running lakehouse jobs:

```bash
uv pip install -e ".[columnar]"
```

## Basic Use

```python
from openmed.integrations.lakehouse_redact import redact_lakehouse_table

result = redact_lakehouse_table(
    "tables/patient_notes",
    text_columns=["clinical_note", "discharge_summary"],
    snapshot_id="remediate-2026-07",
    batch_size=512,
)
```

The returned `LakehouseRedactionResult` includes:

- `snapshot_path`: the redacted snapshot data directory.
- `manifest`: PHI-free counts, labels, and span offsets.
- `manifest_path`: the written redaction manifest.
- `checkpoint_path`: the per-partition resume checkpoint.
- `metadata_path`: snapshot metadata for the OpenMed-managed version.

Delta Lake and Iceberg metadata directories are ignored during the scan. The
current implementation operates on the Parquet data files available under the
table root; catalog provisioning and metastore commits remain out of scope.

## Dry Run

Use `dry_run=True` to measure impact before writing a new snapshot:

```python
plan = redact_lakehouse_table(
    "tables/patient_notes",
    text_columns=["clinical_note"],
    dry_run=True,
)

print(plan.manifest["affected_rows"])
print(plan.manifest["affected_columns"])
```

Dry runs call the same de-identification path to count affected rows, columns,
labels, and offsets, but they do not create `_openmed_lakehouse`, manifests,
checkpoints, or redacted data files.

## Resume Behavior

The redactor checkpoints after each completed partition. If a run is
interrupted, rerun with the same `snapshot_id` and `resume=True`:

```python
result = redact_lakehouse_table(
    "tables/patient_notes",
    text_columns=["clinical_note"],
    snapshot_id="remediate-2026-07",
    resume=True,
)
```

Completed partitions with existing snapshot files are reused. Missing or
interrupted partitions are processed again and then recorded in the checkpoint.

## Privacy Contract

The manifest and checkpoint contain counts, labels, row offsets, and digests
for partitions and files. They do not include raw text cells, redacted text
cells, raw partition values, source table paths, or source file names.

The snapshot data preserves the table's existing relative partition layout so
downstream engines can point at the new version. The output is irreversible
redaction; re-identification and reversible tokenization are not supported.
