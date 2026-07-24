# Batch Processing

OpenMed provides batch processing capabilities for efficiently analyzing multiple texts or files with progress reporting and result aggregation.

## Quick Start

```python
from openmed import BatchProcessor, process_batch

# Simple batch processing
texts = [
    "Patient has diabetes mellitus type 2.",
    "Acute lymphoblastic leukemia diagnosed.",
    "No significant findings.",
]

result = process_batch(texts, model_name="disease_detection_superclinical")

print(f"Processed: {result.successful_items}/{result.total_items}")
print(f"Total time: {result.total_processing_time:.2f}s")
```

## BatchProcessor Class

For more control over batch processing:

```python
from openmed import BatchProcessor

processor = BatchProcessor(
    model_name="disease_detection_superclinical",
    batch_size=16,
    confidence_threshold=0.5,
    group_entities=False,  # preserve separate BIO entities in diagnosis lists
    continue_on_error=True,  # Don't stop on individual failures
)

# Process texts
result = processor.process_texts(texts)

# Process files
result = processor.process_files(["/path/to/file1.txt", "/path/to/file2.txt"])

# Process directory
result = processor.process_directory(
    "/path/to/notes/",
    pattern="*.txt",
    recursive=True,
)
```

## Crash-safe checkpoints and resume

Long-running batches can durably commit progress every N items. A checkpoint
contains only hashes, item indexes and statuses, committed-output byte offsets,
and output hashes. It never stores raw input text, source paths, model output,
or entity values.

```python
from pathlib import Path

from openmed import BatchProcessor

processor = BatchProcessor(
    model_name="disease_detection_superclinical",
    checkpoint_interval=25,
)

result = processor.process_texts(
    texts,
    output_path=Path("results.json"),
    checkpoint_path=Path("results.checkpoint.json"),
)
```

If the process or host stops, rerun the same batch with the same ordered input,
model settings, output format, and paths:

```python
result = processor.process_texts(
    texts,
    output_path=Path("results.json"),
    checkpoint_path=Path("results.checkpoint.json"),
    resume_from_checkpoint=True,
)
```

`BatchProcessor.resume_from_checkpoint()` is also available when the caller
already has `BatchItem` objects. Resume verifies the input and configuration
fingerprints, the committed journal prefix, and the final output when the
checkpoint is complete. It refuses to continue if an output is missing,
truncated, or has a different SHA-256 digest.

Each checkpoint first writes and fsyncs a same-directory part file, atomically
renames it, and then atomically commits the checkpoint that points to its valid
byte prefix. If power fails between those commits, resume discards the
uncommitted tail. At most `checkpoint_interval` items are processed again.
Final result files also use temp-file, fsync, and atomic-rename semantics.

The checkpoint JSON is PHI-free. For `process_texts`, `process_files`, and
`process_directory`, `<checkpoint-path>.part` is the committed result journal
and has the same sensitivity as the requested result file; protect it with the
same permissions. The `openmed pii batch` journal is status-only, while each
de-identified output file is hashed and verified separately.

The general CLI enables checkpoints whenever `--output` is present:

```bash
openmed batch \
  --input-dir /data/notes \
  --output /data/results.json \
  --output-format json \
  --checkpoint-interval 25

# After an interruption, repeat the same command and add --resume.
openmed batch \
  --input-dir /data/notes \
  --output /data/results.json \
  --output-format json \
  --checkpoint-interval 25 \
  --resume
```

Use `--checkpoint-path` to override the default
`<output>.checkpoint.json`. `--resume` requires `--output` so the committed
result can be verified.

For atomic per-file de-identification:

```bash
openmed pii batch \
  --input-dir /data/raw-notes \
  --output-dir /data/deidentified \
  --checkpoint-interval 25

# Resume after a power failure.
openmed pii batch \
  --input-dir /data/raw-notes \
  --output-dir /data/deidentified \
  --checkpoint-interval 25 \
  --resume
```

The PII command defaults to
`<output-dir>/.openmed-batch.checkpoint.json`. On resume, every file recorded as
complete must still match its committed size and hash; a changed output stops
the run with a clear integrity error instead of silently mixing results.

## Operations

`BatchProcessor` supports three operations:

| Operation | Result type | Use when |
| --- | --- | --- |
| `analyze_text` | `PredictionResult` | Clinical or biomedical NER. |
| `extract_pii` | `PredictionResult` | PII detection across many records. |
| `deidentify` | `DeidentificationResult` | Batch masking, removal, replacement, hashing, or date shifting. |

`batch_size` controls how many documents are sent through each batch helper.
For PII operations, OpenMed reuses the same loader or privacy-filter pipeline
inside each batch instead of rebuilding it for every item.

## Batch PII Extraction

```python
from openmed import BatchProcessor

texts = [
    "Patient John Doe, DOB 01/15/1970, phone (555) 123-4567.",
    "Jane Roe emailed jane.roe@example.org from Boston.",
]

processor = BatchProcessor(
    operation="extract_pii",
    model_name="pii_detection",
    batch_size=16,
    confidence_threshold=0.5,
    use_smart_merging=True,
)

result = processor.process_texts(texts, ids=["note-1", "note-2"])

for item in result.get_successful_results():
    print(item.id)
    for entity in item.result.entities:
        print(f"  {entity.label}: {entity.text}")
```

## Batch De-identification

```python
from openmed import BatchProcessor

processor = BatchProcessor(
    operation="deidentify",
    model_name="pii_detection",
    batch_size=16,
    method="mask",
    confidence_threshold=0.7,
)

result = processor.process_texts(texts)

for item in result.items:
    if item.success:
        print(item.result.deidentified_text)
```

All `deidentify()` options can be passed through the constructor:

```python
processor = BatchProcessor(
    operation="deidentify",
    model_name="pii_detection",
    method="replace",
    lang="pt",
    locale="pt_BR",
    consistent=True,
    seed=42,
)
```

For date shifting:

```python
processor = BatchProcessor(
    operation="deidentify",
    model_name="pii_detection",
    method="shift_dates",
    date_shift_days=180,
)
```

## Dataset Redaction

Use `redact_dataset()` when the source is a tabular or line-delimited dataset
and only specific free-text columns should be de-identified. Supported formats
are `.csv`, `.jsonl`/`.ndjson`, and `.parquet`; CSV and JSONL rows are streamed,
and Parquet input is processed in row batches when `pyarrow` is installed.
Columns are never inferred automatically: pass the free-text columns explicitly.

```python
from openmed import redact_dataset

result = redact_dataset(
    "notes.csv",
    text_columns=["note", "comment"],
    output_path="notes.redacted.csv",
    policy="strict_no_leak",
)

print(result.summary.to_dict())
```

The console entry point exposes the same path:

```bash
openmed redact-dataset notes.csv \
  --text-columns note,comment \
  --policy strict_no_leak \
  --output notes.redacted.csv
```

The audit summary contains aggregate counts only, including total spans,
per-label counts, and a residual-leakage estimate. It does not include raw
cell values or detected entity text.

## Progress Tracking

Track progress with `on_progress`. The callback receives a frozen
`BatchProgress` record with counts, the current zero-based item index, and
elapsed time only. It does not receive source text, file content, model output,
or item metadata, so it is safe to use for progress bars and logs.

```python
from openmed import BatchProgress


def on_progress(progress: BatchProgress) -> None:
    print(
        f"[{progress.completed}/{progress.total}] "
        f"index={progress.current_index} elapsed={progress.elapsed:.1f}s"
    )


result = processor.process_texts(texts, on_progress=on_progress)
```

Existing callers can still use `progress_callback(current, total, item_result)`
when they need per-result status, but avoid logging the `item_result` payload in
PHI workflows because model outputs may contain source-derived text.

```python
def progress_callback(current, total, item_result):
    status = "OK" if item_result.success else "FAILED"
    print(f"[{current}/{total}] {status}")


result = processor.process_texts(texts, progress_callback=progress_callback)
```

## Streaming Results

For memory-efficient processing of large batches:

```python
for item_result in processor.iter_process(texts):
    if item_result.success:
        for entity in item_result.result.entities:
            print(f"{item_result.id}: {entity.label} - {entity.text}")
```

## Result Structure

### BatchResult

The `BatchResult` object contains:

- `total_items`: Total number of items processed
- `successful_items`: Number of successful items
- `failed_items`: Number of failed items
- `success_rate`: Success percentage
- `total_processing_time`: Total time in seconds
- `average_processing_time`: Average time per item
- `items`: List of `BatchItemResult` objects

```python
result = processor.process_texts(texts)

print(result.summary())
# Output:
# Batch Processing Summary
# ========================
# Model: disease_detection_superclinical
# Total items: 3
# Successful: 3
# Failed: 0
# Success rate: 100.0%
# Total time: 1.23s
# Average time per item: 0.410s
```

### BatchItemResult

Each item result contains:

- `id`: Item identifier
- `success`: Whether processing succeeded
- `result`: `PredictionResult` or `DeidentificationResult` (if successful)
- `error`: Error message (if failed)
- `processing_time`: Time taken for this item
- `source`: Source file path (if applicable)

## Error Handling

By default, batch processing continues on individual item errors:

```python
processor = BatchProcessor(
    model_name="disease_detection_superclinical",
    continue_on_error=True,  # Default
)

result = processor.process_texts(texts)

# Check for failures
for item in result.get_failed_results():
    print(f"Failed: {item.id} - {item.error}")
```

If a PII batch helper fails and `continue_on_error=True`, OpenMed falls back
to item-level processing so one bad record does not discard the rest of the
batch. Set `continue_on_error=False` to raise the batch exception immediately.

To stop on first error:

```python
processor = BatchProcessor(
    model_name="disease_detection_superclinical",
    continue_on_error=False,
)

try:
    result = processor.process_texts(texts)
except Exception as e:
    print(f"Processing stopped: {e}")
```

## Export Results

Export batch results to JSON:

```python
import json

result = processor.process_texts(texts)

# Export full results
with open("results.json", "w") as f:
    json.dump(result.to_dict(), f, indent=2)

# Export summary only
summary = result.summary()
```
