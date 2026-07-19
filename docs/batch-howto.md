# Batch Processing How-to

This guide walks through two complete data-engineering jobs: de-identifying a
column in a CSV file and redacting a directory of text files. Both examples
create synthetic input data, reuse one `BatchProcessor`, report PHI-safe
progress, and write to a separate output path.

For constructor options, result types, and the complete API, see the
[BatchProcessor reference](batch-processing.md#batchprocessor-class).

!!! warning "Validate before releasing data"
    De-identification is probabilistic. Validate the output and residual risk
    against your own data and compliance requirements before sharing it. The
    names, identifiers, and contact details below are entirely synthetic.

## Prerequisites

Install OpenMed with its Hugging Face dependencies:

```bash
uv pip install "openmed[hf]"
```

The first model-backed run may download model artifacts. Warm the model cache
before a scheduled job; after that, you can use
[local-only offline mode](configuration.md#local-only-offline-mode) to prevent
network access during processing.

## De-identify a CSV column

This standalone script creates a synthetic CSV, de-identifies its `note`
column, and writes `note_deidentified` to a new CSV. It keeps the source file
unchanged and stops before writing any output if a record fails.

```python
import csv
from pathlib import Path

from openmed import BatchProcessor, BatchProgress

work_dir = Path("batch-demo")
work_dir.mkdir(exist_ok=True)
input_path = work_dir / "notes.csv"
output_path = work_dir / "notes.deidentified.csv"

synthetic_rows = [
    {
        "record_id": "synthetic-001",
        "note": "Casey Example called from 555-0100 on 01/15/2020.",
    },
    {
        "record_id": "synthetic-002",
        "note": "Email Jordan Sample at jordan.sample@example.org.",
    },
]

with input_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=["record_id", "note"])
    writer.writeheader()
    writer.writerows(synthetic_rows)

with input_path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))


def report_progress(progress: BatchProgress) -> None:
    print(
        f"Processed {progress.completed}/{progress.total} "
        f"in {progress.elapsed:.1f}s"
    )


processor = BatchProcessor(
    operation="deidentify",
    model_name="pii_detection",
    method="mask",
    batch_size=16,
    confidence_threshold=0.7,
    continue_on_error=True,
)

result = processor.process_texts(
    [row["note"] for row in rows],
    ids=[row["record_id"] for row in rows],
    on_progress=report_progress,
)

failures = result.get_failed_results()
if failures:
    failed_ids = ", ".join(item.id for item in failures)
    raise RuntimeError(f"De-identification failed for: {failed_ids}")

for row, item in zip(rows, result.items, strict=True):
    row["note_deidentified"] = item.result.deidentified_text

with output_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=["record_id", "note", "note_deidentified"],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {result.successful_items} rows to {output_path}")
```

The example keeps the original `note` for comparison because the input is
synthetic. For real data, omit the raw column from the output unless downstream
access is explicitly authorized. If you only need dataset redaction, the
streaming `redact_dataset()` helper can write selected CSV, JSONL, or Parquet
columns directly; see [Dataset Redaction](batch-processing.md#dataset-redaction).

## Redact a directory into a mirrored output tree

`process_directory()` finds matching files in sorted order and returns each
source path with its result. This standalone script uses that source path to
preserve subdirectories under a separate output root. It never overwrites the
input notes.

```python
from pathlib import Path

from openmed import BatchProcessor, BatchProgress

input_dir = Path("batch-demo/raw-notes")
output_dir = Path("batch-demo/redacted-notes")

synthetic_notes = {
    Path("ward-a/note-1.txt"): (
        "Synthetic patient Casey Example has MRN 00123 and phone 555-0100."
    ),
    Path("ward-b/note-2.txt"): (
        "Synthetic patient Jordan Sample uses jordan.sample@example.org."
    ),
}
for relative_path, text in synthetic_notes.items():
    source_path = input_dir / relative_path
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(text, encoding="utf-8")


def report_progress(progress: BatchProgress) -> None:
    percent = 100 * progress.completed / progress.total
    print(f"{percent:5.1f}% ({progress.completed}/{progress.total})")


processor = BatchProcessor(
    operation="deidentify",
    model_name="pii_detection",
    method="mask",
    batch_size=16,
    confidence_threshold=0.7,
    continue_on_error=True,
)

result = processor.process_directory(
    input_dir,
    pattern="*.txt",
    recursive=True,
    on_progress=report_progress,
)

failures = result.get_failed_results()
if failures:
    failed_sources = ", ".join(item.source or item.id for item in failures)
    raise RuntimeError(f"De-identification failed for: {failed_sources}")

for item in result.get_successful_results():
    source_path = Path(item.source)
    relative_path = source_path.relative_to(input_dir)
    destination = output_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(item.result.deidentified_text, encoding="utf-8")

print(f"Wrote {result.successful_items} files under {output_dir}")
```

Use a distinct output root, then review the completed tree before replacing or
publishing anything. The fail-closed check above avoids silently copying raw
input when one item cannot be processed.

## Report long-job progress safely

[OM-078](https://github.com/maziyarpanahi/openmed/issues/243) added the
`BatchProcessor` `on_progress` callback used in both examples. It runs after
each completed item and receives an immutable `BatchProgress` value with only:

- `completed`: number of completed items
- `total`: number of items in the job
- `current_index`: zero-based index of the item just completed
- `elapsed`: seconds since processing started

These fields contain no source text, file content, identifiers, or model
output, so the callback is suitable for progress bars and operational metrics.
Avoid logging a `BatchItemResult` from the legacy `progress_callback` API in a
PHI workflow because its payload can contain source-derived text. See
[Progress Tracking](batch-processing.md#progress-tracking) for both callback
signatures.

## Tune memory and throughput

- Create one processor per worker and reuse it across jobs. Reusing the loaded
  model avoids repeated initialization and model-cache work.
- Start with `batch_size=8` or `16`, benchmark representative document lengths,
  and increase gradually while watching resident memory and accelerator memory.
  A larger batch may improve throughput, but it can also exhaust memory.
- `batch_size` controls inference chunks; it does not cap total job memory.
  `process_texts()` retains all inputs and results, while `process_directory()`
  discovers and reads the matching files before returning one `BatchResult`.
- For a corpus larger than memory, enumerate files in bounded windows, process
  and persist one window at a time, then release it before continuing. When
  input texts are already in a bounded sequence, `iter_process()` avoids
  retaining an aggregate result; see
  [Streaming Results](batch-processing.md#streaming-results).
- Keep progress reporting metadata-only. Record counts, timings, and failure
  IDs that are non-sensitive job keys; never emit raw notes or redacted model
  payloads to general-purpose logs.
- `continue_on_error=True` lets the job finish collecting item failures. Treat
  those failures explicitly before publishing output, as both examples do.

For all batch operations, constructor parameters, error semantics, and result
fields, continue with the [BatchProcessor reference](batch-processing.md).
