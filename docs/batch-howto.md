# Batch Processing How-To

This guide shows three complete batch workflows using synthetic text:

1. de-identify a text column in a CSV and write a new column;
2. de-identify every `.txt` file in a directory into a mirrored output tree;
3. report progress for a long-running batch without logging source text.

The examples use the same `BatchProcessor` API described in the
[Batch Processing reference](batch-processing.md). Replace the model name with
one available in your environment. Do not use real patient data in examples,
tests, logs, or issue reports.

## Setup

Install the development or runtime extra, then choose a local model:

```bash
uv pip install "openmed[hf]"
```

The snippets below use `pii_detection` and synthetic records such as `Casey
Example`. The first run may download model artifacts. Review the model and
dataset terms before using them in a deployment.

## De-identify a CSV column

Read one explicit text column, process it as a batch, and write a new
de-identified column. Columns are never inferred automatically.

```python
import csv
from pathlib import Path

from openmed import BatchProcessor


input_path = Path("synthetic-notes.csv")
output_path = Path("synthetic-notes.redacted.csv")

with input_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

text_column = "note"
texts = [row[text_column] for row in rows]
processor = BatchProcessor(
    operation="deidentify",
    model_name="pii_detection",
    method="mask",
    batch_size=16,
)
result = processor.process_texts(texts, ids=[row["id"] for row in rows])

for row, item in zip(rows, result.items, strict=True):
    if item.success:
        row[f"{text_column}_redacted"] = item.result.deidentified_text
    else:
        row[f"{text_column}_redacted"] = ""
        print(f"Skipped {item.id}: {item.error}")

with output_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=[*rows[0], "note_redacted"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {result.successful_items} of {result.total_items} rows")
```

For example, `synthetic-notes.csv` can contain only synthetic values:

```csv
id,note
note-1,"Casey Example called 212-555-0198 about a refill."
note-2,"Jordan Sample emailed jordan@example.org from Boston."
```

Keep the original input protected. The output contains masked text, but it is
still derived data that should follow the same access policy as the input.

## Redact a directory of text files

`process_files_to_directory` preserves each input path relative to
`input_root`, writes each de-identified file atomically, and supports
checkpointing for interruption-safe reruns.

```python
from pathlib import Path

from openmed import BatchProcessor


input_root = Path("synthetic-notes")
output_root = Path("synthetic-notes-redacted")
files = sorted(input_root.rglob("*.txt"))

processor = BatchProcessor(
    operation="deidentify",
    model_name="pii_detection",
    method="mask",
    batch_size=16,
)
result = processor.process_files_to_directory(
    files,
    input_root=input_root,
    output_dir=output_root,
    checkpoint_path=output_root / ".openmed-batch.checkpoint.json",
    checkpoint_interval=25,
)

print(f"Wrote {result.successful_items} of {result.total_items} files")
```

With this layout:

```text
synthetic-notes/
├── january/note-1.txt
└── february/note-2.txt
```

the output files are written to `synthetic-notes-redacted/january/note-1.txt`
and `synthetic-notes-redacted/february/note-2.txt`. The checkpoint contains
status and hashes, not raw text or model output. After an interruption, repeat
the call with the same ordered input and add `resume_from_checkpoint=True`.

## Add progress callbacks

Use `on_progress` for a PHI-safe progress record. It reports counts, the
zero-based current index, and elapsed time, but not source text, model output,
or item metadata.

```python
from openmed import BatchProcessor, BatchProgress


def report(progress: BatchProgress) -> None:
    print(
        f"[{progress.completed}/{progress.total}] "
        f"item={progress.current_index} "
        f"elapsed={progress.elapsed:.1f}s"
    )


processor = BatchProcessor(
    operation="deidentify",
    model_name="pii_detection",
    method="mask",
    batch_size=16,
)
result = processor.process_texts(
    [
        "Casey Example called 212-555-0198.",
        "Jordan Sample emailed jordan@example.org.",
    ],
    ids=["synthetic-1", "synthetic-2"],
    on_progress=report,
)
```

Avoid printing `item.result` or the input text from a callback. The legacy
`progress_callback(current, total, item_result)` API is still available when
per-item status is required, but `on_progress` is the safer default for
privacy-sensitive workflows.

## Tune throughput and validate the docs

Increase `batch_size` only after measuring memory use for the selected model.
For long jobs, combine a checkpoint interval with progress reporting so a
restart does not repeat the entire batch. Keep synthetic fixtures small when
testing the workflow.

Build the documentation with the same strict command used by CI:

```bash
mkdocs build --strict
```
