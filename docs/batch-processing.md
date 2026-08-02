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
    group_entities=True,
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

## Distributed runs: start, resume, report

A distributed run splits a corpus into deterministic shards, records each
shard's state in a durable manifest, and can be resumed by run id after an
interruption. Everything stays local: shards run on local workers and the
manifest is an ordinary JSON file under the run directory.

```bash
# Start a run. --run-id is yours to choose and is stored verbatim, so it must
# not embed a patient identifier or any other record-derived value.
openmed batch-run start \
  --run-dir runs/nightly-2026-08-02 \
  --input-dir /data/notes \
  --run-id nightly-2026-08-02 \
  --shards 16 \
  --workers 4

# Resume after an interruption. Only shards that failed, went missing, or no
# longer match their recorded digest are recomputed. Set --max-attempts so a
# shard that fails every time eventually makes the run exhausted; without it a
# scripted retry loop never terminates.
openmed batch-run resume \
  --run-dir runs/nightly-2026-08-02 \
  --input-dir /data/notes \
  --workers 4 \
  --max-attempts 3

# Inspect a run without changing it.
openmed batch-run report --run-dir runs/nightly-2026-08-02 --format markdown
```

Exit codes follow the repository's gate convention: `0` when every shard
finished with a usable output, `1` when shards remain outstanding or have spent
their attempt budget, and `2` for a usage error. A run that produced nothing
never exits `0`. All three commands accept `--json` and emit a single envelope.

`--max-attempts` is unset by default, which means unbounded: shards stay merely
outstanding no matter how often they fail, and the run reports `in_progress`
rather than `exhausted`. Pass it whenever a resume runs unattended.

`--run-id` must be a single token — letters, digits and `._:+/-`, no spaces, at
most 128 characters — and is rejected before anything is written. The manifest
stores it verbatim and every later `report` and `resume` reads it back, so a run
id that a report could not publish would leave the run unreadable with no
recovery short of editing the manifest by hand.

Note file stems are the document ids, so they must be unique across
`--input-dir`; a collision is reported with the two offending positions.

`--pattern` follows the platform's own glob semantics, which are case-sensitive
on Linux and macOS and case-insensitive on Windows. `--pattern '*.txt'`
therefore matches `NOTE.TXT` on Windows and not elsewhere, so the same directory
can yield a different document set — and hence a different `plan_fingerprint` —
on different platforms. Fingerprints are comparable across runs on one platform,
not across platforms. Give the pattern an explicit case (`--pattern '*.TXT'`) or
normalise extensions beforehand if a run has to reproduce on both.

### What a report contains, and what it deliberately omits

Reports carry per-shard counts, attempt counters, durations, failures and
fingerprints. They contain no note text, no document identifiers and no
exception messages.

| Emitted | Why it is safe to publish |
| --- | --- |
| `run_id` | Operator-supplied, length-bounded and control-character free. Reproduced rather than hashed because it is the key tying a report to its manifest. |
| `plan_fingerprint`, per-shard `fingerprint` | sha256 over document hashes. |
| `output_digest`, `output_bytes` | Digest and size of a shard's output. |
| `status`, `attempts`, `document_count` | Enum values and counters. |
| `duration_seconds` per shard; `created_at`, `updated_at`, `generated_at` at the top level | Timings. |
| `worker_ref` | Hashed worker reference; the raw worker id is never published. |
| `error_type` | Exception class name only, never its message. |

Two omissions are deliberate. `output_path` is never published: a relative path
is safe to *store*, but a filename is chosen by whoever configured the run and
is not worth putting in an artifact that gets pasted into tickets. `worker_id`
is never published either — the manifest bounds its length and rejects control
characters, which does nothing to a host named after the ward it serves — so
reports carry `worker_ref` instead. The same worker yields the same reference
on every path of a report, so "which shards is this worker running" is still
answerable.

Shard output lines carry only de-identified text. They deliberately carry no
per-record identifier or hash of one: an unsalted digest over a fixed namespace
is reversible by enumeration across a low-entropy id space such as medical
record numbers, so it would be a pseudonym rather than a protection. Records
stay joinable because sharding is deterministic — replanning the same corpus
with the same shard count reproduces each shard's document order, so line `n` of
a shard output is document `n` of that shard.

A zero-document shard that reports as `completed` is normal. Empty shards
publish an empty output so that they settle rather than being re-queued by every
later run.

An empty straggler list means two different things, and the report says which:
`straggler_detection_enabled` is `false` when too few shards had finished to
establish a baseline, and `true` when detection ran and found nothing lagging.

The CLI runs shards on local worker threads. To drive the same manifest from a
process pool, a Ray cluster or a Spark cluster, see the next section.

## Distributed shard execution

For corpora too large for a single process, a run can be split into
deterministic shards and executed across worker processes, a Ray cluster, or a
Spark cluster. All three backends sit behind one `ShardExecutor` protocol, so
the surrounding code is identical and only the executor changes.

Importing `openmed.processing` imports neither Ray nor PySpark. Each adapter
imports its backend only when it runs.

```python
from openmed.processing import (
    LocalShardExecutor,
    build_run_manifest,
    plan_document_shards,
    run_shard_plan,
)

def handler(shard):
    # Return the bytes this shard publishes. Must be deterministic.
    return b"".join(redact(doc) for doc in load(shard.document_ids))

plan = plan_document_shards(documents, shard_count=64)
manifest = build_run_manifest(run_id="corpus-2026-08", plan=plan)

result = run_shard_plan(
    plan, handler,
    manifest=manifest,
    root="/data/runs/corpus-2026-08",
    executor=LocalShardExecutor(max_workers=8, use_processes=True),
)
```

### Ray

```bash
pip install "openmed[ray]"
```

```python
from openmed.processing import RayShardExecutor

executor = RayShardExecutor(num_cpus=1, max_in_flight=32)
executor.ensure_available()          # see "Failing before the manifest" below
result = run_shard_plan(plan, handler, manifest=manifest, root=root, executor=executor)
```

`max_in_flight` bounds how many shards are submitted at once; leave it unset to
submit the whole plan. Extra keyword arguments are forwarded to `ray.remote`.
A runtime is started or attached to automatically unless `auto_init=False`.

### Spark

```bash
pip install "openmed[spark]"
```

```python
from openmed.processing import SparkShardExecutor

executor = SparkShardExecutor(session=spark)
executor.ensure_available()
result = run_shard_plan(plan, handler, manifest=manifest, root=root, executor=executor)
```

The plan is parallelized with one slice per shard by default. A session is
never created implicitly: pass `session=`, or leave it unset to reuse the
active session.

### Failing before the manifest

`run_shard_plan` marks every shard `RUNNING` and increments its attempt counter
*before* it calls the executor. A backend that turns out to be missing or
unreachable at that point has already cost each shard an attempt. Calling
`ensure_available()` first — it imports the backend, and for Ray also starts or
attaches to the runtime — fails before any of that bookkeeping happens.

### Retries and duplicate execution

Neither adapter configures retry policy. Ray retries a task on worker death by
default, and Spark may run speculative duplicates, so a shard can execute more
than once. This is safe because the write path is idempotent: a worker
publishes through a temporary file and an atomic replace, and a re-executed
shard is compared against the digest it published earlier before its output is
replaced.

The one case this does not cover is a handler that is not deterministic. Such a
shard is reported as a digest mismatch and its existing output is left
untouched, rather than being silently overwritten — so make `handler` a pure
function of the shard it is given.

### Errors

Workers return PHI-free metadata only: a shard failure is recorded as an
exception *type* name, never a message, since messages can quote document text.

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
