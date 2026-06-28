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
