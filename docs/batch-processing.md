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

## Progress Tracking

Track progress with a callback:

```python
def progress_callback(current, total, item_result):
    status = "OK" if item_result.success else "FAILED"
    print(f"[{current}/{total}] {item_result.id}: {status}")

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
- `result`: PredictionResult (if successful)
- `error`: Error message (if failed)
- `processing_time`: Time taken for this item
- `source`: Source file path (if applicable)

## CLI Usage

Batch processing from the command line:

```bash
# Process multiple texts
openmed batch --texts "Text one" "Text two" "Text three" --model disease_detection_superclinical

# Process files
openmed batch --input-files file1.txt file2.txt --output-format json

# Process directory
openmed batch --input-dir /path/to/notes --pattern "*.txt" --recursive

# Output to file
openmed batch --input-dir ./notes --output results.json --output-format json
```

### CLI Options

- `--model`: Model to use (default: disease_detection_superclinical)
- `--input-dir`: Directory containing files
- `--input-files`: List of specific files
- `--texts`: List of text strings
- `--pattern`: Glob pattern for directory (default: *.txt)
- `--recursive`: Search directories recursively
- `--output`: Output file path
- `--output-format`: `json` or `summary` (default: summary)
- `--confidence-threshold`: Minimum confidence
- `--group-entities`: Group adjacent entities
- `--quiet`: Suppress progress output

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
