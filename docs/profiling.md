# Performance Profiling

OpenMed includes built-in performance profiling utilities to measure and optimize your NLP pipelines.

## Quick Start

```python
from openmed import enable_profiling, get_profile_report, analyze_text

# Enable global profiling
profiler = enable_profiling()

# Run your analysis
result = analyze_text(
    "Patient has diabetes mellitus type 2.",
    model_name="disease_detection_superclinical",
)

# Get the report
report = get_profile_report()
print(report.format_report())
```

## Profiler Class

For fine-grained control:

```python
from openmed import Profiler

profiler = Profiler(enabled=True)

# Measure operations
with profiler.measure("model_loading"):
    model = load_model("disease_detection_superclinical")

with profiler.measure("inference"):
    result = model(text)

with profiler.measure("postprocessing"):
    formatted = format_predictions(result, text)

# Generate report
report = profiler.report()
print(f"Total time: {report.total_duration * 1000:.2f}ms")
```

## Context Managers

### measure()

The primary way to time operations:

```python
profiler = Profiler(enabled=True)

with profiler.measure("my_operation", metadata={"input_size": len(text)}):
    result = process(text)
```

### timed()

Quick timing with automatic logging:

```python
from openmed import timed

with timed("model inference") as timer:
    result = model(text)

print(f"Took {timer.elapsed_ms:.2f}ms")
# Also logs: "model inference took 123.45ms"
```

## Timer Class

For manual timing:

```python
from openmed import Timer

timer = Timer()
timer.start()

# Do work
result = analyze_text(text)

elapsed = timer.stop()
print(f"Analysis took {elapsed:.2f}ms")

# Properties
print(f"Seconds: {timer.elapsed_s}")
print(f"Milliseconds: {timer.elapsed_ms}")
```

## Profile Decorator

Decorate functions to automatically profile them:

```python
from openmed import profile, enable_profiling, get_profile_report

enable_profiling()

@profile("data_preprocessing")
def preprocess(text):
    # preprocessing logic
    return cleaned_text

@profile()  # Uses function name
def run_inference(text):
    return analyze_text(text)

# Call functions
result = run_inference(preprocess(text))

# Get timings
report = get_profile_report()
for timing in report.timings:
    print(f"{timing.name}: {timing.duration * 1000:.2f}ms")
```

## Profile Reports

### ProfileReport Structure

```python
report = profiler.report()

# Properties
report.total_duration      # Total time in seconds
report.timing_count        # Number of measurements
report.timings            # List of TimingResult objects
report.metadata           # Custom metadata dict

# Methods
report.get_timing("model_loading")           # Get specific timing
report.get_timings_by_prefix("inference.")   # Filter by prefix
report.summary()                             # Dict summary
report.to_dict()                             # Full dict representation
report.format_report()                       # Human-readable string
```

### Example Output

```
Performance Profile Report
==================================================
Total Duration: 1234.56ms
Operations: 4

Timings:
--------------------------------------------------
  inference.forward                    856.23ms ( 69.4%)
  model_loading                        312.45ms ( 25.3%)
  postprocessing                        45.67ms (  3.7%)
  preprocessing                         20.21ms (  1.6%)
```

## Inference Metrics

Track detailed inference metrics:

```python
from openmed.utils.profiling import InferenceMetrics

metrics = InferenceMetrics(
    text_length=500,
    token_count=120,
    entity_count=5,
    inference_time_ms=150.0,
    preprocessing_time_ms=10.0,
    postprocessing_time_ms=5.0,
    total_time_ms=165.0,
)

print(f"Tokens/sec: {metrics.tokens_per_second:.0f}")
print(f"Chars/sec: {metrics.chars_per_second:.0f}")
```

## Batch Metrics

Aggregate metrics for batch processing:

```python
from openmed.utils.profiling import BatchMetrics, InferenceMetrics

items = [
    InferenceMetrics(text_length=100, entity_count=2, total_time_ms=50),
    InferenceMetrics(text_length=200, entity_count=3, total_time_ms=80),
    InferenceMetrics(text_length=150, entity_count=1, total_time_ms=60),
]

batch = BatchMetrics(items=items, total_time_ms=190)

print(batch.format_report())
# Output:
# Batch Processing Metrics
# ========================================
# Items processed: 3
# Total characters: 450
# Total entities: 6
# Total time: 190.00ms
# Avg time/item: 63.33ms
# Throughput: 15.79 items/sec
# Throughput: 2368 chars/sec
```

## Global Profiling

Enable profiling globally for an entire session:

```python
from openmed import (
    enable_profiling,
    disable_profiling,
    get_profile_report,
)

# Enable profiling
profiler = enable_profiling()

# All analyze_text calls will be profiled
result1 = analyze_text(text1)
result2 = analyze_text(text2)

# Get combined report
report = get_profile_report()
print(report.format_report())

# Disable when done
disable_profiling()
```

## Session Tracking

Track overall session duration:

```python
profiler = Profiler(enabled=True)
profiler.start()  # Start session timer

with profiler.measure("op1"):
    do_work()

with profiler.measure("op2"):
    do_more_work()

profiler.stop()  # Stop session timer

report = profiler.report()
print(f"Session duration: {report.metadata['session_duration_ms']:.2f}ms")
```

## Adding Custom Metadata

```python
profiler = Profiler(enabled=True)

# Add metadata
profiler.add_metadata("model", "disease_detection_superclinical")
profiler.add_metadata("batch_size", 32)

# Add manual timing
profiler.add_timing("external_api_call", 0.5, metadata={"endpoint": "/api/v1"})

report = profiler.report()
print(report.metadata)
# {'model': 'disease_detection_superclinical', 'batch_size': 32}
```

## Disabled Profiler (Zero Overhead)

When profiling is disabled, all operations are no-ops:

```python
profiler = Profiler(enabled=False)

# These have virtually no overhead
with profiler.measure("operation"):
    result = expensive_operation()

report = profiler.report()
assert report.timing_count == 0  # No timings recorded
```

## Best Practices

1. **Enable only when needed**: Use `enabled=False` in production unless actively profiling
2. **Use prefixes for organization**: `model.load`, `model.infer`, `post.format`
3. **Add metadata for context**: Include batch sizes, model names, input characteristics
4. **Profile at multiple levels**: Overall pipeline + individual components
5. **Reset between runs**: Call `profiler.reset()` for clean measurements
