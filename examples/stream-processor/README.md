# Stream processor record de-identification

OpenMed's framework-neutral stream helpers make record-level de-identification
easy to place between a source and sink. The map function copies every record,
redacts only the configured free-text field, and sends text to
`openmed.processing.process_batch` in bounded micro-batches. Its `open()`
lifecycle is idempotent, so one resident processor is reused by each parallel
subtask.

This offline job uses synthetic records and a list-backed sink. It exercises the
same `open`/`map_batch`/`invoke` lifecycle that a cluster adapter can call, but
does not require a stream-processing cluster:

```python
from openmed.integrations.stream_processor import (
    StreamDeidentifyMapFunction,
    StreamSink,
    run_stream_job,
)

source = [
    {
        "event_id": "evt-2",
        "note": "Patient Jane Roe called 555-0101",
        "facility": "synthetic-clinic",
    },
    {
        "event_id": "evt-1",  # out-of-order records are safe
        "note": "Patient Jane Roe called 555-0101",  # duplicates are safe
        "facility": "synthetic-clinic",
    },
]

mapper = StreamDeidentifyMapFunction(
    text_field="note",
    policy="hipaa_safe_harbor",
    batch_size=32,
)
redacted_records = []
sink = StreamSink(redacted_records.append)

written = run_stream_job(source, mapper, sink)
assert written == 2
assert source[0]["note"] == "Patient Jane Roe called 555-0101"
assert redacted_records[0]["facility"] == "synthetic-clinic"
assert "Jane Roe" not in redacted_records[0]["note"]
```

For a parallel runtime, construct one `StreamDeidentifyMapFunction` per subtask,
call `open(runtime_context)` during subtask initialization, pass record groups to
`map_batch`, and forward each returned record to `StreamSink.invoke`. Do not
share one map-function instance between concurrent subtasks.
