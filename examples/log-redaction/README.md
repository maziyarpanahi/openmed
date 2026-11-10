# Log Redaction Example

This example shows how to use OpenMed as an NDJSON transform for structured log
events before they reach a centralized observability store.

Input events are one JSON object per line:

```json
{"timestamp":"2026-07-01T10:00:00Z","service":"api","message":"Patient Jane Roe called 555-0100","trace_id":"abc"}
{"timestamp":"2026-07-01T10:00:01Z","service":"worker","error":{"message":"Follow up with john.doe@example.org"},"trace_id":"def"}
```

Run the stdin/stdout transform:

```bash
python -m openmed.integrations.log_redactor \
  --field message \
  --field error.message \
  < raw-events.ndjson \
  > redacted-events.ndjson
```

The transform preserves event order and non-target fields while replacing
configured text fields:

```json
{"timestamp":"2026-07-01T10:00:00Z","service":"api","message":"Patient [NAME] called [PHONE]","trace_id":"abc"}
{"timestamp":"2026-07-01T10:00:01Z","service":"worker","error":{"message":"Follow up with [EMAIL]"},"trace_id":"def"}
```

Python callers can embed the same filter:

```python
from openmed.integrations.log_redactor import redact_log_events

events = [
    {
        "service": "api",
        "message": "Patient Jane Roe called 555-0100",
        "trace_id": "abc",
    },
]

redacted = list(
    redact_log_events(
        events,
        message_fields=("message",),
        batch_size=16,
    )
)
```

Diagnostics report only counts, line numbers, and safe failure summaries. They
do not echo raw event bodies.
