# Structure-aware tool-call redaction

Tool arguments and results are often JSON objects, arrays, or JSON-encoded
strings. Redacting the whole serialized payload can corrupt the JSON and make
the trace impossible to replay. `openmed.traces.tool_calls` walks only the
configured content paths and applies a text redactor to string leaves. Numbers,
booleans, nulls, arrays, object keys, and unconfigured fields remain intact.

## Offline usage

Pass a deterministic callable when processing local traces or tests. The input
is deep-copied, so the source trace is not changed:

```python
from openmed.traces.tool_calls import redact_tool_call


def redact_text(value: str) -> str:
    return value.replace("synthetic-name-001", "[NAME]")


trace_record = {
    "function": {
        "name": "lookup",
        "arguments": {"name": "synthetic-name-001", "limit": 5},
    },
    "result": {"matches": [], "message": "synthetic-name-001 not found"},
}

redacted = redact_tool_call(trace_record, text_redactor=redact_text)
```

The default content paths are `arguments`, `function.arguments`, and
`result`. For a trace envelope with nested calls, provide explicit dot paths;
`*` matches every object key or array item:

```python
redacted = redact_tool_call(
    envelope,
    content_paths=(
        "messages.*.tool_calls.*.function.arguments",
        "messages.*.tool_calls.*.result",
    ),
    text_redactor=redact_text,
)
```

JSON Pointer paths such as `/messages/0/tool_calls/0/result` and a nested
segment sequence such as `(("messages", 0, "tool_calls", 0, "result"),)` are
also accepted. Paths that target a JSON-encoded object or array are decoded,
redacted, and emitted as compact, sorted JSON. A malformed JSON string is
passed to the text redactor as-is; its path is recorded in the safe report, but
the payload is never put in an exception or report.

## Safe reports

Use `redact_tool_call_with_report` when counts are useful for audit metadata:

```python
from openmed.traces.tool_calls import redact_tool_call_with_report

processed = redact_tool_call_with_report(
    trace_record,
    text_redactor=redact_text,
)
safe_report = processed.report.to_dict()
```

The report contains counts and structural paths only. Object-key path segments
are represented by stable SHA-256 digest prefixes while array indexes and
wildcards retain their structural form, so data-derived object keys cannot leak
through report or exception text. The report never stores source strings,
replacement strings, or the original payload. The module has no network side
effects at import time; an explicit `text_redactor` is the recommended path for
deterministic offline workflows. When it is omitted, the default OpenMed
redactor runs under the enforced local-only network guard and therefore
requires a cached model or local model configuration.
