# JSONL trace content walker

Agent traces are records, not plain text. A trace can carry roles, timestamps,
tool calls, replay identifiers, and nested content in the same JSON object.
The JSONL walker keeps that structure intact while exposing only configured
string values to a local redaction callback.

The implementation is in openmed.traces.jsonl and has no network or model
dependency. A caller supplies the redaction function, so an offline policy can
be used in tests and in air-gapped workflows.

## Find redactable content

walk_trace_content accepts JSONL text, a path, a text stream, or an iterable
of lines. It yields immutable TraceContentLocation values:

~~~python
from openmed.traces.jsonl import walk_trace_content

records = [
    '{"trace_id":"trace-001","events":[{"role":"user",'
    '"content":"synthetic note"}]}'
]

locations = walk_trace_content(
    records,
    content_paths=("events.*.content",),
)

for location in locations:
    print(location.line_number, location.json_path, location.value)
~~~

The location contains the physical line number, a tuple path such as
("events", 0, "content"), and the selected string. It does not include a
copy of the surrounding record. This makes it possible to pass only the
redactable value to a privacy component.

Paths may be dotted strings or tuples of object keys and array indexes:

| Pattern | Meaning |
| --- | --- |
| content | A top-level content field |
| events.*.content | Content in every item of events |
| messages[0].content | Content in the first message |
| **.content | A content field at any nesting depth |

The default is **.content. For a trace schema with stricter policy boundaries,
pass explicit paths so similarly named fields outside the intended content
locations are not selected. Only string values at matched locations are
yielded; numbers, booleans, null, objects, and arrays are not coerced.

## Rewrite without flattening

rewrite_trace_jsonl is the streaming rewrite path. It parses one record,
transforms only selected strings, and yields the rewritten line:

~~~python
from openmed.traces.jsonl import rewrite_trace_jsonl

def redact(value: str) -> str:
    return value.replace("synthetic", "[REDACTED]")

rewritten = rewrite_trace_jsonl(
    records,
    redact,
    content_paths=("events.*.content",),
)
output = "".join(rewritten)
~~~

The rewrite retains object insertion order, identifiers, scalar types, list
positions, and every unconfigured field. A transform must return a string;
returning another type would change the trace schema and is rejected. Use
write_trace_jsonl to stream directly to a path or text output. Path-based input
and output must be different files; the writer rejects the same file before it
opens the destination so a lazy input stream cannot be truncated. Use a
separate output for validation before any caller-managed replacement.

## Malformed records

Malformed JSON, duplicate object keys, non-standard numeric constants, and
non-object records raise TraceJSONLLineError. The exception exposes the
one-based line_number and a short message, but never includes the source line
or a value from it. Transform failures use TraceJSONLTransformError, which
follows the same value-free rule and also identifies the configured JSON path.
Common structural keys remain readable in that diagnostic; nonstandard object
keys are replaced by deterministic `key_sha256_...` segments so a key that
contains patient data cannot be echoed through an exception.
Unreadable path/stream inputs and unwritable destinations raise
`TraceJSONLIOError` with a value-free message; underlying filesystem paths and
custom stream exception text are not propagated.

Blank physical lines are passed through unchanged by rewrites and do not
produce locations. All behavior is deterministic and local-first.
