# Preflight PHI gate for agent context

`openmed.guard` provides a deterministic, local check for sensitive values in
agent context and tool-output payloads. It runs before the caller dispatches a
request, does not load a model, and makes no mandatory network call.

The default scanner reuses OpenMed's structured-identifier safety sweep. It is
useful for common email, phone, date, identifier, and address patterns, but it
is not a complete PHI detector or a compliance certification. A deployment
with domain-specific identifiers should supply its own local scanner.

## Fail closed

Fail-closed mode is the default. A finding raises `PreflightBlockedError`, and
the exception message does not contain the supplied payload. The exception's
`report` contains only categories, character offsets within each string leaf,
payload indexes, lengths, and one-way hashes:

```python
from openmed.guard import PreflightBlockedError, preflight_context

try:
    result = preflight_context(context, tool_outputs)
except PreflightBlockedError as error:
    safe_report = error.report.to_dict()
    # Do not serialize context or arbitrary tool results.
```

Use `inspect_context` when a caller needs a non-raising result with
`result.allowed == False`.

## Redact and continue

`redact_then_continue` replaces each detected span with a stable token and
returns the same nested shape for mappings, lists, and tuples:

```python
from openmed.guard import preflight_context

result = preflight_context(
    context,
    tool_outputs,
    policy="redact_then_continue",
)
dispatch(result.context, result.tool_outputs)
audit_log(result.report.to_dict())
```

The report never includes the redacted payload. Offsets refer to the original
string leaf and are zero-based, end-exclusive. `channel` is either `context`
or `tool_output`; `payload_index` identifies the leaf within that channel's
deterministic traversal.

## Custom local scanners

A scanner receives one validated string leaf at a time and returns mappings or
tuples with a non-sensitive category and offsets. It must not return matched
text:

```python
def scan_local(text):
    marker = "SYNTHETIC_IDENTIFIER"
    start = text.find(marker)
    if start < 0:
        return ()
    return ({"category": "LOCAL_IDENTIFIER", "start": start, "end": start + len(marker)},)


result = preflight_context(
    context,
    policy="redact_then_continue",
    scanner=scan_local,
)
```

Keep reports, exceptions, and audit records to the guard-owned `report` or
`to_dict()` output. The guard is a defensive preflight layer; callers remain
responsible for choosing detectors appropriate to their data and workflow.
