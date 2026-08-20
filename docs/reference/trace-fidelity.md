# Trace fidelity verification

`openmed.traces.fidelity` checks that a privacy transformation did not make a
trace unusable for replay or training. It compares the input and output
locally and deterministically. No model, file, telemetry, or network service is
required.

Only declared content paths may change. The default policy permits fields named
`content` at any depth; use an explicit policy when a trace uses another
content field:

```python
from openmed.traces.fidelity import verify_trace_fidelity

before = {
    "trace_id": "trace-synthetic-1",
    "messages": [
        {"role": "user", "content": "SYNTHETIC_INPUT"},
        {"role": "assistant", "content": "SYNTHETIC_OUTPUT"},
    ],
    "label": "accepted",
}
after = {
    "trace_id": "trace-synthetic-1",
    "messages": [
        {"role": "user", "content": "SYNTHETIC_REDACTED_INPUT"},
        {"role": "assistant", "content": "SYNTHETIC_REDACTED_OUTPUT"},
    ],
    "label": "accepted",
}

report = verify_trace_fidelity(before, after)
if not report.passed:
    raise RuntimeError(report.summary())
```

Content values may change, but their shape and scalar types must remain stable.
Identifiers, timestamps, message order, tool-call identifiers and references,
and training labels are always compared, including when nested below a broad
content path. For a nonstandard content field, pass a dotted path or a path
pattern:

```python
report = verify_trace_fidelity(
    before,
    after,
    content_paths=["messages.*.prompt"],
)
```

`*` matches one object key or array item. `**` matches zero or more levels, so
`**.content` covers nested content fields. An empty list makes the comparison
strict. The aliases `content_fields`, `allowed_content_fields`, and
`allowed_paths` are available for callers that use those terms; provide only
one spelling per call.

## Value-free reports

`TraceFidelityReport.to_dict()`, `to_json()`, and `summary()` contain issue
codes, safe paths, coarse type names, and counts. They never include the input
or output field values. Common structural keys remain readable; nonstandard
caller-controlled keys are represented by deterministic `key_sha256_...`
segments so a key containing patient data cannot leak through a failing path
or configured-path summary. Useful checks include:

```python
report.message_order_valid
report.call_linkage_valid
report.identifiers_valid
report.timestamps_valid
report.training_labels_valid
report.scalar_types_valid
report.failing_paths
```

Use `assert_trace_fidelity()` when a failing report should raise
`TraceFidelityError`. The exception retains the value-free report for callers
that need structured diagnostics.

This verifier is a structural gate, not a compliance certification or a
guarantee that a trace contains no sensitive content.
