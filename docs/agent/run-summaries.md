# Privacy-safe agent run summaries

`openmed.agent.RunSummary` produces deterministic metadata for dashboards and
evidence bundles without copying prompts, tool arguments, tool outputs,
evidence text, filesystem paths, credentials, or exception text.

## Event contract

Each `RunEvent` contains only:

- a bounded workflow identifier without path or URL syntax;
- a typed [`WorkflowOutcome`](outcome-reasons.md) from the closed outcome and
  reason-code vocabulary;
- a bounded non-negative tool-call count and finite duration; and
- optional lowercase SHA-256 artifact digests.

Inputs and aggregate totals are bounded. Duplicate event digests, malformed
identifiers, incomplete outcome-count mappings, non-finite values, and direct
construction that bypasses canonical ordering fail with stable value-free
errors.

Serialized summaries carry the exact schema identifier
`openmed.agent.run_summary.v1`. Use `RunSummary.from_dict()` or
`RunSummary.from_json()` at trust boundaries. Both reject missing or unknown
fields, unsupported versions, invalid counts, and unsafe strings;
`from_json()` additionally rejects duplicate keys, non-standard non-finite
numbers, malformed JSON, and documents larger than 1 MiB. Validation errors
name only stable fields and error codes, never submitted values.

## Example

```python
from openmed.agent import OutcomeClass, RunEvent, RunSummary, WorkflowOutcome

event = RunEvent(
    workflow_id="clinical-review",
    outcome=WorkflowOutcome(OutcomeClass.SUCCESS, "completed"),
    tool_call_count=3,
    duration_seconds=2.5,
    artifact_digests=("sha256:" + "a" * 64,),
)

summary = RunSummary.from_events([event])
json_payload = summary.to_json()
assert RunSummary.from_json(json_payload) == summary
markdown_report = summary.to_markdown()
```

JSON keys, outcome rows, workflow identifiers, and artifact digests have stable
ordering. Artifact contents and workflow content are never read by the summary
layer.

Strict parsing rejects mapping objects in sequence fields and checks duration
bounds before converting numbers to floats. Malformed JSON and unreadable
mappings produce field-only failures without retaining source exceptions.
