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
markdown_report = summary.to_markdown()
```

JSON keys, outcome rows, workflow identifiers, and artifact digests have stable
ordering. Artifact contents and workflow content are never read by the summary
layer.
