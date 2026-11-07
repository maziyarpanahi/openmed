# Workflow run rollups

`openmed.agent.WorkflowRollup` turns validated run metadata into one stable row
per workflow. It is intended for operator reports, benchmark summaries, and
release evidence that need totals without reopening raw events.

## Inputs and privacy boundary

Build a rollup with `WorkflowRollup.from_events()` from validated
[`RunEvent`](run-summaries.md) objects, or use
`WorkflowRollup.from_summaries()` with summaries that each contain zero or one
workflow identifier. An empty summary is ignored. A non-empty summary with no
workflow identifier, or any summary with multiple workflow identifiers, is
rejected because its metrics cannot be attributed safely to a workflow.

Both input count and aggregate totals are bounded. Raw mappings are rejected,
so prompts, tool arguments, tool outputs, evidence text, paths, credentials,
and exception text cannot enter the rollup. Rendered output contains only the
schema identifier, workflow identifiers, outcome counts, run counts, tool-call
counts, durations, and artifact digest counts. Digest values are never
rendered or retained in rows.

## Example

```python
from openmed.agent import (
    OutcomeClass,
    RunEvent,
    WorkflowOutcome,
    WorkflowRollup,
)

events = [
    RunEvent(
        workflow_id="clinical-review",
        outcome=WorkflowOutcome(OutcomeClass.SUCCESS, "completed"),
        tool_call_count=3,
        duration_seconds=2.5,
        artifact_digests=("sha256:" + "a" * 64,),
    ),
]

rollup = WorkflowRollup.from_events(events)
json_report = rollup.to_json()
markdown_report = rollup.to_markdown()
```

JSON and Markdown rows are ordered by workflow identifier. Outcome columns use
the closed outcome vocabulary in stable order. The overall fields are derived
from the rows, so outcome, run, tool-call, duration, and artifact digest counts
reconcile exactly.

Artifact digests are deduplicated within each workflow. The overall artifact
digest count is the sum of those per-workflow unique counts. A digest associated
with two workflows therefore contributes once to each workflow and twice to
the reconciled overall count; this preserves row-level reconciliation without
claiming cross-workflow artifact identity.
