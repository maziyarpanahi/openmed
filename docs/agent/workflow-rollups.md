# Agent Workflow Rollups

`openmed.agent.rollup_workflows()` turns validated
[run events](run-summaries.md) into one stable, metadata-only row per workflow
without reopening raw traces. It reports per-workflow run counts, outcome
counts, tool-call counts, durations, and unique artifact digests, plus overall
totals that reconcile exactly with a `RunSummary` built from the same events.

## What is rolled up

| Per workflow | Overall |
| --- | --- |
| Run count | Total run count |
| Outcome counts (all five classes) | Total outcome counts |
| Tool-call count | Total tool-call count |
| Duration (seconds) | Total duration |
| Unique SHA-256 artifact digests | Unique digests across all workflows |

Entries are sorted by workflow identifier, so identical inputs always produce
identical JSON and Markdown regardless of event order. Duplicate workflow
identifiers are grouped into a single entry, and repeated artifact digests are
deduplicated per workflow.

## Example

```python
from openmed.agent import (
    OutcomeClass,
    RunEvent,
    WorkflowOutcome,
    rollup_workflows,
)

events = [
    RunEvent(
        workflow_id="intake",
        outcome=WorkflowOutcome(OutcomeClass.SUCCESS, "completed"),
        tool_call_count=3,
        duration_seconds=1.5,
    ),
    RunEvent(
        workflow_id="intake",
        outcome=WorkflowOutcome(OutcomeClass.SUCCESS, "completed"),
        tool_call_count=1,
        duration_seconds=0.5,
    ),
]

rollup = rollup_workflows(events)
assert rollup.entries[0].workflow_id == "intake"
assert rollup.entries[0].run_count == 2
assert rollup.to_dict()["tool_call_count"] == 4
json_payload = rollup.to_json()
markdown_report = rollup.to_markdown()
```

## Output contract

- `to_json()` returns compact JSON with sorted keys and the schema identifier
  `openmed.agent.workflow_rollup.v1`; `to_dict()` keeps a fixed field order.
- `to_markdown()` renders one table row per workflow plus the reconciled
  totals in a stable order.
- Overall totals reconcile exactly with `RunSummary.from_events()` over the
  same events: identical outcome counts, tool-call count, duration, and unique
  artifact digests.

Every input must be a validated `RunEvent`; any other item fails closed.
Failures raise `WorkflowRollupError` with a stable `field: code` message that
never repeats the submitted value. Prompts, tool payloads, evidence text,
paths, and credentials never enter output.

## Out of scope

The rollup does not rank workflows, retain event payloads, or aggregate across
tenants or sites. Use it to produce stable per-workflow rows and reconcile
against a run summary before applying your own reporting policy.
