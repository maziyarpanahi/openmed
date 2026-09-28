# Privacy-safe Agent Run Diffs

`openmed.agent.diff_run_summaries()` compares two
[privacy-safe run summaries](run-summaries.md) and reports what changed using
only aggregate metadata. It never sees raw traces, prompts, tool arguments,
clinical outputs, paths, or credentials, because both inputs must already be
validated `RunSummary` values.

## What is compared

| Field | Diff output |
| --- | --- |
| Workflow identifiers | Sorted `added` and `removed` identifiers |
| Outcome counts | Signed delta for every closed outcome class |
| Tool calls | Signed integer delta |
| Duration | Signed delta in seconds |
| Artifact digests | Sorted `added` and `removed` SHA-256 digests |

Every delta is `after - before`. Swapping the two inputs negates every delta
and swaps each `added` set with its `removed` set. Equal summaries produce a
diff whose `changed` property is `False`, with zero deltas and empty sets.

## Example

```python
from openmed.agent import (
    OutcomeClass,
    RunEvent,
    RunSummary,
    WorkflowOutcome,
    diff_run_summaries,
)

before = RunSummary.from_events(
    [
        RunEvent(
            workflow_id="intake",
            outcome=WorkflowOutcome(OutcomeClass.SUCCESS, "completed"),
            tool_call_count=3,
            duration_seconds=1.5,
        )
    ]
)
after = RunSummary.from_events(
    [
        RunEvent(
            workflow_id="intake",
            outcome=WorkflowOutcome(OutcomeClass.FAILED, "timeout"),
            tool_call_count=5,
            duration_seconds=4.0,
        )
    ]
)

diff = diff_run_summaries(before, after)
assert diff.changed
assert diff.outcome_count_deltas["failed"] == 1
assert diff.tool_call_count_delta == 2
json_payload = diff.to_json()
markdown_report = diff.to_markdown()
```

## Output contract

- `to_json()` returns compact JSON with sorted keys and the schema identifier
  `openmed.agent.run_diff.v1`; `to_dict()` keeps a fixed field order.
- `to_markdown()` renders workflow, outcome, execution, and artifact tables in
  a stable order. Non-zero deltas carry an explicit `+` or `-` sign.
- A zero duration delta is always serialized as `0.0`, never `-0.0`.

Direct construction of `RunSummaryDiff` validates the same vocabulary as
summaries: bounded identifiers, closed outcome names, integer counts, finite
durations, and lowercase SHA-256 digests. Identifiers and digests must be
sorted, unique, and never both added and removed. Failures raise
`RunDiffError` with a stable `field: code` message that never repeats the
submitted value.

## Out of scope

The diff does not decide whether a change is a regression, fetch summaries
from storage, or compare raw event order. Use it to review differences, and
apply your own acceptance policy to the result.
