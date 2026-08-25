# Agent Run Summaries

`openmed.agent.run_summary` provides privacy-safe summaries of agent runs.

The summary layer accepts structured run metadata and intentionally does not
publish prompts, tool arguments, tool outputs, evidence text, filesystem
paths, or credentials.

## Run events

A `RunEvent` represents one agent workflow execution:

- `workflow_id` — safe identifier for the workflow.
- `outcome` — one of `success`, `failure`, or `abstained`.
- `tool_call_count` — number of tool calls made during the run.
- `duration_seconds` — non-negative execution duration.
- `artifact_digests` — optional SHA-256 digests identifying generated artifacts.

Example:

```python
from openmed.agent.run_summary import RunEvent

event = RunEvent(
    workflow_id="clinical-review",
    outcome="success",
    tool_call_count=3,
    duration_seconds=2.5,
    artifact_digests=("sha256:" + "a" * 64,),
)