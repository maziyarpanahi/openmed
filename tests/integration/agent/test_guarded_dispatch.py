"""Exercise authority enforcement on the real stateful workflow executor."""

import json
from dataclasses import replace

import pytest

from openmed.agent.workflows.recovery import RecoveryReason
from openmed.mcp.workflow import WorkflowRunner, WorkflowStateStore
from tests.fixtures.agent.guarded_dispatch import PRIVATE, DispatchHarness

pytestmark = pytest.mark.integration


def test_registered_tool_executes_once_and_runner_resumes_metadata():
    h = DispatchHarness()
    runner = WorkflowRunner(
        store=WorkflowStateStore(), executors={h.spec.name: h.adapter()}
    )
    pipeline = {
        "steps": [
            {
                "id": "summary",
                "tool": h.spec.name,
                "inputs": h.arguments,
                "allow_raw_output": True,
            }
        ]
    }
    first = runner.run(pipeline, session_id="session", workflow_id="workflow")
    resumed = runner.run(pipeline, session_id="session", workflow_id="workflow")
    assert first["status"] == resumed["status"] == "completed"
    assert h.tools.calls == 1
    assert resumed["trace"][0]["status"] == "resumed"
    assert PRIVATE not in json.dumps(first)
    assert "private_output" not in json.dumps(first)
    assert first["final_output"]["outcome"]["outcome_class"] == "success"


@pytest.mark.parametrize(
    "failure", ["authority", "arguments", "cancellation", "uncertain"]
)
def test_runner_stops_without_retry_or_downstream_execution(failure):
    h = DispatchHarness()
    authority = h.authority
    if failure == "authority":
        authority = replace(authority, ticket=None)
    elif failure == "arguments":
        h.arguments = {"text": PRIVATE + " changed"}
    elif failure == "cancellation":
        h.cancelled = True
    else:
        h.tools.uncertain = True
    downstream = []
    runner = WorkflowRunner(
        store=WorkflowStateStore(),
        executors={
            h.spec.name: h.adapter(authority=authority),
            "later": lambda: downstream.append(True),
        },
    )
    result = runner.run(
        {
            "steps": [
                {
                    "id": "summary",
                    "tool": h.spec.name,
                    "inputs": h.arguments,
                    "retry": {"max_attempts": 4},
                },
                {"id": "later", "tool": "later"},
            ]
        }
    )
    assert result["status"] == "failed"
    assert result["trace"][0]["attempt_count"] == 1
    assert h.tools.calls == (1 if failure == "uncertain" else 0)
    assert downstream == []
    assert PRIVATE not in json.dumps(result)
    if failure in {"cancellation", "uncertain"}:
        reason = result["trace"][0]["dispatch"]["recovery"]["reason"]
        assert reason == (
            RecoveryReason.WORKFLOW_ABORTED.value
            if failure == "cancellation"
            else RecoveryReason.AMBIGUOUS_EFFECT.value
        )
