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


@pytest.mark.parametrize("stop_phase", [None, "approval_recorded", "dispatching"])
def test_real_durable_admission_controls_registered_dispatch(tmp_path, stop_phase):
    from openmed.agent.admission import (
        AdmissionState,
        EffectAdmissionController,
        SQLiteAdmissionStore,
    )
    from openmed.agent.approvals.tokens import ApprovalTokenSigner
    from tests.fixtures.agent.guarded_dispatch import KEY, ROLE

    h = DispatchHarness()
    ledger, anchor = tmp_path / "admission.db", tmp_path / "anchor.db"
    store = SQLiteAdmissionStore(ledger, anchor, KEY)
    store.initialize(now=1)
    control = EffectAdmissionController(store)
    control.enable(workflow_id=h.binding.workflow_id, now=2)
    generation = control.require_admitted(h.binding.workflow_id).generation
    preview = h.adapter(admission=control, admission_generation=generation)
    token = ApprovalTokenSigner(KEY, clock=lambda: h.now).issue(
        action_digest=preview.action_digest(h.arguments),
        reviewer_role=ROLE,
        expires_at=100,
        nonce_source=lambda n: b"d" * n,
    )
    authority = replace(h.authority, approval=token)
    append = h.effects.append

    def stop_after_append(checkpoint):
        append(checkpoint)
        if checkpoint.phase.value == stop_phase:
            control.stop(now=3)

    h.effects.append = stop_after_append
    result = h.adapter(
        authority=authority,
        admission=control,
        admission_generation=generation,
    ).dispatch(h.arguments)
    assert h.tools.calls == (1 if stop_phase is None else 0)
    assert result.outcome.outcome_class.value == (
        "success" if stop_phase is None else "policy_denied"
    )
    assert PRIVATE not in json.dumps(result.to_dict())
    if stop_phase is not None:
        reopened = EffectAdmissionController(SQLiteAdmissionStore(ledger, anchor, KEY))
        assert reopened.status(h.binding.workflow_id).state is AdmissionState.STOPPED
        retry = h.adapter(
            authority=authority,
            admission=reopened,
            admission_generation=generation,
        ).dispatch(h.arguments)
        assert retry.outcome.outcome_class.value == "policy_denied"
        assert h.tools.calls == 0
