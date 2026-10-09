"""Synthetic HTTP adapter contract checks, not clinical or authority validation."""

import json
from dataclasses import replace

import pytest

from openmed.agent.action_phases import ActionPhase
from openmed.agent.approvals.tokens import ApprovalReceipt
from openmed.agent.correlation import ActionId, RunId
from openmed.agent.identifiers import ToolId, WorkflowId
from openmed.agent.outcomes import OutcomeClass, WorkflowOutcome
from openmed.agent.workflows.recovery import CompensationLimit, EffectKind, EffectRecord
from openmed.service.governed_workflows import (
    MAX_WORKFLOW_REQUEST_BYTES,
    WorkflowReference,
    WorkflowServiceError,
    WorkflowView,
    parse_workflow_json,
    validate_workflow_receipt,
    validate_workflow_view,
    workflow_receipt_digest,
)

DIGEST = "sha256:" + "a" * 64
STATE = "sha256:" + "b" * 64


def reference():
    return WorkflowReference(
        RunId.parse("run_" + "1" * 32),
        WorkflowId.parse("workflow:test.example/review@1.0.0"),
        DIGEST,
        STATE,
        "req_" + "2" * 32,
    )


def effect(ordinal=0, action="3"):
    return EffectRecord.create(
        ordinal=ordinal,
        run_id=reference().run_id,
        action_id=ActionId.parse("act_" + action * 32),
        tool_id=ToolId.parse("tool:test.example/fhir@1.0.0"),
        kind=EffectKind.FHIR_WRITE,
        operation_digest=DIGEST,
        approval_required=True,
        compensation_limit=CompensationLimit.PROPOSE_ONLY,
    )


def receipt():
    return ApprovalReceipt(DIGEST, "role:test.example/reviewer", STATE, 100, 200)


def test_reference_roundtrip_and_closed_repr():
    value = reference()
    assert WorkflowReference.from_dict(value.to_dict()) == value
    assert value.run_id.value not in repr(value)
    assert value.action_digest not in repr(value)
    value.require_mutation()


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": "unknown"},
        {"run_id": "patient"},
        {"workflow_id": "bad"},
        {"action_digest": True},
        {"expected_state_digest": "bad"},
        {"request_id": "patient"},
        {"role": "admin"},
        {"approval": True},
        {"clinical_text": "Synthetic private marker"},
    ],
)
def test_reference_rejects_untrusted_fields(change):
    value = reference().to_dict()
    value.update(change)
    with pytest.raises(WorkflowServiceError) as error:
        WorkflowReference.from_dict(value)
    assert error.value.code == "workflow_invalid_input"
    assert "Synthetic private marker" not in str(error.value)


@pytest.mark.parametrize("field", ["expected_state_digest", "request_id"])
def test_mutation_requires_state_and_idempotency(field):
    value = replace(reference(), **{field: None})
    with pytest.raises(WorkflowServiceError):
        value.require_mutation()


def test_existing_effect_and_outcome_records_are_preserved():
    value = WorkflowView(
        reference(),
        STATE,
        ActionPhase.WAITING_REVIEW,
        (effect(),),
        WorkflowOutcome(OutcomeClass.REVIEW_REQUIRED, "human_gate"),
    )
    wire = validate_workflow_view(value, reference()).to_dict()
    assert wire["effects"] == [effect().to_dict()]
    assert wire["outcome"] == value.outcome.to_dict()
    assert wire["proposed_effect_count"] == 1
    assert wire["committed_effect_count"] == 0
    assert value.to_dict() == wire
    assert "Synthetic private marker" not in json.dumps(wire)


def test_response_rejects_pending_effect_in_completed_phase():
    with pytest.raises(WorkflowServiceError):
        WorkflowView(reference(), STATE, ActionPhase.COMPLETED, (effect(),))


def test_response_rejects_duplicate_actions_and_bad_order():
    with pytest.raises(WorkflowServiceError):
        WorkflowView(reference(), STATE, ActionPhase.READY, (effect(), effect(1)))
    with pytest.raises(WorkflowServiceError):
        WorkflowView(reference(), STATE, ActionPhase.READY, (effect(1),))


def test_response_rejects_effect_bound_to_another_run():
    e = effect()
    object.__setattr__(e, "idempotency_key", "idem_" + "f" * 64)
    with pytest.raises(WorkflowServiceError):
        WorkflowView(reference(), STATE, ActionPhase.READY, (e,))


@pytest.mark.parametrize(
    "change",
    [
        {"run_id": RunId.parse("run_" + "f" * 32)},
        {"workflow_id": WorkflowId.parse("workflow:test.example/other")},
        {"action_digest": "sha256:" + "f" * 64},
    ],
)
def test_service_result_cannot_cross_run_workflow_or_action(change):
    view = WorkflowView(replace(reference(), **change), STATE, ActionPhase.READY)
    with pytest.raises(WorkflowServiceError) as error:
        validate_workflow_view(view, reference())
    assert error.value.code == "workflow_conflict"


def test_modified_frozen_service_results_are_revalidated():
    view = WorkflowView(reference(), STATE, ActionPhase.READY)
    object.__setattr__(view, "state_digest", "Synthetic private marker")
    with pytest.raises(WorkflowServiceError) as error:
        validate_workflow_view(view, reference())
    assert error.value.code == "workflow_invalid_result"
    assert "Synthetic private marker" not in str(error.value)


@pytest.mark.parametrize(
    "now,code",
    [
        (99, "workflow_receipt_future"),
        (200, "workflow_receipt_expired"),
        (201, "workflow_receipt_expired"),
        (True, "workflow_service_failed"),
    ],
)
def test_receipt_time_limits(now, code):
    with pytest.raises(WorkflowServiceError) as error:
        validate_workflow_receipt(reference(), receipt(), now=now)
    assert error.value.code == code


def test_receipt_is_actual_public_contract_and_requires_exact_action():
    value = receipt()
    assert validate_workflow_receipt(
        reference(), value, now=100
    ) == workflow_receipt_digest(value)
    assert ApprovalReceipt.from_dict(value.to_dict()) == value
    with pytest.raises(WorkflowServiceError) as error:
        validate_workflow_receipt(
            reference(), replace(value, action_digest=STATE), now=100
        )
    assert error.value.code == "workflow_conflict"


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b"[]",
        b"null",
        b"\xff",
        b"{" + b'"x":[' * 10 + b"0" + b"]}" * 10,
        json.dumps({"x": list(range(257))}).encode(),
    ],
)
def test_strict_json_rejects_unsafe_or_excessive_input(raw):
    with pytest.raises(WorkflowServiceError) as error:
        parse_workflow_json(raw)
    assert error.value.code == "workflow_invalid_input"


def test_json_byte_limit_and_roundtrip():
    value = reference().to_dict()
    assert parse_workflow_json(json.dumps(value).encode()) == value
    with pytest.raises(WorkflowServiceError) as error:
        parse_workflow_json(b" " * (MAX_WORKFLOW_REQUEST_BYTES + 1))
    assert error.value.code == "workflow_request_too_large"
