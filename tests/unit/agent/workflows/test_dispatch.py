"""Independent authority mutations and content-free recovery negative controls."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from openmed.agent.action_phases import ActionPhase
from openmed.agent.approvals.tokens import ApprovalTokenSigner
from openmed.agent.correlation import RunId
from openmed.agent.outcomes import OutcomeClass
from openmed.agent.permissions.access_tickets import RecordSelector, ToolAction
from openmed.agent.permissions.grants import CapabilityGrantSigner
from openmed.agent.workflows.recovery import (
    EffectObservation,
    ObservationState,
    RecoveryDisposition,
    RecoveryPhase,
    RecoveryReason,
    validate_checkpoint_lineage,
)
from tests.fixtures.agent.guarded_dispatch import (
    ACTION,
    DATA,
    KEY,
    PRIVATE,
    ROLE,
    DispatchHarness,
)


def assert_private_free(result):
    serialized = json.dumps(result.to_dict()) + repr(result)
    for secret in (
        PRIVATE,
        "Alice",
        "1970",
        "/private/record",
        "API_SECRET",
        "private_output",
    ):
        assert secret not in serialized


def test_execute_once_and_bind_approval_effect_outcome():
    harness = DispatchHarness()
    adapter = harness.adapter()
    result = adapter.dispatch(harness.arguments)
    assert harness.tools.calls == 1
    assert harness.tools.arguments == [harness.arguments]
    assert result.phase is ActionPhase.COMPLETED
    assert result.outcome.outcome_class is OutcomeClass.SUCCESS
    assert result.recovery.disposition is RecoveryDisposition.COMPLETE
    assert result.checkpoint.plan_digest == adapter.action_digest(harness.arguments)
    assert (
        result.checkpoint.effects[0].operation_digest == result.checkpoint.plan_digest
    )
    assert result.checkpoint.approval_action_digest == result.checkpoint.plan_digest
    validate_checkpoint_lineage(next(iter(harness.effects.lineages.values())))
    assert_private_free(result)
    assert (
        adapter.dispatch(harness.arguments).outcome.outcome_class
        is OutcomeClass.REVIEW_REQUIRED
    )
    assert harness.tools.calls == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_grant",
        "signature",
        "grant_expiry",
        "grant_scope",
        "grant_request",
        "missing_ticket",
        "ticket_expiry",
        "run",
        "purpose",
        "data",
        "selector",
        "tool_action",
        "ticket_request",
        "projection",
        "registered_identity",
        "missing_approval",
        "approval_signature",
        "approval_expiry",
        "approval_role",
        "approval_arguments",
        "extra_argument",
        "argument_type",
        "nan",
    ],
)
def test_independent_corruption_never_invokes(mutation):
    h = DispatchHarness()
    a = h.authority
    if mutation == "missing_grant":
        a = replace(a, grant=None)
    elif mutation == "signature":
        a = replace(a, grant=replace(a.grant, signature="hmac-sha256:" + "0" * 64))
    elif mutation == "grant_expiry":
        a = replace(
            a,
            grant=CapabilityGrantSigner(KEY).issue(
                [a.grant_request.as_constraint()], expires_at=10
            ),
        )
    elif mutation == "grant_scope":
        request = replace(a.grant_request, action="action:org.example/other@1.0.0")
        a = replace(
            a,
            grant=CapabilityGrantSigner(KEY).issue(
                [request.as_constraint()], expires_at=100
            ),
        )
    elif mutation == "grant_request":
        a = replace(
            a,
            grant_request=replace(a.grant_request, tool="tool:org.example/other@1.0.0"),
        )
    elif mutation == "missing_ticket":
        a = replace(a, ticket=None)
    elif mutation == "ticket_expiry":
        a = replace(a, ticket=replace(a.ticket, expires_at=10))
    elif mutation == "run":
        a = replace(a, ticket=replace(a.ticket, run_id=RunId("run_" + "c" * 32)))
    elif mutation == "purpose":
        a = replace(
            a, ticket=replace(a.ticket, purpose="purpose:org.example/other@1.0.0")
        )
    elif mutation == "data":
        a = replace(
            a,
            ticket=replace(
                a.ticket, permitted_data_classes=("data:org.example/other@1.0.0",)
            ),
        )
    elif mutation == "selector":
        a = replace(
            a,
            ticket=replace(
                a.ticket,
                record_selectors=(
                    RecordSelector(
                        a.ticket.record_selectors[0].kind, "hmac-sha256:" + "2" * 64
                    ),
                ),
            ),
        )
    elif mutation == "tool_action":
        a = replace(
            a,
            ticket=replace(
                a.ticket,
                permitted_tool_actions=(
                    ToolAction("tool:org.example/other@1.0.0", ACTION),
                ),
            ),
        )
    elif mutation == "ticket_request":
        a = replace(
            a,
            ticket_request=replace(
                a.ticket_request, purpose="purpose:org.example/other@1.0.0"
            ),
        )
    elif mutation == "projection":
        # A valid plan from another schema still cannot authorize this tool.
        other = DispatchHarness(
            schema={
                **h.spec.input_schema,
                "properties": {"other": h.spec.input_schema["properties"]["text"]},
                "required": ["other"],
            }
        )
        a = replace(a, projection=other.authority.projection)
    elif mutation == "registered_identity":
        h.tools.get = lambda name: replace(h.spec, version="2.0.0")
    elif mutation == "missing_approval":
        a = replace(a, approval=None)
    elif mutation == "approval_signature":
        a = replace(
            a, approval=replace(a.approval, signature="hmac-sha256:" + "0" * 64)
        )
    elif mutation in {"approval_expiry", "approval_role"}:
        a = replace(
            a,
            approval=ApprovalTokenSigner(KEY).issue(
                action_digest=a.approval.action_digest,
                reviewer_role=ROLE
                if mutation == "approval_expiry"
                else "role:org.example/operator@1.0.0",
                expires_at=10 if mutation == "approval_expiry" else 100,
                nonce_source=lambda n: b"z" * n,
            ),
        )
    elif mutation == "approval_arguments":
        h.arguments = {"text": PRIVATE + " changed"}
    elif mutation == "extra_argument":
        h.arguments[PRIVATE] = PRIVATE
    elif mutation == "argument_type":
        h.arguments["text"] = 42
    elif mutation == "nan":
        h.arguments["text"] = float("nan")
    result = h.adapter(authority=a).dispatch(h.arguments)
    assert h.tools.calls == 0
    assert result.outcome.outcome_class is not OutcomeClass.SUCCESS
    assert_private_free(result)


@pytest.mark.parametrize(
    "phase",
    [
        RecoveryPhase.APPROVAL_RECORDED,
        RecoveryPhase.DISPATCHING,
        RecoveryPhase.COMPLETED,
        RecoveryPhase.RECONCILING,
        RecoveryPhase.ABORTED,
    ],
)
def test_storage_failure_cannot_repeat_tool(phase):
    h = DispatchHarness()
    h.effects.fail_phase = phase
    if phase is RecoveryPhase.RECONCILING:
        h.tools.uncertain = True
    elif phase is RecoveryPhase.ABORTED:
        h.authority = replace(h.authority, approval=None)
    adapter = h.adapter()
    first = adapter.dispatch(h.arguments)
    second = adapter.dispatch(h.arguments)
    assert h.tools.calls <= 1
    if phase in {RecoveryPhase.APPROVAL_RECORDED, RecoveryPhase.DISPATCHING}:
        assert h.tools.calls == 0
    assert second.outcome.outcome_class is OutcomeClass.REVIEW_REQUIRED
    assert_private_free(first)
    assert_private_free(second)


def test_uncertain_claim_never_invokes_or_reclaims():
    h = DispatchHarness()
    h.effects.fail_claim = True
    first = h.adapter().dispatch(h.arguments)
    second = h.adapter().dispatch(h.arguments)
    assert h.tools.calls == 0
    assert first.recovery.reason is RecoveryReason.AMBIGUOUS_EFFECT
    assert second.recovery.reason is RecoveryReason.AMBIGUOUS_EFFECT


@pytest.mark.parametrize(
    "boundary", ["before", "approval", "dispatch", "during", "after"]
)
def test_cancellation_is_typed_and_private_free(boundary):
    h = DispatchHarness()
    if boundary == "before":
        h.cancelled = True
    elif boundary == "approval":
        consume = h.approvals.consume

        def cancel_after_approval(*args, **kwargs):
            result = consume(*args, **kwargs)
            h.cancelled = True
            return result

        h.approvals.consume = cancel_after_approval
    elif boundary == "dispatch":
        append = h.effects.append

        def cancel_after_append(checkpoint):
            append(checkpoint)
            if checkpoint.phase is RecoveryPhase.DISPATCHING:
                h.cancelled = True

        h.effects.append = cancel_after_append
    elif boundary == "during":
        h.tools.error = asyncio.CancelledError(PRIVATE)
    else:
        h.tools.after_invoke = lambda: setattr(h, "cancelled", True)
    result = h.adapter().dispatch(h.arguments)
    assert h.tools.calls == (1 if boundary in {"during", "after"} else 0)
    assert result.phase is ActionPhase.ABORTED
    assert result.recovery.reason is RecoveryReason.WORKFLOW_ABORTED
    assert result.recovery.disposition is RecoveryDisposition.REVIEW_REQUIRED
    assert_private_free(result)


@pytest.mark.parametrize(
    "mode", ["uncertain", "exception", "observe_exception", "mismatch", "absent"]
)
def test_commit_uncertainty_requires_review_without_retry(mode):
    h = DispatchHarness()
    if mode == "uncertain":
        h.tools.uncertain = True
    elif mode == "exception":
        h.tools.error = RuntimeError(PRIVATE)
    elif mode == "observe_exception":

        def fail(effect):
            raise RuntimeError(PRIVATE)

        h.effects.observe = fail
    else:
        observe = h.effects.observe
        h.effects.observe = lambda effect: (
            replace(
                observe(effect),
                operation_digest="sha256:" + "0" * 64,
            )
            if mode == "mismatch"
            else EffectObservation(
                effect.action_id,
                effect.operation_digest,
                effect.idempotency_key,
                ObservationState.ABSENT,
            )
        )
    adapter = h.adapter()
    result = adapter.dispatch(h.arguments)
    assert result.phase is ActionPhase.WAITING_REVIEW
    assert result.recovery.reason is RecoveryReason.AMBIGUOUS_EFFECT
    adapter.dispatch(h.arguments)
    assert h.tools.calls == 1
    assert_private_free(result)


def test_concurrent_dispatch_and_changed_arguments_share_action_reservation():
    h = DispatchHarness()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: h.adapter().dispatch(h.arguments), range(2)))
    assert h.tools.calls == 1
    assert sum(r.outcome.outcome_class is OutcomeClass.SUCCESS for r in results) == 1
    changed = {"text": "different synthetic arguments"}
    token = ApprovalTokenSigner(KEY).issue(
        action_digest=h.adapter().action_digest(changed),
        reviewer_role=ROLE,
        expires_at=100,
        nonce_source=lambda n: b"y" * n,
    )
    result = h.adapter(authority=replace(h.authority, approval=token)).dispatch(changed)
    assert result.recovery.reason is RecoveryReason.AMBIGUOUS_EFFECT
    assert h.tools.calls == 1


def test_write_cannot_disable_approval_but_read_only_policy_can():
    write = DispatchHarness(approval_required=False)
    result = write.adapter(authority=replace(write.authority, approval=None)).dispatch(
        write.arguments
    )
    assert result.outcome.outcome_class is OutcomeClass.POLICY_DENIED
    assert write.tools.calls == 0
    read = DispatchHarness(read_only=True, approval_required=False)
    result = read.adapter(authority=replace(read.authority, approval=None)).dispatch(
        read.arguments
    )
    assert result.outcome.outcome_class is OutcomeClass.SUCCESS
    assert result.checkpoint.approval_receipt_digest is None


@pytest.mark.parametrize(
    "phase", [RecoveryPhase.APPROVAL_RECORDED, RecoveryPhase.DISPATCHING]
)
def test_expiry_while_recording_approval_prevents_invocation(phase):
    h = DispatchHarness()
    append = h.effects.append

    def delayed(checkpoint):
        append(checkpoint)
        if checkpoint.phase is phase:
            h.now = 100

    h.effects.append = delayed
    result = h.adapter().dispatch(h.arguments)
    assert h.tools.calls == 0
    assert_private_free(result)


def test_nested_unprojected_field_cannot_hide_under_authorized_parent():
    h = DispatchHarness(
        schema={
            "type": "object",
            "x-openmed-purpose": "purpose:org.example/summary@1.0.0",
            "properties": {
                "text": {
                    "type": "object",
                    "x-openmed-purpose": "purpose:org.example/summary@1.0.0",
                    "x-openmed-minimum-data": "required",
                    "x-openmed-data-class": DATA,
                    "properties": {
                        "allowed": {
                            "type": "string",
                            "x-openmed-purpose": "purpose:org.example/summary@1.0.0",
                            "x-openmed-minimum-data": "required",
                            "x-openmed-data-class": DATA,
                        }
                    },
                    "additionalProperties": False,
                    "required": ["allowed"],
                }
            },
            "required": ["text"],
            "additionalProperties": False,
        }
    )
    result = h.adapter().dispatch({"text": {PRIVATE: PRIVATE}})
    assert h.tools.calls == 0
    assert_private_free(result)
