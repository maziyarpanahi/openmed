"""Synthetic approval, effect and recovery boundary composition."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from openmed.agent.admission import (
    AdmissionError,
    EffectAdmissionController,
    SQLiteAdmissionStore,
    dispatch_with_admission,
)
from openmed.agent.approvals.tokens import (
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
    dispatch_with_approval_token,
)
from openmed.agent.correlation import ActionId, RunId
from openmed.agent.identifiers import ToolId, WorkflowId
from openmed.agent.workflows.recovery import (
    CompensationLimit,
    EffectKind,
    EffectObservation,
    EffectRecord,
    ObservationState,
    RecoveryCheckpoint,
    RecoveryDisposition,
    RecoveryPhase,
    recover_workflow,
)

pytestmark = pytest.mark.integration
KEY = b"synthetic-admission-integration-key"
WORKFLOW = WorkflowId("workflow:org.example/synthetic-writes")
ACTION = "sha256:" + "a" * 64
ROLE = "role:org.example/operator"


def controller_at(tmp_path: Path) -> EffectAdmissionController:
    return EffectAdmissionController(
        SQLiteAdmissionStore(tmp_path / "ledger.db", tmp_path / "anchor.db", KEY)
    )


@pytest.mark.parametrize("kind", [EffectKind.FHIR_WRITE, EffectKind.OMOP_BATCH])
def test_stop_after_preview_blocks_valid_approval_before_effect(
    tmp_path: Path, kind: EffectKind
) -> None:
    store = SQLiteAdmissionStore(tmp_path / "ledger.db", tmp_path / "anchor.db", KEY)
    store.initialize(now=100)
    controller = EffectAdmissionController(store)
    controller.enable(workflow_id=WORKFLOW, now=101)
    generation = controller.require_admitted(WORKFLOW).generation
    token = ApprovalTokenSigner(KEY).issue(
        action_digest=ACTION, reviewer_role=ROLE, expires_at=200
    )
    verifier = ApprovalTokenVerifier(KEY, InMemoryApprovalNonceStore())
    effect = Mock(name=kind.value, return_value="opaque-commit-reference")
    controller.stop(now=102)
    with pytest.raises(AdmissionError, match="admission_stopped"):
        dispatch_with_approval_token(
            token,
            action_digest=ACTION,
            reviewer_role=ROLE,
            verifier=verifier,
            now=103,
            dispatch=lambda: dispatch_with_admission(
                effect,
                workflow_id=WORKFLOW,
                admission=controller,
                generation=generation,
            ),
        )
    effect.assert_not_called()
    # Read-only preview/review work remains callable; no admission check is used.
    preview = Mock(return_value=ACTION)
    assert preview() == ACTION


def test_restart_cannot_resume_stopped_recovery(tmp_path: Path) -> None:
    store = SQLiteAdmissionStore(tmp_path / "ledger.db", tmp_path / "anchor.db", KEY)
    store.initialize(now=100)
    controller = EffectAdmissionController(store)
    controller.enable(workflow_id=WORKFLOW, now=101)
    generation = controller.require_admitted(WORKFLOW).generation
    run = RunId("run_" + "1" * 32)
    effect = EffectRecord.create(
        ordinal=0,
        run_id=run,
        action_id=ActionId("act_" + "2" * 32),
        tool_id=ToolId("tool:org.example/fhir"),
        kind=EffectKind.FHIR_WRITE,
        operation_digest=ACTION,
        approval_required=True,
        compensation_limit=CompensationLimit.PROPOSE_ONLY,
    )
    checkpoint = RecoveryCheckpoint.create(
        workflow_id=WORKFLOW,
        run_id=run,
        sequence=0,
        phase=RecoveryPhase.APPROVAL_RECORDED,
        plan_digest=ACTION,
        effects=(effect,),
        approval_action_digest=ACTION,
        approval_receipt_digest="sha256:" + "b" * 64,
        approval_expires_at=200,
    )
    observation = EffectObservation(
        action_id=effect.action_id,
        operation_digest=ACTION,
        idempotency_key=effect.idempotency_key,
        state=ObservationState.ABSENT,
    )
    decision = recover_workflow((checkpoint,), (observation,), now=103)
    assert decision.disposition is RecoveryDisposition.RESUME
    # Recovery planning is read-only; a plan never grants admission to resume.
    controller.stop(now=102)
    restarted = controller_at(tmp_path)
    with pytest.raises(AdmissionError, match="admission_stopped"):
        restarted.require_admitted(checkpoint.workflow_id, generation=generation)
    restarted.enable(now=104)
    with pytest.raises(AdmissionError, match="stale_generation"):
        restarted.require_admitted(checkpoint.workflow_id, generation=generation)
    fresh = restarted.require_admitted(WORKFLOW).generation
    callback = Mock(return_value="opaque-commit-reference")
    assert (
        dispatch_with_admission(
            callback, workflow_id=WORKFLOW, admission=restarted, generation=fresh
        )
        == "opaque-commit-reference"
    )
    assert callback.call_count == 1
