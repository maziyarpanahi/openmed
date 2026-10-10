"""Run the synthetic governed clinical workflow without network or patient data.

Run from the repository root with
``python -m examples.agent.governed_clinical_workflow``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from openmed.agent.approvals.side_effect_preview import (
    ResourceWrite,
    WorkflowState,
    WriteIntent,
    WriteKind,
    render_side_effect_preview,
    require_current_preview,
)
from openmed.agent.approvals.tokens import (
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
    dispatch_with_approval_token,
)
from openmed.agent.audit.action_ledger import ActionLedger, ActionState
from openmed.agent.audit.replay_verifier import (
    FrozenReplayEvidence,
    SignedReplayManifest,
    capture_replay_step,
    verify_replay,
)
from openmed.agent.permissions.grants import (
    CapabilityGrantRequest,
    CapabilityGrantSigner,
    CapabilityGrantVerifier,
    dispatch_with_capability_grant,
)
from openmed.agent.tools import plan_data_projection
from openmed.agent.workflows.recovery import (
    CheckpointJournal,
    EffectKind,
    RecoveryCheckpoint,
    RecoveryDisposition,
    RecoveryPhase,
    advance_checkpoint,
    recover_workflow,
)
from openmed.interop.fhir.concurrency_guard import VersionEvidence, guard_update
from openmed.interop.fhir.conditional_writes import (
    ConditionalWriteKind,
    assess_matches,
    build_conditional_write_plan,
)
from openmed.interop.omop import (
    OmopMutation,
    OmopMutationBatch,
    VocabularyConcept,
    VocabularySnapshot,
)
from openmed.interop.omop_rollback_manifest import (
    OmopRollbackInstruction,
    RollbackStrategy,
    build_omop_rollback_manifest,
)
from tests.fixtures.agent.governed_workflow import (
    ACTION_ID,
    ACTOR_ROLE,
    DATA_CLASS,
    EXPIRES_AT,
    GRANT_CONSTRAINT,
    KEY,
    NOW,
    PRIVATE_MARKER,
    PURPOSE,
    REVIEWER_ROLE,
    RUN_ID,
    WORKFLOW_ID,
    SyntheticEffectSink,
    reviewed_tool_schema,
    synthetic_effects,
)


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def run_synthetic_workflow(directory: Path, *, interrupt_after: int = 0) -> dict:
    """Exercise reviewed local, FHIR, and OMOP effects with restart recovery.

    Args:
        directory: New local directory for content-free ledger and checkpoints.
        interrupt_after: Number of effects committed before a simulated restart.

    Returns:
        Deterministic, content-free evidence for comparison across restarts.
    """

    if interrupt_after not in range(4):
        raise ValueError("invalid_interruption_point")
    directory.mkdir(parents=True, exist_ok=True)
    grant = CapabilityGrantSigner(KEY).issue([GRANT_CONSTRAINT], expires_at=EXPIRES_AT)
    request = CapabilityGrantRequest(**GRANT_CONSTRAINT.to_dict())
    grant_verifier = CapabilityGrantVerifier(KEY)
    projection = plan_data_projection(
        reviewed_tool_schema(),
        workflow_purpose=PURPOSE,
        granted_data_classes=(DATA_CLASS,),
    )
    assert projection.field_paths == ("/status",)

    intent = WriteIntent(
        action_id="11111111-1111-4111-8111-111111111111",
        writes=(
            ResourceWrite(
                WriteKind.UPDATE,
                "Observation",
                "res_" + "1" * 32,
                {"status": "preliminary", "valueString": PRIVATE_MARKER},
                {"status": "final", "valueString": PRIVATE_MARKER},
            ),
            ResourceWrite(
                WriteKind.CREATE,
                "OmopPerson",
                "res_" + "2" * 32,
                None,
                {"personSourceValue": PRIVATE_MARKER},
            ),
        ),
        workflow_before=WorkflowState.AWAITING_REVIEW,
        workflow_after=WorkflowState.APPROVED,
    )
    preview = render_side_effect_preview(intent, secret=KEY)
    require_current_preview(
        intent,
        observed_before=(dict(intent.writes[0].before), None),
        observed_workflow_state=WorkflowState.AWAITING_REVIEW,
        approved_digest=preview.digest,
        secret=KEY,
    )
    action_digest = "sha256:" + preview.digest
    frozen = FrozenReplayEvidence(
        grant,
        request,
        b"synthetic-tool-contract-v1",
        b"synthetic-policy-v1",
        b"pending",
        b"synthetic-model-v1",
    )
    first_step = capture_replay_step(ACTION_ID, NOW, frozen, commitment_key=KEY)
    ledger = ActionLedger(directory / "ledger")
    entry_fields = dict(
        run_id=RUN_ID,
        action_id=ACTION_ID,
        actor_role=ACTOR_ROLE,
        grant_digest=first_step.grant_digest,
        tool_digest=first_step.tool_digest,
        resource_refs=(_digest(b"synthetic-resource"),),
    )
    ledger.record(state=ActionState.PROPOSED, **entry_fields)
    token = ApprovalTokenSigner(KEY).issue(
        action_digest=action_digest,
        reviewer_role=REVIEWER_ROLE,
        expires_at=EXPIRES_AT,
        nonce="nonce_" + "1" * 32,
    )
    dispatched, receipt = dispatch_with_approval_token(
        token,
        action_digest=action_digest,
        reviewer_role=REVIEWER_ROLE,
        verifier=ApprovalTokenVerifier(KEY, InMemoryApprovalNonceStore()),
        dispatch=lambda: dispatch_with_capability_grant(
            grant,
            request,
            grant_verifier,
            lambda: "planned",
            now=NOW,
        ),
        now=NOW,
    )
    assert dispatched == "planned"
    ledger.record(state=ActionState.APPROVED, **entry_fields)

    fhir_plan = build_conditional_write_plan(
        ConditionalWriteKind.UPDATE,
        "Observation",
        {"identifier": "urn:synthetic|reviewed-observation"},
        operation_id=UUID(intent.action_id),
        secret=KEY,
    )
    assess_matches(fhir_plan, 1, search_complete=True).require_ready()
    version = VersionEvidence("4", "2026-01-01T00:00:00Z")
    assert guard_update(version, version).if_match == 'W/"4"'
    batch = OmopMutationBatch(
        (
            OmopMutation.insert(
                "person",
                {
                    "person_id": 101,
                    "person_source_value": PRIVATE_MARKER,
                },
            ),
        )
    )
    omop_preview = batch.preview()
    omop_approval = batch.bind_approval(
        omop_preview,
        approved_preview_digest=omop_preview.preview_digest,
        approval_receipt_digest=receipt.token_digest,
    )
    vocabulary = VocabularySnapshot(
        {"SYNTHETIC": "reference-v1"},
        (VocabularyConcept(101, "SYNTHETIC", standard_concept="S"),),
    )
    rollback = build_omop_rollback_manifest(
        batch,
        vocabulary,
        (
            OmopRollbackInstruction(
                0, RollbackStrategy.DELETE_INSERTED_ROW, _digest(b"before-image")
            ),
        ),
    )

    effects = synthetic_effects()
    checkpoint = RecoveryCheckpoint.create(
        workflow_id=WORKFLOW_ID,
        run_id=RUN_ID,
        sequence=0,
        phase=RecoveryPhase.APPROVAL_RECORDED,
        plan_digest=action_digest,
        effects=effects,
        approval_action_digest=action_digest,
        approval_receipt_digest=receipt.token_digest,
        approval_expires_at=EXPIRES_AT,
    )
    journal = CheckpointJournal(directory / "checkpoints")
    journal.append(checkpoint)
    sink = SyntheticEffectSink()
    decision = recover_workflow(journal.load(), sink.observe(effects), now=NOW)
    assert decision.disposition is RecoveryDisposition.RESUME
    dispatching = advance_checkpoint(checkpoint, decision)
    journal.append(dispatching)
    ledger.record(state=ActionState.ATTEMPTED, **entry_fields)

    def commit(effect) -> None:
        if effect.kind is EffectKind.LOCAL_TOOL:
            sink.apply(effect.idempotency_key, _digest(b"planned"))
        elif effect.kind is EffectKind.FHIR_WRITE:
            sink.apply(
                effect.idempotency_key,
                _digest(fhir_plan.idempotency_key.encode("ascii")),
            )
        else:
            sink.omop_key = effect.idempotency_key
            result = batch.commit(sink, approval=omop_approval)
            assert result.status.value == "committed"

    for effect in effects[:interrupt_after]:
        commit(effect)

    restarted = CheckpointJournal(directory / "checkpoints")
    lineage = restarted.load()
    recovered = recover_workflow(lineage, sink.observe(effects), now=NOW)
    reconciled = advance_checkpoint(lineage[-1], recovered)
    restarted.append(reconciled)
    if recovered.disposition is RecoveryDisposition.RESUME:
        retry = set(recovered.retry_idempotency_keys)
        for effect in effects:
            if effect.idempotency_key in retry:
                commit(effect)
        final_decision = recover_workflow(
            restarted.load(), sink.observe(effects), now=NOW
        )
        restarted.append(advance_checkpoint(restarted.load()[-1], final_decision))
    assert restarted.load()[-1].phase is RecoveryPhase.COMPLETED
    ledger.record(state=ActionState.COMMITTED, **entry_fields)

    frozen = replace(
        frozen,
        tool_response=json.dumps(
            sink.commits, sort_keys=True, separators=(",", ":")
        ).encode("ascii"),
    )
    step = capture_replay_step(ACTION_ID, NOW, frozen, commitment_key=KEY)
    manifest = SignedReplayManifest.sign(
        run_id=RUN_ID,
        ledger_head_digest=ledger.load()[-1].entry_digest,
        steps=(step,),
        key=KEY,
    )
    replay = verify_replay(
        manifest,
        ledger.load(),
        (frozen,),
        signing_key=KEY,
        grant_verifier=grant_verifier,
    )
    assert replay.matched
    assert set(sink.commit_counts.values()) == {1}
    return {
        "effect_count": len(sink.commits),
        "effects": dict(sorted(sink.commits.items())),
        "ledger_head_digest": ledger.load()[-1].entry_digest,
        "replay_artifact_digest": step.artifact_digest,
        "replay_matched": replay.matched,
        "rollback_manifest_digest": rollback.manifest_digest,
        "review_packet": {
            "preview_digest": preview.digest,
            "omop_preview_digest": omop_preview.preview_digest,
            "rollback": rollback.to_dict(),
        },
    }


if __name__ == "__main__":
    with TemporaryDirectory(prefix="openmed-synthetic-governed-") as temporary:
        result = run_synthetic_workflow(Path(temporary), interrupt_after=1)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
