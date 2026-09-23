"""Tests for content-free durable workflow recovery."""

from __future__ import annotations

import json
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from openmed.agent.correlation import ActionId, RunId
from openmed.agent.identifiers import ToolId, WorkflowId
from openmed.agent.workflows.recovery import (
    RECOVERY_CHECKPOINT_SCHEMA_VERSION,
    CheckpointJournal,
    CompensationLimit,
    EffectKind,
    EffectObservation,
    EffectRecord,
    EffectState,
    ObservationState,
    RecoveryCheckpoint,
    RecoveryDecision,
    RecoveryDisposition,
    RecoveryError,
    RecoveryPhase,
    RecoveryReason,
    advance_checkpoint,
    derive_idempotency_key,
    recover_workflow,
    validate_checkpoint_lineage,
)

RUN_ID = RunId("run_" + "1" * 32)
WORKFLOW_ID = WorkflowId("workflow:org.openmed/reconcile@1.0.0")
PLAN_DIGEST = "sha256:" + "a" * 64
APPROVAL_RECEIPT_DIGEST = "sha256:" + "b" * 64
COMMIT_DIGESTS = tuple("sha256:" + digit * 64 for digit in ("4", "5", "6"))


def _action(index: int) -> ActionId:
    return ActionId(f"act_{index:032x}")


def _tool(index: int) -> ToolId:
    return ToolId(f"tool:org.openmed/tool-{index}@1.0.0")


def _effects(*, approval_required: bool = True) -> tuple[EffectRecord, ...]:
    return tuple(
        EffectRecord.create(
            ordinal=index,
            run_id=RUN_ID,
            action_id=_action(index + 1),
            tool_id=_tool(index + 1),
            kind=kind,
            operation_digest=f"sha256:{index + 1:064x}",
            approval_required=approval_required,
            compensation_limit=(
                CompensationLimit.NONE
                if kind is EffectKind.LOCAL_TOOL
                else CompensationLimit.PROPOSE_ONLY
            ),
        )
        for index, kind in enumerate(EffectKind)
    )


def _checkpoint(
    *,
    phase: RecoveryPhase = RecoveryPhase.APPROVAL_RECORDED,
    approval: bool = True,
    effects: tuple[EffectRecord, ...] | None = None,
) -> RecoveryCheckpoint:
    return RecoveryCheckpoint.create(
        workflow_id=WORKFLOW_ID,
        run_id=RUN_ID,
        sequence=0,
        phase=phase,
        plan_digest=PLAN_DIGEST,
        effects=_effects() if effects is None else effects,
        approval_action_digest=PLAN_DIGEST if approval else None,
        approval_receipt_digest=APPROVAL_RECEIPT_DIGEST if approval else None,
        approval_expires_at=100 if approval else None,
    )


def _observations(
    effects: tuple[EffectRecord, ...],
    states: tuple[ObservationState, ...],
) -> tuple[EffectObservation, ...]:
    return tuple(
        EffectObservation(
            action_id=effect.action_id,
            operation_digest=effect.operation_digest,
            idempotency_key=effect.idempotency_key,
            state=state,
            commit_evidence_digest=(
                COMMIT_DIGESTS[index] if state is ObservationState.COMMITTED else None
            ),
        )
        for index, (effect, state) in enumerate(zip(effects, states, strict=True))
    )


def test_idempotency_keys_are_stable_and_bind_every_effect_identity() -> None:
    effect = _effects()[0]
    expected = derive_idempotency_key(
        run_id=RUN_ID,
        action_id=effect.action_id,
        tool_id=effect.tool_id,
        kind=effect.kind,
        operation_digest=effect.operation_digest,
    )

    assert effect.idempotency_key == expected
    assert expected.startswith("idem_")
    changed = derive_idempotency_key(
        run_id=RUN_ID,
        action_id=effect.action_id,
        tool_id=effect.tool_id,
        kind=EffectKind.FHIR_WRITE,
        operation_digest=effect.operation_digest,
    )
    assert changed != expected


def test_checkpoint_round_trips_with_only_content_free_metadata() -> None:
    checkpoint = _checkpoint()

    assert RecoveryCheckpoint.from_json(checkpoint.to_json()) == checkpoint
    assert json.loads(checkpoint.to_json()) == checkpoint.to_dict()
    assert checkpoint.schema_version == RECOVERY_CHECKPOINT_SCHEMA_VERSION
    assert "approval_token" not in checkpoint.to_json()
    assert "payload" not in checkpoint.to_json()
    assert "clinical" not in checkpoint.to_json()
    assert "effect_count=3" in repr(checkpoint)
    assert checkpoint.run_id.serialize() not in repr(checkpoint)


def test_checkpoint_rejects_unknown_payload_without_echoing_it() -> None:
    sentinel = "Synthetic Person has condition Z99.999; bearer secret"
    payload = _checkpoint().to_dict()
    payload["tool_payload"] = sentinel

    with pytest.raises(RecoveryError, match="unknown_field") as caught:
        RecoveryCheckpoint.from_dict(payload)

    rendered = "".join(traceback.format_exception(caught.value))
    assert sentinel not in rendered

    token_payload = _checkpoint().to_dict()
    token_payload["approval_token"] = "signed-bearer-value"
    with pytest.raises(RecoveryError, match="unknown_field"):
        RecoveryCheckpoint.from_dict(token_payload)


def test_checkpoint_digest_detects_tampering() -> None:
    payload = _checkpoint().to_dict()
    payload["checkpoint_digest"] = "sha256:" + "f" * 64

    with pytest.raises(RecoveryError, match="checkpoint_digest_mismatch"):
        RecoveryCheckpoint.from_dict(payload)


def test_journal_is_durable_append_only_and_accepts_identical_retry(
    tmp_path: Path,
) -> None:
    journal = CheckpointJournal(tmp_path / "journal")
    first = _checkpoint()
    journal.append(first)
    assert journal.append(first) == first

    decision = recover_workflow(
        (first,),
        _observations(first.effects, (ObservationState.ABSENT,) * 3),
        now=10,
    )
    second = advance_checkpoint(first, decision)
    journal.append(second)

    assert CheckpointJournal(tmp_path / "journal").load() == (first, second)
    assert second.previous_checkpoint_digest == first.checkpoint_digest
    assert second.recovery_evidence_digest == decision.evidence_digest


def test_journal_rejects_conflict_and_tampered_file(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal"
    journal = CheckpointJournal(journal_path)
    first = _checkpoint()
    journal.append(first)

    conflicting = RecoveryCheckpoint.create(
        workflow_id=WORKFLOW_ID,
        run_id=RUN_ID,
        sequence=0,
        phase=RecoveryPhase.PLANNED,
        plan_digest=PLAN_DIGEST,
        effects=_effects(approval_required=False),
    )
    with pytest.raises(RecoveryError, match="checkpoint_conflict"):
        journal.append(conflicting)

    path = journal_path / "checkpoint-00000000000000000000.json"
    payload = json.loads(path.read_text())
    payload["checkpoint_digest"] = "sha256:" + "f" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(RecoveryError, match="checkpoint_digest_mismatch"):
        journal.load()


def test_recovery_reconciles_commits_and_retries_only_proven_absent_effects() -> None:
    checkpoint = _checkpoint()
    observations = _observations(
        checkpoint.effects,
        (
            ObservationState.COMMITTED,
            ObservationState.ABSENT,
            ObservationState.COMMITTED,
        ),
    )

    decision = recover_workflow((checkpoint,), observations, now=10)

    assert decision.disposition is RecoveryDisposition.RESUME
    assert decision.reason is RecoveryReason.SAFE_TO_RESUME
    assert decision.retry_effect_ids == (_action(2).serialize(),)
    assert decision.retry_idempotency_keys == (checkpoint.effects[1].idempotency_key,)
    assert decision.committed_effects == (
        (_action(1).serialize(), COMMIT_DIGESTS[0]),
        (_action(3).serialize(), COMMIT_DIGESTS[2]),
    )
    assert decision.to_json() == decision.to_json()

    advanced = advance_checkpoint(checkpoint, decision)
    assert tuple(effect.state for effect in advanced.effects) == (
        EffectState.COMMITTED,
        EffectState.PENDING,
        EffectState.COMMITTED,
    )
    assert validate_checkpoint_lineage((checkpoint, advanced))[-1] == advanced


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (
            lambda observation: EffectObservation(
                action_id=observation.action_id,
                operation_digest=observation.operation_digest,
                idempotency_key=observation.idempotency_key,
                state=ObservationState.AMBIGUOUS,
            ),
            RecoveryReason.AMBIGUOUS_EFFECT,
        ),
        (
            lambda observation: EffectObservation(
                action_id=observation.action_id,
                operation_digest="sha256:" + "e" * 64,
                idempotency_key=observation.idempotency_key,
                state=ObservationState.ABSENT,
            ),
            RecoveryReason.EFFECT_MISMATCH,
        ),
    ],
)
def test_ambiguous_or_mismatched_effects_fail_closed(
    mutator: Any, reason: RecoveryReason
) -> None:
    checkpoint = _checkpoint()
    observations = list(
        _observations(
            checkpoint.effects,
            (
                ObservationState.COMMITTED,
                ObservationState.ABSENT,
                ObservationState.ABSENT,
            ),
        )
    )
    observations[1] = mutator(observations[1])

    decision = recover_workflow((checkpoint,), observations, now=10)

    assert decision.disposition is RecoveryDisposition.REVIEW_REQUIRED
    assert decision.reason is reason
    assert decision.retry_effect_ids == ()
    assert decision.compensation_effect_ids == ()


def test_partial_fhir_commit_produces_only_a_compensation_proposal() -> None:
    checkpoint = _checkpoint()
    observations = _observations(
        checkpoint.effects,
        (
            ObservationState.ABSENT,
            ObservationState.COMMITTED,
            ObservationState.AMBIGUOUS,
        ),
    )

    decision = recover_workflow((checkpoint,), observations, now=10)

    assert decision.disposition is RecoveryDisposition.REVIEW_REQUIRED
    assert decision.reason is RecoveryReason.AMBIGUOUS_EFFECT
    assert decision.compensation_effect_ids == (_action(2).serialize(),)
    assert decision.retry_effect_ids == ()


@pytest.mark.parametrize(
    ("approval", "now", "reason"),
    [
        (False, 10, RecoveryReason.APPROVAL_MISSING),
        (True, 100, RecoveryReason.APPROVAL_EXPIRED),
        (True, 101, RecoveryReason.APPROVAL_EXPIRED),
    ],
)
def test_missing_or_expired_approval_receipts_cannot_authorize_recovery(
    approval: bool, now: int, reason: RecoveryReason
) -> None:
    checkpoint = _checkpoint(
        phase=(RecoveryPhase.APPROVAL_RECORDED if approval else RecoveryPhase.PLANNED),
        approval=approval,
    )
    observations = _observations(checkpoint.effects, (ObservationState.ABSENT,) * 3)

    decision = recover_workflow((checkpoint,), observations, now=now)

    assert decision.disposition is RecoveryDisposition.REVIEW_REQUIRED
    assert decision.reason is reason
    assert decision.retry_idempotency_keys == ()
    review_checkpoint = advance_checkpoint(checkpoint, decision)
    assert review_checkpoint.phase is RecoveryPhase.RECONCILING
    assert review_checkpoint.recovery_evidence_digest == decision.evidence_digest


def test_committed_effect_disappearance_fails_closed() -> None:
    original = _checkpoint()
    first_decision = recover_workflow(
        (original,),
        _observations(
            original.effects,
            (
                ObservationState.COMMITTED,
                ObservationState.ABSENT,
                ObservationState.ABSENT,
            ),
        ),
        now=10,
    )
    advanced = advance_checkpoint(original, first_decision)

    decision = recover_workflow(
        (original, advanced),
        _observations(advanced.effects, (ObservationState.ABSENT,) * 3),
        now=10,
    )

    assert decision.disposition is RecoveryDisposition.REVIEW_REQUIRED
    assert decision.reason is RecoveryReason.AMBIGUOUS_EFFECT


def test_terminal_phases_never_resume() -> None:
    first = _checkpoint()
    complete = RecoveryCheckpoint.create(
        workflow_id=first.workflow_id,
        run_id=first.run_id,
        sequence=0,
        phase=RecoveryPhase.COMPLETED,
        plan_digest=first.plan_digest,
        effects=tuple(
            replace(
                effect,
                state=EffectState.COMMITTED,
                commit_evidence_digest=COMMIT_DIGESTS[index],
            )
            for index, effect in enumerate(first.effects)
        ),
        approval_action_digest=first.approval_action_digest,
        approval_receipt_digest=first.approval_receipt_digest,
        approval_expires_at=first.approval_expires_at,
    )
    decision = recover_workflow((complete,), (), now=10)

    assert decision.disposition is RecoveryDisposition.COMPLETE
    assert decision.reason is RecoveryReason.ALREADY_COMPLETE
    assert advance_checkpoint(complete, decision) is complete


def test_duplicate_or_incomplete_observations_are_rejected() -> None:
    checkpoint = _checkpoint()
    one = _observations(checkpoint.effects, (ObservationState.ABSENT,) * 3)[0]

    with pytest.raises(RecoveryError, match="observation_coverage_mismatch"):
        recover_workflow((checkpoint,), (one,), now=10)
    with pytest.raises(RecoveryError, match="duplicate_observation"):
        recover_workflow((checkpoint,), (one, one), now=10)


def test_advance_rejects_a_decision_for_an_unknown_effect() -> None:
    checkpoint = _checkpoint()
    decision = RecoveryDecision.create(
        disposition=RecoveryDisposition.COMPLETE,
        reason=RecoveryReason.EFFECTS_RECONCILED,
        source_checkpoint_digest=checkpoint.checkpoint_digest,
        committed_effects=(
            (ActionId("act_" + "f" * 32).serialize(), COMMIT_DIGESTS[0]),
        ),
    )

    with pytest.raises(RecoveryError, match="unknown_committed_effect"):
        advance_checkpoint(checkpoint, decision)
