"""Synthetic interruption tests across local, FHIR, and OMOP effects."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from openmed.agent.correlation import ActionId, RunId
from openmed.agent.identifiers import ToolId, WorkflowId
from openmed.agent.workflows.recovery import (
    CheckpointJournal,
    CompensationLimit,
    EffectKind,
    EffectObservation,
    EffectRecord,
    ObservationState,
    RecoveryCheckpoint,
    RecoveryDisposition,
    RecoveryPhase,
    RecoveryReason,
    advance_checkpoint,
    recover_workflow,
)

RUN_ID = RunId("run_" + "a" * 32)
WORKFLOW_ID = WorkflowId("workflow:org.openmed/recovery-test@1.0.0")
PLAN_DIGEST = "sha256:" + "1" * 64
RECEIPT_DIGEST = "sha256:" + "2" * 64


class SyntheticSink:
    """Idempotent content-free stand-in for three effect adapters."""

    def __init__(self) -> None:
        self.commits: dict[str, str] = {}
        self.commit_counts: Counter[str] = Counter()

    def apply(self, key: str, evidence_digest: str) -> None:
        if key not in self.commits:
            self.commits[key] = evidence_digest
            self.commit_counts[key] += 1

    def observe(
        self, effects: tuple[EffectRecord, ...]
    ) -> tuple[EffectObservation, ...]:
        return tuple(
            EffectObservation(
                action_id=effect.action_id,
                operation_digest=effect.operation_digest,
                idempotency_key=effect.idempotency_key,
                state=(
                    ObservationState.COMMITTED
                    if effect.idempotency_key in self.commits
                    else ObservationState.ABSENT
                ),
                commit_evidence_digest=self.commits.get(effect.idempotency_key),
            )
            for effect in effects
        )


def _effects() -> tuple[EffectRecord, ...]:
    return tuple(
        EffectRecord.create(
            ordinal=index,
            run_id=RUN_ID,
            action_id=ActionId(f"act_{index + 1:032x}"),
            tool_id=ToolId(f"tool:org.openmed/recovery-{index + 1}@1.0.0"),
            kind=kind,
            operation_digest=f"sha256:{index + 3:064x}",
            approval_required=True,
            compensation_limit=(
                CompensationLimit.NONE
                if kind is EffectKind.LOCAL_TOOL
                else CompensationLimit.PROPOSE_ONLY
            ),
        )
        for index, kind in enumerate(EffectKind)
    )


def _initial_checkpoint() -> RecoveryCheckpoint:
    return RecoveryCheckpoint.create(
        workflow_id=WORKFLOW_ID,
        run_id=RUN_ID,
        sequence=0,
        phase=RecoveryPhase.APPROVAL_RECORDED,
        plan_digest=PLAN_DIGEST,
        effects=_effects(),
        approval_action_digest=PLAN_DIGEST,
        approval_receipt_digest=RECEIPT_DIGEST,
        approval_expires_at=1_000,
    )


@pytest.mark.parametrize("commits_before_interruption", [0, 1, 2, 3])
def test_interruption_reconciles_or_retries_every_effect_exactly_once(
    tmp_path: Path, commits_before_interruption: int
) -> None:
    journal = CheckpointJournal(tmp_path / "journal")
    initial = _initial_checkpoint()
    journal.append(initial)
    sink = SyntheticSink()

    first = recover_workflow(journal.load(), sink.observe(initial.effects), now=10)
    assert first.disposition is RecoveryDisposition.RESUME
    dispatching = advance_checkpoint(initial, first)
    journal.append(dispatching)

    for index, effect in enumerate(dispatching.effects[:commits_before_interruption]):
        sink.apply(effect.idempotency_key, f"sha256:{index + 7:064x}")

    # A new journal instance models process/device restart after the interruption.
    restarted = CheckpointJournal(tmp_path / "journal")
    lineage = restarted.load()
    recovered = recover_workflow(lineage, sink.observe(lineage[-1].effects), now=10)
    reconciled = advance_checkpoint(lineage[-1], recovered)
    restarted.append(reconciled)

    if recovered.disposition is RecoveryDisposition.RESUME:
        retry_keys = set(recovered.retry_idempotency_keys)
        for index, effect in enumerate(reconciled.effects):
            if effect.idempotency_key in retry_keys:
                sink.apply(effect.idempotency_key, f"sha256:{index + 7:064x}")
        final_decision = recover_workflow(
            restarted.load(), sink.observe(reconciled.effects), now=10
        )
        final_checkpoint = advance_checkpoint(restarted.load()[-1], final_decision)
        restarted.append(final_checkpoint)
    else:
        assert recovered.disposition is RecoveryDisposition.COMPLETE

    final_lineage = restarted.load()
    if final_lineage[-1].phase is not RecoveryPhase.COMPLETED:
        final_decision = recover_workflow(
            final_lineage, sink.observe(final_lineage[-1].effects), now=10
        )
        final_checkpoint = advance_checkpoint(final_lineage[-1], final_decision)
        restarted.append(final_checkpoint)
        final_lineage = restarted.load()

    assert final_lineage[-1].phase is RecoveryPhase.COMPLETED
    assert set(sink.commit_counts.values()) == {1}
    assert set(sink.commits) == {effect.idempotency_key for effect in initial.effects}


@pytest.mark.parametrize(
    ("phase", "expected_disposition", "expected_reason"),
    [
        (
            RecoveryPhase.PLANNED,
            RecoveryDisposition.REVIEW_REQUIRED,
            RecoveryReason.APPROVAL_MISSING,
        ),
        (
            RecoveryPhase.APPROVAL_RECORDED,
            RecoveryDisposition.RESUME,
            RecoveryReason.SAFE_TO_RESUME,
        ),
        (
            RecoveryPhase.DISPATCHING,
            RecoveryDisposition.RESUME,
            RecoveryReason.SAFE_TO_RESUME,
        ),
        (
            RecoveryPhase.RECONCILING,
            RecoveryDisposition.RESUME,
            RecoveryReason.SAFE_TO_RESUME,
        ),
        (
            RecoveryPhase.ABORTED,
            RecoveryDisposition.REVIEW_REQUIRED,
            RecoveryReason.WORKFLOW_ABORTED,
        ),
    ],
)
def test_interruption_at_each_nonterminal_checkpoint_resumes_or_fails_closed(
    phase: RecoveryPhase,
    expected_disposition: RecoveryDisposition,
    expected_reason: RecoveryReason,
) -> None:
    approval = phase is not RecoveryPhase.PLANNED
    checkpoint = RecoveryCheckpoint.create(
        workflow_id=WORKFLOW_ID,
        run_id=RUN_ID,
        sequence=0,
        phase=phase,
        plan_digest=PLAN_DIGEST,
        effects=_effects(),
        approval_action_digest=PLAN_DIGEST if approval else None,
        approval_receipt_digest=RECEIPT_DIGEST if approval else None,
        approval_expires_at=1_000 if approval else None,
    )
    sink = SyntheticSink()

    decision = recover_workflow((checkpoint,), sink.observe(checkpoint.effects), now=10)

    assert decision.disposition is expected_disposition
    assert decision.reason is expected_reason
