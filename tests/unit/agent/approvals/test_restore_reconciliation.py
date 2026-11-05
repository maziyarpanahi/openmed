"""Offline tests for post-restore approval and authority reconciliation."""

from __future__ import annotations

import json

import pytest

from openmed.agent.approvals.restore_reconciliation import (
    RESTORE_RECONCILIATION_SCHEMA_VERSION,
    ApprovalRestoreDisposition,
    InMemoryRecoveryEpochStore,
    RecoveryEpoch,
    RestoredApproval,
    RestoredSnapshot,
    RestoredWorkflow,
    RestoreReason,
    RestoreReconciler,
    RestoreReconciliationError,
    WorkflowRestoreDisposition,
    epoch_action_digest,
    require_current_epoch,
)
from openmed.agent.approvals.tokens import (
    ApprovalActionMismatchError,
    ApprovalReplayError,
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
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
    RetiredJournalRecord,
    advance_checkpoint,
    recover_workflow,
)
from tests.fixtures.agent.authority_status import SyntheticAuthority

KEY = b"synthetic-local-approval-key-32-bytes"
ROLE = "role:org.example/clinical-reviewer@1.0.0"
ACTION = "sha256:" + "a" * 64
SNAPSHOT = "sha256:" + "5" * 64
RECEIPT = "sha256:" + "6" * 64
COMMIT_EVIDENCE = "sha256:" + "7" * 64
RUN_ID = RunId("run_" + "1" * 32)
WORKFLOW_ID = WorkflowId("workflow:org.example/restore@1.0.0")
SENTINEL = "SENTINEL-PRIVATE-NOTE-0001"
NOW = 10


class Clock:
    def __init__(self, now: int = NOW) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


class RecordingNonceStore(InMemoryApprovalNonceStore):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def claim(self, nonce_digest: str, *, expires_at: int, now: int) -> bool:
        self.calls += 1
        return super().claim(nonce_digest, expires_at=expires_at, now=now)


class FailingNonceStore:
    def claim(self, nonce_digest: str, *, expires_at: int, now: int) -> bool:
        raise RuntimeError(SENTINEL)


class NonBoolNonceStore:
    def claim(self, nonce_digest: str, *, expires_at: int, now: int) -> object:
        return 1


class Sink:
    """Observation-only effect status source; it exposes no delete operation."""

    def __init__(self) -> None:
        self.commits: dict[str, str] = {}
        self.calls = 0
        self.fail = False
        self.ambiguous = False

    def __call__(self, checkpoint: RecoveryCheckpoint):
        self.calls += 1
        if self.fail:
            raise RuntimeError(SENTINEL)
        return tuple(
            EffectObservation(
                action_id=effect.action_id,
                operation_digest=effect.operation_digest,
                idempotency_key=effect.idempotency_key,
                state=(
                    ObservationState.AMBIGUOUS
                    if self.ambiguous
                    else ObservationState.COMMITTED
                    if effect.idempotency_key in self.commits
                    else ObservationState.ABSENT
                ),
                commit_evidence_digest=(
                    None if self.ambiguous else self.commits.get(effect.idempotency_key)
                ),
            )
            for effect in checkpoint.effects
        )


def _token(*, action: str = ACTION, nonce_byte: int = 1, expires_at: int = 500):
    signer = ApprovalTokenSigner(KEY, clock=lambda: NOW)
    return signer.issue(
        action_digest=action,
        reviewer_role=ROLE,
        expires_at=expires_at,
        nonce_source=lambda size: bytes([nonce_byte]) * size,
    )


def _effects(*, approval_required: bool, count: int = 2):
    return tuple(
        EffectRecord.create(
            ordinal=index,
            run_id=RUN_ID,
            action_id=ActionId(f"act_{index + 1:032x}"),
            tool_id=ToolId(f"tool:org.example/restore-{index + 1}@1.0.0"),
            kind=EffectKind.LOCAL_TOOL,
            operation_digest=f"sha256:{index + 10:064x}",
            approval_required=approval_required,
            compensation_limit=CompensationLimit.NONE,
        )
        for index in range(count)
    )


def _lineage(*, approval_required: bool, commit_first: bool = False):
    approval = (
        {
            "approval_action_digest": ACTION,
            "approval_receipt_digest": RECEIPT,
            "approval_expires_at": 500,
        }
        if approval_required
        else {}
    )
    initial = RecoveryCheckpoint.create(
        workflow_id=WORKFLOW_ID,
        run_id=RUN_ID,
        sequence=0,
        phase=(
            RecoveryPhase.APPROVAL_RECORDED
            if approval_required
            else RecoveryPhase.PLANNED
        ),
        plan_digest=ACTION,
        effects=_effects(approval_required=approval_required),
        **approval,
    )
    if not commit_first:
        return (initial,)
    sink = Sink()
    sink.commits[initial.effects[0].idempotency_key] = COMMIT_EVIDENCE
    decision = recover_workflow((initial,), sink(initial), now=NOW)
    assert decision.disposition is RecoveryDisposition.RESUME
    return (initial, advance_checkpoint(initial, decision))


def _reconciler(nonce_store=None, *, authority=None, clock=None, epochs=None):
    return RestoreReconciler(
        InMemoryApprovalNonceStore() if nonce_store is None else nonce_store,
        InMemoryRecoveryEpochStore() if epochs is None else epochs,
        authority=authority,
        clock=Clock() if clock is None else clock,
    )


def _snapshot(*, approvals=(), workflows=(), recorded_epoch: int = 0):
    return RestoredSnapshot(
        snapshot_digest=SNAPSHOT,
        recorded_epoch=recorded_epoch,
        approvals=tuple(approvals),
        workflows=tuple(workflows),
    )


# Approvals and nonces -------------------------------------------------------


def test_backup_taken_before_consumption_cannot_make_token_spendable() -> None:
    token = _token()
    # The live store was consumed after the backup; restore replaced it with
    # the pre-consumption copy, so it holds no claim for this nonce.
    restored_store = InMemoryApprovalNonceStore()
    snapshot = _snapshot(approvals=(RestoredApproval.from_token(token),))

    report = _reconciler(restored_store).reconcile(snapshot, observe=Sink())

    (outcome,) = report.approvals
    assert outcome.disposition is ApprovalRestoreDisposition.INVALIDATED
    assert outcome.reason is RestoreReason.NONCE_BURNED
    verifier = ApprovalTokenVerifier(KEY, restored_store, clock=lambda: NOW)
    with pytest.raises(ApprovalReplayError):
        verifier.consume(token, action_digest=ACTION, reviewer_role=ROLE)


def test_negative_control_restored_store_alone_would_replay() -> None:
    token = _token()
    live = InMemoryApprovalNonceStore()
    ApprovalTokenVerifier(KEY, live, clock=lambda: NOW).consume(
        token, action_digest=ACTION, reviewer_role=ROLE
    )
    restored_store = InMemoryApprovalNonceStore()
    receipt = ApprovalTokenVerifier(KEY, restored_store, clock=lambda: NOW).consume(
        token, action_digest=ACTION, reviewer_role=ROLE
    )
    assert receipt.code == "approved"


def test_authoritative_journal_claim_is_reported_without_new_claim() -> None:
    token = _token()
    live = InMemoryApprovalNonceStore()
    ApprovalTokenVerifier(KEY, live, clock=lambda: NOW).consume(
        token, action_digest=ACTION, reviewer_role=ROLE
    )
    report = _reconciler(live).reconcile(
        _snapshot(approvals=(RestoredApproval.from_token(token),)), observe=Sink()
    )
    (outcome,) = report.approvals
    assert outcome.disposition is ApprovalRestoreDisposition.INVALIDATED
    assert outcome.reason is RestoreReason.NONCE_ALREADY_CLAIMED
    assert report.complete


def test_reconciliation_is_idempotent_and_never_revives_a_burned_nonce() -> None:
    token = _token()
    store = InMemoryApprovalNonceStore()
    epochs = InMemoryRecoveryEpochStore()
    snapshot = _snapshot(approvals=(RestoredApproval.from_token(token),))
    first = _reconciler(store, epochs=epochs).reconcile(snapshot, observe=Sink())
    second = _reconciler(store, epochs=epochs).reconcile(snapshot, observe=Sink())
    assert first.approvals[0].reason is RestoreReason.NONCE_BURNED
    assert second.approvals[0].reason is RestoreReason.NONCE_ALREADY_CLAIMED
    assert second.epoch.epoch > first.epoch.epoch


def test_expired_approval_is_invalidated_without_a_claim() -> None:
    store = RecordingNonceStore()
    approval = RestoredApproval.from_token(_token(expires_at=50))
    report = _reconciler(store, clock=Clock(60)).reconcile(
        _snapshot(approvals=(approval,)), observe=Sink()
    )
    assert report.approvals[0].reason is RestoreReason.APPROVAL_EXPIRED
    assert report.approvals[0].disposition is ApprovalRestoreDisposition.INVALIDATED
    assert store.calls == 0


def test_skew_extends_retention_to_the_verifier_window() -> None:
    token = _token(expires_at=50)
    approval = RestoredApproval.from_token(token, clock_skew_seconds=30)
    assert approval.expires_at == 80
    store = InMemoryApprovalNonceStore()
    report = _reconciler(store, clock=Clock(60)).reconcile(
        _snapshot(approvals=(approval,)), observe=Sink()
    )
    assert report.approvals[0].reason is RestoreReason.NONCE_BURNED
    verifier = ApprovalTokenVerifier(
        KEY, store, clock=lambda: 60, clock_skew_seconds=30
    )
    with pytest.raises(ApprovalReplayError):
        verifier.consume(token, action_digest=ACTION, reviewer_role=ROLE)


@pytest.mark.parametrize("store", [FailingNonceStore(), NonBoolNonceStore()])
def test_unavailable_nonce_journal_holds_restore(store) -> None:
    report = _reconciler(store).reconcile(
        _snapshot(approvals=(RestoredApproval.from_token(_token()),)), observe=Sink()
    )
    (outcome,) = report.approvals
    assert outcome.disposition is ApprovalRestoreDisposition.HELD
    assert outcome.reason is RestoreReason.NONCE_JOURNAL_UNAVAILABLE
    assert not report.complete
    assert report.held_count == 1
    with pytest.raises(RestoreReconciliationError) as raised:
        report.require_complete()
    assert raised.value.code == "restore_reconciliation_incomplete"
    assert SENTINEL not in str(raised.value)
    assert SENTINEL not in report.to_json()


# Recovery epochs ------------------------------------------------------------


def test_restore_opens_a_new_epoch_above_recorded_and_retained_high_water() -> None:
    epochs = InMemoryRecoveryEpochStore()
    first = _reconciler(epochs=epochs).reconcile(
        _snapshot(recorded_epoch=7), observe=Sink()
    )
    assert first.epoch.epoch == 8
    assert first.epoch.previous_epoch == 7
    assert first.epoch.snapshot_digest == SNAPSHOT
    # An older snapshot cannot rewind the independently retained epoch.
    second = _reconciler(epochs=epochs).reconcile(
        _snapshot(recorded_epoch=2), observe=Sink()
    )
    assert second.epoch.epoch == 9
    assert epochs.current() == 9
    require_current_epoch(second.epoch, epochs)
    with pytest.raises(RestoreReconciliationError) as raised:
        require_current_epoch(first.epoch, epochs)
    assert raised.value.code == "stale_recovery_epoch"


def test_epoch_bound_action_digest_rejects_pre_restore_tokens() -> None:
    epochs = InMemoryRecoveryEpochStore()
    report = _reconciler(epochs=epochs).reconcile(_snapshot(), observe=Sink())
    bound = epoch_action_digest(ACTION, report.epoch)
    assert bound != ACTION
    # A token issued before restore and absent from the snapshot.
    unlisted = _token(nonce_byte=9)
    store = InMemoryApprovalNonceStore()
    verifier = ApprovalTokenVerifier(KEY, store, clock=lambda: NOW)
    with pytest.raises(ApprovalActionMismatchError):
        verifier.consume(unlisted, action_digest=bound, reviewer_role=ROLE)
    with pytest.raises(ApprovalReplayError):
        verifier.consume(unlisted, action_digest=bound, reviewer_role=ROLE)
    later = _reconciler(epochs=epochs).reconcile(_snapshot(), observe=Sink())
    assert epoch_action_digest(ACTION, later.epoch) != bound
    fresh = _token(action=epoch_action_digest(ACTION, later.epoch), nonce_byte=3)
    verifier.consume(
        fresh,
        action_digest=epoch_action_digest(ACTION, later.epoch),
        reviewer_role=ROLE,
    )


class RewindingEpochStore(InMemoryRecoveryEpochStore):
    def advance(self, *, floor: int) -> int:
        return floor


class FailingEpochStore:
    def current(self) -> int:
        raise RuntimeError(SENTINEL)

    def advance(self, *, floor: int) -> int:
        raise RuntimeError(SENTINEL)


@pytest.mark.parametrize(
    ("epochs", "code"),
    [
        (RewindingEpochStore(), "epoch_not_monotonic"),
        (FailingEpochStore(), "recovery_epoch_unavailable"),
    ],
)
def test_untrustworthy_epoch_store_fails_closed(epochs, code) -> None:
    store = RecordingNonceStore()
    with pytest.raises(RestoreReconciliationError) as raised:
        _reconciler(store, epochs=epochs).reconcile(
            _snapshot(approvals=(RestoredApproval.from_token(_token()),)),
            observe=Sink(),
        )
    assert raised.value.code == code
    assert SENTINEL not in str(raised.value)
    assert store.calls == 0


def test_unavailable_clock_fails_closed() -> None:
    def clock() -> int:
        raise RuntimeError(SENTINEL)

    with pytest.raises(RestoreReconciliationError) as raised:
        _reconciler(clock=clock).reconcile(_snapshot(), observe=Sink())
    assert raised.value.code == "clock_unavailable"
    assert SENTINEL not in str(raised.value)


def test_recovery_epoch_detects_tampering() -> None:
    epoch = RecoveryEpoch.create(
        epoch=2, previous_epoch=1, snapshot_digest=SNAPSHOT, opened_at=NOW
    )
    with pytest.raises(RestoreReconciliationError) as raised:
        RecoveryEpoch(
            epoch=3,
            previous_epoch=1,
            snapshot_digest=SNAPSHOT,
            opened_at=NOW,
            epoch_digest=epoch.epoch_digest,
        )
    assert raised.value.code == "epoch_digest_mismatch"
    with pytest.raises(RestoreReconciliationError):
        RecoveryEpoch.create(
            epoch=1, previous_epoch=1, snapshot_digest=SNAPSHOT, opened_at=NOW
        )


# Workflows, authority and effects -------------------------------------------


def _authority_workflow(fixture, *, approval_required=False, commit_first=False):
    binding = fixture.runtime.bind(fixture.contracts)
    return RestoredWorkflow(
        lineage=_lineage(
            approval_required=approval_required, commit_first=commit_first
        ),
        authority=binding,
    )


@pytest.mark.parametrize("revoked_index", [0, 1, 2])
def test_revoked_authority_stays_unusable_when_snapshot_records_it_active(
    revoked_index,
) -> None:
    fixture = SyntheticAuthority()
    workflow = _authority_workflow(fixture, commit_first=True)
    fixture.provider.revoke(fixture.contracts[revoked_index])
    sink = Sink()
    report = _reconciler(authority=fixture.runtime, clock=fixture.clock).reconcile(
        _snapshot(workflows=(workflow,)), observe=sink
    )
    (outcome,) = report.workflows
    assert outcome.disposition is WorkflowRestoreDisposition.FRESH_REVIEW_REQUIRED
    assert outcome.reason is RestoreReason.AUTHORITY_REVOKED
    assert outcome.retry_idempotency_keys == ()
    assert sink.calls == 0
    # Committed effects recorded before the backup are retained as evidence.
    assert outcome.committed_effects == (
        (workflow.lineage[-1].effects[0].action_id.serialize(), COMMIT_EVIDENCE),
    )


def test_expired_authority_stays_unusable() -> None:
    fixture = SyntheticAuthority()
    workflow = _authority_workflow(fixture)
    fixture.clock.now = 150
    sink = Sink()
    report = _reconciler(authority=fixture.runtime, clock=fixture.clock).reconcile(
        _snapshot(workflows=(workflow,)), observe=sink
    )
    (outcome,) = report.workflows
    assert outcome.disposition is WorkflowRestoreDisposition.FRESH_REVIEW_REQUIRED
    assert outcome.reason is RestoreReason.AUTHORITY_EXPIRED
    assert sink.calls == 0


def test_generation_change_requires_fresh_review() -> None:
    fixture = SyntheticAuthority()
    workflow = _authority_workflow(fixture)
    key = (fixture.contracts[0].kind, fixture.contracts[0].digest)
    fixture.provider.records[key] = (1, False)
    report = _reconciler(authority=fixture.runtime, clock=fixture.clock).reconcile(
        _snapshot(workflows=(workflow,)), observe=Sink()
    )
    assert report.workflows[0].reason is RestoreReason.AUTHORITY_CHANGED
    assert (
        report.workflows[0].disposition
        is WorkflowRestoreDisposition.FRESH_REVIEW_REQUIRED
    )


def test_unavailable_current_authority_keeps_work_held() -> None:
    fixture = SyntheticAuthority()
    workflow = _authority_workflow(fixture, commit_first=True)
    fixture.provider.unavailable = True
    sink = Sink()
    report = _reconciler(authority=fixture.runtime, clock=fixture.clock).reconcile(
        _snapshot(workflows=(workflow,)), observe=sink
    )
    (outcome,) = report.workflows
    assert outcome.disposition is WorkflowRestoreDisposition.HELD
    assert outcome.reason is RestoreReason.AUTHORITY_UNAVAILABLE
    assert outcome.retry_idempotency_keys == ()
    assert len(outcome.committed_effects) == 1
    assert sink.calls == 0
    assert not report.complete
    assert "synthetic-private-provider-payload" not in report.to_json()


def test_bound_workflow_without_runtime_is_held() -> None:
    fixture = SyntheticAuthority()
    workflow = _authority_workflow(fixture)
    report = _reconciler().reconcile(_snapshot(workflows=(workflow,)), observe=Sink())
    assert report.workflows[0].disposition is WorkflowRestoreDisposition.HELD
    assert report.workflows[0].reason is RestoreReason.AUTHORITY_UNAVAILABLE


def test_active_authority_without_approval_resumes_idempotently() -> None:
    fixture = SyntheticAuthority()
    workflow = _authority_workflow(fixture)
    report = _reconciler(authority=fixture.runtime, clock=fixture.clock).reconcile(
        _snapshot(workflows=(workflow,)), observe=Sink()
    )
    (outcome,) = report.workflows
    assert outcome.disposition is WorkflowRestoreDisposition.RESUME
    assert outcome.reason is RestoreReason.RESUME_SAFE
    assert outcome.retry_idempotency_keys == tuple(
        effect.idempotency_key for effect in workflow.lineage[-1].effects
    )
    assert outcome.recovery_evidence_digest is not None
    assert report.complete


def test_pre_restore_approval_receipt_cannot_authorize_resume() -> None:
    workflow = RestoredWorkflow(lineage=_lineage(approval_required=True))
    report = _reconciler().reconcile(_snapshot(workflows=(workflow,)), observe=Sink())
    (outcome,) = report.workflows
    assert outcome.disposition is WorkflowRestoreDisposition.FRESH_REVIEW_REQUIRED
    assert outcome.reason is RestoreReason.APPROVAL_INVALIDATED
    assert outcome.retry_idempotency_keys == ()
    assert outcome.historical_approval_receipt_digest == RECEIPT


def test_expired_recorded_approval_requires_fresh_review() -> None:
    workflow = RestoredWorkflow(lineage=_lineage(approval_required=True))
    report = _reconciler(clock=Clock(600)).reconcile(
        _snapshot(workflows=(workflow,)), observe=Sink()
    )
    assert report.workflows[0].reason is RestoreReason.APPROVAL_EXPIRED
    assert (
        report.workflows[0].disposition
        is WorkflowRestoreDisposition.FRESH_REVIEW_REQUIRED
    )


def test_effect_committed_after_backup_is_reconciled_not_repeated() -> None:
    lineage = _lineage(approval_required=False)
    sink = Sink()
    sink.commits[lineage[-1].effects[1].idempotency_key] = COMMIT_EVIDENCE
    report = _reconciler().reconcile(
        _snapshot(workflows=(RestoredWorkflow(lineage=lineage),)), observe=sink
    )
    (outcome,) = report.workflows
    assert outcome.disposition is WorkflowRestoreDisposition.RESUME
    assert outcome.retry_idempotency_keys == (lineage[-1].effects[0].idempotency_key,)
    assert outcome.committed_effects == (
        (lineage[-1].effects[1].action_id.serialize(), COMMIT_EVIDENCE),
    )


def test_all_effects_committed_live_completes_with_approval_history() -> None:
    lineage = _lineage(approval_required=True)
    sink = Sink()
    for effect in lineage[-1].effects:
        sink.commits[effect.idempotency_key] = COMMIT_EVIDENCE
    report = _reconciler().reconcile(
        _snapshot(workflows=(RestoredWorkflow(lineage=lineage),)), observe=sink
    )
    (outcome,) = report.workflows
    assert outcome.disposition is WorkflowRestoreDisposition.COMPLETE
    assert outcome.reason is RestoreReason.EFFECTS_COMPLETE
    assert len(outcome.committed_effects) == 2
    assert outcome.historical_approval_receipt_digest == RECEIPT


def test_unavailable_effect_status_is_held() -> None:
    sink = Sink()
    sink.fail = True
    workflow = RestoredWorkflow(lineage=_lineage(approval_required=False))
    report = _reconciler().reconcile(_snapshot(workflows=(workflow,)), observe=sink)
    assert report.workflows[0].disposition is WorkflowRestoreDisposition.HELD
    assert report.workflows[0].reason is RestoreReason.EFFECT_STATUS_UNAVAILABLE
    assert SENTINEL not in report.to_json()


def test_malformed_effect_status_is_held() -> None:
    workflow = RestoredWorkflow(lineage=_lineage(approval_required=False))
    report = _reconciler().reconcile(
        _snapshot(workflows=(workflow,)), observe=lambda checkpoint: ()
    )
    assert report.workflows[0].disposition is WorkflowRestoreDisposition.HELD
    assert report.workflows[0].reason is RestoreReason.EFFECT_STATUS_UNAVAILABLE


def test_ambiguous_effect_requires_effect_review() -> None:
    sink = Sink()
    sink.ambiguous = True
    workflow = RestoredWorkflow(lineage=_lineage(approval_required=False))
    report = _reconciler().reconcile(_snapshot(workflows=(workflow,)), observe=sink)
    assert report.workflows[0].disposition is WorkflowRestoreDisposition.REVIEW_REQUIRED
    assert report.workflows[0].reason is RestoreReason.EFFECTS_NEED_REVIEW
    assert report.workflows[0].recovery_reason == "ambiguous_effect"


def test_terminal_and_retired_lineages_skip_authority_and_sinks() -> None:
    lineage = _lineage(approval_required=False, commit_first=True)
    sink = Sink()
    for effect in lineage[-1].effects:
        sink.commits[effect.idempotency_key] = COMMIT_EVIDENCE
    decision = recover_workflow(lineage, sink(lineage[-1]), now=NOW)
    completed = (*lineage, advance_checkpoint(lineage[-1], decision))
    retired = RetiredJournalRecord(
        final_checkpoint_digest=completed[-1].checkpoint_digest,
        recovery_evidence_digest=decision.evidence_digest,
        checkpoint_recovery_evidence_digest=decision.evidence_digest,
        phase=RecoveryPhase.COMPLETED,
        checkpoint_count=3,
        effect_count=2,
        committed_effect_count=2,
        record_digest=_retired_digest(completed, decision),
    )
    fixture = SyntheticAuthority()
    binding = fixture.runtime.bind(fixture.contracts)
    fixture.provider.unavailable = True
    probe = Sink()
    reconciler = _reconciler(authority=fixture.runtime, clock=fixture.clock)
    outcomes = [
        reconciler.reconcile(
            _snapshot(workflows=(RestoredWorkflow(lineage=item, authority=binding),)),
            observe=probe,
        ).workflows[0]
        for item in (completed, retired)
    ]
    assert [outcome.disposition for outcome in outcomes] == [
        WorkflowRestoreDisposition.COMPLETE,
        WorkflowRestoreDisposition.RETIRED,
    ]
    assert outcomes[0].committed_effects
    assert outcomes[1].source_digest == completed[-1].checkpoint_digest
    assert probe.calls == 0
    # The same restored lineage cannot appear twice in one snapshot.
    with pytest.raises(RestoreReconciliationError) as raised:
        _snapshot(
            workflows=(
                RestoredWorkflow(lineage=completed),
                RestoredWorkflow(lineage=retired),
            )
        )
    assert raised.value.code == "duplicate_restored_workflow"


def _retired_digest(completed, decision) -> str:
    import hashlib

    values = {
        "final_checkpoint_digest": completed[-1].checkpoint_digest,
        "recovery_evidence_digest": decision.evidence_digest,
        "checkpoint_recovery_evidence_digest": decision.evidence_digest,
        "phase": "completed",
        "checkpoint_count": 3,
        "effect_count": 2,
        "committed_effect_count": 2,
        "schema_version": "openmed.agent.retired_journal.v1",
    }
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


# Snapshot validation and report privacy -------------------------------------


def test_snapshot_rejects_duplicates_and_malformed_state() -> None:
    approval = RestoredApproval.from_token(_token())
    with pytest.raises(RestoreReconciliationError) as raised:
        _snapshot(approvals=(approval, approval))
    assert raised.value.code == "duplicate_restored_approval"
    workflow = RestoredWorkflow(lineage=_lineage(approval_required=False))
    with pytest.raises(RestoreReconciliationError) as raised:
        _snapshot(workflows=(workflow, workflow))
    assert raised.value.code == "duplicate_restored_workflow"
    with pytest.raises(RestoreReconciliationError) as raised:
        RestoredSnapshot(
            snapshot_digest=SENTINEL, recorded_epoch=0, approvals=(), workflows=()
        )
    assert raised.value.code == "invalid_digest"
    assert SENTINEL not in str(raised.value)
    with pytest.raises(RestoreReconciliationError) as raised:
        RestoredWorkflow(lineage=(_lineage(approval_required=False)[0], SENTINEL))
    assert raised.value.code == "invalid_restored_lineage"
    with pytest.raises(RestoreReconciliationError) as raised:
        RestoredWorkflow(lineage=_lineage(approval_required=False), authority=SENTINEL)
    assert raised.value.code == "invalid_restored_authority"
    with pytest.raises(RestoreReconciliationError):
        RestoredApproval(token_digest=SENTINEL, nonce_digest=SENTINEL, expires_at=1)
    with pytest.raises(RestoreReconciliationError):
        RestoredApproval.from_token(SENTINEL)


def test_report_is_deterministic_content_free_and_never_claims_review() -> None:
    token = _token()
    lineage = _lineage(approval_required=True)

    def run():
        return _reconciler().reconcile(
            _snapshot(
                approvals=(RestoredApproval.from_token(token),),
                workflows=(RestoredWorkflow(lineage=lineage),),
            ),
            observe=Sink(),
        )

    first, second = run(), run()
    assert first.to_json() == second.to_json()
    payload = json.loads(first.to_json())
    assert payload["schema_version"] == RESTORE_RECONCILIATION_SCHEMA_VERSION
    assert payload["report_digest"] == first.report_digest
    assert payload["counts"] == {
        "approvals": 1,
        "workflows": 1,
        "invalidated_approvals": 1,
        "held": 0,
        "fresh_review_required": 1,
    }
    serialized = first.to_json()
    assert token.nonce not in serialized
    assert token.signature not in serialized
    assert '"approved"' not in serialized
    assert "review_repeated" not in serialized
    assert first.workflows[0].historical_approval_receipt_digest == RECEIPT
    for text in (
        repr(first),
        repr(first.epoch),
        repr(RestoredApproval.from_token(token)),
    ):
        assert token.nonce not in text
        assert SNAPSHOT not in text


def test_reconciler_rejects_invalid_ports() -> None:
    with pytest.raises(RestoreReconciliationError) as raised:
        RestoreReconciler(object(), InMemoryRecoveryEpochStore())
    assert raised.value.code == "invalid_nonce_store"
    with pytest.raises(RestoreReconciliationError) as raised:
        RestoreReconciler(InMemoryApprovalNonceStore(), object())
    assert raised.value.code == "invalid_epoch_store"
    with pytest.raises(RestoreReconciliationError) as raised:
        RestoreReconciler(
            InMemoryApprovalNonceStore(), InMemoryRecoveryEpochStore(), authority=1
        )
    assert raised.value.code == "invalid_authority_runtime"
    with pytest.raises(RestoreReconciliationError) as raised:
        _reconciler().reconcile(SENTINEL, observe=Sink())
    assert raised.value.code == "invalid_snapshot"
    with pytest.raises(RestoreReconciliationError) as raised:
        _reconciler().reconcile(_snapshot(), observe=SENTINEL)
    assert raised.value.code == "invalid_effect_observer"


def test_authority_rollback_requires_fresh_review() -> None:
    from openmed.agent.permissions.revocation import AuthorityStatus

    fixture = SyntheticAuthority()
    workflow = _authority_workflow(fixture)
    contract = fixture.contracts[0]
    fixture.store.observe(AuthorityStatus(contract.kind, contract.digest, 2, False, 10))
    report = _reconciler(authority=fixture.runtime, clock=fixture.clock).reconcile(
        _snapshot(workflows=(workflow,)), observe=Sink()
    )
    assert report.workflows[0].reason is RestoreReason.AUTHORITY_ROLLBACK
    assert (
        report.workflows[0].disposition
        is WorkflowRestoreDisposition.FRESH_REVIEW_REQUIRED
    )


def test_aborted_lineage_requires_effect_review_without_sink_reads() -> None:
    (initial,) = _lineage(approval_required=False)
    aborted = RecoveryCheckpoint.create(
        workflow_id=WORKFLOW_ID,
        run_id=RUN_ID,
        sequence=1,
        phase=RecoveryPhase.ABORTED,
        plan_digest=ACTION,
        effects=initial.effects,
        previous_checkpoint_digest=initial.checkpoint_digest,
    )
    sink = Sink()
    report = _reconciler().reconcile(
        _snapshot(workflows=(RestoredWorkflow(lineage=(initial, aborted)),)),
        observe=sink,
    )
    assert report.workflows[0].disposition is WorkflowRestoreDisposition.REVIEW_REQUIRED
    assert report.workflows[0].recovery_reason == "workflow_aborted"
    assert sink.calls == 0


@pytest.mark.parametrize("value", [-1, "10", True, None])
def test_malformed_clock_value_fails_closed(value) -> None:
    with pytest.raises(RestoreReconciliationError) as raised:
        _reconciler(clock=lambda: value).reconcile(_snapshot(), observe=Sink())
    assert raised.value.code == "clock_unavailable"
