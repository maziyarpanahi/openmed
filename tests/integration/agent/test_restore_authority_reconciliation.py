"""Offline backup/restore anti-replay tests with synthetic fixture snapshots."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from openmed.agent.approvals import (
    ApprovalActionMismatchError,
    ApprovalReplayError,
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    SQLiteApprovalNonceStore,
)
from openmed.agent.approvals.restore_reconciliation import (
    ApprovalRestoreDisposition,
    RestoredApproval,
    RestoredSnapshot,
    RestoredWorkflow,
    RestoreReason,
    RestoreReconciler,
    WorkflowRestoreDisposition,
    epoch_action_digest,
    require_current_epoch,
)
from openmed.agent.correlation import ActionId
from openmed.agent.identifiers import ToolId, WorkflowId
from openmed.agent.workflows.recovery import (
    CheckpointJournal,
    CompensationLimit,
    EffectKind,
    EffectObservation,
    EffectRecord,
    ObservationState,
    RecoveryCheckpoint,
    RecoveryPhase,
)
from tests.fixtures.agent.authority_status import ACTION_DIGEST, SyntheticAuthority

pytestmark = pytest.mark.integration

KEY = b"synthetic-local-approval-key-32-bytes"
ROLE = "role:org.example/clinical-reviewer@1.0.0"
SNAPSHOT_DIGEST = "sha256:" + "b" * 64
COMMIT_EVIDENCE = "sha256:" + "c" * 64


class SyntheticDurableEpochs:
    """Test-only durable epoch high water kept outside every backup."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS epoch "
                "(id INTEGER PRIMARY KEY, value INTEGER)"
            )
            connection.execute("INSERT OR IGNORE INTO epoch VALUES (0, 0)")

    def current(self) -> int:
        with sqlite3.connect(self.path) as connection:
            return connection.execute("SELECT value FROM epoch").fetchone()[0]

    def advance(self, *, floor: int) -> int:
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT value FROM epoch").fetchone()[0]
            value = max(current, floor) + 1
            connection.execute("UPDATE epoch SET value = ?", (value,))
            return value


class SyntheticSink:
    """Idempotent effect sink; restore may observe it but never delete from it."""

    def __init__(self) -> None:
        self.commits: dict[str, str] = {}
        self.observed = 0

    def apply(self, key: str) -> None:
        self.commits.setdefault(key, COMMIT_EVIDENCE)

    def observe(self, checkpoint: RecoveryCheckpoint):
        self.observed += 1
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
            for effect in checkpoint.effects
        )


def _checkpoint(fixture, *, approval_required: bool, receipt=None):
    effects = tuple(
        EffectRecord.create(
            ordinal=index,
            run_id=fixture.ticket.run_id,
            action_id=ActionId(f"act_{index + 1:032x}"),
            tool_id=ToolId(fixture.constraint.tool),
            kind=kind,
            operation_digest=f"sha256:{index + (20 if approval_required else 40):064x}",
            approval_required=approval_required,
            compensation_limit=CompensationLimit.PROPOSE_ONLY,
        )
        for index, kind in enumerate((EffectKind.FHIR_WRITE, EffectKind.OMOP_BATCH))
    )
    approval = (
        {
            "approval_action_digest": ACTION_DIGEST,
            "approval_receipt_digest": receipt.token_digest,
            "approval_expires_at": 90,
        }
        if receipt is not None
        else {}
    )
    return RecoveryCheckpoint.create(
        workflow_id=WorkflowId("workflow:org.example/restore@1.0.0"),
        run_id=fixture.ticket.run_id,
        sequence=0,
        phase=(
            RecoveryPhase.APPROVAL_RECORDED
            if receipt is not None
            else RecoveryPhase.PLANNED
        ),
        plan_digest=ACTION_DIGEST,
        effects=effects,
        **approval,
    )


def _restore_file(backup: Path, live: Path) -> None:
    live.unlink()
    shutil.copy2(backup, live)


def test_backup_before_consumption_cannot_revive_spent_or_unlisted_tokens(
    tmp_path,
) -> None:
    fixture = SyntheticAuthority()
    nonce_path = tmp_path / "live" / "nonces.sqlite"
    nonce_path.parent.mkdir()
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    epochs = SyntheticDurableEpochs(tmp_path / "independent-epochs.sqlite")
    signer = ApprovalTokenSigner(KEY, clock=fixture.clock)
    verifier = ApprovalTokenVerifier(
        KEY, SQLiteApprovalNonceStore(nonce_path), clock=fixture.clock
    )

    # Reviewed token recorded as outstanding in the backup.
    spent = signer.issue(action_digest=ACTION_DIGEST, reviewer_role=ROLE, expires_at=90)
    shutil.copy2(nonce_path, backup_dir / "nonces.sqlite")
    snapshot = RestoredSnapshot(
        snapshot_digest=SNAPSHOT_DIGEST,
        recorded_epoch=0,
        approvals=(RestoredApproval.from_token(spent),),
        workflows=(),
    )

    # After the backup: consume the token, then issue one the backup never saw.
    verifier.consume(spent, action_digest=ACTION_DIGEST, reviewer_role=ROLE)
    unlisted = signer.issue(
        action_digest=ACTION_DIGEST, reviewer_role=ROLE, expires_at=90
    )

    # Negative control: the restored nonce file alone would accept a replay.
    control = tmp_path / "control.sqlite"
    shutil.copy2(backup_dir / "nonces.sqlite", control)
    ApprovalTokenVerifier(
        KEY, SQLiteApprovalNonceStore(control), clock=fixture.clock
    ).consume(spent, action_digest=ACTION_DIGEST, reviewer_role=ROLE)

    _restore_file(backup_dir / "nonces.sqlite", nonce_path)
    restored_store = SQLiteApprovalNonceStore(nonce_path)
    report = RestoreReconciler(
        restored_store, epochs, authority=fixture.runtime, clock=fixture.clock
    ).reconcile(snapshot, observe=SyntheticSink().observe)
    assert report.complete
    assert report.approvals[0].disposition is ApprovalRestoreDisposition.INVALIDATED
    assert report.approvals[0].reason is RestoreReason.NONCE_BURNED

    # A separate process-like verifier over the durable store still refuses.
    restarted = ApprovalTokenVerifier(
        KEY, SQLiteApprovalNonceStore(nonce_path), clock=fixture.clock
    )
    with pytest.raises(ApprovalReplayError):
        restarted.consume(spent, action_digest=ACTION_DIGEST, reviewer_role=ROLE)
    bound = epoch_action_digest(ACTION_DIGEST, report.epoch)
    require_current_epoch(report.epoch, epochs)
    with pytest.raises(ApprovalActionMismatchError):
        restarted.consume(unlisted, action_digest=bound, reviewer_role=ROLE)
    fresh = signer.issue(action_digest=bound, reviewer_role=ROLE, expires_at=90)
    restarted.consume(fresh, action_digest=bound, reviewer_role=ROLE)


def test_restore_reconciles_effects_and_authority_against_current_status(
    tmp_path,
) -> None:
    fixture = SyntheticAuthority()
    epochs = SyntheticDurableEpochs(tmp_path / "independent-epochs.sqlite")
    nonce_store = SQLiteApprovalNonceStore(tmp_path / "nonces.sqlite")
    sink = SyntheticSink()
    binding = fixture.runtime.bind(fixture.contracts)
    signer = ApprovalTokenSigner(KEY, clock=fixture.clock)
    token = signer.issue(action_digest=ACTION_DIGEST, reviewer_role=ROLE, expires_at=90)
    receipt = ApprovalTokenVerifier(KEY, nonce_store, clock=fixture.clock).consume(
        token, action_digest=ACTION_DIGEST, reviewer_role=ROLE
    )
    gated = _checkpoint(fixture, approval_required=True, receipt=receipt)
    open_run = _checkpoint(fixture, approval_required=False)
    journals = {}
    for name, checkpoint in (("gated", gated), ("open", open_run)):
        journal = CheckpointJournal(tmp_path / "journals" / name)
        journal.append(checkpoint)
        journals[name] = journal

    # One gated effect commits after the backup; it must not be repeated.
    sink.apply(gated.effects[0].idempotency_key)
    lineage = {name: journal.load() for name, journal in journals.items()}
    snapshot = RestoredSnapshot(
        snapshot_digest=SNAPSHOT_DIGEST,
        recorded_epoch=epochs.current(),
        approvals=(),
        workflows=(
            RestoredWorkflow(lineage=lineage["gated"], authority=binding),
            RestoredWorkflow(lineage=lineage["open"], authority=binding),
        ),
    )

    # Current authority status is unavailable: everything protected stays held.
    fixture.provider.unavailable = True
    held = RestoreReconciler(
        nonce_store, epochs, authority=fixture.runtime, clock=fixture.clock
    ).reconcile(snapshot, observe=sink.observe)
    assert [w.disposition for w in held.workflows] == [
        WorkflowRestoreDisposition.HELD,
        WorkflowRestoreDisposition.HELD,
    ]
    assert sink.observed == 0
    assert not held.complete

    fixture.provider.unavailable = False
    restored = RestoreReconciler(
        nonce_store, epochs, authority=fixture.runtime, clock=fixture.clock
    ).reconcile(snapshot, observe=sink.observe)
    assert restored.epoch.epoch > held.epoch.epoch
    gated_outcome, open_outcome = restored.workflows
    assert gated_outcome.disposition is WorkflowRestoreDisposition.FRESH_REVIEW_REQUIRED
    assert gated_outcome.reason is RestoreReason.APPROVAL_INVALIDATED
    assert gated_outcome.retry_idempotency_keys == ()
    assert gated_outcome.historical_approval_receipt_digest == receipt.token_digest
    assert gated_outcome.committed_effects == (
        (gated.effects[0].action_id.serialize(), COMMIT_EVIDENCE),
    )
    assert open_outcome.disposition is WorkflowRestoreDisposition.RESUME
    assert open_outcome.retry_idempotency_keys == tuple(
        effect.idempotency_key for effect in open_run.effects
    )

    # Revocation after the backup: the snapshot binding still records it active.
    fixture.provider.revoke(fixture.contracts[0])
    revoked = RestoreReconciler(
        nonce_store, epochs, authority=fixture.runtime, clock=fixture.clock
    ).reconcile(snapshot, observe=sink.observe)
    for outcome in revoked.workflows:
        assert outcome.disposition is WorkflowRestoreDisposition.FRESH_REVIEW_REQUIRED
        assert outcome.reason is RestoreReason.AUTHORITY_REVOKED
        assert outcome.retry_idempotency_keys == ()
    assert revoked.workflows[0].committed_effects == ()  # recorded tip had none
    # Committed effects remain in the sink; restore never deletes them.
    assert sink.commits == {gated.effects[0].idempotency_key: COMMIT_EVIDENCE}
    # Journals are retained unchanged as historical audit evidence.
    for name, journal in journals.items():
        assert journal.load() == lineage[name]
