"""Synthetic OS-level faults against the durable action-ledger audit store."""

from __future__ import annotations

import errno
import os
import tempfile

import pytest

from openmed.agent.audit.action_ledger import ActionLedger, ActionState
from openmed.agent.correlation import RunId
from openmed.agent.workflows.recovery import ObservationState
from openmed.service.required_audit import (
    ActionLedgerAuditStore,
    CommitBoundary,
    EvidenceState,
    RequiredAuditCoordinator,
    RequiredAuditOutcome,
    RequiredAuditPolicy,
    RequiredAuditReason,
)

pytestmark = pytest.mark.integration

RUN = RunId("run_" + "7" * 32)
OP = "fhir.write_back"
KEY = "SYNTHETICrequestKEY0001"
DIGEST = "sha256:" + "9" * 64
POLICY = RequiredAuditPolicy({OP: CommitBoundary.INVOCATION})
LEDGER_MODULE = "openmed.agent.audit.action_ledger.os"


def build(tmp_path):
    store = ActionLedgerAuditStore(
        ActionLedger(tmp_path / "ledger"), run_id=RUN, operations=POLICY.operations
    )
    return RequiredAuditCoordinator(POLICY, store)


class Sink:
    """Synthetic idempotent effect sink keyed only by opaque effect keys."""

    def __init__(self):
        self.applied: list[str] = []

    def apply(self, effect_key):
        self.applied.append(effect_key)

    def observe(self, operation, effect_key):
        if effect_key in self.applied:
            return ObservationState.COMMITTED
        return ObservationState.ABSENT


def disk_full_after(monkeypatch, entries_allowed):
    """Fail ledger entry writes with ENOSPC after ``entries_allowed`` entries."""
    real_write = os.write
    real_mkstemp = tempfile.mkstemp
    ordinals: dict[int, int] = {}
    created = {"count": 0}

    def mkstemp(*args, **kwargs):
        descriptor, name = real_mkstemp(*args, **kwargs)
        if kwargs.get("prefix") == ".entry-":
            ordinals[descriptor] = created["count"]
            created["count"] += 1
        return descriptor, name

    def write(descriptor, data):
        if ordinals.get(descriptor, -1) >= entries_allowed:
            raise OSError(errno.ENOSPC, "synthetic")
        return real_write(descriptor, data)

    monkeypatch.setattr(f"{LEDGER_MODULE}.write", write)
    monkeypatch.setattr("openmed.agent.audit.action_ledger.tempfile.mkstemp", mkstemp)


def test_disk_full_on_intent_denies_and_leaves_no_entry(tmp_path, monkeypatch):
    coord = build(tmp_path)
    sink = Sink()
    disk_full_after(monkeypatch, 0)

    receipt = coord.perform(
        OP, request_key=KEY, operation_digest=DIGEST, effect=sink.apply
    )

    assert receipt.outcome is RequiredAuditOutcome.DENIED
    assert receipt.reason is RequiredAuditReason.INTENT_NOT_DURABLE
    assert sink.applied == []
    monkeypatch.undo()
    assert ActionLedger(tmp_path / "ledger").load() == ()


def test_disk_full_after_effect_then_restart_completes_once(tmp_path, monkeypatch):
    coord = build(tmp_path)
    sink = Sink()
    disk_full_after(monkeypatch, 2)  # intent and attempt fit, commit does not

    receipt = coord.perform(
        OP, request_key=KEY, operation_digest=DIGEST, effect=sink.apply
    )

    assert receipt.outcome is RequiredAuditOutcome.RECONCILE_REQUIRED
    assert receipt.evidence is EvidenceState.PENDING
    assert len(sink.applied) == 1
    monkeypatch.undo()

    restarted = build(tmp_path)
    retried = restarted.perform(
        OP, request_key=KEY, operation_digest=DIGEST, effect=sink.apply
    )
    assert retried.outcome is RequiredAuditOutcome.RECONCILE_REQUIRED
    assert len(sink.applied) == 1

    first = restarted.reconcile_open(observer=sink)
    second = restarted.reconcile(receipt, observer=sink)

    assert [r.outcome for r in first] == [RequiredAuditOutcome.COMPLETED]
    assert first[0].receipt_id == receipt.receipt_id
    assert second.reason is RequiredAuditReason.RECONCILED_EXISTING
    assert second.evidence_digest == first[0].evidence_digest
    states = [entry.state for entry in ActionLedger(tmp_path / "ledger").load()]
    assert states == [
        ActionState.PROPOSED,
        ActionState.ATTEMPTED,
        ActionState.COMMITTED,
    ]
    assert len(sink.applied) == 1


def test_interrupted_directory_sync_is_reconciled_from_ledger(tmp_path, monkeypatch):
    coord = build(tmp_path)
    sink = Sink()
    ledger_dir = str(tmp_path / "ledger")
    real_open = os.open
    seen = {"count": 0}

    def open_(path, flags, *args, **kwargs):
        if str(path) == ledger_dir:
            seen["count"] += 1
            if seen["count"] == 3:  # commit entry linked, ack interrupted
                raise OSError(errno.EIO, "synthetic")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(f"{LEDGER_MODULE}.open", open_)

    receipt = coord.perform(
        OP, request_key=KEY, operation_digest=DIGEST, effect=sink.apply
    )
    monkeypatch.undo()

    assert receipt.outcome is RequiredAuditOutcome.RECONCILE_REQUIRED
    assert receipt.reason is RequiredAuditReason.COMMIT_NOT_DURABLE
    result = build(tmp_path).reconcile(receipt, observer=sink)
    assert result.outcome is RequiredAuditOutcome.COMPLETED
    assert result.reason is RequiredAuditReason.RECONCILED_EXISTING
    assert result.fully_evidenced
    entries = ActionLedger(tmp_path / "ledger").load()
    assert [e.state for e in entries].count(ActionState.COMMITTED) == 1
    assert result.evidence_digest == entries[-1].entry_digest
    assert KEY not in "".join(
        p.read_text() for p in (tmp_path / "ledger").glob("*.json")
    )
