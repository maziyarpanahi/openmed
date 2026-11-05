"""Synthetic offline tests for fail-closed required audit at effect boundaries."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.agent.audit.action_ledger import ActionLedger, ActionState
from openmed.agent.correlation import RunId
from openmed.agent.workflows.recovery import ObservationState
from openmed.compliance.audit_chain import HashChainAuditLog
from openmed.service.required_audit import (
    ActionLedgerAuditStore,
    AuditStage,
    CommitBoundary,
    EvidenceState,
    RequiredAuditCoordinator,
    RequiredAuditError,
    RequiredAuditOutcome,
    RequiredAuditPolicy,
    RequiredAuditReason,
    RequiredAuditRecord,
    RequiredAuditStoreError,
    derive_receipt_id,
)

RUN = RunId("run_" + "4" * 32)
OP = "jobs.terminal_state"
ACK_OP = "omop.local_batch"
SENTINEL = "SENTINELmrn4417SYNTHETIC"
DIGEST = "sha256:" + "d" * 64
OTHER_DIGEST = "sha256:" + "e" * 64
POLICY = RequiredAuditPolicy(
    {OP: CommitBoundary.INVOCATION, ACK_OP: CommitBoundary.ACKNOWLEDGED}
)


class FaultyStore:
    """Wrap a durable store and inject faults at chosen audit stages."""

    def __init__(self, inner, faults=None):
        self.inner = inner
        self.faults = dict(faults or {})
        self.read_fault = False

    def append(self, record):
        fault = self.faults.pop(record.stage, None)
        if fault == "disk_full":
            raise RequiredAuditStoreError("audit_storage_full")
        if fault == "rejected":
            raise RequiredAuditStoreError("audit_append_rejected")
        digest = self.inner.append(record)
        if fault == "ack_interrupted":
            raise RequiredAuditStoreError("audit_unavailable")
        return digest

    def history(self, receipt_id):
        if self.read_fault:
            raise RequiredAuditStoreError("audit_read_failed")
        return self.inner.history(receipt_id)

    def open_receipt_ids(self):
        if self.read_fault:
            raise RequiredAuditStoreError("audit_read_failed")
        return self.inner.open_receipt_ids()


class Effect:
    """Count invocations of one synthetic effect and optionally fail."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def __call__(self, effect_key):
        self.calls.append(effect_key)
        if self.error is not None:
            raise self.error


class Observer:
    """Return one configured observation and count queries."""

    def __init__(self, state=ObservationState.COMMITTED):
        self.state = state
        self.calls = []

    def observe(self, operation, effect_key):
        self.calls.append((operation, effect_key))
        if isinstance(self.state, Exception):
            raise self.state
        return self.state


def ledger_store(tmp_path: Path, policy=POLICY) -> ActionLedgerAuditStore:
    return ActionLedgerAuditStore(
        ActionLedger(tmp_path / "ledger"), run_id=RUN, operations=policy.operations
    )


def coordinator(tmp_path, faults=None, policy=POLICY, diagnostics=None):
    store = FaultyStore(ledger_store(tmp_path, policy), faults)
    return RequiredAuditCoordinator(policy, store, diagnostics=diagnostics), store


def perform(coord, effect, *, operation=OP, key=SENTINEL, digest=DIGEST):
    return coord.perform(
        operation, request_key=key, operation_digest=digest, effect=effect
    )


def ledger_states(tmp_path):
    return [entry.state for entry in ActionLedger(tmp_path / "ledger").load()]


# Policy and configuration -------------------------------------------------


def test_policy_rejects_invalid_declarations_with_fixed_codes():
    with pytest.raises(RequiredAuditError) as error:
        RequiredAuditPolicy({})
    assert str(error.value) == "empty_policy"
    with pytest.raises(RequiredAuditError) as error:
        RequiredAuditPolicy({SENTINEL + " ": CommitBoundary.INVOCATION})
    assert str(error.value) == "invalid_operation"
    assert SENTINEL not in repr(error.value)
    with pytest.raises(RequiredAuditError):
        RequiredAuditPolicy({OP: "invocation"})
    with pytest.raises(RequiredAuditError) as error:
        RequiredAuditPolicy({OP: CommitBoundary.INVOCATION}, max_open_receipts=0)
    assert str(error.value) == "invalid_open_receipt_bound"
    with pytest.raises(RequiredAuditError):
        RequiredAuditPolicy({OP: CommitBoundary.INVOCATION}, max_open_receipts=5000)
    assert POLICY.boundary_for(ACK_OP) is CommitBoundary.ACKNOWLEDGED
    assert POLICY.boundary_for("undeclared.op") is None


def test_unconfigured_required_store_denies_even_with_diagnostics(tmp_path):
    diagnostics = HashChainAuditLog()
    coord = RequiredAuditCoordinator(POLICY, None, diagnostics=diagnostics)
    effect = Effect()

    receipt = perform(coord, effect)

    assert receipt.outcome is RequiredAuditOutcome.DENIED
    assert receipt.reason is RequiredAuditReason.REQUIRED_AUDIT_UNCONFIGURED
    assert receipt.evidence is EvidenceState.NONE
    assert not receipt.fully_evidenced
    assert effect.calls == []
    assert diagnostics.records  # diagnostics may observe, never substitute


def test_diagnostic_sink_cannot_be_installed_as_required_store():
    with pytest.raises(RequiredAuditError) as error:
        RequiredAuditCoordinator(POLICY, HashChainAuditLog())
    assert str(error.value) == "invalid_required_store"


def test_undeclared_operation_is_denied_before_any_record(tmp_path):
    coord, _ = coordinator(tmp_path)
    effect = Effect()

    receipt = perform(coord, effect, operation="undeclared.op")

    assert receipt.outcome is RequiredAuditOutcome.DENIED
    assert receipt.reason is RequiredAuditReason.OPERATION_UNDECLARED
    assert effect.calls == []
    assert ledger_states(tmp_path) == []


def test_invalid_arguments_raise_value_free_codes(tmp_path):
    coord, _ = coordinator(tmp_path)
    with pytest.raises(RequiredAuditError) as error:
        perform(coord, Effect(), key="short")
    assert str(error.value) == "invalid_request_key"
    with pytest.raises(RequiredAuditError) as error:
        perform(coord, Effect(), digest=SENTINEL)
    assert str(error.value) == "invalid_operation_digest"
    assert SENTINEL not in str(error.value)
    with pytest.raises(RequiredAuditError):
        coord.perform(OP, request_key=SENTINEL, operation_digest=DIGEST, effect=None)


# Success and idempotent replay ---------------------------------------------


def test_success_is_fully_evidenced_by_the_required_store(tmp_path):
    diagnostics = HashChainAuditLog()
    coord, _ = coordinator(tmp_path, diagnostics=diagnostics)
    effect = Effect()

    receipt = perform(coord, effect)

    assert receipt.outcome is RequiredAuditOutcome.COMPLETED
    assert receipt.reason is RequiredAuditReason.COMMITTED
    assert receipt.evidence is EvidenceState.COMPLETE
    assert receipt.fully_evidenced
    assert receipt.recorded_stages == (
        AuditStage.INTENT,
        AuditStage.ATTEMPT,
        AuditStage.COMMIT,
    )
    assert effect.calls == [receipt.effect_key]
    assert receipt.receipt_id == derive_receipt_id(OP, SENTINEL)
    entries = ActionLedger(tmp_path / "ledger").load()
    assert [entry.state for entry in entries] == [
        ActionState.PROPOSED,
        ActionState.ATTEMPTED,
        ActionState.COMMITTED,
    ]
    assert receipt.evidence_digest == entries[-1].entry_digest


def test_same_request_key_replays_receipt_without_new_effect(tmp_path):
    coord, _ = coordinator(tmp_path)
    effect = Effect()
    first = perform(coord, effect)

    second = perform(coord, effect)

    assert effect.calls == [first.effect_key]
    assert second.outcome is RequiredAuditOutcome.COMPLETED
    assert second.reason is RequiredAuditReason.ALREADY_RECORDED
    assert second.evidence_digest == first.evidence_digest
    assert len(ledger_states(tmp_path)) == 3


def test_reused_request_key_with_different_payload_is_denied(tmp_path):
    coord, _ = coordinator(tmp_path)
    effect = Effect()
    perform(coord, effect)

    conflict = perform(coord, effect, digest=OTHER_DIGEST)

    assert conflict.outcome is RequiredAuditOutcome.DENIED
    assert conflict.reason is RequiredAuditReason.IDEMPOTENCY_CONFLICT
    assert len(effect.calls) == 1


# Fault matrix: before the commit boundary -> denied -------------------------


@pytest.mark.parametrize("fault", ["disk_full", "rejected", "ack_interrupted"])
@pytest.mark.parametrize("stage", [AuditStage.INTENT, AuditStage.ATTEMPT])
@pytest.mark.parametrize("operation", [OP, ACK_OP])
def test_faults_before_effect_boundary_deny_without_invoking(
    tmp_path, fault, stage, operation
):
    coord, _ = coordinator(tmp_path, {stage: fault})
    effect = Effect()

    receipt = perform(coord, effect, operation=operation)

    assert receipt.outcome is RequiredAuditOutcome.DENIED
    expected = (
        RequiredAuditReason.INTENT_NOT_DURABLE
        if stage is AuditStage.INTENT
        else RequiredAuditReason.ATTEMPT_NOT_DURABLE
    )
    assert receipt.reason is expected
    assert effect.calls == []
    assert not receipt.fully_evidenced
    definitive = stage is AuditStage.INTENT and fault != "ack_interrupted"
    assert receipt.evidence is (
        EvidenceState.NONE if definitive else EvidenceState.PENDING
    )


# Fault matrix: after the commit boundary -> reconcile required --------------


@pytest.mark.parametrize("fault", ["disk_full", "rejected", "ack_interrupted"])
@pytest.mark.parametrize("operation", [OP, ACK_OP])
def test_commit_record_faults_after_effect_require_reconcile(
    tmp_path, fault, operation
):
    coord, _ = coordinator(tmp_path, {AuditStage.COMMIT: fault})
    effect = Effect()

    receipt = perform(coord, effect, operation=operation)

    assert receipt.outcome is RequiredAuditOutcome.RECONCILE_REQUIRED
    assert receipt.reason is RequiredAuditReason.COMMIT_NOT_DURABLE
    assert receipt.evidence is EvidenceState.PENDING
    assert not receipt.fully_evidenced
    assert receipt.evidence_digest is None
    assert len(effect.calls) == 1


def test_invocation_boundary_treats_adapter_failure_as_uncertain(tmp_path):
    coord, _ = coordinator(tmp_path)
    effect = Effect(RuntimeError(SENTINEL))

    receipt = perform(coord, effect)

    assert receipt.outcome is RequiredAuditOutcome.RECONCILE_REQUIRED
    assert receipt.reason is RequiredAuditReason.EFFECT_OUTCOME_UNKNOWN
    assert ledger_states(tmp_path) == [ActionState.PROPOSED, ActionState.ATTEMPTED]
    assert SENTINEL not in json.dumps(receipt.to_dict())


def test_acknowledged_boundary_records_abort_for_adapter_failure(tmp_path):
    coord, _ = coordinator(tmp_path)
    effect = Effect(RuntimeError(SENTINEL))

    receipt = perform(coord, effect, operation=ACK_OP)

    assert receipt.outcome is RequiredAuditOutcome.DENIED
    assert receipt.reason is RequiredAuditReason.EFFECT_NOT_COMMITTED
    assert receipt.evidence is EvidenceState.COMPLETE
    assert ledger_states(tmp_path)[-1] is ActionState.REJECTED


def test_acknowledged_abort_fault_leaves_pending_denial(tmp_path):
    coord, _ = coordinator(tmp_path, {AuditStage.ABORT: "disk_full"})

    receipt = perform(coord, Effect(RuntimeError("x")), operation=ACK_OP)

    assert receipt.outcome is RequiredAuditOutcome.DENIED
    assert receipt.reason is RequiredAuditReason.ABORT_NOT_DURABLE
    assert receipt.evidence is EvidenceState.PENDING


# Committed effects are never retried blindly --------------------------------


def test_retry_after_commit_fault_never_reinvokes_effect(tmp_path):
    coord, _ = coordinator(tmp_path, {AuditStage.COMMIT: "disk_full"})
    effect = Effect()
    first = perform(coord, effect)

    retried = perform(coord, effect)

    assert len(effect.calls) == 1
    assert retried.outcome is RequiredAuditOutcome.RECONCILE_REQUIRED
    assert retried.reason is RequiredAuditReason.RECONCILE_PENDING
    assert retried.receipt_id == first.receipt_id


def test_reconcile_has_no_effect_port_and_ambiguity_stays_pending(tmp_path):
    coord, _ = coordinator(tmp_path, {AuditStage.COMMIT: "rejected"})
    effect = Effect()
    receipt = perform(coord, effect)
    observer = Observer(ObservationState.AMBIGUOUS)

    result = coord.reconcile(receipt, observer=observer)

    assert result.outcome is RequiredAuditOutcome.RECONCILE_REQUIRED
    assert result.reason is RequiredAuditReason.EFFECT_AMBIGUOUS
    assert observer.calls == [(OP, receipt.effect_key)]
    assert len(effect.calls) == 1


@pytest.mark.parametrize("failure", [RuntimeError(SENTINEL), "not-a-state"])
def test_observer_failure_stays_reconcile_required(tmp_path, failure):
    coord, _ = coordinator(tmp_path, {AuditStage.COMMIT: "disk_full"})
    receipt = perform(coord, Effect())

    result = coord.reconcile(receipt, observer=Observer(failure))

    assert result.outcome is RequiredAuditOutcome.RECONCILE_REQUIRED
    assert result.reason is RequiredAuditReason.OBSERVER_UNAVAILABLE
    assert SENTINEL not in json.dumps(result.to_dict())


# Recovery completes the same receipt exactly once ---------------------------


def test_reconcile_after_disk_full_completes_same_receipt_once(tmp_path):
    coord, _ = coordinator(tmp_path, {AuditStage.COMMIT: "disk_full"})
    effect = Effect()
    pending = perform(coord, effect)
    observer = Observer(ObservationState.COMMITTED)

    first = coord.reconcile(pending, observer=observer)
    second = coord.reconcile(pending, observer=observer)

    assert first.receipt_id == pending.receipt_id
    assert first.outcome is RequiredAuditOutcome.COMPLETED
    assert first.reason is RequiredAuditReason.RECONCILED_COMMIT
    assert first.fully_evidenced
    assert second.reason is RequiredAuditReason.RECONCILED_EXISTING
    assert second.evidence_digest == first.evidence_digest
    assert ledger_states(tmp_path).count(ActionState.COMMITTED) == 1
    assert len(observer.calls) == 1
    assert len(effect.calls) == 1


def test_interrupted_ack_reconciles_without_duplicate_record(tmp_path):
    coord, _ = coordinator(tmp_path, {AuditStage.COMMIT: "ack_interrupted"})
    pending = perform(coord, Effect())
    observer = Observer(ObservationState.ABSENT)

    result = coord.reconcile(pending, observer=observer)

    assert result.outcome is RequiredAuditOutcome.COMPLETED
    assert result.reason is RequiredAuditReason.RECONCILED_EXISTING
    assert observer.calls == []  # durable commit record wins over observation
    assert ledger_states(tmp_path).count(ActionState.COMMITTED) == 1


def test_reconcile_absent_effect_records_abort(tmp_path):
    coord, _ = coordinator(tmp_path)
    pending = perform(coord, Effect(RuntimeError("x")))

    result = coord.reconcile(pending, observer=Observer(ObservationState.ABSENT))

    assert result.outcome is RequiredAuditOutcome.DENIED
    assert result.reason is RequiredAuditReason.RECONCILED_ABSENT
    assert result.evidence is EvidenceState.COMPLETE
    assert ledger_states(tmp_path)[-1] is ActionState.REJECTED


def test_reconcile_intent_only_closes_without_observation(tmp_path):
    coord, _ = coordinator(tmp_path, {AuditStage.ATTEMPT: "rejected"})
    pending = perform(coord, Effect())
    observer = Observer()

    result = coord.reconcile(pending, observer=observer)

    assert result.outcome is RequiredAuditOutcome.DENIED
    assert result.reason is RequiredAuditReason.RECONCILED_NOT_ATTEMPTED
    assert observer.calls == []
    assert ledger_states(tmp_path) == [ActionState.PROPOSED, ActionState.REJECTED]


def test_reconcile_with_no_durable_intent_reports_nothing_recorded(tmp_path):
    coord, _ = coordinator(tmp_path, {AuditStage.INTENT: "disk_full"})
    pending = perform(coord, Effect())

    result = coord.reconcile(pending, observer=Observer())

    assert result.outcome is RequiredAuditOutcome.DENIED
    assert result.reason is RequiredAuditReason.NO_DURABLE_INTENT
    assert result.evidence is EvidenceState.NONE


def test_reconcile_commit_fault_is_retried_only_as_audit_append(tmp_path):
    coord, store = coordinator(tmp_path, {AuditStage.COMMIT: "disk_full"})
    effect = Effect()
    pending = perform(coord, effect)
    store.faults[AuditStage.COMMIT] = "disk_full"

    still = coord.reconcile(pending, observer=Observer())
    done = coord.reconcile(still, observer=Observer())

    assert still.outcome is RequiredAuditOutcome.RECONCILE_REQUIRED
    assert still.reason is RequiredAuditReason.COMMIT_NOT_DURABLE
    assert done.outcome is RequiredAuditOutcome.COMPLETED
    assert len(effect.calls) == 1


def test_reconcile_read_failure_keeps_outcome(tmp_path):
    coord, store = coordinator(tmp_path, {AuditStage.COMMIT: "disk_full"})
    pending = perform(coord, Effect())
    store.read_fault = True

    result = coord.reconcile(pending, observer=Observer())

    assert result.outcome is RequiredAuditOutcome.RECONCILE_REQUIRED
    assert result.reason is RequiredAuditReason.AUDIT_READ_FAILED


def test_reconcile_rejects_receipt_for_other_payload(tmp_path):
    coord, _ = coordinator(tmp_path, {AuditStage.COMMIT: "disk_full"})
    pending = perform(coord, Effect())
    tampered = replace(pending, binding_digest=OTHER_DIGEST)

    result = coord.reconcile(tampered, observer=Observer())

    assert result.outcome is RequiredAuditOutcome.RECONCILE_REQUIRED
    assert result.reason is RequiredAuditReason.EVIDENCE_MISMATCH
    assert ledger_states(tmp_path).count(ActionState.COMMITTED) == 0


# Bounded open evidence and admission -----------------------------------------


def test_open_receipt_bound_denies_new_admission(tmp_path):
    policy = RequiredAuditPolicy({OP: CommitBoundary.INVOCATION}, max_open_receipts=1)
    coord, store = coordinator(tmp_path, policy=policy)
    store.faults[AuditStage.COMMIT] = "disk_full"
    first = perform(coord, Effect(), key="A" * 16)
    effect = Effect()

    blocked = perform(coord, effect, key="B" * 16)

    assert first.outcome is RequiredAuditOutcome.RECONCILE_REQUIRED
    assert blocked.outcome is RequiredAuditOutcome.DENIED
    assert blocked.reason is RequiredAuditReason.RECONCILE_BACKLOG_FULL
    assert effect.calls == []
    coord.reconcile(first, observer=Observer())
    assert perform(coord, effect, key="B" * 16).fully_evidenced


def test_read_failure_denies_admission(tmp_path):
    coord, store = coordinator(tmp_path)
    store.read_fault = True
    effect = Effect()

    receipt = perform(coord, effect)

    assert receipt.outcome is RequiredAuditOutcome.DENIED
    assert receipt.reason is RequiredAuditReason.AUDIT_READ_FAILED
    assert effect.calls == []


def test_concurrent_same_receipt_is_not_invoked_twice(tmp_path):
    coord, _ = coordinator(tmp_path)
    nested = []

    def effect(effect_key):
        nested.append(perform(coord, Effect()))

    outer = perform(coord, effect)

    assert outer.outcome is RequiredAuditOutcome.COMPLETED
    assert nested[0].outcome is RequiredAuditOutcome.RECONCILE_REQUIRED
    assert nested[0].reason is RequiredAuditReason.RECEIPT_IN_FLIGHT


def test_open_receipts_are_rebuilt_from_required_store(tmp_path):
    coord, _ = coordinator(tmp_path, {AuditStage.COMMIT: "disk_full"})
    pending = perform(coord, Effect())

    restarted = RequiredAuditCoordinator(POLICY, ledger_store(tmp_path))
    rebuilt = restarted.open_receipts()

    assert [receipt.receipt_id for receipt in rebuilt] == [pending.receipt_id]
    assert rebuilt[0].effect_key == pending.effect_key
    assert rebuilt[0].outcome is RequiredAuditOutcome.RECONCILE_REQUIRED
    results = restarted.reconcile_open(observer=Observer())
    assert [result.outcome for result in results] == [RequiredAuditOutcome.COMPLETED]
    assert restarted.open_receipts() == ()


# Diagnostics are optional and never evidence --------------------------------


class BrokenSink:
    def append(self, event_type, payload):
        raise OSError(SENTINEL)


def test_diagnostic_failure_never_changes_required_outcome(tmp_path):
    coord, _ = coordinator(tmp_path, diagnostics=BrokenSink())

    receipt = perform(coord, Effect())

    assert receipt.fully_evidenced


def test_diagnostic_success_cannot_cover_missing_required_record(tmp_path):
    diagnostics = HashChainAuditLog()
    coord, _ = coordinator(
        tmp_path, {AuditStage.COMMIT: "disk_full"}, diagnostics=diagnostics
    )

    receipt = perform(coord, Effect())

    assert diagnostics.verify() and diagnostics.records
    assert receipt.outcome is RequiredAuditOutcome.RECONCILE_REQUIRED
    assert not receipt.fully_evidenced
    restarted = RequiredAuditCoordinator(POLICY, ledger_store(tmp_path))
    assert [r.receipt_id for r in restarted.open_receipts()] == [receipt.receipt_id]


# Value-free evidence ----------------------------------------------------------


def test_sentinel_never_reaches_receipts_ledger_or_diagnostics(tmp_path):
    diagnostics = HashChainAuditLog()
    coord, _ = coordinator(
        tmp_path, {AuditStage.COMMIT: "ack_interrupted"}, diagnostics=diagnostics
    )
    receipt = perform(coord, Effect())
    reconciled = coord.reconcile(receipt, observer=Observer())

    rendered = [
        json.dumps(receipt.to_dict()),
        json.dumps(reconciled.to_dict()),
        repr(receipt),
        json.dumps(diagnostics.to_payload()),
    ]
    rendered += [
        path.read_text() for path in sorted((tmp_path / "ledger").glob("*.json"))
    ]
    assert all(SENTINEL not in text for text in rendered)
    assert receipt.to_dict()["schema_version"] == "openmed.service.required_audit.v1"


def test_store_error_rejects_unknown_codes():
    assert str(RequiredAuditStoreError(SENTINEL)) == "audit_unavailable"
    assert RequiredAuditStoreError("audit_storage_full").not_written
    assert not RequiredAuditStoreError("audit_unavailable").not_written


def test_record_validates_value_free_fields():
    with pytest.raises(RequiredAuditError):
        RequiredAuditRecord(SENTINEL, OP, AuditStage.INTENT, DIGEST)
    with pytest.raises(RequiredAuditError):
        RequiredAuditRecord(derive_receipt_id(OP, SENTINEL), OP, "intent", DIGEST)


def test_reconcile_refuses_forged_effect_key(tmp_path):
    coord, _ = coordinator(tmp_path, {AuditStage.COMMIT: "disk_full"})
    pending = perform(coord, Effect())
    observer = Observer()

    result = coord.reconcile(
        replace(pending, effect_key="idem_" + "0" * 64), observer=observer
    )

    assert result.reason is RequiredAuditReason.EVIDENCE_MISMATCH
    assert observer.calls == []


def test_ledger_store_rejects_second_terminal_record(tmp_path):
    store = ledger_store(tmp_path)
    receipt_id = derive_receipt_id(OP, SENTINEL)
    for stage in (AuditStage.INTENT, AuditStage.ATTEMPT, AuditStage.COMMIT):
        store.append(RequiredAuditRecord(receipt_id, OP, stage, DIGEST))

    for stage in (AuditStage.COMMIT, AuditStage.ABORT):
        with pytest.raises(RequiredAuditStoreError) as error:
            store.append(RequiredAuditRecord(receipt_id, OP, stage, DIGEST))
        assert error.value.code == "audit_append_rejected"
    assert len(store.history(receipt_id)) == 3
    assert store.open_receipt_ids() == ()


def test_ledger_store_validates_configuration(tmp_path):
    ledger = ActionLedger(tmp_path / "ledger")
    with pytest.raises(RequiredAuditError) as error:
        ActionLedgerAuditStore(ledger, run_id=RUN, operations=[OP], actor_role="x")
    assert str(error.value) == "invalid_actor_role"
    with pytest.raises(RequiredAuditError):
        ActionLedgerAuditStore(ledger, run_id=RUN, operations=[])
    with pytest.raises(RequiredAuditError):
        ActionLedgerAuditStore(object(), run_id=RUN, operations=[OP])


def test_unknown_operation_in_ledger_fails_closed_on_read(tmp_path):
    coord, _ = coordinator(tmp_path, {AuditStage.COMMIT: "disk_full"})
    perform(coord, Effect())
    narrowed = RequiredAuditPolicy({ACK_OP: CommitBoundary.ACKNOWLEDGED})
    restarted = RequiredAuditCoordinator(narrowed, ledger_store(tmp_path, narrowed))

    with pytest.raises(RequiredAuditStoreError) as error:
        restarted.open_receipts()
    assert error.value.code == "audit_read_failed"
    effect = Effect()
    denied = perform(restarted, effect, operation=ACK_OP, key="C" * 16)
    assert denied.reason is RequiredAuditReason.AUDIT_READ_FAILED
    assert effect.calls == []


class GarbageStore(FaultyStore):
    def append(self, record):
        self.inner.append(record)
        return SENTINEL


def test_store_returning_non_digest_ack_is_not_evidence(tmp_path):
    store = GarbageStore(ledger_store(tmp_path))
    coord = RequiredAuditCoordinator(POLICY, store)
    effect = Effect()

    receipt = perform(coord, effect)

    assert receipt.outcome is RequiredAuditOutcome.DENIED
    assert receipt.reason is RequiredAuditReason.INTENT_NOT_DURABLE
    assert receipt.evidence is EvidenceState.PENDING
    assert effect.calls == []
