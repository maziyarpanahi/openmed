"""Offline negative controls for service continuation authorization."""

from __future__ import annotations

import asyncio
import json
import traceback

import pytest

from openmed.agent.permissions.revocation import (
    AuthorityKind,
    AuthorityReason,
    InMemoryAuthorityGenerationStore,
)
from openmed.agent.workflows.recovery import EffectState
from openmed.agent.workflows.scheduler import CancellationSignal
from openmed.service.auth import AuthPrincipal
from openmed.service.continuation_auth import (
    CONTINUATION_OUTCOME_VERSION,
    CONTINUATION_RECORD_VERSION,
    ContinuationBoundary,
    ContinuationDisposition,
    ContinuationError,
    ContinuationGuard,
    ContinuationInterrupted,
    ContinuationReason,
    ContinuationRecord,
    GuardedStream,
    InMemoryContinuationLedger,
    ReconciliationState,
    consent_authority,
    continuation_error_event,
    principal_authority,
    run_continuation,
    work_digest,
)
from tests.fixtures.service.continuation_authority import (
    CONSENT_DIGEST,
    POLICY_REVISION,
    SENTINEL,
    WORK_ID,
    ContinuationAuthority,
    SyntheticAsyncSource,
    SyntheticSink,
    SyntheticSource,
    pending_effect,
)

_OUTCOME_FIELDS = {
    "schema_version",
    "boundary",
    "disposition",
    "reason_code",
    "reconciliation",
    "compensation",
    "work_digest",
    "epoch_digest",
    "authority_count",
    "checkpoint",
    "delivered_output_count",
    "withheld_output_count",
    "committed_effect_count",
    "pending_effect_count",
    "committed_effect_digest",
}


def _admit(fixture: ContinuationAuthority, checkpoints=("documents", "effects")):
    return fixture.guard.admit(WORK_ID, fixture.contracts, checkpoints=checkpoints)


def _assert_no_sentinel(*values: object) -> None:
    for value in values:
        assert SENTINEL not in repr(value)
        assert SENTINEL not in str(value)
        if hasattr(value, "to_dict"):
            assert SENTINEL not in json.dumps(value.to_dict(), sort_keys=True)


# -- Admission, epoch binding and persisted metadata -------------------------


def test_admission_binds_an_authorization_epoch_to_content_free_metadata():
    fixture = ContinuationAuthority()
    record = _admit(fixture)

    payload = record.to_dict()
    assert payload["schema_version"] == CONTINUATION_RECORD_VERSION
    assert set(payload) == {
        "schema_version",
        "admission_id",
        "work_digest",
        "checkpoints",
        "authority",
    }
    assert payload["work_digest"] == work_digest(WORK_ID)
    assert payload["checkpoints"] == ["documents", "effects"]
    assert {v["kind"] for v in payload["authority"]["versions"]} == {
        AuthorityKind.GRANT.value,
        AuthorityKind.PURPOSE_TICKET.value,
    }
    restored = ContinuationRecord.from_dict(json.loads(json.dumps(payload)))
    assert restored == record
    assert restored.epoch_digest == record.epoch_digest
    assert record.epoch_digest.startswith("sha256:")
    encoded = json.dumps(payload)
    assert SENTINEL not in encoded and WORK_ID not in encoded
    assert "Jane" not in repr(record)

    # A second admission of the same work under the same authority is a new
    # epoch, so cancelling one admission never blocks a fresh resubmission.
    assert _admit(fixture).epoch_digest != record.epoch_digest


def test_admission_requires_active_known_authority():
    for setup, reason in [
        ("revoked", ContinuationReason.AUTHORITY_REVOKED),
        ("unknown", ContinuationReason.AUTHORITY_UNAVAILABLE),
        ("expired", ContinuationReason.AUTHORITY_EXPIRED),
    ]:
        fixture = ContinuationAuthority()
        if setup == "revoked":
            fixture.provider.revoke(fixture.consent)
        elif setup == "unknown":
            fixture.provider.records.clear()
        else:
            fixture.clock.now = 100
        with pytest.raises(ContinuationInterrupted) as error:
            _admit(fixture)
        outcome = error.value.outcome
        assert outcome.boundary is ContinuationBoundary.ADMISSION
        assert outcome.reason is reason
        assert outcome.epoch_digest is None
        assert error.value.code == reason.value
        _assert_no_sentinel(error.value, outcome)


def test_principal_authority_commits_to_scope_and_policy_revision():
    fixture = ContinuationAuthority()
    base = fixture.scope
    assert base.kind is AuthorityKind.GRANT
    assert base.expires_at == 100
    assert SENTINEL not in repr(base)
    narrowed = AuthPrincipal(
        subject=SENTINEL,
        credential_type="api_key",
        scopes=("deidentify",),
        key_id="synthetic-key",
    )
    variants = [
        principal_authority(narrowed, policy_revision=POLICY_REVISION, expires_at=100),
        principal_authority(
            fixture.principal,
            policy_revision="policy:org.example/deid@4",
            expires_at=100,
        ),
    ]
    assert all(v.digest != base.digest for v in variants)
    reordered = AuthPrincipal(
        subject=SENTINEL,
        credential_type="api_key",
        scopes=("jobs", "deidentify"),
        key_id="synthetic-key",
    )
    assert (
        principal_authority(reordered, policy_revision=POLICY_REVISION, expires_at=100)
        == base
    )
    consent = consent_authority(CONSENT_DIGEST, expires_at=100)
    assert consent.kind is AuthorityKind.PURPOSE_TICKET
    assert consent.digest != CONSENT_DIGEST


def test_invalid_inputs_fail_with_fixed_codes_that_never_echo_values():
    fixture = ContinuationAuthority()
    record = _admit(fixture)
    cases = [
        lambda: work_digest(""),
        lambda: work_digest(SENTINEL * 20),
        lambda: fixture.guard.admit(WORK_ID, (), checkpoints=()),
        lambda: fixture.guard.admit(
            WORK_ID, fixture.contracts, checkpoints=(SENTINEL,)
        ),
        lambda: fixture.guard.admit(WORK_ID, fixture.contracts, checkpoints=("a", "a")),
        lambda: fixture.guard.check(record, WORK_ID, SENTINEL),
        lambda: fixture.guard.check(SENTINEL, WORK_ID, ContinuationBoundary.RESUME),
        lambda: consent_authority(SENTINEL, expires_at=100),
        lambda: principal_authority(
            fixture.principal, policy_revision=SENTINEL + " x", expires_at=100
        ),
        lambda: principal_authority(object(), policy_revision="p", expires_at=100),
        lambda: ContinuationGuard(object()),
    ]
    for case in cases:
        with pytest.raises(ContinuationError) as error:
            case()
        assert error.value.code.startswith("invalid_")
        _assert_no_sentinel(error.value)
        assert error.value.__cause__ is None


# -- Boundary revalidation ----------------------------------------------------


@pytest.mark.parametrize(
    "boundary",
    [
        ContinuationBoundary.EXECUTION_START,
        ContinuationBoundary.RESUME,
        ContinuationBoundary.CHECKPOINT,
        ContinuationBoundary.EFFECT_DISPATCH,
        ContinuationBoundary.PROTECTED_OUTPUT,
    ],
)
def test_every_boundary_performs_a_fresh_lookup_and_stops_on_revocation(boundary):
    fixture = ContinuationAuthority()
    record = _admit(fixture)
    checkpoint = "documents" if boundary is ContinuationBoundary.CHECKPOINT else None
    calls = fixture.provider.calls
    outcome = fixture.guard.check(record, WORK_ID, boundary, checkpoint=checkpoint)
    assert outcome.disposition is ContinuationDisposition.CONTINUE
    assert outcome.reason is ContinuationReason.ACTIVE
    assert fixture.provider.calls == calls + len(fixture.contracts)

    fixture.provider.revoke(fixture.consent)
    outcome = fixture.guard.check(record, WORK_ID, boundary, checkpoint=checkpoint)
    assert outcome.disposition is ContinuationDisposition.STOPPED
    assert outcome.reason is ContinuationReason.AUTHORITY_REVOKED
    assert outcome.boundary is boundary
    assert outcome.epoch_digest == record.epoch_digest
    assert set(outcome.to_dict()) == _OUTCOME_FIELDS
    assert outcome.to_dict()["schema_version"] == CONTINUATION_OUTCOME_VERSION


def test_expiry_stops_and_integrity_failures_quarantine():
    cases = {
        "expired": (ContinuationReason.AUTHORITY_EXPIRED, "stopped"),
        "unavailable": (ContinuationReason.AUTHORITY_UNAVAILABLE, "quarantined"),
        "stale": (ContinuationReason.AUTHORITY_STALE, "quarantined"),
        "generation": (ContinuationReason.AUTHORITY_GENERATION_CHANGED, "quarantined"),
        "work_mismatch": (ContinuationReason.WORK_MISMATCH, "quarantined"),
        "undeclared": (ContinuationReason.UNDECLARED_CHECKPOINT, "quarantined"),
        "missing_checkpoint": (
            ContinuationReason.UNDECLARED_CHECKPOINT,
            "quarantined",
        ),
    }
    for case, (reason, disposition) in cases.items():
        fixture = ContinuationAuthority()
        record = _admit(fixture)
        work_id = WORK_ID
        boundary = ContinuationBoundary.RESUME
        checkpoint = None
        if case == "expired":
            fixture.clock.now = 100
        elif case == "unavailable":
            fixture.provider.unavailable = True
        elif case == "stale":
            status = fixture.provider.get_status(
                fixture.scope.kind, fixture.scope.digest
            )
            fixture.provider.override = type(status)(
                status.kind,
                status.digest,
                status.generation,
                status.revoked,
                status.observed_at - 1,
            )
        elif case == "generation":
            key = (fixture.scope.kind, fixture.scope.digest)
            fixture.provider.records[key] = (1, False)
        elif case == "work_mismatch":
            work_id = "job-synthetic-other"
        elif case == "undeclared":
            boundary = ContinuationBoundary.CHECKPOINT
            checkpoint = "not_declared"
        else:
            boundary = ContinuationBoundary.CHECKPOINT
        outcome = fixture.guard.check(record, work_id, boundary, checkpoint=checkpoint)
        assert outcome.reason is reason, case
        assert outcome.disposition.value == disposition, case
        _assert_no_sentinel(outcome)


def test_every_runtime_denial_reason_maps_to_a_closed_continuation_reason():
    for reason in AuthorityReason:
        if reason is AuthorityReason.ACTIVE:
            continue
        assert ContinuationReason(reason.value).value == reason.value


def test_terminal_epoch_cannot_continue_even_with_active_authority():
    fixture = ContinuationAuthority()
    record = _admit(fixture)
    cancelled = fixture.guard.cancel(record, WORK_ID)
    assert cancelled.disposition is ContinuationDisposition.STOPPED
    assert cancelled.reason is ContinuationReason.CANCELLED
    outcome = fixture.guard.check(record, WORK_ID, ContinuationBoundary.RESUME)
    assert outcome.reason is ContinuationReason.TERMINAL
    assert outcome.disposition is ContinuationDisposition.STOPPED


def test_ledger_failure_quarantines_without_continuing():
    class BrokenLedger:
        def is_closed(self, epoch_digest):
            raise RuntimeError(SENTINEL)

        def close(self, epoch_digest):
            raise RuntimeError(SENTINEL)

    fixture = ContinuationAuthority()
    record = _admit(fixture)
    broken = ContinuationGuard(fixture.runtime, ledger=BrokenLedger())
    outcome = broken.check(record, WORK_ID, ContinuationBoundary.RESUME)
    assert outcome.reason is ContinuationReason.LEDGER_UNAVAILABLE
    assert outcome.disposition is ContinuationDisposition.QUARANTINED
    _assert_no_sentinel(outcome)


# -- Restart cannot revive stale consent ---------------------------------------


def test_restarted_worker_cannot_revive_revoked_consent_from_metadata():
    fixture = ContinuationAuthority()
    persisted = json.dumps(_admit(fixture).to_dict())
    fixture.provider.revoke(fixture.consent)

    # Fresh process: empty generation store and ledger, same authority service.
    restarted = fixture.restart()
    record = ContinuationRecord.from_dict(json.loads(persisted))
    outcome = restarted.guard.check(record, WORK_ID, ContinuationBoundary.RESUME)
    assert outcome.disposition is ContinuationDisposition.STOPPED
    assert outcome.reason is ContinuationReason.AUTHORITY_REVOKED

    # Re-pinning the restored binding to the provider's current generation is
    # still revoked: generations never resurrect a tombstoned artifact.
    payload = json.loads(persisted)
    for version in payload["authority"]["versions"]:
        version["generation"] = 1
    forged = ContinuationRecord.from_dict(payload)
    outcome = restarted.guard.check(forged, WORK_ID, ContinuationBoundary.RESUME)
    assert outcome.disposition is not ContinuationDisposition.CONTINUE


def test_restored_metadata_cannot_carry_cached_consent_or_status():
    fixture = ContinuationAuthority()
    payload = _admit(fixture).to_dict()
    for extra in [
        {"authorized": True},
        {"status": "active"},
        {"consent_active": True},
        {"observed_at": 10},
    ]:
        with pytest.raises(ContinuationError) as error:
            ContinuationRecord.from_dict({**payload, **extra})
        assert error.value.code == "invalid_continuation_record"
    for missing in list(payload):
        broken = dict(payload)
        broken.pop(missing)
        with pytest.raises(ContinuationError):
            ContinuationRecord.from_dict(broken)
    with pytest.raises(ContinuationError):
        ContinuationRecord.from_dict({**payload, "schema_version": "v0"})


def test_restart_with_durable_ledger_keeps_quarantine_and_rollback_detection():
    fixture = ContinuationAuthority()
    record = _admit(fixture)
    assert fixture.guard.cancel(record, WORK_ID).reason is ContinuationReason.CANCELLED
    restarted = fixture.restart(ledger=fixture.ledger)
    outcome = restarted.guard.check(record, WORK_ID, ContinuationBoundary.RESUME)
    assert outcome.reason is ContinuationReason.TERMINAL

    # A durable generation store rejects a provider rolling back a revocation.
    fixture = ContinuationAuthority()
    record = _admit(fixture)
    fixture.provider.revoke(fixture.scope)
    assert (
        fixture.guard.check(record, WORK_ID, ContinuationBoundary.RESUME).reason
        is ContinuationReason.AUTHORITY_REVOKED
    )
    fixture.provider.records[(fixture.scope.kind, fixture.scope.digest)] = (0, False)
    restarted = fixture.restart(store=fixture.store)
    outcome = restarted.guard.check(record, WORK_ID, ContinuationBoundary.RESUME)
    assert outcome.reason is ContinuationReason.AUTHORITY_ROLLBACK
    assert outcome.disposition is ContinuationDisposition.QUARANTINED


# -- Queued execution ---------------------------------------------------------


def test_queued_work_completes_and_releases_outputs_after_final_check():
    fixture = ContinuationAuthority()
    record = _admit(fixture)
    sink = SyntheticSink()

    def worker(ctx):
        ctx.checkpoint("documents")
        ctx.emit({"document": 0})
        effect = pending_effect(0)
        ctx.commit_effect(effect, lambda: sink.commit(effect.idempotency_key))
        ctx.checkpoint("effects")
        ctx.emit({"document": 1})

    result = run_continuation(fixture.guard, record, WORK_ID, worker)
    outcome = result.outcome
    assert outcome.disposition is ContinuationDisposition.COMPLETED
    assert outcome.reason is ContinuationReason.COMPLETED
    assert outcome.boundary is ContinuationBoundary.PROTECTED_OUTPUT
    assert result.outputs == ({"document": 0}, {"document": 1})
    assert outcome.delivered_output_count == 2
    assert outcome.committed_effect_count == 1
    assert outcome.reconciliation is ReconciliationState.COMMITTED_EFFECTS_RETAINED
    assert [e.state for e in result.effects] == [EffectState.COMMITTED]
    assert len(sink.commits) == 1
    # Completion closes the epoch: the same admission cannot be replayed.
    again = run_continuation(fixture.guard, record, WORK_ID, worker, resumed=True)
    assert again.outcome.reason is ContinuationReason.TERMINAL
    assert again.outputs == ()
    assert len(sink.commits) == 1


def test_revocation_before_queued_execution_never_starts_the_worker():
    fixture = ContinuationAuthority()
    record = _admit(fixture)
    fixture.provider.revoke(fixture.scope)
    started = []
    result = run_continuation(
        fixture.guard, record, WORK_ID, lambda ctx: started.append(True)
    )
    assert started == []
    assert result.outcome.boundary is ContinuationBoundary.EXECUTION_START
    assert result.outcome.reason is ContinuationReason.AUTHORITY_REVOKED
    assert result.outcome.reconciliation is ReconciliationState.NO_EFFECTS


def test_revocation_mid_run_keeps_committed_effects_and_discards_buffer():
    fixture = ContinuationAuthority()
    record = _admit(fixture)
    sink = SyntheticSink()
    reached = []

    def worker(ctx):
        ctx.emit(SENTINEL)
        first = pending_effect(0)
        ctx.commit_effect(first, lambda: sink.commit(first.idempotency_key))
        fixture.provider.revoke(fixture.consent)
        ctx.checkpoint("effects")
        reached.append(True)
        second = pending_effect(1)
        ctx.commit_effect(second, lambda: sink.commit(second.idempotency_key))

    result = run_continuation(fixture.guard, record, WORK_ID, worker)
    outcome = result.outcome
    assert reached == []
    assert len(sink.commits) == 1
    assert result.outputs == ()
    assert outcome.boundary is ContinuationBoundary.CHECKPOINT
    assert outcome.checkpoint == "effects"
    assert outcome.disposition is ContinuationDisposition.STOPPED
    assert outcome.delivered_output_count == 0
    assert outcome.withheld_output_count == 1
    assert outcome.committed_effect_count == 1
    assert outcome.pending_effect_count == 0
    assert outcome.reconciliation is ReconciliationState.COMMITTED_EFFECTS_RETAINED
    assert outcome.to_dict()["compensation"] == "not_attempted"
    assert outcome.committed_effect_digest.startswith("sha256:")
    assert [e.state for e in result.effects] == [EffectState.COMMITTED]
    _assert_no_sentinel(result, outcome)


def test_effect_dispatch_is_rechecked_immediately_before_each_effect():
    fixture = ContinuationAuthority()
    record = _admit(fixture)
    sink = SyntheticSink()

    def worker(ctx):
        effect = pending_effect(0)
        fixture.provider.revoke(fixture.scope)
        ctx.commit_effect(effect, lambda: sink.commit(effect.idempotency_key))

    result = run_continuation(fixture.guard, record, WORK_ID, worker)
    assert sink.commits == []
    assert result.outcome.boundary is ContinuationBoundary.EFFECT_DISPATCH
    assert result.effects == ()
    assert result.outcome.reconciliation is ReconciliationState.NO_EFFECTS


def test_revocation_after_worker_returns_withholds_all_buffered_output():
    fixture = ContinuationAuthority()
    record = _admit(fixture)

    def worker(ctx):
        ctx.emit(SENTINEL)
        ctx.emit(SENTINEL)
        fixture.clock.now = 100

    result = run_continuation(fixture.guard, record, WORK_ID, worker)
    assert result.outputs == ()
    assert result.outcome.boundary is ContinuationBoundary.PROTECTED_OUTPUT
    assert result.outcome.reason is ContinuationReason.AUTHORITY_EXPIRED
    assert result.outcome.withheld_output_count == 2
    _assert_no_sentinel(result)


def test_worker_and_dispatch_failures_quarantine_without_leaking():
    fixture = ContinuationAuthority()
    record = _admit(fixture)

    def failing(ctx):
        ctx.emit(SENTINEL)
        raise RuntimeError(SENTINEL)

    result = run_continuation(fixture.guard, record, WORK_ID, failing)
    assert result.outcome.reason is ContinuationReason.WORKER_FAILED
    assert result.outcome.disposition is ContinuationDisposition.QUARANTINED
    assert result.outputs == ()
    _assert_no_sentinel(result)

    fixture = ContinuationAuthority()
    record = _admit(fixture)
    sink = SyntheticSink()
    sink.fail = True

    def ambiguous(ctx):
        effect = pending_effect(0)
        ctx.commit_effect(effect, lambda: sink.commit(effect.idempotency_key))

    result = run_continuation(fixture.guard, record, WORK_ID, ambiguous)
    outcome = result.outcome
    assert outcome.reason is ContinuationReason.WORKER_FAILED
    assert outcome.pending_effect_count == 1
    assert outcome.committed_effect_count == 0
    assert outcome.reconciliation is ReconciliationState.PENDING_EFFECTS_UNCONFIRMED
    assert [e.state for e in result.effects] == [EffectState.PENDING]
    _assert_no_sentinel(result)


def test_invalid_commit_evidence_is_never_recorded_as_committed():
    fixture = ContinuationAuthority()
    record = _admit(fixture)

    def worker(ctx):
        ctx.commit_effect(pending_effect(0), lambda: SENTINEL)

    result = run_continuation(fixture.guard, record, WORK_ID, worker)
    assert (
        result.outcome.reconciliation is ReconciliationState.PENDING_EFFECTS_UNCONFIRMED
    )
    assert result.outcome.committed_effect_count == 0
    _assert_no_sentinel(result)


def test_cancellation_signal_stops_and_never_claims_to_undo_effects():
    fixture = ContinuationAuthority()
    record = _admit(fixture)
    sink = SyntheticSink()
    signal = CancellationSignal()

    def worker(ctx):
        effect = pending_effect(0)
        ctx.commit_effect(effect, lambda: sink.commit(effect.idempotency_key))
        ctx.emit({"document": 0})
        signal.cancel()
        ctx.checkpoint("effects")

    result = run_continuation(
        fixture.guard, record, WORK_ID, worker, cancellation=signal
    )
    outcome = result.outcome
    assert outcome.reason is ContinuationReason.CANCELLED
    assert outcome.disposition is ContinuationDisposition.STOPPED
    assert outcome.committed_effect_count == 1
    assert outcome.withheld_output_count == 1
    assert outcome.to_dict()["compensation"] == "not_attempted"
    assert len(sink.commits) == 1
    assert result.outputs == ()


def test_interruption_from_a_different_epoch_is_not_trusted():
    first = ContinuationAuthority()
    other_record = _admit(first)
    other = first.guard.cancel(other_record, WORK_ID)
    fixture = ContinuationAuthority()
    record = _admit(fixture)

    def worker(ctx):
        raise ContinuationInterrupted(other)

    result = run_continuation(fixture.guard, record, WORK_ID, worker)
    assert result.outcome.epoch_digest == record.epoch_digest
    assert result.outcome.reason is ContinuationReason.WORKER_FAILED


# -- Streams ------------------------------------------------------------------


def test_revoked_stream_releases_no_later_protected_chunk():
    fixture = ContinuationAuthority()
    record = _admit(fixture, checkpoints=())
    source = SyntheticSource(["chunk-0", "chunk-1", SENTINEL, SENTINEL])
    stream = GuardedStream(fixture.guard, record, WORK_ID, source)
    received = []
    with pytest.raises(ContinuationInterrupted) as error:
        for chunk in stream:
            received.append(chunk)
            if len(received) == 2:
                fixture.provider.revoke(fixture.consent)
    assert received == ["chunk-0", "chunk-1"]
    assert source.closed
    assert source.read == 3
    outcome = stream.outcome
    assert outcome is error.value.outcome
    assert outcome.boundary is ContinuationBoundary.PROTECTED_OUTPUT
    assert outcome.reason is ContinuationReason.AUTHORITY_REVOKED
    assert outcome.delivered_output_count == 2
    assert outcome.withheld_output_count == 1
    rendered = "".join(traceback.format_exception(error.value))
    assert SENTINEL not in rendered
    _assert_no_sentinel(outcome, error.value)


def test_stream_terminal_event_replaces_the_withheld_chunk():
    fixture = ContinuationAuthority()
    record = _admit(fixture, checkpoints=())
    source = SyntheticSource(["chunk-0", SENTINEL])
    stream = GuardedStream(
        fixture.guard,
        record,
        WORK_ID,
        source,
        terminal=continuation_error_event,
    )
    received = []
    for item in stream:
        received.append(item)
        fixture.clock.now = 100
    assert received[0] == "chunk-0"
    event = received[1]
    assert event["type"] == "error"
    assert event["error"]["code"] == ContinuationReason.AUTHORITY_EXPIRED.value
    assert event["error"]["details"]["delivered_output_count"] == 1
    assert len(received) == 2
    assert SENTINEL not in json.dumps(event)


def test_stream_start_is_checked_before_reading_the_source():
    fixture = ContinuationAuthority()
    record = _admit(fixture, checkpoints=())
    fixture.provider.revoke(fixture.scope)
    source = SyntheticSource([SENTINEL])
    stream = GuardedStream(fixture.guard, record, WORK_ID, source)
    with pytest.raises(ContinuationInterrupted) as error:
        list(stream)
    assert error.value.outcome.boundary is ContinuationBoundary.EXECUTION_START
    assert source.read == 0
    assert source.closed


def test_stream_completion_source_failure_and_consumer_disconnect():
    fixture = ContinuationAuthority()
    record = _admit(fixture, checkpoints=())
    stream = GuardedStream(fixture.guard, record, WORK_ID, SyntheticSource(["a"]))
    assert list(stream) == ["a"]
    assert stream.outcome.disposition is ContinuationDisposition.COMPLETED
    with pytest.raises(ContinuationError):
        list(stream)
    replay = GuardedStream(fixture.guard, record, WORK_ID, SyntheticSource(["a"]))
    with pytest.raises(ContinuationInterrupted) as error:
        list(replay)
    assert error.value.outcome.reason is ContinuationReason.TERMINAL

    fixture = ContinuationAuthority()
    record = _admit(fixture, checkpoints=())
    source = SyntheticSource(["a", "b"], fail_at=1)
    stream = GuardedStream(fixture.guard, record, WORK_ID, source)
    with pytest.raises(ContinuationInterrupted) as error:
        list(stream)
    assert error.value.outcome.reason is ContinuationReason.SOURCE_FAILED
    assert error.value.outcome.disposition is ContinuationDisposition.QUARANTINED
    assert error.value.__cause__ is None and error.value.__context__ is None
    assert source.closed

    fixture = ContinuationAuthority()
    record = _admit(fixture, checkpoints=())
    source = SyntheticSource(["a", "b", "c"])
    stream = GuardedStream(fixture.guard, record, WORK_ID, source)
    iterator = iter(stream)
    assert next(iterator) == "a"
    iterator.close()
    assert source.closed
    assert stream.outcome.reason is ContinuationReason.CANCELLED
    assert stream.outcome.delivered_output_count == 1


def test_async_stream_stops_on_revocation_and_on_task_cancellation():
    async def revoked():
        fixture = ContinuationAuthority()
        record = _admit(fixture, checkpoints=())
        source = SyntheticAsyncSource(["chunk-0", SENTINEL])
        stream = GuardedStream(
            fixture.guard, record, WORK_ID, source, terminal=continuation_error_event
        )
        received = []
        async for item in stream:
            received.append(item)
            fixture.provider.revoke(fixture.consent)
        return received, source, stream.outcome

    received, source, outcome = asyncio.run(revoked())
    assert received[0] == "chunk-0"
    assert received[1]["error"]["code"] == ContinuationReason.AUTHORITY_REVOKED.value
    assert source.closed
    assert outcome.withheld_output_count == 1

    async def cancelled():
        fixture = ContinuationAuthority()
        record = _admit(fixture, checkpoints=())
        source = SyntheticAsyncSource(["a", "b", "c"])
        stream = GuardedStream(fixture.guard, record, WORK_ID, source)
        signal = CancellationSignal()
        stream_with_signal = GuardedStream(
            fixture.guard,
            fixture.guard.admit(WORK_ID, fixture.contracts),
            WORK_ID,
            SyntheticAsyncSource(["x", "y"]),
            cancellation=signal,
        )
        seen = []
        iterator = stream.__aiter__()
        seen.append(await iterator.__anext__())
        await iterator.aclose()
        signalled = []
        with pytest.raises(ContinuationInterrupted) as error:
            async for item in stream_with_signal:
                signalled.append(item)
                signal.cancel()
        return seen, source, stream.outcome, signalled, error.value.outcome

    seen, source, outcome, signalled, signal_outcome = asyncio.run(cancelled())
    assert seen == ["a"]
    assert source.closed
    assert outcome.reason is ContinuationReason.CANCELLED
    assert signalled == ["x"]
    assert signal_outcome.reason is ContinuationReason.CANCELLED
    assert signal_outcome.withheld_output_count == 0


def test_guard_rejects_records_from_another_process_ledger_consistently():
    fixture = ContinuationAuthority()
    record = _admit(fixture)
    ledger = InMemoryContinuationLedger()
    assert ledger.is_closed(record.epoch_digest) is False
    assert ledger.close(record.epoch_digest) is True
    assert ledger.is_closed(record.epoch_digest) is True
    assert ledger.close(record.epoch_digest) is True
    guard = ContinuationGuard(fixture.runtime, ledger=ledger)
    outcome = guard.check(record, WORK_ID, ContinuationBoundary.RESUME)
    assert outcome.reason is ContinuationReason.TERMINAL


def test_completion_withholds_outputs_when_the_epoch_cannot_be_closed():
    class UnclosableLedger:
        def is_closed(self, epoch_digest):
            return False

        def close(self, epoch_digest):
            return False

    fixture = ContinuationAuthority()
    record = _admit(fixture)
    guard = ContinuationGuard(fixture.runtime, ledger=UnclosableLedger())
    result = run_continuation(guard, record, WORK_ID, lambda ctx: ctx.emit(SENTINEL))
    assert result.outputs == ()
    assert result.outcome.reason is ContinuationReason.LEDGER_UNAVAILABLE
    assert result.outcome.delivered_output_count == 0
    assert result.outcome.withheld_output_count == 1
    _assert_no_sentinel(result)


def test_worker_swallowing_an_interruption_cannot_release_later_output():
    fixture = ContinuationAuthority()
    record = _admit(fixture)

    def worker(ctx):
        fixture.provider.revoke(fixture.consent)
        try:
            ctx.checkpoint("documents")
        except ContinuationInterrupted:
            pass
        ctx.emit(SENTINEL)
        ctx.emit(SENTINEL)

    result = run_continuation(fixture.guard, record, WORK_ID, worker)
    assert result.outputs == ()
    assert result.outcome.reason is ContinuationReason.AUTHORITY_REVOKED
    assert result.outcome.checkpoint == "documents"
    assert result.outcome.withheld_output_count == 2
    _assert_no_sentinel(result)
