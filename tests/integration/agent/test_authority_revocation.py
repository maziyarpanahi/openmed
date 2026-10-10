"""Offline preview/dispatch and durable restart tests with synthetic adapters."""

import json
import sqlite3
from pathlib import Path

import pytest

from openmed.agent.correlation import ActionId
from openmed.agent.identifiers import ToolId, WorkflowId
from openmed.agent.permissions.revocation import (
    AuthorityBinding,
    AuthorityBoundary,
    AuthorityDeniedError,
    AuthorityRuntime,
)
from openmed.agent.permissions.runtime_authority import (
    dispatch_with_revocable_grant,
    recover_with_revocable_authority,
)
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
)
from tests.fixtures.agent.authority_status import ACTION_DIGEST, SyntheticAuthority

pytestmark = pytest.mark.integration


class SyntheticDurableGenerations:
    """Test-only SQLite implementation of the injected atomic store contract."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS generations "
                "(kind TEXT, digest TEXT, generation INTEGER, revoked INTEGER, "
                "PRIMARY KEY (kind, digest))"
            )

    def observe(self, status) -> bool:
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            key = (status.kind.value, status.digest)
            previous = connection.execute(
                "SELECT generation, revoked FROM generations WHERE kind=? AND digest=?",
                key,
            ).fetchone()
            if previous is not None:
                generation, revoked = previous
                if (
                    status.generation < generation
                    or (
                        status.generation == generation
                        and status.revoked != bool(revoked)
                    )
                    or (revoked and not status.revoked)
                ):
                    return False
            connection.execute(
                "INSERT OR REPLACE INTO generations VALUES (?, ?, ?, ?)",
                (*key, status.generation, int(status.revoked)),
            )
            return True


def _checkpoint(fixture):
    effect = EffectRecord.create(
        ordinal=0,
        run_id=fixture.ticket.run_id,
        action_id=ActionId("act_" + "1" * 32),
        tool_id=ToolId(fixture.constraint.tool),
        kind=EffectKind.LOCAL_TOOL,
        operation_digest=ACTION_DIGEST,
        approval_required=False,
        compensation_limit=CompensationLimit.NONE,
    )
    checkpoint = RecoveryCheckpoint.create(
        workflow_id=WorkflowId("workflow:org.example/synthetic@1.0.0"),
        run_id=fixture.ticket.run_id,
        sequence=0,
        phase=RecoveryPhase.PLANNED,
        plan_digest=ACTION_DIGEST,
        effects=(effect,),
    )
    observation = EffectObservation(
        action_id=effect.action_id,
        operation_digest=effect.operation_digest,
        idempotency_key=effect.idempotency_key,
        state=ObservationState.ABSENT,
    )
    return checkpoint, observation


@pytest.mark.parametrize("revoked_index", [0, 1, 2, 3, 4])
def test_restart_cannot_restore_authority_revoked_since_preview(
    tmp_path, revoked_index
):
    fixture = SyntheticAuthority()
    store_path = tmp_path / "independent-high-water.sqlite"
    runtime = AuthorityRuntime(
        fixture.provider, SyntheticDurableGenerations(store_path), clock=fixture.clock
    )
    preview = runtime.bind(fixture.contracts)
    # Run recovery storage and high-water storage have separate lifecycles.
    saved_preview = json.loads(json.dumps(preview.to_dict()))
    checkpoint, observation = _checkpoint(fixture)
    journal = CheckpointJournal(tmp_path / "run-journal")
    journal.append(checkpoint)
    contract = fixture.contracts[revoked_index]
    fixture.provider.revoke(contract)
    with pytest.raises(AuthorityDeniedError, match="authority_revoked"):
        runtime.check(preview, AuthorityBoundary.EFFECT_DISPATCH)

    restarted = AuthorityRuntime(
        fixture.provider, SyntheticDurableGenerations(store_path), clock=fixture.clock
    )
    restored = AuthorityBinding.from_dict(saved_preview)
    lineage = CheckpointJournal(tmp_path / "run-journal").load()

    def observations():
        pytest.fail("revoked recovery performed a sensitive sink read")
        yield observation

    with pytest.raises(AuthorityDeniedError, match="authority_revoked"):
        recover_with_revocable_authority(
            lineage, observations(), runtime=restarted, binding=restored, now=10
        )
    # Roll back the provider's active state, leaving independently retained
    # high water intact. A fresh process still rejects the old generation.
    fixture.provider.records[(contract.kind, contract.digest)] = (0, False)
    with pytest.raises(AuthorityDeniedError, match="authority_status_rollback"):
        recover_with_revocable_authority(
            lineage, (observation,), runtime=restarted, binding=restored, now=10
        )
    assert journal.load() == (checkpoint,)
    assert restored == preview


def test_new_active_generation_cannot_refresh_restored_preview(tmp_path):
    fixture = SyntheticAuthority()
    preview = fixture.runtime.bind(fixture.contracts)
    contract = fixture.contracts[0]
    fixture.provider.records[(contract.kind, contract.digest)] = (1, False)
    checkpoint, observation = _checkpoint(fixture)
    restarted = AuthorityRuntime(
        fixture.provider,
        SyntheticDurableGenerations(tmp_path / "high-water.sqlite"),
        clock=fixture.clock,
    )
    with pytest.raises(AuthorityDeniedError, match="authority_generation_changed"):
        recover_with_revocable_authority(
            (checkpoint,),
            (observation,),
            runtime=restarted,
            binding=AuthorityBinding.from_dict(preview.to_dict()),
            now=10,
        )


def test_valid_resume_plan_does_not_bypass_next_effect_status_check():
    fixture = SyntheticAuthority()
    binding = fixture.runtime.bind(fixture.contracts)
    checkpoint, observation = _checkpoint(fixture)
    decision = recover_with_revocable_authority(
        (checkpoint,), (observation,), runtime=fixture.runtime, binding=binding, now=10
    )
    assert decision.disposition is RecoveryDisposition.RESUME
    fixture.provider.revoke(fixture.contracts[0])
    with pytest.raises(AuthorityDeniedError, match="authority_revoked"):
        dispatch_with_revocable_grant(
            fixture.grant,
            fixture.grant_request,
            fixture.grant_verifier,
            lambda: pytest.fail("revoked resumed effect executed"),
            runtime=fixture.runtime,
            binding=binding,
        )


def test_live_expiry_and_outage_fail_closed_before_recovery_observations():
    fixture = SyntheticAuthority()
    binding = fixture.runtime.bind(fixture.contracts)
    checkpoint, observation = _checkpoint(fixture)
    fixture.clock.now = 100
    with pytest.raises(AuthorityDeniedError, match="authority_expired"):
        recover_with_revocable_authority(
            (checkpoint,),
            (observation,),
            runtime=fixture.runtime,
            binding=binding,
            now=10,  # Historical reconciliation time cannot override runtime time.
        )
    fixture.provider.unavailable = True
    with pytest.raises(AuthorityDeniedError, match="authority_status_unavailable"):
        recover_with_revocable_authority(
            (checkpoint,),
            (observation,),
            runtime=fixture.runtime,
            binding=binding,
            now=10,
        )


@pytest.mark.parametrize("revoke_phase", [None, "approval_recorded", "dispatching"])
def test_revocation_composes_on_actual_guarded_dispatch_boundary(revoke_phase):
    from openmed.agent.permissions.access_tickets import AccessTicketVerifier
    from openmed.agent.permissions.revocation import InMemoryAuthorityGenerationStore
    from openmed.agent.permissions.runtime_authority import (
        grant_authority,
        read_with_revocable_ticket,
        ticket_authority,
    )
    from tests.fixtures.agent.authority_status import SyntheticStatusProvider
    from tests.fixtures.agent.guarded_dispatch import PRIVATE, DispatchHarness

    h = DispatchHarness()
    provider = SyntheticStatusProvider(lambda: h.now)
    contracts = (
        grant_authority(h.authority.grant),
        ticket_authority(h.authority.ticket),
    )
    provider.register(contracts)
    runtime = AuthorityRuntime(
        provider, InMemoryAuthorityGenerationStore(), clock=lambda: h.now
    )
    binding = runtime.bind(contracts)
    invoke = h.tools.invoke

    def governed(spec, arguments, *, effect):
        return dispatch_with_revocable_grant(
            h.authority.grant,
            h.authority.grant_request,
            h.grants,
            lambda: read_with_revocable_ticket(
                h.authority.ticket,
                h.authority.ticket_request,
                AccessTicketVerifier(clock=lambda: h.now),
                lambda: invoke(spec, arguments, effect=effect),
                runtime=runtime,
                binding=binding,
            ),
            runtime=runtime,
            binding=binding,
            now=h.now,
        )

    h.tools.invoke = governed
    append = h.effects.append

    def revoke_after_storage(checkpoint):
        append(checkpoint)
        if checkpoint.phase.value == revoke_phase:
            provider.revoke(contracts[1])

    h.effects.append = revoke_after_storage
    result = h.adapter().dispatch(h.arguments)
    assert h.tools.calls == (1 if revoke_phase is None else 0)
    assert result.outcome.outcome_class.value == (
        "success" if revoke_phase is None else "review_required"
    )
    assert PRIVATE not in json.dumps(result.to_dict())
