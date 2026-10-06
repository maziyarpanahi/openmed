"""Synthetic offline safety controls for confirmed journal retirement."""

from __future__ import annotations

import json
import os
import stat
import traceback
from dataclasses import replace
from pathlib import Path

import pytest

import openmed.agent.workflows.recovery as recovery
from openmed.agent.correlation import ActionId, RunId
from openmed.agent.identifiers import ToolId, WorkflowId
from openmed.agent.outcomes import OutcomeClass, WorkflowOutcome
from openmed.agent.run_summary import RunEvent, RunSummary
from openmed.agent.workflows import (
    CheckpointJournal,
    CompensationLimit,
    EffectKind,
    EffectObservation,
    EffectRecord,
    JournalRetentionPolicy,
    JournalRetirementPlan,
    ObservationState,
    RecoveryCheckpoint,
    RecoveryDecision,
    RecoveryDisposition,
    RecoveryError,
    RecoveryPhase,
    RecoveryReason,
    RetiredJournalRecord,
    advance_checkpoint,
    recover_workflow,
)


def terminal_journal(directory: Path) -> tuple[CheckpointJournal, RecoveryCheckpoint]:
    journal = CheckpointJournal(directory)
    effect = EffectRecord.create(
        ordinal=0,
        run_id=RunId("run_" + "1" * 32),
        action_id=ActionId("act_" + "2" * 32),
        tool_id=ToolId("tool:org.openmed/synthetic@1.0.0"),
        kind=EffectKind.LOCAL_TOOL,
        operation_digest="sha256:" + "a" * 64,
        approval_required=False,
        compensation_limit=CompensationLimit.NONE,
    )
    first = RecoveryCheckpoint.create(
        workflow_id=WorkflowId("workflow:org.openmed/synthetic@1.0.0"),
        run_id=RunId("run_" + "1" * 32),
        sequence=0,
        phase=RecoveryPhase.DISPATCHING,
        plan_digest="sha256:" + "b" * 64,
        effects=(effect,),
    )
    journal.append(first)
    decision = recover_workflow(
        (first,),
        (
            EffectObservation(
                effect.action_id,
                effect.operation_digest,
                effect.idempotency_key,
                ObservationState.COMMITTED,
                "sha256:" + "c" * 64,
            ),
        ),
        now=10,
    )
    final = advance_checkpoint(first, decision)
    journal.append(final)
    os.utime(directory / "checkpoint-00000000000000000001.json", ns=(10**10, 10**10))
    return journal, final


def exported_summary(final: RecoveryCheckpoint) -> RunSummary:
    decision = recover_workflow((final,), (), now=10) if final.sequence == 0 else None
    # The terminal decision is independent of clock and predecessor records.
    if decision is None:
        decision = RecoveryDecision.create(
            disposition=RecoveryDisposition.COMPLETE,
            reason=RecoveryReason.ALREADY_COMPLETE,
            source_checkpoint_digest=final.checkpoint_digest,
        )
    anchors = [final.checkpoint_digest, decision.evidence_digest]
    if final.recovery_evidence_digest is not None:
        anchors.append(final.recovery_evidence_digest)
    return RunSummary.from_events(
        (
            RunEvent(
                "synthetic_workflow",
                WorkflowOutcome(OutcomeClass.SUCCESS, "completed"),
                tool_call_count=1,
                artifact_digests=tuple(anchors),
            ),
        )
    )


def test_confirmed_retirement_preserves_summary_and_cannot_resume(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "journal"
    journal, final = terminal_journal(directory)
    summary = RunSummary.from_json(exported_summary(final).to_json())
    before = journal.load()
    plan = journal.plan_retirement(JournalRetentionPolicy(50), now=100)
    assert plan is not None
    assert JournalRetirementPlan.from_json(plan.to_json()) == plan
    assert plan.dry_run
    assert journal.load() == before
    assert plan == journal.plan_retirement(JournalRetentionPolicy(50), now=100)
    assert plan.record.checkpoint_count == 2
    assert plan.record.effect_count == plan.record.committed_effect_count == 1

    record = journal.retire(plan, confirmation=plan.confirmation_token, now=100)
    assert record.verifies_summary(summary)
    assert RetiredJournalRecord.from_json(record.to_json()) == record
    assert CheckpointJournal(directory).load() == record
    assert not list(directory.glob("checkpoint-*.json"))
    decision = recover_workflow(journal.load(), (), now=100)
    assert decision.disposition is RecoveryDisposition.RETIRED
    assert decision.reason is RecoveryReason.JOURNAL_RETIRED
    assert not decision.retry_idempotency_keys
    assert not decision.retry_effect_ids
    assert not decision.compensation_effect_ids
    assert json.loads(decision.to_json())["disposition"] == "retired"
    with pytest.raises(RecoveryError, match="journal_retired"):
        journal.append(final)
    with pytest.raises(RecoveryError, match="journal_retired"):
        advance_checkpoint(final, decision)
    assert journal.plan_retirement(JournalRetentionPolicy(50), now=1000) is None
    assert journal.retire(plan, confirmation=plan.confirmation_token, now=101) == record
    if os.name != "nt":
        assert stat.S_IMODE((directory / "retired.json").stat().st_mode) == 0o600


@pytest.mark.parametrize("now", [0, 9, 10, 59, 60])
def test_minimum_age_is_exclusive_and_future_times_are_untouched(
    tmp_path: Path, now: int
) -> None:
    journal, _ = terminal_journal(tmp_path / "journal")
    before = journal.load()
    assert journal.plan_retirement(JournalRetentionPolicy(50), now=now) is None
    assert journal.load() == before


@pytest.mark.parametrize("age", [-1, True, 1.5, "50", 2**63])
def test_invalid_policy_is_value_free(age: object) -> None:
    with pytest.raises(RecoveryError, match="invalid_non_negative_integer"):
        JournalRetentionPolicy(age)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "phase",
    [
        RecoveryPhase.PLANNED,
        RecoveryPhase.DISPATCHING,
        RecoveryPhase.RECONCILING,
        RecoveryPhase.ABORTED,
    ],
)
def test_inflight_and_aborted_pending_effects_are_untouched(
    tmp_path: Path, phase: RecoveryPhase
) -> None:
    source, _ = terminal_journal(tmp_path / "source")
    first = source.load()[0]
    checkpoint = RecoveryCheckpoint.create(
        workflow_id=first.workflow_id,
        run_id=first.run_id,
        sequence=0,
        phase=phase,
        plan_digest=first.plan_digest,
        effects=first.effects,
    )
    directory = tmp_path / "journal"
    journal = CheckpointJournal(directory)
    journal.append(checkpoint)
    os.utime(directory / "checkpoint-00000000000000000000.json", (10, 10))
    assert journal.plan_retirement(JournalRetentionPolicy(0), now=100) is None
    assert journal.load() == (checkpoint,)


def test_aborted_proven_committed_runs_can_retire(
    tmp_path: Path,
) -> None:
    source, final = terminal_journal(tmp_path / "source")
    for index, effects in enumerate((final.effects,)):
        directory = tmp_path / str(index)
        journal = CheckpointJournal(directory)
        aborted = RecoveryCheckpoint.create(
            workflow_id=final.workflow_id,
            run_id=final.run_id,
            sequence=0,
            phase=RecoveryPhase.ABORTED,
            plan_digest=final.plan_digest,
            effects=effects,
        )
        journal.append(aborted)
        os.utime(directory / "checkpoint-00000000000000000000.json", (10, 10))
        plan = journal.plan_retirement(JournalRetentionPolicy(0), now=100)
        assert plan is not None
        record = journal.retire(plan, confirmation=plan.confirmation_token, now=100)
        assert record.phase is RecoveryPhase.ABORTED
        assert (
            recover_workflow(record, (), now=100).disposition
            is RecoveryDisposition.RETIRED
        )


@pytest.mark.parametrize(
    "confirmation", [None, "", "confirm:wrong", "患者 synthetic secret"]
)
def test_confirmation_is_required_before_publication_or_deletion(
    tmp_path: Path, confirmation: object
) -> None:
    directory = tmp_path / "journal"
    journal, _ = terminal_journal(directory)
    plan = journal.plan_retirement(JournalRetentionPolicy(0), now=100)
    before = journal.load()
    with pytest.raises(RecoveryError, match="confirmation_required"):
        journal.retire(plan, confirmation=confirmation, now=100)  # type: ignore[arg-type]
    assert journal.load() == before
    assert not (directory / "retired.json").exists()


def test_plan_is_bound_to_journal_and_mtime_and_clock(tmp_path: Path) -> None:
    directory = tmp_path / "journal"
    journal, _ = terminal_journal(directory)
    plan = journal.plan_retirement(JournalRetentionPolicy(0), now=100)
    assert plan is not None
    other, _ = terminal_journal(tmp_path / "other")
    with pytest.raises(RecoveryError, match="retirement_plan_stale"):
        other.retire(plan, confirmation=plan.confirmation_token, now=100)
    with pytest.raises(RecoveryError, match="clock_moved_backwards"):
        journal.retire(plan, confirmation=plan.confirmation_token, now=99)
    os.utime(directory / "checkpoint-00000000000000000001.json", (20, 20))
    with pytest.raises(RecoveryError, match="retirement_plan_stale"):
        journal.retire(plan, confirmation=plan.confirmation_token, now=100)
    with pytest.raises(RecoveryError, match="retirement_plan_digest_mismatch"):
        replace(plan, planned_at=101)
    assert len(journal.load()) == 2


@pytest.mark.parametrize("problem", ["gap", "tamper", "unknown", "symlink"])
def test_ambiguous_journal_is_never_modified(tmp_path: Path, problem: str) -> None:
    directory = tmp_path / "journal"
    journal, _ = terminal_journal(directory)
    first = directory / "checkpoint-00000000000000000000.json"
    if problem == "gap":
        first.unlink()
    elif problem == "tamper":
        first.write_text('{"checkpoint_digest":"synthetic protected payload"}')
    elif problem == "unknown":
        (directory / "private-input.txt").write_text("Synthetic patient 王小明")
    else:
        target = tmp_path / "private-input.txt"
        target.write_bytes(first.read_bytes())
        first.unlink()
        first.symlink_to(target)
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    with pytest.raises(RecoveryError):
        journal.plan_retirement(JournalRetentionPolicy(0), now=100)
    after = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    assert before == after


@pytest.mark.parametrize(
    "field,value",
    [
        ("effect_count", 2),
        ("checkpoint_count", 0),
        ("phase", "dispatching"),
        ("record_digest", "sha256:" + "f" * 64),
        ("schema_version", "unknown"),
        ("patient", "Synthetic Person private payload"),
    ],
)
def test_record_tampering_and_payloads_fail_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    journal, _ = terminal_journal(tmp_path / "journal")
    plan = journal.plan_retirement(JournalRetentionPolicy(0), now=100)
    assert plan is not None
    payload = plan.record.to_dict()
    payload[field] = value
    with pytest.raises(RecoveryError) as caught:
        RetiredJournalRecord.from_json(json.dumps(payload))
    assert "Synthetic Person" not in "".join(traceback.format_exception(caught.value))


def test_summary_missing_anchors_fails_verification(tmp_path: Path) -> None:
    journal, final = terminal_journal(tmp_path / "journal")
    plan = journal.plan_retirement(JournalRetentionPolicy(0), now=100)
    summary = exported_summary(final)
    assert plan is not None
    for anchor in summary.artifact_digests:
        changed = replace(
            summary,
            artifact_digests=tuple(d for d in summary.artifact_digests if d != anchor),
        )
        assert not plan.record.verifies_summary(changed)
    with pytest.raises(RecoveryError, match="invalid_summary"):
        plan.record.verifies_summary({"patient": "synthetic"})  # type: ignore[arg-type]


def test_private_paths_never_escape_reports_or_io_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = "Synthetic-Person-患者-bearer-secret"
    journal, _ = terminal_journal(tmp_path / sentinel)
    plan = journal.plan_retirement(JournalRetentionPolicy(0), now=100)
    assert plan is not None
    assert sentinel not in plan.to_json() + repr(plan) + plan.record.to_json()

    def fail(*args: object) -> None:
        raise OSError(sentinel)

    monkeypatch.setattr(recovery.os, "replace", fail)
    with pytest.raises(RecoveryError, match="retirement_io_failed") as caught:
        journal.retire(plan, confirmation=plan.confirmation_token, now=100)
    assert sentinel not in "".join(traceback.format_exception(caught.value))
    assert len(journal.load()) == 2


def test_symlink_terminal_record_fails_closed(tmp_path: Path) -> None:
    directory = tmp_path / "journal"
    journal, _ = terminal_journal(directory)
    plan = journal.plan_retirement(JournalRetentionPolicy(0), now=100)
    assert plan is not None
    journal.retire(plan, confirmation=plan.confirmation_token, now=100)
    target = tmp_path / "record.json"
    (directory / "retired.json").rename(target)
    (directory / "retired.json").symlink_to(target)
    with pytest.raises(RecoveryError, match="unsafe_retired_file"):
        journal.load()


def test_retired_decision_rejects_retry_fields(tmp_path: Path) -> None:
    journal, final = terminal_journal(tmp_path / "journal")
    with pytest.raises(RecoveryError, match="invalid_retired_decision"):
        RecoveryDecision.create(
            disposition=RecoveryDisposition.RETIRED,
            reason=RecoveryReason.JOURNAL_RETIRED,
            source_checkpoint_digest=final.checkpoint_digest,
            committed_effects=(
                (final.effects[0].action_id.serialize(), "sha256:" + "d" * 64),
            ),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("dry_run", False),
        ("schema_version", "unknown"),
        ("planned_at", 101),
        ("checkpoint_digests", []),
        ("path", "Synthetic protected path"),
    ],
)
def test_restored_plan_rejects_changed_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    journal, _ = terminal_journal(tmp_path / "journal")
    plan = journal.plan_retirement(JournalRetentionPolicy(0), now=100)
    assert plan is not None
    payload = plan.to_dict()
    payload[field] = value
    with pytest.raises(RecoveryError):
        JournalRetirementPlan.from_json(json.dumps(payload))


def test_journal_lock_symlink_and_interrupted_acquisition_fail_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("POSIX advisory lock and no-follow coverage")
    import fcntl

    journal, _ = terminal_journal(tmp_path / "journal")
    close = os.close
    closed = []

    def interrupt(*args: object) -> None:
        raise KeyboardInterrupt

    def count_close(descriptor: int) -> None:
        closed.append(descriptor)
        close(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(fcntl, "flock", interrupt)
        patch.setattr(os, "close", count_close)
        with pytest.raises(KeyboardInterrupt):
            journal.load()
    assert len(closed) == 1
    assert len(journal.load()) == 2
    lock = tmp_path / "journal" / ".journal-lock"
    lock.unlink()
    target = tmp_path / "synthetic-private"
    target.write_text("untouched")
    lock.symlink_to(target)
    with pytest.raises(RecoveryError, match="journal_lock_failed"):
        journal.load()
    assert target.read_text() == "untouched"
