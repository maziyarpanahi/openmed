"""Restart, durability and process-lock tests with synthetic local journals."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import openmed.agent.workflows.recovery as recovery
from openmed.agent.workflows import (
    CheckpointJournal,
    JournalRetentionPolicy,
    JournalRetirementPlan,
    RecoveryDisposition,
    RecoveryError,
    RetiredJournalRecord,
    recover_workflow,
)
from tests.unit.agent.workflows.test_journal_retirement import (
    exported_summary,
    terminal_journal,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "boundary",
    [
        "write",
        "file_sync",
        "publish",
        "directory_sync",
        "unlink0",
        "unlink1",
        "cleanup_sync",
    ],
)
def test_interruption_always_exposes_full_lineage_or_sealed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    directory = tmp_path / "journal"
    journal, final = terminal_journal(directory)
    original = journal.load()
    summary = exported_summary(final)
    plan = journal.plan_retirement(JournalRetentionPolicy(0), now=100)
    assert plan is not None
    unlink = Path.unlink
    sync = recovery._fsync_directory
    deleted = 0
    sync_calls = 0

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    def interrupted_unlink(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal deleted
        if path.name.startswith("checkpoint-"):
            if boundary == f"unlink{deleted}":
                raise KeyboardInterrupt
            deleted += 1
        unlink(path, *args, **kwargs)

    def interrupted_sync(path: Path) -> None:
        nonlocal sync_calls
        sync_calls += 1
        if (boundary == "directory_sync" and sync_calls == 1) or (
            boundary == "cleanup_sync" and sync_calls == 3
        ):
            raise KeyboardInterrupt
        sync(path)

    with monkeypatch.context() as patch:
        if boundary == "write":
            patch.setattr(RetiredJournalRecord, "to_json", interrupt)
        elif boundary == "file_sync":
            patch.setattr(os, "fsync", interrupt)
        elif boundary == "publish":
            patch.setattr(os, "replace", interrupt)
        patch.setattr(Path, "unlink", interrupted_unlink)
        patch.setattr(recovery, "_fsync_directory", interrupted_sync)
        with pytest.raises(KeyboardInterrupt):
            journal.retire(plan, confirmation=plan.confirmation_token, now=100)

    plan = JournalRetirementPlan.from_json(plan.to_json())
    restarted = CheckpointJournal(directory)
    state = restarted.load()
    if boundary in {"write", "file_sync", "publish"}:
        assert state == original
        assert len(list(directory.glob("checkpoint-*.json"))) == 2
    else:
        assert isinstance(state, RetiredJournalRecord)
        assert state.verifies_summary(summary)
        decision = recover_workflow(state, (), now=101)
        assert decision.disposition is RecoveryDisposition.RETIRED
        assert decision.retry_idempotency_keys == ()
    assert (
        restarted.retire(plan, confirmation=plan.confirmation_token, now=101)
        == plan.record
    )
    assert restarted.load() == plan.record
    assert not list(directory.glob("checkpoint-*.json"))


def test_failed_cleanup_is_controlled_and_retry_checks_remaining_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "journal"
    journal, _ = terminal_journal(directory)
    plan = journal.plan_retirement(JournalRetentionPolicy(0), now=100)
    assert plan is not None
    unlink = Path.unlink

    def fail(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith("checkpoint-"):
            raise OSError("synthetic private-path secret")
        unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail)
        with pytest.raises(RecoveryError, match="retirement_io_failed"):
            journal.retire(plan, confirmation=plan.confirmation_token, now=100)
    assert journal.load() == plan.record
    (directory / "checkpoint-00000000000000000000.json").write_text("{}")
    with pytest.raises(RecoveryError):
        journal.retire(plan, confirmation=plan.confirmation_token, now=101)
    assert len(list(directory.glob("checkpoint-*.json"))) == 2
    assert journal.load() == plan.record


def test_existing_journal_instance_waits_for_cross_process_retirement_lock(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "journal"
    journal, _ = terminal_journal(directory)
    script = """
import sys
from openmed.agent.workflows import CheckpointJournal, JournalRetentionPolicy
journal = CheckpointJournal(sys.argv[1])
print('started', flush=True)
plan = journal.plan_retirement(JournalRetentionPolicy(0), now=100)
record = journal.retire(plan, confirmation=plan.confirmation_token, now=100)
print(record.record_digest, flush=True)
"""
    with journal._locked():
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(directory)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "started"
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.2)
        assert not (directory / "retired.json").exists()
    output, errors = process.communicate(timeout=20)
    assert process.returncode == 0, errors
    assert isinstance(journal.load(), RetiredJournalRecord)
    assert output.strip() == journal.load().record_digest
