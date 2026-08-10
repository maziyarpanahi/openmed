"""Offline regression tests for transactional trace-redaction recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.multimodal import (
    TraceRecoveryError,
    recover_trace_redaction,
    redact_trace_file,
    trace_fingerprint,
)

SYNTHETIC_VALUE = "SYNTHETIC-PATIENT-771"
SYNTHETIC_REPLACEMENT = "[PATIENT]"


class InjectedCrash(RuntimeError):
    """Synthetic process interruption used to leave a staged transaction."""


def _redact(text: str) -> str:
    return text.replace(SYNTHETIC_VALUE, SYNTHETIC_REPLACEMENT)


def _crash_after_staging(state) -> None:
    if state.phase == "staged":
        raise InjectedCrash("synthetic interruption")


def test_trace_redaction_journal_is_value_free_and_deterministic(tmp_path: Path):
    source = f"trace event for {SYNTHETIC_VALUE}\n"
    trace_path = tmp_path / "synthetic-trace.jsonl"
    journal_path = tmp_path / "trace-recovery.json"
    trace_path.write_text(source, encoding="utf-8")

    result = redact_trace_file(trace_path, _redact, journal_path=journal_path)

    expected = source.replace(SYNTHETIC_VALUE, SYNTHETIC_REPLACEMENT)
    assert trace_path.read_text(encoding="utf-8") == expected
    assert result.input_fingerprint == trace_fingerprint(source)
    assert result.output_fingerprint == trace_fingerprint(expected)
    assert result.recovery_attempts == 0

    journal_text = journal_path.read_text(encoding="utf-8")
    journal = json.loads(journal_text)
    assert journal["phase"] == "committed"
    assert journal["recovery_decision"] == "none"
    assert journal["input_fingerprint"] == result.input_fingerprint
    assert journal["output_fingerprint"] == result.output_fingerprint
    assert SYNTHETIC_VALUE not in journal_text
    assert SYNTHETIC_VALUE not in str(result.to_audit_report())

    calls = 0

    def should_not_run(_text: str) -> str:
        nonlocal calls
        calls += 1
        raise AssertionError("completed transactions must be idempotent")

    repeated = redact_trace_file(
        trace_path,
        should_not_run,
        journal_path=journal_path,
    )
    assert calls == 0
    assert repeated.phase == "committed"
    assert trace_path.read_text(encoding="utf-8") == expected


def test_staged_trace_transaction_resumes_and_repeated_recovery_is_idempotent(
    tmp_path: Path,
):
    trace_path = tmp_path / "synthetic-trace.jsonl"
    journal_path = tmp_path / "trace-recovery.json"
    original = f"start {SYNTHETIC_VALUE} end\n"
    trace_path.write_text(original, encoding="utf-8")

    with pytest.raises(InjectedCrash):
        redact_trace_file(
            trace_path,
            _redact,
            journal_path=journal_path,
            phase_hook=_crash_after_staging,
        )

    assert trace_path.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".openmed-trace-stage-*.bin"))

    recovered = recover_trace_redaction(trace_path, journal_path=journal_path)
    assert recovered.resumed is True
    assert recovered.recovery_decision == "resume"
    assert recovered.recovery_attempts == 1
    assert SYNTHETIC_VALUE not in trace_path.read_text(encoding="utf-8")

    before = trace_path.read_bytes()
    repeated = recover_trace_redaction(trace_path, journal_path=journal_path)
    assert repeated.resumed is False
    assert repeated.recovery_decision == "already_complete"
    assert trace_path.read_bytes() == before
    assert SYNTHETIC_VALUE not in journal_path.read_text(encoding="utf-8")


def test_rollback_removes_only_owned_staging_artifact(tmp_path: Path):
    trace_path = tmp_path / "synthetic-trace.jsonl"
    journal_path = tmp_path / "trace-recovery.json"
    original = f"keep {SYNTHETIC_VALUE}\n"
    trace_path.write_text(original, encoding="utf-8")
    unrelated = tmp_path / ".unrelated-artifact"
    unrelated.write_text("keep me", encoding="utf-8")

    with pytest.raises(InjectedCrash):
        redact_trace_file(
            trace_path,
            _redact,
            journal_path=journal_path,
            phase_hook=_crash_after_staging,
        )

    # A hard stop during the staging write may leave a partial owned artifact.
    next(tmp_path.glob(".openmed-trace-stage-*.bin")).write_text(
        "synthetic partial stage",
        encoding="utf-8",
    )
    rolled_back = recover_trace_redaction(
        trace_path,
        journal_path=journal_path,
        decision="rollback",
    )

    assert rolled_back.phase == "rolled_back"
    assert rolled_back.recovery_decision == "rollback"
    assert trace_path.read_text(encoding="utf-8") == original
    assert unrelated.read_text(encoding="utf-8") == "keep me"
    assert not list(tmp_path.glob(".openmed-trace-stage-*.bin"))


def test_recovery_is_bounded_and_rejects_tampered_owned_artifact(tmp_path: Path):
    trace_path = tmp_path / "synthetic-trace.jsonl"
    journal_path = tmp_path / "trace-recovery.json"
    original = f"event {SYNTHETIC_VALUE}\n"
    trace_path.write_text(original, encoding="utf-8")

    with pytest.raises(InjectedCrash):
        redact_trace_file(
            trace_path,
            _redact,
            journal_path=journal_path,
            phase_hook=_crash_after_staging,
        )

    stage_path = next(tmp_path.glob(".openmed-trace-stage-*.bin"))
    stage_path.write_text("synthetic tampering", encoding="utf-8")

    with pytest.raises(TraceRecoveryError) as excinfo:
        recover_trace_redaction(trace_path, journal_path=journal_path)
    assert excinfo.value.reason == "owned_artifact_conflict"
    assert SYNTHETIC_VALUE not in str(excinfo.value)
    assert trace_path.read_text(encoding="utf-8") == original

    def interrupt_recovery(state) -> None:
        if state.recovery_attempts == 2:
            raise InjectedCrash("synthetic recovery interruption")

    # Repair the synthetic staging artifact, then interrupt one bounded
    # recovery attempt before the commit phase.
    stage_path.write_text(
        original.replace(SYNTHETIC_VALUE, SYNTHETIC_REPLACEMENT),
        encoding="utf-8",
    )
    with pytest.raises(InjectedCrash):
        recover_trace_redaction(
            trace_path,
            journal_path=journal_path,
            max_recovery_attempts=2,
            phase_hook=interrupt_recovery,
        )

    with pytest.raises(TraceRecoveryError) as limit_excinfo:
        recover_trace_redaction(
            trace_path,
            journal_path=journal_path,
            max_recovery_attempts=2,
        )
    assert limit_excinfo.value.reason == "recovery_attempt_limit"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["phase"] == "blocked"
    assert trace_path.read_text(encoding="utf-8") == original


def test_redactor_errors_are_safe_to_log(tmp_path: Path):
    trace_path = tmp_path / "synthetic-trace.jsonl"
    trace_path.write_text(SYNTHETIC_VALUE, encoding="utf-8")

    def failing_redactor(_text: str) -> str:
        raise RuntimeError(f"unexpected value {SYNTHETIC_VALUE}")

    with pytest.raises(TraceRecoveryError) as excinfo:
        redact_trace_file(trace_path, failing_redactor)

    assert excinfo.value.reason == "redactor_failed"
    assert SYNTHETIC_VALUE not in str(excinfo.value)
