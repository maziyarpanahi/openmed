"""Focused tests for transactional in-place trace redaction."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import openmed.traces.transaction as transaction_module
from openmed.traces.transaction import (
    TransactionConflictError,
    TransactionRedactionError,
    TransactionValidationError,
    TransactionWriteError,
    transactional_redact,
)

SYNTHETIC_VALUE = "Synthetic-Trace-Value-001"
SYNTHETIC_REPLACEMENT = "[REDACTED]"


def test_transaction_replaces_atomically_creates_collision_safe_backup_and_preserves_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "trace.jsonl"
    original = f"event={SYNTHETIC_VALUE}\n"
    target.write_text(original, encoding="utf-8", newline="")
    os.chmod(target, 0o640)
    original_time = 1_700_000_123_456_789_000
    os.utime(target, ns=(original_time, original_time))
    existing_backup = target.with_name(target.name + ".bak")
    existing_backup.write_text("existing backup\n", encoding="utf-8")
    replace_sources: list[Path] = []

    original_replace = os.replace

    def record_replace(
        source: str | os.PathLike[str], destination: str | os.PathLike[str]
    ) -> None:
        replace_sources.append(Path(source))
        original_replace(source, destination)

    monkeypatch.setattr("openmed.traces.transaction.os.replace", record_replace)

    result = transactional_redact(
        target,
        lambda text: text.replace(SYNTHETIC_VALUE, SYNTHETIC_REPLACEMENT),
    )

    assert result.changed is True
    assert result.backup_path == target.with_name("trace.jsonl.bak.1")
    target_stat = target.stat()
    assert (target_stat.st_mode & 0o777) == 0o640
    assert target_stat.st_atime_ns == original_time
    assert target_stat.st_mtime_ns == original_time
    assert target.read_text(encoding="utf-8") == f"event={SYNTHETIC_REPLACEMENT}\n"
    assert result.backup_path.read_text(encoding="utf-8") == original
    assert existing_backup.read_text(encoding="utf-8") == "existing backup\n"
    assert replace_sources and replace_sources[0].parent == target.parent
    assert replace_sources[0] != target
    assert not replace_sources[0].exists()

    assert result.to_dict() == {
        "changed": True,
        "backup_created": True,
        "original_bytes": len(original.encode()),
        "replacement_bytes": len(f"event={SYNTHETIC_REPLACEMENT}\n".encode()),
    }


def test_redactor_failure_is_value_free_and_leaves_source_untouched(
    tmp_path: Path,
) -> None:
    target = tmp_path / "trace.jsonl"
    target.write_text(SYNTHETIC_VALUE, encoding="utf-8")

    def fail(text: str) -> str:
        raise RuntimeError(f"failed while handling {text}")

    with pytest.raises(TransactionRedactionError) as error:
        transactional_redact(target, fail)

    assert SYNTHETIC_VALUE not in str(error.value)
    assert target.read_text(encoding="utf-8") == SYNTHETIC_VALUE
    assert list(tmp_path.glob(".openmed-transaction-*.tmp")) == []
    assert not target.with_name(target.name + ".bak").exists()


def test_validation_mismatch_rolls_back_without_creating_backup(tmp_path: Path) -> None:
    target = tmp_path / "trace.jsonl"
    target.write_text(SYNTHETIC_VALUE, encoding="utf-8")

    with pytest.raises(TransactionValidationError):
        transactional_redact(
            target,
            lambda _text: SYNTHETIC_REPLACEMENT,
            validator=lambda _candidate: False,
        )

    assert target.read_text(encoding="utf-8") == SYNTHETIC_VALUE
    assert list(tmp_path.glob(".openmed-transaction-*.tmp")) == []
    assert not target.with_name(target.name + ".bak").exists()


def test_source_change_is_detected_before_atomic_exchange(tmp_path: Path) -> None:
    target = tmp_path / "trace.jsonl"
    target.write_text(SYNTHETIC_VALUE, encoding="utf-8")

    def mutate_source(_text: str) -> str:
        target.write_text("changed outside transaction", encoding="utf-8")
        return SYNTHETIC_REPLACEMENT

    with pytest.raises(TransactionConflictError):
        transactional_redact(target, mutate_source)

    assert target.read_text(encoding="utf-8") == "changed outside transaction"
    assert list(tmp_path.glob(".openmed-transaction-*.tmp")) == []
    assert not target.with_name(target.name + ".bak").exists()


def test_atomic_exchange_failure_rolls_back_temp_and_backup(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "trace.jsonl"
    target.write_text(SYNTHETIC_VALUE, encoding="utf-8")

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("synthetic disk failure")

    monkeypatch.setattr("openmed.traces.transaction.os.replace", fail_replace)

    with pytest.raises(TransactionWriteError):
        transactional_redact(target, lambda _text: SYNTHETIC_REPLACEMENT)

    assert target.read_text(encoding="utf-8") == SYNTHETIC_VALUE
    assert list(tmp_path.glob(".openmed-transaction-*.tmp")) == []
    assert not target.with_name(target.name + ".bak").exists()


def test_interruption_cleans_transaction_artifacts(tmp_path: Path) -> None:
    target = tmp_path / "trace.jsonl"
    target.write_text(SYNTHETIC_VALUE, encoding="utf-8")

    def interrupt(_text: str) -> str:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        transactional_redact(target, interrupt)

    assert target.read_text(encoding="utf-8") == SYNTHETIC_VALUE
    assert list(tmp_path.glob(".openmed-transaction-*.tmp")) == []


def test_unchanged_candidate_does_not_create_backup(tmp_path: Path) -> None:
    target = tmp_path / "trace.jsonl"
    target.write_text(SYNTHETIC_VALUE, encoding="utf-8")

    result = transactional_redact(target, lambda text: text)

    assert result.changed is False
    assert result.backup_path is None
    assert not target.with_name(target.name + ".bak").exists()


def test_metadata_fallback_omits_unsupported_follow_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "candidate.tmp"
    target.touch()
    original_chmod = os.chmod
    original_utime = os.utime
    calls: list[str] = []

    def chmod_without_follow_symlinks(path: Path, mode: int) -> None:
        calls.append("chmod")
        original_chmod(path, mode)

    def utime_without_follow_symlinks(
        path: Path,
        *,
        ns: tuple[int, int],
    ) -> None:
        calls.append("utime")
        original_utime(path, ns=ns)

    monkeypatch.delattr(transaction_module.os, "fchmod", raising=False)
    monkeypatch.setattr(transaction_module.os, "supports_follow_symlinks", set())
    monkeypatch.setattr(transaction_module.os, "chmod", chmod_without_follow_symlinks)
    monkeypatch.setattr(transaction_module.os, "utime", utime_without_follow_symlinks)

    transaction_module._write_payload(
        target,
        b"synthetic replacement",
        mode=0o600,
        timestamps=(1_700_000_123_456_789_000,) * 2,
    )

    assert target.read_bytes() == b"synthetic replacement"
    assert calls == ["chmod", "utime"]
