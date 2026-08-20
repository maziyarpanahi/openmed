"""Focused tests for transactional in-place trace redaction."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import openmed.traces.transaction as transaction_module
from openmed.traces.transaction import (
    TransactionConflictError,
    TransactionReadError,
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
    original_mode = target.stat().st_mode & 0o777
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
    assert (target_stat.st_mode & 0o777) == original_mode
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


def test_callback_result_failures_are_value_free(tmp_path: Path) -> None:
    target = tmp_path / "trace.jsonl"
    target.write_text(SYNTHETIC_VALUE, encoding="utf-8")

    class FailingDecision:
        def __bool__(self) -> bool:
            raise RuntimeError(SYNTHETIC_VALUE)

    with pytest.raises(TransactionRedactionError) as encoding_error:
        transactional_redact(target, lambda _text: "\ud800")
    assert SYNTHETIC_VALUE not in str(encoding_error.value)

    with pytest.raises(TransactionValidationError) as validation_error:
        transactional_redact(
            target,
            lambda _text: SYNTHETIC_REPLACEMENT,
            validator=lambda _candidate: FailingDecision(),  # type: ignore[return-value]
        )
    assert SYNTHETIC_VALUE not in str(validation_error.value)
    assert target.read_text(encoding="utf-8") == SYNTHETIC_VALUE


def test_string_subclass_hooks_cannot_change_validated_replacement_bytes(
    tmp_path: Path,
) -> None:
    target = tmp_path / "trace.jsonl"
    target.write_text(SYNTHETIC_VALUE, encoding="utf-8")

    class DeceptiveText(str):
        def encode(self, *args: object, **kwargs: object) -> bytes:
            del args, kwargs
            return SYNTHETIC_VALUE.encode("utf-8")

    result = transactional_redact(
        target,
        lambda _text: DeceptiveText(SYNTHETIC_REPLACEMENT),
        validator=lambda candidate: candidate == SYNTHETIC_REPLACEMENT,
        backup=False,
    )

    assert result.changed is True
    assert target.read_text(encoding="utf-8") == SYNTHETIC_REPLACEMENT


def test_relative_target_is_bound_before_callback_changes_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_directory = tmp_path / "source"
    other_directory = tmp_path / "other"
    source_directory.mkdir()
    other_directory.mkdir()
    target = source_directory / "trace.jsonl"
    other_target = other_directory / "trace.jsonl"
    target.write_text(SYNTHETIC_VALUE, encoding="utf-8")
    other_target.write_text("unrelated local trace", encoding="utf-8")
    monkeypatch.chdir(source_directory)

    def change_directory(_text: str) -> str:
        os.chdir(other_directory)
        return SYNTHETIC_REPLACEMENT

    result = transactional_redact(
        Path("trace.jsonl"),
        change_directory,
        backup=False,
    )

    assert result.path == target
    assert target.read_text(encoding="utf-8") == SYNTHETIC_REPLACEMENT
    assert other_target.read_text(encoding="utf-8") == "unrelated local trace"


def test_source_bytes_are_read_from_the_audited_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "trace.jsonl"
    decoy = tmp_path / "decoy.jsonl"
    target.write_text(SYNTHETIC_VALUE, encoding="utf-8")
    decoy.write_text("SYNTHETIC_DECOY_VALUE", encoding="utf-8")
    original_read_bytes = Path.read_bytes
    seen: list[str] = []

    def swap_before_path_read(path: Path) -> bytes:
        if path == target:
            target.unlink()
            target.symlink_to(decoy)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", swap_before_path_read)

    result = transactional_redact(
        target,
        lambda text: seen.append(text) or SYNTHETIC_REPLACEMENT,
        backup=False,
    )

    assert result.changed is True
    assert seen == [SYNTHETIC_VALUE]
    assert target.read_text(encoding="utf-8") == SYNTHETIC_REPLACEMENT


def test_final_symlink_is_rejected_before_source_text_reaches_redactor(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    target = tmp_path / "trace.jsonl"
    source.write_text(SYNTHETIC_VALUE, encoding="utf-8")
    try:
        target.symlink_to(source)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    seen: list[str] = []

    with pytest.raises(TransactionReadError):
        transactional_redact(
            target,
            lambda text: seen.append(text) or SYNTHETIC_REPLACEMENT,
        )

    assert seen == []
    assert source.read_text(encoding="utf-8") == SYNTHETIC_VALUE


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


def test_interruption_after_atomic_replace_keeps_the_recovery_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "trace.jsonl"
    target.write_text(SYNTHETIC_VALUE, encoding="utf-8")
    original_replace = os.replace

    def replace_then_interrupt(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
    ) -> None:
        original_replace(source, destination)
        raise KeyboardInterrupt

    monkeypatch.setattr("openmed.traces.transaction.os.replace", replace_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        transactional_redact(target, lambda _text: SYNTHETIC_REPLACEMENT)

    backup = target.with_name(target.name + ".bak")
    assert target.read_text(encoding="utf-8") == SYNTHETIC_REPLACEMENT
    assert backup.read_text(encoding="utf-8") == SYNTHETIC_VALUE
    assert list(tmp_path.glob(".openmed-transaction-*.tmp")) == []


def test_unchanged_candidate_does_not_create_backup(tmp_path: Path) -> None:
    target = tmp_path / "trace.jsonl"
    target.write_text(SYNTHETIC_VALUE, encoding="utf-8")

    result = transactional_redact(target, lambda text: text)

    assert result.changed is False
    assert result.backup_path is None
    assert not target.with_name(target.name + ".bak").exists()


def test_unchanged_candidate_still_detects_source_conflicts(tmp_path: Path) -> None:
    target = tmp_path / "trace.jsonl"
    target.write_text(SYNTHETIC_VALUE, encoding="utf-8")

    def mutate_source(text: str) -> str:
        target.write_text("changed outside transaction", encoding="utf-8")
        return text

    with pytest.raises(TransactionConflictError):
        transactional_redact(target, mutate_source)

    assert target.read_text(encoding="utf-8") == "changed outside transaction"
    assert not target.with_name(target.name + ".bak").exists()


def test_temporary_payload_uses_exclusive_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "trace.jsonl"
    target.write_text(SYNTHETIC_VALUE, encoding="utf-8")
    original_write = transaction_module._write_payload
    temporary_descriptors: list[int | None] = []

    def record_write(
        path: Path,
        payload: bytes,
        *,
        file_descriptor: int | None = None,
        mode: int | None,
        timestamps: tuple[int, int] | None,
    ) -> None:
        if path.name.startswith(transaction_module.TEMPORARY_FILE_PREFIX):
            temporary_descriptors.append(file_descriptor)
        original_write(
            path,
            payload,
            file_descriptor=file_descriptor,
            mode=mode,
            timestamps=timestamps,
        )

    monkeypatch.setattr(transaction_module, "_write_payload", record_write)

    transactional_redact(
        target,
        lambda _text: SYNTHETIC_REPLACEMENT,
        backup=False,
    )

    assert len(temporary_descriptors) == 1
    assert temporary_descriptors[0] is not None


def test_path_and_encoding_errors_do_not_expose_values(tmp_path: Path) -> None:
    sensitive = "PatientJaneDoe"
    target = tmp_path / "trace.jsonl"
    target.write_text(SYNTHETIC_VALUE, encoding="utf-8")

    with pytest.raises(TransactionReadError) as encoding_error:
        transactional_redact(target, lambda text: text, encoding=sensitive)
    assert sensitive not in str(encoding_error.value)

    class FailingPath:
        def __fspath__(self) -> str:
            raise RuntimeError(sensitive)

    with pytest.raises(ValueError) as path_error:
        transactional_redact(FailingPath(), lambda text: text)
    assert sensitive not in str(path_error.value)


def test_result_repr_does_not_expose_paths(tmp_path: Path) -> None:
    sensitive = "PatientJaneDoe"
    target = tmp_path / f"{sensitive}.jsonl"
    target.write_text(SYNTHETIC_VALUE, encoding="utf-8")

    result = transactional_redact(target, lambda text: text)

    assert sensitive not in repr(result)


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
