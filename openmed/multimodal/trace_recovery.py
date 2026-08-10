"""Crash-safe, value-free recovery for in-place trace redaction.

Trace redaction normally produces a new value in memory and then replaces the
source file.  A process can stop between those operations, however, leaving a
staging file without an explicit record of what it belongs to.  This module
keeps a small, atomically-written journal next to the target and commits only a
fingerprint-verified staging artifact.

The journal contains no source or output text.  It stores only content
fingerprints, an opaque target identity fingerprint, a phase, and a bounded
recovery-attempt count.  Recovery is local and deterministic; it never calls a
network service.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal, Mapping

TRACE_RECOVERY_SCHEMA_VERSION = 1
DEFAULT_MAX_RECOVERY_ATTEMPTS = 3
DEFAULT_MAX_TRACE_BYTES = 64 * 1024 * 1024
MAX_JOURNAL_BYTES = 8 * 1024

TraceRecoveryPhase = Literal[
    "prepared",
    "staged",
    "committing",
    "committed",
    "rolled_back",
    "blocked",
]
RecoveryDecision = Literal[
    "none",
    "commit",
    "resume",
    "rollback",
    "already_complete",
    "already_rolled_back",
    "blocked",
]
TraceRedactor = Callable[[str], str | bytes]
TraceRecoveryHook = Callable[["TraceRecoveryJournal"], None]

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PHASES = frozenset(
    {"prepared", "staged", "committing", "committed", "rolled_back", "blocked"}
)
_DECISIONS = frozenset(
    {
        "none",
        "commit",
        "resume",
        "rollback",
        "already_complete",
        "already_rolled_back",
        "blocked",
    }
)


class TraceRecoveryError(RuntimeError):
    """Raised when a trace transaction cannot be recovered safely.

    ``reason`` is deliberately a short constant-like code.  The exception
    never includes a path, input text, output text, or an underlying OS error
    message, so callers can safely send it to an error logger.
    """

    def __init__(self, reason: str) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", reason):
            reason = "recovery_failed"
        self.reason = reason
        super().__init__(f"trace recovery failed: {reason}")


def trace_fingerprint(value: str | bytes | bytearray | memoryview) -> str:
    """Return a stable SHA-256 fingerprint without retaining ``value``."""

    if isinstance(value, str):
        payload = value.encode("utf-8")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
    else:
        raise TypeError("trace fingerprint input must be text or bytes")
    return _digest(payload)


@dataclass(frozen=True)
class TraceRecoveryJournal:
    """Value-free durable state for one trace redaction transaction."""

    target_fingerprint: str
    input_fingerprint: str
    output_fingerprint: str
    phase: TraceRecoveryPhase
    recovery_decision: RecoveryDecision = "none"
    recovery_attempts: int = 0
    staging_fingerprint: str | None = None
    schema_version: int = TRACE_RECOVERY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_journal(self)

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-safe journal payload."""

        return {
            "schema_version": self.schema_version,
            "target_fingerprint": self.target_fingerprint,
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "staging_fingerprint": self.staging_fingerprint,
            "phase": self.phase,
            "recovery_decision": self.recovery_decision,
            "recovery_attempts": self.recovery_attempts,
        }

    def to_audit_report(self) -> dict[str, object]:
        """Return an audit-safe summary with no path or raw value."""

        return {
            "type": "trace_redaction_recovery",
            **self.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TraceRecoveryJournal":
        """Parse and validate a journal payload without exposing bad input."""

        try:
            schema_version = payload["schema_version"]
            target_fingerprint = payload["target_fingerprint"]
            input_fingerprint = payload["input_fingerprint"]
            output_fingerprint = payload["output_fingerprint"]
            phase = payload["phase"]
            recovery_decision = payload.get("recovery_decision", "none")
            recovery_attempts = payload.get("recovery_attempts", 0)
            staging_fingerprint = payload.get("staging_fingerprint")
        except (KeyError, TypeError):
            raise TraceRecoveryError("journal_invalid") from None

        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise TraceRecoveryError("journal_invalid")
        if not isinstance(target_fingerprint, str):
            raise TraceRecoveryError("journal_invalid")
        if not isinstance(input_fingerprint, str):
            raise TraceRecoveryError("journal_invalid")
        if not isinstance(output_fingerprint, str):
            raise TraceRecoveryError("journal_invalid")
        if not isinstance(phase, str) or phase not in _PHASES:
            raise TraceRecoveryError("journal_invalid")
        if (
            not isinstance(recovery_decision, str)
            or recovery_decision not in _DECISIONS
        ):
            raise TraceRecoveryError("journal_invalid")
        if not isinstance(recovery_attempts, int) or isinstance(
            recovery_attempts, bool
        ):
            raise TraceRecoveryError("journal_invalid")
        if staging_fingerprint is not None and not isinstance(staging_fingerprint, str):
            raise TraceRecoveryError("journal_invalid")

        try:
            return cls(
                target_fingerprint=target_fingerprint,
                input_fingerprint=input_fingerprint,
                output_fingerprint=output_fingerprint,
                phase=phase,  # type: ignore[arg-type]
                recovery_decision=recovery_decision,  # type: ignore[arg-type]
                recovery_attempts=recovery_attempts,
                staging_fingerprint=staging_fingerprint,
                schema_version=schema_version,
            )
        except (TypeError, ValueError):
            raise TraceRecoveryError("journal_invalid") from None


@dataclass(frozen=True)
class TraceRedactionResult:
    """PHI-free result for a trace redaction or recovery operation."""

    target_fingerprint: str
    input_fingerprint: str
    output_fingerprint: str
    phase: TraceRecoveryPhase
    recovery_decision: RecoveryDecision
    recovery_attempts: int
    resumed: bool
    changed: bool
    output_size: int

    def to_audit_report(self) -> dict[str, object]:
        """Return a serializable report containing fingerprints only."""

        return {
            "type": "trace_redaction_recovery",
            "schema_version": TRACE_RECOVERY_SCHEMA_VERSION,
            "target_fingerprint": self.target_fingerprint,
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "phase": self.phase,
            "recovery_decision": self.recovery_decision,
            "recovery_attempts": self.recovery_attempts,
            "resumed": self.resumed,
            "changed": self.changed,
            "output_size": self.output_size,
        }


def redact_trace_file(
    path: str | Path,
    redactor: TraceRedactor,
    *,
    journal_path: str | Path | None = None,
    recovery: Literal["resume", "rollback"] = "resume",
    encoding: str = "utf-8",
    max_bytes: int = DEFAULT_MAX_TRACE_BYTES,
    max_recovery_attempts: int = DEFAULT_MAX_RECOVERY_ATTEMPTS,
    phase_hook: TraceRecoveryHook | None = None,
) -> TraceRedactionResult:
    """Redact a UTF-8 trace file with an atomic, recoverable replacement.

    ``redactor`` receives the source text and may return text or UTF-8 bytes.
    The source is never written to a journal or staging artifact.  If a prior
    transaction is pending, ``recovery`` controls whether its verified output
    is resumed or its owned staging artifact is rolled back before starting a
    new transaction.  A completed transaction is returned unchanged, making a
    repeated call idempotent.

    ``phase_hook`` is intended for crash-injection tests and operational
    checkpoints.  It runs after each durable phase transition; exceptions from
    it intentionally leave the journal in that phase for a later recovery.
    """

    _validate_recovery_options(recovery, encoding, max_bytes, max_recovery_attempts)
    target = _prepare_target(path)
    journal = _resolve_journal(target, journal_path)
    existing = _load_journal(journal)
    target_fingerprint = _target_fingerprint(target)

    if existing is not None:
        _validate_target_ownership(existing, target_fingerprint)
        if existing.phase != "rolled_back":
            recovered = _recover_transaction(
                target,
                journal,
                existing,
                decision=recovery,
                redactor=redactor,
                encoding=encoding,
                max_bytes=max_bytes,
                max_recovery_attempts=max_recovery_attempts,
                phase_hook=phase_hook,
            )
            if recovered.phase == "committed":
                return recovered

    source = _read_file(target, max_bytes)
    input_fingerprint = _digest(source)
    output = _redact_bytes(source, redactor, encoding)
    if len(output) > max_bytes:
        raise TraceRecoveryError("trace_size_limit")
    output_fingerprint = _digest(output)
    mode = _file_mode(target)

    prepared = TraceRecoveryJournal(
        target_fingerprint=target_fingerprint,
        input_fingerprint=input_fingerprint,
        output_fingerprint=output_fingerprint,
        staging_fingerprint=output_fingerprint,
        phase="prepared",
    )
    _write_journal(journal, prepared)
    _call_hook(phase_hook, prepared)

    staging = _staging_path(target, output_fingerprint)
    _write_staging(
        staging,
        output,
        mode,
        output_fingerprint,
        max_bytes=max_bytes,
        allow_existing=False,
    )
    staged = replace(prepared, phase="staged")
    _write_journal(journal, staged)
    _call_hook(phase_hook, staged)

    committed = _commit_staging(
        target,
        staging,
        journal,
        staged,
        recovery_decision="none",
        max_bytes=max_bytes,
        phase_hook=phase_hook,
    )
    return _result_from_journal(
        committed,
        resumed=False,
        output_size=len(output),
    )


def recover_trace_redaction(
    path: str | Path,
    *,
    journal_path: str | Path | None = None,
    decision: Literal["resume", "rollback"] = "resume",
    max_bytes: int = DEFAULT_MAX_TRACE_BYTES,
    max_recovery_attempts: int = DEFAULT_MAX_RECOVERY_ATTEMPTS,
    phase_hook: TraceRecoveryHook | None = None,
) -> TraceRedactionResult:
    """Resume or roll back a pending trace transaction using local state only.

    Recovery can resume a journaled staging artifact without a redactor.  If
    the staging artifact is missing, recovery safely rolls back instead of
    inventing output; callers that want to recompute output should call
    :func:`redact_trace_file`, which supplies the redactor.
    """

    _validate_recovery_options(decision, "utf-8", max_bytes, max_recovery_attempts)
    target = _prepare_target(path)
    journal_path_value = _resolve_journal(target, journal_path)
    journal = _load_journal(journal_path_value)
    if journal is None:
        raise TraceRecoveryError("no_recovery_state")
    _validate_target_ownership(journal, _target_fingerprint(target))
    return _recover_transaction(
        target,
        journal_path_value,
        journal,
        decision=decision,
        redactor=None,
        encoding="utf-8",
        max_bytes=max_bytes,
        max_recovery_attempts=max_recovery_attempts,
        phase_hook=phase_hook,
    )


# These names make the transactional behavior discoverable to callers that
# already use "in place" terminology while keeping one implementation.
redact_trace_in_place = redact_trace_file
transactional_trace_redact = redact_trace_file


def _recover_transaction(
    target: Path,
    journal_path: Path,
    journal: TraceRecoveryJournal,
    *,
    decision: Literal["resume", "rollback"],
    redactor: TraceRedactor | None,
    encoding: str,
    max_bytes: int,
    max_recovery_attempts: int,
    phase_hook: TraceRecoveryHook | None,
) -> TraceRedactionResult:
    """Recover one validated journal without touching unrelated artifacts."""

    staging = _staging_path(target, journal.output_fingerprint)
    current_fingerprint = _file_fingerprint(target, max_bytes)

    if journal.phase == "blocked":
        raise TraceRecoveryError("recovery_blocked")

    if journal.phase == "rolled_back":
        if current_fingerprint != journal.input_fingerprint:
            raise TraceRecoveryError("target_changed")
        return _result_from_journal(journal, resumed=False, output_size=0)

    if (
        journal.phase == "committed"
        or current_fingerprint == journal.output_fingerprint
    ):
        if current_fingerprint != journal.output_fingerprint:
            _block_transaction(journal_path, journal, phase_hook)
            raise TraceRecoveryError("target_changed")
        _remove_owned_staging(staging, journal.output_fingerprint, max_bytes)
        complete = replace(
            journal,
            phase="committed",
            recovery_decision="already_complete",
        )
        if complete != journal:
            _write_journal(journal_path, complete)
            _call_hook(phase_hook, complete)
        return _result_from_journal(
            complete,
            resumed=journal.phase != "committed",
            output_size=_file_size(target, max_bytes),
        )

    if current_fingerprint != journal.input_fingerprint:
        _block_transaction(journal_path, journal, phase_hook)
        raise TraceRecoveryError("target_changed")

    attempts = journal.recovery_attempts + 1
    if attempts > max_recovery_attempts:
        blocked = replace(
            journal,
            phase="blocked",
            recovery_decision="blocked",
            recovery_attempts=journal.recovery_attempts,
        )
        _write_journal(journal_path, blocked)
        _call_hook(phase_hook, blocked)
        raise TraceRecoveryError("recovery_attempt_limit")

    pending = replace(
        journal,
        recovery_decision=decision,
        recovery_attempts=attempts,
    )
    _write_journal(journal_path, pending)
    _call_hook(phase_hook, pending)

    if decision == "rollback":
        _remove_owned_staging(
            staging,
            journal.output_fingerprint,
            max_bytes,
            verify_fingerprint=False,
        )
        rolled_back = replace(
            pending,
            phase="rolled_back",
            recovery_decision="rollback",
        )
        _write_journal(journal_path, rolled_back)
        _call_hook(phase_hook, rolled_back)
        return _result_from_journal(rolled_back, resumed=True, output_size=0)

    if _path_exists(staging):
        _ensure_owned_staging(staging, journal.output_fingerprint, max_bytes)
    elif redactor is not None:
        source = _read_file(target, max_bytes)
        if _digest(source) != journal.input_fingerprint:
            _block_transaction(journal_path, pending, phase_hook)
            raise TraceRecoveryError("target_changed")
        output = _redact_bytes(source, redactor, encoding)
        if len(output) > max_bytes:
            raise TraceRecoveryError("trace_size_limit")
        if _digest(output) != journal.output_fingerprint:
            _block_transaction(journal_path, pending, phase_hook)
            raise TraceRecoveryError("output_fingerprint_mismatch")
        _write_staging(
            staging,
            output,
            _file_mode(target),
            journal.output_fingerprint,
            max_bytes=max_bytes,
            allow_existing=False,
        )
        pending = replace(pending, phase="staged")
        _write_journal(journal_path, pending)
        _call_hook(phase_hook, pending)
    else:
        _remove_owned_staging(staging, journal.output_fingerprint, max_bytes)
        rolled_back = replace(
            pending,
            phase="rolled_back",
            recovery_decision="rollback",
        )
        _write_journal(journal_path, rolled_back)
        _call_hook(phase_hook, rolled_back)
        return _result_from_journal(rolled_back, resumed=True, output_size=0)

    committed = _commit_staging(
        target,
        staging,
        journal_path,
        pending,
        recovery_decision="resume",
        max_bytes=max_bytes,
        phase_hook=phase_hook,
    )
    return _result_from_journal(
        committed,
        resumed=True,
        output_size=_file_size(target, max_bytes),
    )


def _commit_staging(
    target: Path,
    staging: Path,
    journal_path: Path,
    journal: TraceRecoveryJournal,
    *,
    recovery_decision: RecoveryDecision,
    max_bytes: int,
    phase_hook: TraceRecoveryHook | None,
) -> TraceRecoveryJournal:
    """Atomically replace the target after rechecking both fingerprints."""

    _ensure_owned_staging(staging, journal.output_fingerprint, max_bytes)
    committing = replace(
        journal,
        phase="committing",
        recovery_decision=recovery_decision,
    )
    _write_journal(journal_path, committing)
    _call_hook(phase_hook, committing)

    if _file_fingerprint(target, max_bytes) != journal.input_fingerprint:
        raise TraceRecoveryError("target_changed")
    try:
        os.replace(staging, target)
        _fsync_directory(target.parent)
    except OSError:
        raise TraceRecoveryError("commit_failed") from None

    if _file_fingerprint(target, max_bytes) != journal.output_fingerprint:
        raise TraceRecoveryError("output_verification_failed")
    committed = replace(
        committing,
        phase="committed",
        recovery_decision=recovery_decision,
    )
    _write_journal(journal_path, committed)
    _call_hook(phase_hook, committed)
    return committed


def _result_from_journal(
    journal: TraceRecoveryJournal,
    *,
    resumed: bool,
    output_size: int,
) -> TraceRedactionResult:
    return TraceRedactionResult(
        target_fingerprint=journal.target_fingerprint,
        input_fingerprint=journal.input_fingerprint,
        output_fingerprint=journal.output_fingerprint,
        phase=journal.phase,
        recovery_decision=journal.recovery_decision,
        recovery_attempts=journal.recovery_attempts,
        resumed=resumed,
        changed=journal.input_fingerprint != journal.output_fingerprint,
        output_size=output_size,
    )


def _redact_bytes(source: bytes, redactor: TraceRedactor, encoding: str) -> bytes:
    try:
        redacted = redactor(source.decode(encoding))
    except TraceRecoveryError:
        raise
    except Exception:
        raise TraceRecoveryError("redactor_failed") from None
    if isinstance(redacted, str):
        try:
            return redacted.encode(encoding)
        except UnicodeError:
            raise TraceRecoveryError("redactor_result_invalid") from None
    if isinstance(redacted, (bytes, bytearray, memoryview)):
        return bytes(redacted)
    raise TraceRecoveryError("redactor_result_invalid")


def _write_staging(
    path: Path,
    content: bytes,
    mode: int,
    expected_fingerprint: str,
    *,
    max_bytes: int,
    allow_existing: bool,
) -> None:
    if _path_exists(path):
        if not allow_existing:
            raise TraceRecoveryError("owned_artifact_conflict")
        _ensure_owned_staging(path, expected_fingerprint, max_bytes)
        return

    file_descriptor: int | None = None
    created = False
    try:
        file_descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        created = True
        with os.fdopen(file_descriptor, "wb") as handle:
            file_descriptor = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, mode)
        _fsync_directory(path.parent)
    except FileExistsError:
        if allow_existing:
            _ensure_owned_staging(path, expected_fingerprint, max_bytes)
            return
        raise TraceRecoveryError("owned_artifact_conflict") from None
    except (OSError, ValueError):
        raise TraceRecoveryError("staging_write_failed") from None
    finally:
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except OSError:
                pass
        if created and _path_exists(path):
            try:
                if _file_fingerprint(path, max_bytes) != expected_fingerprint:
                    path.unlink()
            except (OSError, TraceRecoveryError):
                pass


def _ensure_owned_staging(
    path: Path, expected_fingerprint: str, max_bytes: int
) -> None:
    try:
        if path.is_symlink() or not path.is_file():
            raise TraceRecoveryError("owned_artifact_conflict")
        actual = _file_fingerprint(path, max_bytes)
    except TraceRecoveryError:
        raise
    except OSError:
        raise TraceRecoveryError("owned_artifact_unreadable") from None
    if actual != expected_fingerprint:
        raise TraceRecoveryError("owned_artifact_conflict")


def _remove_owned_staging(
    path: Path,
    expected_fingerprint: str,
    max_bytes: int = DEFAULT_MAX_TRACE_BYTES,
    *,
    verify_fingerprint: bool = True,
) -> None:
    if not _path_exists(path):
        return
    if verify_fingerprint:
        _ensure_owned_staging(path, expected_fingerprint, max_bytes)
    else:
        try:
            if path.is_symlink() or not path.is_file():
                raise TraceRecoveryError("owned_artifact_conflict")
        except TraceRecoveryError:
            raise
        except OSError:
            raise TraceRecoveryError("owned_artifact_unreadable") from None
    try:
        path.unlink()
        _fsync_directory(path.parent)
    except OSError:
        raise TraceRecoveryError("staging_cleanup_failed") from None


def _write_journal(path: Path, journal: TraceRecoveryJournal) -> None:
    payload = (
        json.dumps(
            journal.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    if len(payload) > MAX_JOURNAL_BYTES:
        raise TraceRecoveryError("journal_too_large")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise TraceRecoveryError("journal_conflict")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".openmed-trace-journal-{journal.target_fingerprint[7:23]}-",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if _path_exists(temporary):
                temporary.unlink()
    except TraceRecoveryError:
        raise
    except (OSError, ValueError):
        raise TraceRecoveryError("journal_write_failed") from None


def _load_journal(path: Path) -> TraceRecoveryJournal | None:
    if not _path_exists(path):
        return None
    try:
        if path.is_symlink() or not path.is_file():
            raise TraceRecoveryError("journal_conflict")
        if path.stat().st_size > MAX_JOURNAL_BYTES:
            raise TraceRecoveryError("journal_too_large")
        payload = json.loads(path.read_text(encoding="ascii"))
    except TraceRecoveryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise TraceRecoveryError("journal_invalid") from None
    if not isinstance(payload, dict):
        raise TraceRecoveryError("journal_invalid")
    return TraceRecoveryJournal.from_dict(payload)


def _block_transaction(
    journal_path: Path,
    journal: TraceRecoveryJournal,
    phase_hook: TraceRecoveryHook | None,
) -> None:
    blocked = replace(journal, phase="blocked", recovery_decision="blocked")
    _write_journal(journal_path, blocked)
    _call_hook(phase_hook, blocked)


def _validate_journal(journal: TraceRecoveryJournal) -> None:
    if journal.schema_version != TRACE_RECOVERY_SCHEMA_VERSION:
        raise ValueError("unsupported trace recovery schema")
    for value in (
        journal.target_fingerprint,
        journal.input_fingerprint,
        journal.output_fingerprint,
    ):
        if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
            raise ValueError("invalid trace recovery fingerprint")
    if journal.staging_fingerprint is not None and not _DIGEST_RE.fullmatch(
        journal.staging_fingerprint
    ):
        raise ValueError("invalid trace recovery fingerprint")
    if journal.phase not in _PHASES:
        raise ValueError("invalid trace recovery phase")
    if journal.recovery_decision not in _DECISIONS:
        raise ValueError("invalid trace recovery decision")
    if (
        not isinstance(journal.recovery_attempts, int)
        or isinstance(journal.recovery_attempts, bool)
        or journal.recovery_attempts < 0
        or journal.recovery_attempts > 1_000_000
    ):
        raise ValueError("invalid trace recovery attempt count")
    if (
        journal.staging_fingerprint is not None
        and journal.staging_fingerprint != journal.output_fingerprint
    ):
        raise ValueError("staging fingerprint must match output fingerprint")


def _validate_target_ownership(
    journal: TraceRecoveryJournal,
    target_fingerprint: str,
) -> None:
    if journal.target_fingerprint != target_fingerprint:
        raise TraceRecoveryError("journal_target_mismatch")


def _validate_recovery_options(
    decision: str,
    encoding: str,
    max_bytes: int,
    max_recovery_attempts: int,
) -> None:
    if decision not in {"resume", "rollback"}:
        raise TraceRecoveryError("invalid_recovery_decision")
    if not isinstance(encoding, str) or not encoding:
        raise TraceRecoveryError("invalid_encoding")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise TraceRecoveryError("invalid_size_limit")
    if (
        not isinstance(max_recovery_attempts, int)
        or isinstance(max_recovery_attempts, bool)
        or max_recovery_attempts < 1
        or max_recovery_attempts > 1_000_000
    ):
        raise TraceRecoveryError("invalid_recovery_limit")


def _prepare_target(path: str | Path) -> Path:
    try:
        target = Path(path).expanduser().absolute()
        if target.is_symlink():
            raise TraceRecoveryError("symlink_target_unsupported")
        if not target.exists():
            raise TraceRecoveryError("target_missing")
        if not target.is_file():
            raise TraceRecoveryError("target_not_regular")
    except TraceRecoveryError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise TraceRecoveryError("target_invalid") from None
    return target


def _resolve_journal(target: Path, journal_path: str | Path | None) -> Path:
    if journal_path is None:
        return (
            target.parent
            / f".openmed-trace-recovery-{_target_fingerprint(target)[7:]}.json"
        )
    try:
        journal = Path(journal_path).expanduser().absolute()
    except (OSError, RuntimeError, TypeError, ValueError):
        raise TraceRecoveryError("journal_invalid") from None
    if journal == target:
        raise TraceRecoveryError("journal_conflict")
    return journal


def _target_fingerprint(target: Path) -> str:
    return _digest(os.fsencode(str(target)))


def _staging_path(target: Path, output_fingerprint: str) -> Path:
    return target.parent / (
        f".openmed-trace-stage-{_target_fingerprint(target)[7:]}-"
        f"{output_fingerprint[7:]}.bin"
    )


def _read_file(path: Path, max_bytes: int) -> bytes:
    try:
        with path.open("rb") as handle:
            payload = handle.read(max_bytes + 1)
    except OSError:
        raise TraceRecoveryError("target_read_failed") from None
    if len(payload) > max_bytes:
        raise TraceRecoveryError("trace_size_limit")
    return payload


def _file_fingerprint(path: Path, max_bytes: int) -> str:
    return _digest(_read_file(path, max_bytes))


def _file_size(path: Path, max_bytes: int) -> int:
    return len(_read_file(path, max_bytes))


def _file_mode(path: Path) -> int:
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except OSError:
        raise TraceRecoveryError("target_stat_failed") from None


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        raise TraceRecoveryError("artifact_stat_failed") from None
    return True


def _digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _call_hook(hook: TraceRecoveryHook | None, journal: TraceRecoveryJournal) -> None:
    if hook is not None:
        hook(journal)


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability without making the API platform-specific."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


__all__ = [
    "DEFAULT_MAX_RECOVERY_ATTEMPTS",
    "DEFAULT_MAX_TRACE_BYTES",
    "TraceRecoveryError",
    "TraceRecoveryJournal",
    "TraceRedactionResult",
    "recover_trace_redaction",
    "redact_trace_file",
    "redact_trace_in_place",
    "trace_fingerprint",
    "transactional_trace_redact",
]
