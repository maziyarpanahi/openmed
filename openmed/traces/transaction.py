"""Transactional, local-only redaction of trace files.

The module deliberately does not select or download a model.  Callers supply
the text transformation and, when useful, a value-free validation callback.
The source file is read once, the transformed bytes are written and synced to
a sibling temporary file, and the directory entry is replaced only after all
pre-commit checks succeed.
"""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

PathLike: TypeAlias = str | os.PathLike[str]
TextRedactor: TypeAlias = Callable[[str], str]
TextValidator: TypeAlias = Callable[[str], bool]

DEFAULT_BACKUP_SUFFIX = ".bak"
DEFAULT_ENCODING = "utf-8"
TEMPORARY_FILE_PREFIX = ".openmed-transaction-"
MAX_BACKUP_COLLISIONS = 1000

__all__ = [
    "DEFAULT_BACKUP_SUFFIX",
    "DEFAULT_ENCODING",
    "InPlaceRedactionResult",
    "MAX_BACKUP_COLLISIONS",
    "PathLike",
    "RedactionTransaction",
    "TEMPORARY_FILE_PREFIX",
    "TextRedactor",
    "TextValidator",
    "TransactionConflictError",
    "TransactionError",
    "TransactionRedactionError",
    "TransactionReadError",
    "TransactionResult",
    "TransactionValidationError",
    "TransactionWriteError",
    "redact_in_place",
    "redact_trace_in_place",
    "transactional_redact",
]


class TransactionError(RuntimeError):
    """Base error for a failed redaction transaction.

    Error messages are intentionally value-free.  A redactor or validator may
    receive sensitive text, but that text is never copied into an exception.
    """


class TransactionReadError(TransactionError):
    """The source could not be read as the requested text encoding."""


class TransactionRedactionError(TransactionError):
    """The injected text redactor failed or returned an invalid value."""


class TransactionValidationError(TransactionError):
    """The candidate output did not pass the caller's validation callback."""


class TransactionConflictError(TransactionError):
    """The source changed while the candidate output was being prepared."""


class TransactionWriteError(TransactionError):
    """The candidate or backup could not be safely persisted."""


@dataclass(frozen=True, slots=True)
class TransactionResult:
    """PHI-free metadata describing one completed file transaction."""

    path: Path = field(repr=False)
    backup_path: Path | None = field(repr=False)
    changed: bool
    original_bytes: int
    replacement_bytes: int

    @property
    def target_path(self) -> Path:
        """Return the target path under the explicit ``target_path`` name."""

        return self.path

    @property
    def backup(self) -> Path | None:
        """Return the created backup path, if backups were enabled."""

        return self.backup_path

    @property
    def bytes_read(self) -> int:
        """Return the number of source bytes read."""

        return self.original_bytes

    @property
    def bytes_written(self) -> int:
        """Return the number of replacement bytes written."""

        return self.replacement_bytes

    def to_dict(self) -> dict[str, int | bool]:
        """Return a report-safe summary without paths or source values."""

        return {
            "changed": self.changed,
            "backup_created": self.backup_path is not None,
            "original_bytes": self.original_bytes,
            "replacement_bytes": self.replacement_bytes,
        }


RedactionTransaction = TransactionResult
InPlaceRedactionResult = TransactionResult

_FileState: TypeAlias = tuple[int, int, int, int, int, int, int]


def transactional_redact(
    path: PathLike,
    redactor: TextRedactor | None = None,
    *,
    transform: TextRedactor | None = None,
    validator: TextValidator | None = None,
    validate: TextValidator | None = None,
    backup: bool = True,
    backup_path: PathLike | None = None,
    preserve_permissions: bool = True,
    preserve_timestamps: bool = True,
    preserve_metadata: bool | None = None,
    encoding: str = DEFAULT_ENCODING,
) -> TransactionResult:
    """Redact one local text file with an atomic in-place commit.

    Args:
        path: Existing regular file to replace.
        redactor: Deterministic callable receiving the complete decoded file
            and returning its replacement.  The callable is local and is
            never used to load a model or make a network request.
        transform: Compatibility alias for ``redactor``.
        validator: Optional callable receiving only the candidate text.  A
            false result or an exception rejects the transaction.
        validate: Compatibility alias for ``validator``.
        backup: Create an exclusive sibling backup before replacing the file.
        backup_path: Optional preferred backup name.  Existing paths are
            retained and a numeric suffix is selected instead.
        preserve_permissions: Copy the source permission bits to the
            replacement and backup.
        preserve_timestamps: Copy the source access and modification times to
            the replacement and backup.
        preserve_metadata: Set both preservation options at once.
        encoding: Text encoding used for the source and candidate.

    Returns:
        Value-free sizes and paths for the committed replacement.  An
        unchanged candidate is reported without creating a needless backup.

    Raises:
        TransactionError: If reading, redaction, validation, source
            consistency, backup creation, or atomic replacement fails.  The
            target remains unchanged for every failure before the commit.

    The transformation is prepared completely before ``os.replace``.  A
    sibling temporary file is flushed and synced, and backups use exclusive
    creation so an existing backup is never overwritten.
    """

    target = _coerce_path(path)
    resolved_redactor = _resolve_redactor(redactor, transform)
    resolved_validator = _resolve_validator(validator, validate)
    _validate_options(
        backup=backup,
        backup_path=backup_path,
        preserve_permissions=preserve_permissions,
        preserve_timestamps=preserve_timestamps,
        preserve_metadata=preserve_metadata,
        encoding=encoding,
    )
    if preserve_metadata is not None:
        preserve_permissions = preserve_metadata
        preserve_timestamps = preserve_metadata

    preferred_backup = None if backup_path is None else _coerce_path(backup_path)
    if preferred_backup is not None and _same_path(target, preferred_backup):
        raise ValueError("backup path must differ from the target")

    source_state = _read_source_state(target)
    try:
        original_bytes = target.read_bytes()
        original_text = original_bytes.decode(encoding)
    except (LookupError, OSError, UnicodeError):
        raise TransactionReadError("source could not be read") from None

    try:
        replacement_text = resolved_redactor(original_text)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise TransactionRedactionError("redactor failed") from None
    if not isinstance(replacement_text, str):
        raise TransactionRedactionError("redactor must return text")

    try:
        replacement_bytes = replacement_text.encode(encoding)
    except (LookupError, UnicodeError):
        raise TransactionRedactionError("redactor returned unencodable text") from None

    if resolved_validator is not None:
        try:
            valid = resolved_validator(replacement_text)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise TransactionValidationError("candidate validation failed") from None
        if not valid:
            raise TransactionValidationError("candidate validation failed")

    if replacement_bytes == original_bytes:
        _assert_source_unchanged(target, source_state)
        return TransactionResult(
            path=target,
            backup_path=None,
            changed=False,
            original_bytes=len(original_bytes),
            replacement_bytes=len(replacement_bytes),
        )

    temporary_path: Path | None = None
    created_backup: Path | None = None
    committed = False
    mode = stat.S_IMODE(source_state[5]) if preserve_permissions else None
    timestamps = (source_state[6], source_state[3]) if preserve_timestamps else None

    try:
        _assert_source_unchanged(target, source_state)
        temporary_path, temporary_descriptor = _create_temporary_file(target)
        _write_payload(
            temporary_path,
            replacement_bytes,
            file_descriptor=temporary_descriptor,
            mode=mode,
            timestamps=timestamps,
        )
        _assert_source_unchanged(target, source_state)

        if backup:
            created_backup, backup_fd = _reserve_backup_path(
                target,
                preferred_backup,
            )
            try:
                _write_payload(
                    created_backup,
                    original_bytes,
                    file_descriptor=backup_fd,
                    mode=mode,
                    timestamps=timestamps,
                )
            except BaseException:
                _remove_quietly(created_backup)
                created_backup = None
                raise

        _assert_source_unchanged(target, source_state)
        os.replace(os.fspath(temporary_path), os.fspath(target))
        temporary_path = None
        committed = True
    except TransactionError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except OSError:
        raise TransactionWriteError("transaction could not be committed") from None
    except BaseException:
        raise TransactionWriteError("transaction could not be committed") from None
    finally:
        if not committed:
            _remove_quietly(temporary_path)
            _remove_quietly(created_backup)

    return TransactionResult(
        path=target,
        backup_path=created_backup,
        changed=True,
        original_bytes=len(original_bytes),
        replacement_bytes=len(replacement_bytes),
    )


def _coerce_path(value: PathLike) -> Path:
    try:
        return Path(value)
    except Exception:  # noqa: BLE001 - path-like errors may contain PHI
        raise ValueError("path is invalid") from None


def _resolve_redactor(
    redactor: TextRedactor | None,
    transform: TextRedactor | None,
) -> TextRedactor:
    if redactor is not None and transform is not None:
        raise TypeError("pass either redactor or transform")
    resolved = redactor if redactor is not None else transform
    if resolved is None or not callable(resolved):
        raise TypeError("a callable redactor is required")
    return resolved


def _resolve_validator(
    validator: TextValidator | None,
    validate: TextValidator | None,
) -> TextValidator | None:
    if validator is not None and validate is not None:
        raise TypeError("pass either validator or validate")
    resolved = validator if validator is not None else validate
    if resolved is not None and not callable(resolved):
        raise TypeError("validator must be callable")
    return resolved


def _validate_options(
    *,
    backup: bool,
    backup_path: PathLike | None,
    preserve_permissions: bool,
    preserve_timestamps: bool,
    preserve_metadata: bool | None,
    encoding: str,
) -> None:
    if not isinstance(backup, bool):
        raise TypeError("backup must be a boolean")
    if backup_path is not None and not backup:
        raise ValueError("backup_path requires backup=True")
    for value, name in (
        (preserve_permissions, "preserve_permissions"),
        (preserve_timestamps, "preserve_timestamps"),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean")
    if preserve_metadata is not None and not isinstance(preserve_metadata, bool):
        raise TypeError("preserve_metadata must be a boolean")
    if not isinstance(encoding, str) or not encoding:
        raise ValueError("encoding must be a non-empty string")


def _read_source_state(path: Path) -> _FileState:
    try:
        source_stat = os.stat(path, follow_symlinks=False)
    except OSError:
        raise TransactionReadError("source could not be read") from None
    if not stat.S_ISREG(source_stat.st_mode):
        raise TransactionReadError("source must be a regular file")
    return _state_from_stat(source_stat)


def _state_from_stat(source_stat: os.stat_result) -> _FileState:
    return (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_ctime_ns,
        source_stat.st_mode,
        source_stat.st_atime_ns,
    )


def _assert_source_unchanged(path: Path, expected: _FileState) -> None:
    try:
        current_stat = os.stat(path, follow_symlinks=False)
    except OSError:
        raise TransactionConflictError("source changed during transaction") from None
    if not stat.S_ISREG(current_stat.st_mode):
        raise TransactionConflictError("source changed during transaction")
    current = _state_from_stat(current_stat)
    # Reading a file may update atime.  It is preserved on the replacement,
    # but it must not make an otherwise unchanged source look like a conflict.
    if current[:6] != expected[:6]:
        raise TransactionConflictError("source changed during transaction")


def _create_temporary_file(target: Path) -> tuple[Path, int]:
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=TEMPORARY_FILE_PREFIX,
            suffix=".tmp",
            dir=os.fspath(target.parent),
        )
    except OSError:
        raise TransactionWriteError("temporary file could not be created") from None
    return Path(temporary_name), descriptor


def _write_payload(
    path: Path,
    payload: bytes,
    *,
    file_descriptor: int | None = None,
    mode: int | None,
    timestamps: tuple[int, int] | None,
) -> None:
    descriptor = file_descriptor
    try:
        if descriptor is None:
            descriptor = os.open(
                os.fspath(path),
                os.O_WRONLY | os.O_TRUNC | getattr(os, "O_BINARY", 0),
            )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            if mode is not None:
                _chmod_descriptor(handle.fileno(), path, mode)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass

    if timestamps is not None:
        if os.utime in os.supports_follow_symlinks:
            os.utime(path, ns=timestamps, follow_symlinks=False)
        else:
            os.utime(path, ns=timestamps)


def _chmod_descriptor(descriptor: int, path: Path, mode: int) -> None:
    if hasattr(os, "fchmod"):
        os.fchmod(descriptor, mode)
    elif os.chmod in os.supports_follow_symlinks:
        os.chmod(path, mode, follow_symlinks=False)
    else:
        os.chmod(path, mode)


def _reserve_backup_path(
    target: Path,
    preferred: Path | None,
) -> tuple[Path, int]:
    base = preferred or target.with_name(target.name + DEFAULT_BACKUP_SUFFIX)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for index in range(MAX_BACKUP_COLLISIONS):
        candidate = base if index == 0 else base.with_name(f"{base.name}.{index}")
        try:
            descriptor = os.open(os.fspath(candidate), flags, 0o600)
        except FileExistsError:
            continue
        except OSError:
            raise TransactionWriteError("backup could not be created") from None
        return candidate, descriptor
    raise TransactionWriteError("backup name collisions exceeded the limit")


def _same_path(first: Path, second: Path) -> bool:
    try:
        first_name = os.path.normcase(os.path.abspath(os.fspath(first)))
        second_name = os.path.normcase(os.path.abspath(os.fspath(second)))
    except (OSError, TypeError, ValueError):
        return first == second
    return first_name == second_name


def _remove_quietly(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except (FileNotFoundError, OSError):
        pass


redact_in_place = transactional_redact
redact_trace_in_place = transactional_redact
