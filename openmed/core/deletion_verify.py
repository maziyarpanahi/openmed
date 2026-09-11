"""Deterministic, PHI-free verification for local artifact deletion.

Deletion is deliberately a small, local-only transaction.  Callers provide an
explicit path and the SHA-256 fingerprint they expect for every artifact.  The
operation verifies every target before moving anything, quarantines the
targets on the same filesystem, and records only aggregate counts.  No path,
fingerprint, file content, timestamp, or exception from the filesystem is
included in the evidence or public errors.

The module handles regular files.  Directories, symlinks, hard links, path
aliases, and paths outside ``root`` are rejected so a caller cannot
accidentally delete a different object than the one it verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

_FINGERPRINT_RE = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$", re.IGNORECASE)
_FINGERPRINT_PREFIX = "sha256:"
_EVIDENCE_SCHEMA_VERSION = 1
_EVIDENCE_STATUSES = frozenset({"completed", "rejected", "rolled_back"})
_READ_CHUNK_SIZE = 1024 * 1024
MAX_ARTIFACTS: Final = 128
MAX_PATH_LENGTH: Final = 4_096
MAX_RECOVERY_BYTES: Final = 128 * 1024 * 1024
_USE_MEMORY_RECOVERY = os.name == "nt"

PathLike: TypeAlias = str | os.PathLike[str]
EvidenceStatus: TypeAlias = Literal["completed", "rejected", "rolled_back"]

__all__ = [
    "AmbiguousPathError",
    "ArtifactAccessError",
    "ArtifactNotFoundError",
    "DeletionArtifact",
    "DeletionEvidence",
    "DeletionTransactionError",
    "DeletionVerificationError",
    "EvidenceWriteError",
    "FingerprintMismatchError",
    "InvalidDeletionRequest",
    "UnsafePathError",
    "delete_verified_artifacts",
    "fingerprint_file",
    "verify_and_delete",
]


class DeletionVerificationError(RuntimeError):
    """Base class for safe, PHI-free deletion failures."""

    code = "verification_failed"

    def __init__(self) -> None:
        super().__init__(f"deletion verification failed: {self.code}")


class InvalidDeletionRequest(DeletionVerificationError):
    """Raised when an artifact request is malformed."""

    code = "invalid_request"


class UnsafePathError(DeletionVerificationError):
    """Raised when a target is outside the root or uses a symlink."""

    code = "unsafe_path"


class AmbiguousPathError(DeletionVerificationError):
    """Raised when multiple inputs can identify the same filesystem object."""

    code = "ambiguous_path"


class ArtifactNotFoundError(DeletionVerificationError):
    """Raised when an explicitly requested artifact is not present."""

    code = "artifact_not_found"


class ArtifactAccessError(DeletionVerificationError):
    """Raised when a local artifact cannot be read safely."""

    code = "artifact_access"


class FingerprintMismatchError(DeletionVerificationError):
    """Raised when an artifact does not match its expected fingerprint."""

    code = "fingerprint_mismatch"


class DeletionTransactionError(DeletionVerificationError):
    """Raised when a deletion transaction cannot complete or roll back."""

    code = "transaction_failed"


class EvidenceWriteError(DeletionVerificationError):
    """Raised when counts-only evidence cannot be written."""

    code = "evidence_write_failed"


def _coerce_path(value: Any, *, allow_dot: bool) -> Path:
    try:
        raw_path = os.fspath(value)
    except Exception:
        raise InvalidDeletionRequest from None
    if type(raw_path) is not str or len(raw_path) > MAX_PATH_LENGTH:
        raise InvalidDeletionRequest
    path = Path(raw_path)
    if not allow_dot and path == Path("."):
        raise InvalidDeletionRequest
    return path


def _normalize_fingerprint(value: Any) -> str:
    if type(value) is not str:
        raise InvalidDeletionRequest
    match = _FINGERPRINT_RE.fullmatch(value.strip())
    if match is None:
        raise InvalidDeletionRequest
    return f"{_FINGERPRINT_PREFIX}{match.group(1).lower()}"


@dataclass(frozen=True, slots=True)
class DeletionArtifact:
    """One explicit local file and the fingerprint expected before deletion.

    ``path`` is retained only in memory and is excluded from the object's
    representation.  ``fingerprint`` accepts either a bare SHA-256 hex digest
    or the canonical ``sha256:<digest>`` form; the stored value is canonical.
    """

    path: Path = field(repr=False)
    fingerprint: str = field(repr=False)

    def __post_init__(self) -> None:
        path = _coerce_path(self.path, allow_dot=False)
        fingerprint = _normalize_fingerprint(self.fingerprint)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "fingerprint", fingerprint)


@dataclass(frozen=True, slots=True)
class DeletionEvidence:
    """Counts-only evidence for one deletion attempt.

    The serialized form intentionally contains no paths, fingerprints, file
    names, timestamps, or free-form error text.
    """

    requested_count: int
    verified_count: int
    deleted_count: int
    rolled_back_count: int
    status: EvidenceStatus

    def __post_init__(self) -> None:
        counts = (
            self.requested_count,
            self.verified_count,
            self.deleted_count,
            self.rolled_back_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in counts
        ):
            raise ValueError("deletion evidence counts must be integers")
        if any(value < 0 for value in counts):
            raise ValueError("deletion evidence counts must be non-negative")
        if self.status not in _EVIDENCE_STATUSES:
            raise ValueError("deletion evidence status is invalid")

    @property
    def passed(self) -> bool:
        """Return whether every requested artifact was deleted."""

        return (
            self.status == "completed"
            and self.requested_count == self.verified_count == self.deleted_count
            and self.rolled_back_count == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible counts-only mapping."""

        return {
            "schema_version": _EVIDENCE_SCHEMA_VERSION,
            "requested_count": self.requested_count,
            "verified_count": self.verified_count,
            "deleted_count": self.deleted_count,
            "rolled_back_count": self.rolled_back_count,
            "status": self.status,
        }

    def to_json(self) -> str:
        """Return stable JSON suitable for an evidence file."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"


@dataclass(frozen=True, slots=True)
class _PreparedArtifact:
    path: Path = field(repr=False)
    fingerprint: str = field(repr=False)
    state: os.stat_result = field(repr=False)


@dataclass(frozen=True, slots=True)
class _StagedArtifact:
    original: Path = field(repr=False)
    payload: Path = field(repr=False)
    backup: Path = field(repr=False)
    fingerprint: str = field(repr=False)
    mode: int
    recovery_descriptor: int | None = field(default=None, repr=False)
    recovery_bytes: bytes | None = field(default=None, repr=False)


def _same_file_state(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev == second.st_dev
        and first.st_ino == second.st_ino
        and first.st_size == second.st_size
        and first.st_mtime_ns == second.st_mtime_ns
        and first.st_ctime_ns == second.st_ctime_ns
        and first.st_nlink == second.st_nlink
    )


def _same_identity_and_content(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev == second.st_dev
        and first.st_ino == second.st_ino
        and first.st_size == second.st_size
        and first.st_mtime_ns == second.st_mtime_ns
        and first.st_nlink == second.st_nlink
    )


def _safe_os_error(error_type: type[DeletionVerificationError]) -> None:
    raise error_type from None


def _resolve_root(root: PathLike) -> Path:
    root_path = _coerce_path(root, allow_dot=True)
    try:
        root_stat = os.lstat(root_path)
    except FileNotFoundError:
        _safe_os_error(ArtifactNotFoundError)
    except OSError:
        _safe_os_error(ArtifactAccessError)
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        _safe_os_error(UnsafePathError)
    try:
        resolved = root_path.resolve(strict=True)
    except (OSError, RuntimeError):
        _safe_os_error(ArtifactAccessError)
    if resolved == Path(resolved.anchor):
        _safe_os_error(UnsafePathError)
    try:
        resolved_stat = os.stat(resolved)
    except OSError:
        _safe_os_error(ArtifactAccessError)
    if not stat.S_ISDIR(resolved_stat.st_mode):
        _safe_os_error(UnsafePathError)
    return resolved


def _resolve_candidate(root: Path, path: Path) -> Path:
    if any(part in {".", ".."} for part in path.parts):
        _safe_os_error(AmbiguousPathError)
    candidate = path if path.is_absolute() else root / path
    try:
        candidate.relative_to(root)
    except ValueError:
        _safe_os_error(UnsafePathError)
    if candidate == root:
        _safe_os_error(UnsafePathError)
    return candidate


def _assert_no_symlink_components(root: Path, candidate: Path) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        _safe_os_error(UnsafePathError)

    current = root
    for component in relative.parts:
        current /= component
        try:
            item_stat = os.lstat(current)
        except FileNotFoundError:
            _safe_os_error(ArtifactNotFoundError)
        except OSError:
            _safe_os_error(ArtifactAccessError)
        if stat.S_ISLNK(item_stat.st_mode):
            _safe_os_error(UnsafePathError)


def _open_for_hash(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags | no_follow)
    except FileNotFoundError:
        _safe_os_error(ArtifactNotFoundError)
    except OSError:
        _safe_os_error(ArtifactAccessError)
    raise AssertionError("unreachable")


def _hash_descriptor(descriptor: int, expected_state: os.stat_result) -> str:
    try:
        opened_state = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError:
        _safe_os_error(ArtifactAccessError)
    if not stat.S_ISREG(opened_state.st_mode) or not _same_file_state(
        expected_state, opened_state
    ):
        _safe_os_error(DeletionTransactionError)

    digest = hashlib.sha256()
    while True:
        try:
            chunk = os.read(descriptor, _READ_CHUNK_SIZE)
        except OSError:
            _safe_os_error(ArtifactAccessError)
        if not chunk:
            break
        digest.update(chunk)

    try:
        final_state = os.fstat(descriptor)
    except OSError:
        _safe_os_error(ArtifactAccessError)
    if not _same_file_state(opened_state, final_state):
        _safe_os_error(DeletionTransactionError)
    return f"{_FINGERPRINT_PREFIX}{digest.hexdigest()}"


def _hash_open_file(path: Path, expected_state: os.stat_result) -> str:
    descriptor = _open_for_hash(path)
    try:
        return _hash_descriptor(descriptor, expected_state)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def fingerprint_file(path: PathLike) -> str:
    """Return the canonical SHA-256 fingerprint of one regular local file.

    Symlinks, directories, hard links, and unreadable or changing files are
    rejected.  The helper never makes a network request.
    """

    file_path = _coerce_path(path, allow_dot=False)
    if file_path.is_symlink():
        _safe_os_error(UnsafePathError)
    try:
        resolved = file_path.resolve(strict=True)
        file_stat = os.lstat(resolved)
    except FileNotFoundError:
        _safe_os_error(ArtifactNotFoundError)
    except (OSError, RuntimeError):
        _safe_os_error(ArtifactAccessError)
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        _safe_os_error(UnsafePathError)
    if file_stat.st_nlink != 1:
        _safe_os_error(AmbiguousPathError)
    return _hash_open_file(resolved, file_stat)


def _coerce_artifact_item(item: Any) -> DeletionArtifact:
    if isinstance(item, DeletionArtifact):
        return item
    if type(item) is dict:
        if set(item) != {"path", "fingerprint"}:
            raise InvalidDeletionRequest
        return DeletionArtifact(item["path"], item["fingerprint"])
    if type(item) in {tuple, list} and len(item) == 2:
        return DeletionArtifact(item[0], item[1])
    raise InvalidDeletionRequest


def _coerce_artifacts(artifacts: Any) -> list[DeletionArtifact]:
    if isinstance(artifacts, DeletionArtifact):
        return [artifacts]
    if type(artifacts) is dict:
        if len(artifacts) > MAX_ARTIFACTS:
            raise InvalidDeletionRequest
        if set(artifacts) == {"path", "fingerprint"}:
            return [_coerce_artifact_item(artifacts)]
        try:
            return [
                DeletionArtifact(path, fingerprint)
                for path, fingerprint in artifacts.items()
            ]
        except (InvalidDeletionRequest, ValueError, TypeError):
            raise InvalidDeletionRequest from None
    if isinstance(artifacts, (str, bytes, os.PathLike)):
        raise InvalidDeletionRequest
    try:
        iterator = iter(artifacts)
    except Exception:
        raise InvalidDeletionRequest from None
    result: list[DeletionArtifact] = []
    try:
        for item in iterator:
            if len(result) >= MAX_ARTIFACTS:
                raise InvalidDeletionRequest
            result.append(_coerce_artifact_item(item))
    except Exception:
        raise InvalidDeletionRequest from None
    return result


def _prepare_artifact(root: Path, artifact: DeletionArtifact) -> _PreparedArtifact:
    candidate = _resolve_candidate(root, artifact.path)
    _assert_no_symlink_components(root, candidate)
    try:
        file_stat = os.lstat(candidate)
    except FileNotFoundError:
        _safe_os_error(ArtifactNotFoundError)
    except OSError:
        _safe_os_error(ArtifactAccessError)
    if stat.S_ISLNK(file_stat.st_mode):
        _safe_os_error(UnsafePathError)
    if not stat.S_ISREG(file_stat.st_mode):
        _safe_os_error(UnsafePathError)
    if file_stat.st_nlink != 1:
        _safe_os_error(AmbiguousPathError)
    actual = _hash_open_file(candidate, file_stat)
    if actual != artifact.fingerprint:
        _safe_os_error(FingerprintMismatchError)
    return _PreparedArtifact(
        path=candidate,
        fingerprint=artifact.fingerprint,
        state=file_stat,
    )


def _prepare_all(
    root: Path,
    artifacts: list[DeletionArtifact],
    prepared: list[_PreparedArtifact],
) -> None:
    seen_paths: set[Path] = set()
    seen_objects: set[tuple[int, int]] = set()
    candidates = sorted(
        artifacts,
        key=lambda item: (item.path.is_absolute(), item.path.as_posix()),
    )
    for artifact in candidates:
        candidate = _resolve_candidate(root, artifact.path)
        if candidate in seen_paths:
            _safe_os_error(AmbiguousPathError)
        seen_paths.add(candidate)
        item = _prepare_artifact(root, artifact)
        identity = (item.state.st_dev, item.state.st_ino)
        if identity in seen_objects:
            _safe_os_error(AmbiguousPathError)
        seen_objects.add(identity)
        prepared.append(item)
    prepared.sort(key=lambda item: item.path.relative_to(root).as_posix())


def _evidence_target(path: PathLike | None) -> Path | None:
    if path is None:
        return None
    target = _coerce_path(path, allow_dot=False)
    try:
        if target.is_symlink() or target.exists() and target.is_dir():
            _safe_os_error(UnsafePathError)
        target.parent.mkdir(parents=True, exist_ok=True)
    except DeletionVerificationError:
        raise
    except OSError:
        _safe_os_error(EvidenceWriteError)
    return target


def _check_evidence_collision(
    target: Path | None, prepared: Iterable[_PreparedArtifact]
) -> None:
    if target is None:
        return
    try:
        target_resolved = target.resolve(strict=False)
    except (OSError, RuntimeError):
        _safe_os_error(EvidenceWriteError)
    if any(item.path == target_resolved for item in prepared):
        _safe_os_error(AmbiguousPathError)


def _check_requested_evidence_collision(
    root: Path,
    target: Path | None,
    artifacts: Iterable[DeletionArtifact],
) -> None:
    if target is None:
        return
    try:
        target_resolved = target.resolve(strict=False)
        target_state = os.lstat(target)
    except FileNotFoundError:
        target_state = None
    except (OSError, RuntimeError):
        _safe_os_error(EvidenceWriteError)

    for artifact in artifacts:
        candidate = _resolve_candidate(root, artifact.path)
        if candidate == target_resolved:
            _safe_os_error(AmbiguousPathError)
        if target_state is None:
            continue
        try:
            candidate_state = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError:
            _safe_os_error(ArtifactAccessError)
        if (candidate_state.st_dev, candidate_state.st_ino) == (
            target_state.st_dev,
            target_state.st_ino,
        ):
            _safe_os_error(AmbiguousPathError)


def _write_evidence_atomic(target: Path, evidence: DeletionEvidence) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=target.parent,
            prefix=".openmed-deletion-evidence-",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(evidence.to_json())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except (OSError, TypeError, ValueError):
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        _safe_os_error(EvidenceWriteError)


def _prepare_evidence_temporary(target: Path, evidence: DeletionEvidence) -> Path:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=target.parent,
            prefix=".openmed-deletion-evidence-",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(evidence.to_json())
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, TypeError, ValueError):
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        _safe_os_error(EvidenceWriteError)
    if temporary is None:
        _safe_os_error(EvidenceWriteError)
    return temporary


def _remove_path(path: Path) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    except OSError:
        _safe_os_error(DeletionTransactionError)


def _remove_directory(path: Path) -> None:
    try:
        os.rmdir(path)
    except FileNotFoundError:
        return
    except OSError:
        _safe_os_error(DeletionTransactionError)


def _make_transaction_dirs(root: Path) -> tuple[Path, Path]:
    try:
        transaction = Path(tempfile.mkdtemp(prefix=".openmed-deletion-", dir=root))
        payload = transaction / "payload"
        backup = transaction / "backup"
        payload.mkdir()
        backup.mkdir()
        os.chmod(transaction, 0o700)
        os.chmod(payload, 0o700)
        os.chmod(backup, 0o700)
    except (OSError, TypeError, ValueError):
        if "transaction" in locals():
            try:
                _remove_directory(transaction / "payload")
                _remove_directory(transaction / "backup")
                _remove_directory(transaction)
            except DeletionVerificationError:
                pass
        _safe_os_error(DeletionTransactionError)
    return payload, backup


def _copy_descriptor_to_path(descriptor: int, target: Path, mode: int) -> None:
    output_descriptor: int | None = None
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        output_descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_SIZE)
            if not chunk:
                break
            offset = 0
            while offset < len(chunk):
                written = os.write(output_descriptor, chunk[offset:])
                if written <= 0:
                    _safe_os_error(DeletionTransactionError)
                offset += written
        os.chmod(target, mode)
        os.fsync(output_descriptor)
    except DeletionVerificationError:
        raise
    except OSError:
        _safe_os_error(DeletionTransactionError)
    finally:
        if output_descriptor is not None:
            try:
                os.close(output_descriptor)
            except OSError:
                pass


def _stage_all(
    prepared: list[_PreparedArtifact],
    payload: Path,
    backup: Path,
    staged: list[_StagedArtifact],
) -> None:
    for index, item in enumerate(prepared):
        payload_path = payload / f"{index:08d}"
        backup_path = backup / f"{index:08d}"
        try:
            current = os.lstat(item.path)
        except FileNotFoundError:
            _safe_os_error(ArtifactNotFoundError)
        except OSError:
            _safe_os_error(ArtifactAccessError)
        if not stat.S_ISREG(current.st_mode) or not _same_file_state(
            item.state, current
        ):
            _safe_os_error(DeletionTransactionError)
        try:
            os.replace(item.path, payload_path)
            staged_item = _StagedArtifact(
                original=item.path,
                payload=payload_path,
                backup=backup_path,
                fingerprint=item.fingerprint,
                mode=stat.S_IMODE(current.st_mode),
            )
            staged.append(staged_item)
            try:
                staged_state = os.lstat(payload_path)
            except OSError:
                _safe_os_error(DeletionTransactionError)
            if (
                not stat.S_ISREG(staged_state.st_mode)
                or staged_state.st_nlink != 1
                or not _same_identity_and_content(current, staged_state)
            ):
                _safe_os_error(DeletionTransactionError)

            payload_descriptor = _open_for_hash(payload_path)
            try:
                if (
                    _hash_descriptor(payload_descriptor, staged_state)
                    != item.fingerprint
                ):
                    _safe_os_error(FingerprintMismatchError)
                _copy_descriptor_to_path(
                    payload_descriptor,
                    backup_path,
                    staged_item.mode,
                )
            finally:
                try:
                    os.close(payload_descriptor)
                except OSError:
                    pass

            backup_state = os.lstat(backup_path)
            recovery_descriptor = _open_for_hash(backup_path)
            object.__setattr__(
                staged_item,
                "recovery_descriptor",
                recovery_descriptor,
            )
            if (
                backup_state.st_nlink != 1
                or _hash_descriptor(recovery_descriptor, backup_state)
                != item.fingerprint
            ):
                _safe_os_error(DeletionTransactionError)
            if _USE_MEMORY_RECOVERY:
                # Windows cannot unlink or rename the backup while the CRT
                # descriptor is open. Retain a bounded, verified memory copy
                # through cleanup and evidence publication instead.
                remaining = MAX_RECOVERY_BYTES - sum(
                    len(previous.recovery_bytes or b"") for previous in staged
                )
                if backup_state.st_size > remaining:
                    _safe_os_error(DeletionTransactionError)
                os.lseek(recovery_descriptor, 0, os.SEEK_SET)
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = os.read(
                        recovery_descriptor,
                        min(_READ_CHUNK_SIZE, remaining - total + 1),
                    )
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > remaining:
                        _safe_os_error(DeletionTransactionError)
                    chunks.append(chunk)
                recovery_bytes = b"".join(chunks)
                if (
                    f"sha256:{hashlib.sha256(recovery_bytes).hexdigest()}"
                    != item.fingerprint
                ):
                    _safe_os_error(DeletionTransactionError)
                object.__setattr__(staged_item, "recovery_bytes", recovery_bytes)
                os.close(recovery_descriptor)
                object.__setattr__(staged_item, "recovery_descriptor", None)
        except OSError:
            _safe_os_error(DeletionTransactionError)


def _path_state(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError:
        _safe_os_error(DeletionTransactionError)
    raise AssertionError("unreachable")


def _path_matches_fingerprint(path: Path, fingerprint: str) -> bool:
    state = _path_state(path)
    if state is None or not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
        return False
    return _hash_open_file(path, state) == fingerprint


def _verify_staged_before_commit(staged: list[_StagedArtifact]) -> None:
    for item in staged:
        payload_state = _path_state(item.payload)
        backup_state = _path_state(item.backup)
        descriptor = item.recovery_descriptor
        if (
            payload_state is None
            or backup_state is None
            or (descriptor is None and item.recovery_bytes is None)
            or payload_state.st_nlink != 1
            or backup_state.st_nlink != 1
            or (payload_state.st_dev, payload_state.st_ino)
            == (backup_state.st_dev, backup_state.st_ino)
            or _hash_open_file(item.payload, payload_state) != item.fingerprint
            or (
                _hash_descriptor(descriptor, backup_state)
                if descriptor is not None
                else f"sha256:{hashlib.sha256(item.recovery_bytes or b'').hexdigest()}"
            )
            != item.fingerprint
        ):
            _safe_os_error(DeletionTransactionError)


def _restore_from_descriptor(item: _StagedArtifact) -> None:
    descriptor = item.recovery_descriptor
    if descriptor is None and item.recovery_bytes is None:
        _safe_os_error(DeletionTransactionError)

    temporary: Path | None = None
    try:
        if descriptor is not None:
            descriptor_state = os.fstat(descriptor)
            if _hash_descriptor(descriptor, descriptor_state) != item.fingerprint:
                _safe_os_error(DeletionTransactionError)
            os.lseek(descriptor, 0, os.SEEK_SET)
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=item.original.parent,
            prefix=".openmed-deletion-restore-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            if descriptor is None:
                handle.write(item.recovery_bytes or b"")
            else:
                while True:
                    chunk = os.read(descriptor, _READ_CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
            os.chmod(temporary, item.mode)
        restored_state = os.lstat(temporary)
        if _hash_open_file(temporary, restored_state) != item.fingerprint:
            _safe_os_error(DeletionTransactionError)
        os.replace(temporary, item.original)
        temporary = None
    except DeletionVerificationError:
        raise
    except OSError:
        _safe_os_error(DeletionTransactionError)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _restore_staged(staged: list[_StagedArtifact]) -> int:
    restored = 0
    try:
        for item in reversed(staged):
            if _path_state(item.original) is not None:
                _safe_os_error(DeletionTransactionError)

            if _path_matches_fingerprint(item.payload, item.fingerprint):
                os.replace(item.payload, item.original)
            elif _path_matches_fingerprint(item.backup, item.fingerprint):
                os.replace(item.backup, item.original)
            else:
                _restore_from_descriptor(item)

            if _path_state(item.payload) is not None:
                _remove_path(item.payload)
            if _path_state(item.backup) is not None:
                _remove_path(item.backup)
            restored_state = _path_state(item.original)
            if (
                restored_state is None
                or restored_state.st_nlink != 1
                or _hash_open_file(item.original, restored_state) != item.fingerprint
            ):
                _safe_os_error(DeletionTransactionError)
            restored += 1
    except DeletionVerificationError:
        raise
    except OSError:
        _safe_os_error(DeletionTransactionError)
    return restored


def _close_recovery_descriptors(staged: list[_StagedArtifact]) -> None:
    for item in staged:
        object.__setattr__(item, "recovery_bytes", None)
        descriptor = item.recovery_descriptor
        if descriptor is None:
            continue
        try:
            os.close(descriptor)
        except OSError:
            pass
        object.__setattr__(item, "recovery_descriptor", None)


def _cleanup_transaction(
    staged: list[_StagedArtifact], payload: Path, backup: Path
) -> None:
    for item in staged:
        if item.payload.exists():
            _remove_path(item.payload)
        if item.backup.exists():
            _remove_path(item.backup)
    _remove_directory(payload)
    _remove_directory(backup)
    transaction = payload.parent
    _remove_directory(transaction)


def _rollback_transaction(
    staged: list[_StagedArtifact],
    payload: Path,
    backup: Path,
) -> int:
    try:
        if _USE_MEMORY_RECOVERY:
            # A failed memory snapshot can leave a descriptor open. Payloads
            # and backups are still present at that point, so release it first.
            for item in staged:
                if item.recovery_descriptor is not None:
                    os.close(item.recovery_descriptor)
                    object.__setattr__(item, "recovery_descriptor", None)
        restored = _restore_staged(staged)
        _remove_directory(payload)
        _remove_directory(backup)
        _remove_directory(payload.parent)
    except DeletionVerificationError:
        raise DeletionTransactionError from None
    finally:
        _close_recovery_descriptors(staged)
    if restored != len(staged):
        _safe_os_error(DeletionTransactionError)
    return restored


def _delete_payloads(staged: list[_StagedArtifact]) -> None:
    for item in staged:
        _remove_path(item.payload)


def _write_rejection_evidence(
    target: Path | None,
    *,
    requested_count: int,
    verified_count: int,
) -> None:
    if target is None:
        return
    _write_evidence_atomic(
        target,
        DeletionEvidence(
            requested_count=requested_count,
            verified_count=verified_count,
            deleted_count=0,
            rolled_back_count=0,
            status="rejected",
        ),
    )


def delete_verified_artifacts(
    root: PathLike,
    artifacts: Mapping[PathLike, str]
    | DeletionArtifact
    | Iterable[DeletionArtifact | Mapping[str, Any] | tuple[Any, Any] | list[Any]],
    *,
    evidence_path: PathLike | None = None,
) -> DeletionEvidence:
    """Verify and delete files under ``root`` with process-local rollback.

    Args:
        root: Existing directory that bounds all artifact paths.
        artifacts: A path-to-fingerprint mapping, one :class:`DeletionArtifact`,
            or an iterable of artifact records/``(path, fingerprint)`` pairs.
        evidence_path: Optional local JSON path for counts-only evidence.

    Returns:
        A deterministic :class:`DeletionEvidence` record.  A successful empty
        request is a completed no-op with all counts set to zero.

    Raises:
        DeletionVerificationError: If any path, fingerprint, or transaction
            safety check fails.  The exception text contains only a stable
            failure code.
    """

    root_path = _resolve_root(root)
    requested = _coerce_artifacts(artifacts)
    evidence_target = _evidence_target(evidence_path)
    _check_requested_evidence_collision(root_path, evidence_target, requested)

    prepared: list[_PreparedArtifact] = []
    try:
        _prepare_all(root_path, requested, prepared)
    except DeletionVerificationError:
        _write_rejection_evidence(
            evidence_target,
            requested_count=len(requested),
            verified_count=len(prepared),
        )
        raise
    _check_evidence_collision(evidence_target, prepared)

    completed_evidence = DeletionEvidence(
        requested_count=len(prepared),
        verified_count=len(prepared),
        deleted_count=len(prepared),
        rolled_back_count=0,
        status="completed",
    )
    if not prepared:
        if evidence_target is not None:
            _write_evidence_atomic(evidence_target, completed_evidence)
        return completed_evidence

    evidence_temporary = (
        _prepare_evidence_temporary(evidence_target, completed_evidence)
        if evidence_target is not None
        else None
    )
    try:
        payload, backup = _make_transaction_dirs(root_path)
    except DeletionVerificationError:
        if evidence_temporary is not None:
            try:
                evidence_temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    staged: list[_StagedArtifact] = []
    try:
        _stage_all(prepared, payload, backup, staged)
        _verify_staged_before_commit(staged)
        _delete_payloads(staged)
        _cleanup_transaction(staged, payload, backup)
        if evidence_temporary is not None and evidence_target is not None:
            try:
                os.replace(evidence_temporary, evidence_target)
            except OSError:
                _safe_os_error(EvidenceWriteError)
            evidence_temporary = None
        _close_recovery_descriptors(staged)
    except DeletionVerificationError:
        if evidence_temporary is not None:
            try:
                evidence_temporary.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            rolled_back = _rollback_transaction(staged, payload, backup)
        except DeletionVerificationError:
            raise DeletionTransactionError from None
        rollback_evidence = DeletionEvidence(
            requested_count=len(prepared),
            verified_count=len(prepared),
            deleted_count=0,
            rolled_back_count=rolled_back,
            status="rolled_back",
        )
        if evidence_target is not None:
            _write_evidence_atomic(evidence_target, rollback_evidence)
        raise
    except OSError:
        if evidence_temporary is not None:
            try:
                evidence_temporary.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            rolled_back = _rollback_transaction(staged, payload, backup)
        except DeletionVerificationError:
            raise DeletionTransactionError from None
        if evidence_target is not None:
            _write_evidence_atomic(
                evidence_target,
                DeletionEvidence(
                    requested_count=len(prepared),
                    verified_count=len(prepared),
                    deleted_count=0,
                    rolled_back_count=rolled_back,
                    status="rolled_back",
                ),
            )
        raise DeletionTransactionError from None
    return completed_evidence


def verify_and_delete(
    root: PathLike,
    artifacts: Mapping[PathLike, str]
    | DeletionArtifact
    | Iterable[DeletionArtifact | Mapping[str, Any] | tuple[Any, Any] | list[Any]],
    *,
    evidence_path: PathLike | None = None,
) -> DeletionEvidence:
    """Compatibility alias for :func:`delete_verified_artifacts`."""

    return delete_verified_artifacts(
        root,
        artifacts,
        evidence_path=evidence_path,
    )
