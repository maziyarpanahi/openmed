"""Local-only quota and eviction policy for model cache artifacts.

The policy deliberately manages an allow-list rather than scanning and deleting
an arbitrary cache directory.  Callers register the files or directories they
own, and the policy records their content digest, size, access order, and pin
state in a small cache-local manifest.  Reuse verifies the recorded digest
before updating the access order.  Eviction only considers registered,
unpinned artifacts and uses the recorded access order with a stable path tie
breaker.

No operation in this module imports a model hub client or performs network I/O.
The manifest is local bookkeeping; reports and exceptions expose a hash of the
managed relative path instead of the path itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any, Final

logger = logging.getLogger(__name__)

CACHE_POLICY_MANIFEST = ".openmed-cache-policy.json"
CACHE_POLICY_SCHEMA = "openmed.model_cache_policy.v1"
_HASH_CHUNK_SIZE: Final = 1024 * 1024
_SHA256_PATTERN: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_MISSING_DIGEST: Final = "sha256:<missing>"


class ModelCachePolicyError(RuntimeError):
    """Base error for cache ownership, integrity, and eviction failures."""


class CacheOwnershipError(ModelCachePolicyError):
    """Raised when an operation targets an artifact not owned by the policy."""


class CacheIntegrityError(ModelCachePolicyError):
    """Raised when a cached artifact does not match its recorded digest."""

    def __init__(
        self,
        path_hash: str,
        *,
        expected_sha256: str,
        actual_sha256: str,
        reason: str = "artifact digest mismatch",
    ) -> None:
        self.path_hash = path_hash
        self.expected_sha256 = expected_sha256
        self.actual_sha256 = actual_sha256
        self.reason = reason
        super().__init__(
            f"model cache artifact verification failed for {path_hash}: {reason}; "
            f"expected {expected_sha256}, actual {actual_sha256}"
        )


@dataclass(frozen=True)
class CacheArtifact:
    """PHI-free summary of one registered cache artifact.

    ``path_hash`` is a stable hash of the artifact's cache-relative path.  The
    raw path is intentionally not part of this public object.
    """

    path_hash: str
    size_bytes: int
    sha256: str
    last_accessed_ns: int
    pinned: bool
    exists: bool = True
    verified: bool = False

    @property
    def checksum(self) -> str:
        """Return the recorded content checksum."""

        return self.sha256

    def to_dict(self) -> dict[str, Any]:
        """Return a report-safe representation without a raw path."""

        return {
            "path_hash": self.path_hash,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "last_accessed_ns": self.last_accessed_ns,
            "pinned": self.pinned,
            "exists": self.exists,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class EvictionCandidate:
    """PHI-free description of one artifact selected for eviction."""

    path_hash: str
    size_bytes: int
    last_accessed_ns: int

    def to_dict(self) -> dict[str, Any]:
        """Return a report-safe candidate description."""

        return {
            "path_hash": self.path_hash,
            "size_bytes": self.size_bytes,
            "last_accessed_ns": self.last_accessed_ns,
        }


@dataclass(frozen=True)
class EvictionPlan:
    """Deterministic plan for bringing a cache under its configured quota."""

    policy_hash: str
    quota_bytes: int
    current_bytes: int
    additional_bytes: int
    bytes_to_free: int
    pinned_bytes: int
    owned_artifact_count: int
    candidates: tuple[EvictionCandidate, ...]

    @property
    def evictions(self) -> tuple[EvictionCandidate, ...]:
        """Alias for callers that describe the selected candidates as evictions."""

        return self.candidates

    @property
    def bytes_planned(self) -> int:
        """Return the bytes selected for removal."""

        return sum(candidate.size_bytes for candidate in self.candidates)

    @property
    def bytes_freed(self) -> int:
        """Return the planned bytes freed by successful eviction."""

        return self.bytes_planned

    @property
    def remaining_bytes(self) -> int:
        """Return the projected bytes after the planned eviction."""

        return max(self.current_bytes + self.additional_bytes - self.bytes_planned, 0)

    @property
    def quota_satisfied(self) -> bool:
        """Return whether the plan can satisfy the quota."""

        return self.remaining_bytes <= self.quota_bytes

    @property
    def blocked_bytes(self) -> int:
        """Return bytes still over quota when pinned/unsafe entries block cleanup."""

        return max(self.remaining_bytes - self.quota_bytes, 0)

    def to_dict(self) -> dict[str, Any]:
        """Return a report containing hashes, counts, and byte totals only."""

        return {
            "policy_hash": self.policy_hash,
            "quota_bytes": self.quota_bytes,
            "current_bytes": self.current_bytes,
            "additional_bytes": self.additional_bytes,
            "bytes_to_free": self.bytes_to_free,
            "bytes_planned": self.bytes_planned,
            "remaining_bytes": self.remaining_bytes,
            "pinned_bytes": self.pinned_bytes,
            "owned_artifact_count": self.owned_artifact_count,
            "eviction_count": len(self.candidates),
            "evictions": [candidate.to_dict() for candidate in self.candidates],
            "quota_satisfied": self.quota_satisfied,
            "blocked_bytes": self.blocked_bytes,
        }


@dataclass(frozen=True)
class EvictionResult:
    """Report from applying an eviction plan without raw filesystem paths."""

    plan: EvictionPlan
    evicted_path_hashes: tuple[str, ...]
    skipped_path_hashes: tuple[str, ...]
    bytes_freed: int
    remaining_bytes: int
    dry_run: bool = False

    @property
    def evicted_count(self) -> int:
        """Return the number of artifacts removed or already absent."""

        return len(self.evicted_path_hashes)

    @property
    def skipped_count(self) -> int:
        """Return the number of candidates that could not be removed."""

        return len(self.skipped_path_hashes)

    @property
    def quota_satisfied(self) -> bool:
        """Return whether the resulting cache is within quota."""

        return self.remaining_bytes <= self.plan.quota_bytes

    def to_dict(self) -> dict[str, Any]:
        """Return a report containing hashes, counts, and byte totals only."""

        return {
            "plan": self.plan.to_dict(),
            "evicted_path_hashes": list(self.evicted_path_hashes),
            "skipped_path_hashes": list(self.skipped_path_hashes),
            "evicted_count": self.evicted_count,
            "skipped_count": self.skipped_count,
            "bytes_freed": self.bytes_freed,
            "remaining_bytes": self.remaining_bytes,
            "quota_satisfied": self.quota_satisfied,
            "dry_run": self.dry_run,
        }


@dataclass(frozen=True)
class _OwnedArtifact:
    relative_path: str
    size_bytes: int
    sha256: str
    last_accessed_ns: int
    pinned: bool


def _validate_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _normalise_sha256(value: str, *, name: str = "sha256") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 digest")
    candidate = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", candidate):
        candidate = f"sha256:{candidate}"
    if not _SHA256_PATTERN.fullmatch(candidate):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return candidate


def _hash_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _hash_path_text(value: str) -> str:
    return _hash_bytes(value.encode("utf-8"))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _measure_path(path: Path) -> tuple[int, str]:
    """Return size and a deterministic digest for a regular file or directory."""

    if path.is_symlink():
        raise ModelCachePolicyError("symbolic-link artifacts are not managed")
    if path.is_file():
        try:
            return path.stat().st_size, _hash_file(path)
        except OSError as exc:
            raise ModelCachePolicyError("unable to inspect cache artifact") from exc
    if not path.is_dir():
        raise ModelCachePolicyError(
            "cache artifact must be a regular file or directory"
        )

    try:
        children = sorted(
            path.rglob("*"),
            key=lambda child: child.relative_to(path).as_posix(),
        )
        digest = hashlib.sha256()
        digest.update(b"openmed-model-cache-artifact-v1\0")
        total_size = 0
        for child in children:
            if child.is_symlink():
                raise ModelCachePolicyError(
                    "symbolic-link members are not managed in directory artifacts"
                )
            if child.is_dir():
                continue
            if not child.is_file():
                raise ModelCachePolicyError(
                    "cache artifact contains an unsupported file"
                )
            relative = child.relative_to(path).as_posix()
            size = child.stat().st_size
            total_size += size
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(size).encode("ascii"))
            digest.update(b"\0")
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
                    digest.update(chunk)
        return total_size, f"sha256:{digest.hexdigest()}"
    except OSError as exc:
        raise ModelCachePolicyError("unable to inspect cache artifact") from exc


def sha256_path(path: str | Path) -> str:
    """Return the local SHA-256 digest for a file or directory artifact.

    This helper is strictly local and never resolves a model name or contacts a
    remote service.
    """

    try:
        _, digest = _measure_path(Path(path).expanduser())
    except (OSError, ModelCachePolicyError) as exc:
        raise ModelCachePolicyError("unable to inspect cache artifact") from exc
    return digest


def verify_artifact_checksum(
    path: str | Path,
    expected_sha256: str,
) -> str:
    """Verify a local artifact and return its actual digest.

    The exception identifies the input only by a hash, making this safe for
    callers that surface errors in logs or audit reports.
    """

    expected = _normalise_sha256(expected_sha256, name="expected_sha256")
    path_text = Path(path).expanduser().as_posix()
    path_hash = _hash_path_text(path_text)
    try:
        _, actual = _measure_path(Path(path).expanduser())
    except (OSError, ModelCachePolicyError) as exc:
        raise CacheIntegrityError(
            path_hash,
            expected_sha256=expected,
            actual_sha256=_MISSING_DIGEST,
            reason="artifact is unavailable",
        ) from exc
    if actual != expected:
        raise CacheIntegrityError(
            path_hash,
            expected_sha256=expected,
            actual_sha256=actual,
        )
    return actual


class ModelCachePolicy:
    """Manage an explicit local model-cache quota and ownership manifest.

    Args:
        cache_dir: Directory containing artifacts managed by this policy.
        quota_bytes: Maximum total size of registered artifacts.
        pinned_artifacts: Optional paths relative to ``cache_dir`` that are
            protected from eviction.  Paths become owned when registered.
        manifest_name: Cache-local JSON bookkeeping filename.

    Only paths explicitly registered with :meth:`register_artifact` are
    eligible for removal.  Do not register a shared cache root or a directory
    containing artifacts owned by another process.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        quota_bytes: int,
        *,
        pinned_artifacts: Iterable[str | Path] = (),
        manifest_name: str = CACHE_POLICY_MANIFEST,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve(strict=False)
        self.quota_bytes = _validate_nonnegative_int(quota_bytes, name="quota_bytes")
        if (
            not isinstance(manifest_name, str)
            or not manifest_name
            or PurePosixPath(manifest_name).name != manifest_name
            or manifest_name in {".", ".."}
        ):
            raise ValueError("manifest_name must be a single filename")
        self.manifest_path = self.cache_dir / manifest_name
        self._policy_hash = _hash_path_text(self.cache_dir.as_posix())
        self._configured_pins: set[str] = set()
        self._lock = RLock()
        for artifact in pinned_artifacts:
            relative, _ = self._relative_path_for_input(artifact)
            self._configured_pins.add(relative)

    def register_artifact(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        pinned: bool | None = None,
        last_accessed_ns: int | None = None,
    ) -> CacheArtifact:
        """Register an owned artifact after measuring its current contents.

        If ``expected_sha256`` is provided, registration fails closed when the
        local content does not match.  A new artifact receives the current
        wall-clock nanosecond as its access marker; callers can provide an
        explicit marker for reproducible plans and tests.
        """

        if pinned is not None and not isinstance(pinned, bool):
            raise ValueError("pinned must be a boolean or None")
        if last_accessed_ns is not None:
            last_accessed_ns = _validate_nonnegative_int(
                last_accessed_ns,
                name="last_accessed_ns",
            )
        expected = (
            _normalise_sha256(expected_sha256, name="expected_sha256")
            if expected_sha256 is not None
            else None
        )
        with self._lock:
            relative, resolved = self._relative_path_for_input(path)
            path_hash = self._path_hash(relative)
            try:
                size_bytes, actual = _measure_path(resolved)
            except (OSError, ModelCachePolicyError) as exc:
                raise ModelCachePolicyError(
                    "unable to register cache artifact"
                ) from exc
            if expected is not None and actual != expected:
                raise CacheIntegrityError(
                    path_hash,
                    expected_sha256=expected,
                    actual_sha256=actual,
                    reason="artifact digest mismatch during registration",
                )

            records = self._read_records()
            previous = records.get(relative)
            if last_accessed_ns is None:
                last_accessed_ns = (
                    previous.last_accessed_ns
                    if previous is not None
                    else time.time_ns()
                )
            effective_pinned = (
                pinned
                if pinned is not None
                else previous.pinned
                if previous is not None
                else False
            ) or relative in self._configured_pins
            records[relative] = _OwnedArtifact(
                relative_path=relative,
                size_bytes=size_bytes,
                sha256=actual,
                last_accessed_ns=last_accessed_ns,
                pinned=effective_pinned,
            )
            self._write_records(records)
            return self._summary(records[relative], exists=True, verified=True)

    def register(self, path: str | Path, **kwargs: Any) -> CacheArtifact:
        """Alias for :meth:`register_artifact`."""

        return self.register_artifact(path, **kwargs)

    def unregister_artifact(self, path: str | Path) -> bool:
        """Forget ownership of an artifact without deleting it."""

        with self._lock:
            relative, _ = self._relative_path_for_input(path)
            records = self._read_records()
            if relative not in records:
                return False
            del records[relative]
            self._write_records(records)
            return True

    def verify_artifact(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> CacheArtifact:
        """Verify an owned artifact before it is reused."""

        with self._lock:
            relative, resolved = self._relative_path_for_input(path)
            records = self._read_records()
            record = records.get(relative)
            if record is None:
                raise CacheOwnershipError("cache artifact is not owned by this policy")
            self._verify_record(
                relative,
                resolved,
                record,
                expected_sha256=expected_sha256,
            )
            return self._summary(record, exists=True, verified=True)

    def reuse_artifact(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> Path:
        """Verify an owned artifact and mark it as recently used.

        The returned path is local and is intended for the caller's immediate
        model-loading operation.  No remote lookup is performed.
        """

        with self._lock:
            relative, resolved = self._relative_path_for_input(path)
            records = self._read_records()
            record = records.get(relative)
            if record is None:
                raise CacheOwnershipError("cache artifact is not owned by this policy")
            self._verify_record(
                relative,
                resolved,
                record,
                expected_sha256=expected_sha256,
            )
            records[relative] = replace(
                record,
                last_accessed_ns=time.time_ns(),
            )
            self._write_records(records)
            return resolved

    def mark_accessed(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> CacheArtifact:
        """Verify and mark an artifact as recently used, returning its summary."""

        self.reuse_artifact(path, expected_sha256=expected_sha256)
        return self.verify_artifact(path)

    def is_valid(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> bool:
        """Return whether an owned artifact passes checksum verification."""

        try:
            self.verify_artifact(path, expected_sha256=expected_sha256)
        except (
            CacheOwnershipError,
            CacheIntegrityError,
            ModelCachePolicyError,
            ValueError,
        ):
            return False
        return True

    def list_artifacts(self, *, verify: bool = False) -> tuple[CacheArtifact, ...]:
        """Return registered artifact summaries sorted by relative path hash."""

        with self._lock:
            records = self._read_records()
            summaries: list[CacheArtifact] = []
            for relative in sorted(records):
                record = records[relative]
                try:
                    resolved = self._path_for_relative(relative)
                    exists = resolved.exists() and not resolved.is_symlink()
                except (OSError, ModelCachePolicyError):
                    exists = False
                    resolved = None
                if verify:
                    if resolved is None or not exists:
                        raise CacheIntegrityError(
                            self._path_hash(relative),
                            expected_sha256=record.sha256,
                            actual_sha256=_MISSING_DIGEST,
                            reason="artifact is unavailable",
                        )
                    self._verify_record(relative, resolved, record)
                    summaries.append(self._summary(record, exists=True, verified=True))
                else:
                    size_bytes = record.size_bytes
                    if resolved is not None and exists:
                        try:
                            size_bytes, _ = _measure_path(resolved)
                        except (OSError, ModelCachePolicyError):
                            exists = False
                    summaries.append(
                        self._summary(
                            replace(record, size_bytes=size_bytes),
                            exists=exists,
                            verified=False,
                        )
                    )
            return tuple(summaries)

    def inventory(self, *, verify: bool = False) -> tuple[CacheArtifact, ...]:
        """Alias for :meth:`list_artifacts`."""

        return self.list_artifacts(verify=verify)

    def pin_artifact(self, path: str | Path) -> CacheArtifact:
        """Mark an owned artifact as protected from eviction."""

        return self._set_pinned(path, True)

    def unpin_artifact(self, path: str | Path) -> CacheArtifact:
        """Remove an artifact's protection from eviction."""

        return self._set_pinned(path, False)

    def plan_eviction(
        self,
        *,
        additional_bytes: int = 0,
        required_bytes: int | None = None,
    ) -> EvictionPlan:
        """Build a deterministic least-recently-used eviction plan.

        ``additional_bytes`` reserves space for a not-yet-registered artifact.
        ``required_bytes`` is accepted as a descriptive alias for callers that
        are planning a single incoming artifact.
        """

        additional_bytes = _validate_nonnegative_int(
            additional_bytes,
            name="additional_bytes",
        )
        if required_bytes is not None:
            required_bytes = _validate_nonnegative_int(
                required_bytes,
                name="required_bytes",
            )
            if additional_bytes and additional_bytes != required_bytes:
                raise ValueError(
                    "additional_bytes and required_bytes must not disagree"
                )
            additional_bytes = required_bytes

        with self._lock:
            records = self._read_records()
            current_bytes = 0
            pinned_bytes = 0
            candidates: list[tuple[int, str, int]] = []
            for relative, record in records.items():
                try:
                    resolved = self._path_for_relative(relative)
                    size_bytes, _ = _measure_path(resolved)
                except (OSError, ModelCachePolicyError):
                    logger.warning(
                        "Skipping unsafe cache entry path_hash=%s",
                        self._path_hash(relative),
                    )
                    continue
                current_bytes += size_bytes
                pinned = record.pinned or relative in self._configured_pins
                if pinned:
                    pinned_bytes += size_bytes
                else:
                    candidates.append((record.last_accessed_ns, relative, size_bytes))

            bytes_to_free = max(current_bytes + additional_bytes - self.quota_bytes, 0)
            selected: list[EvictionCandidate] = []
            selected_bytes = 0
            for last_accessed_ns, relative, size_bytes in sorted(
                candidates,
                key=lambda item: (item[0], item[1]),
            ):
                if selected_bytes >= bytes_to_free:
                    break
                selected.append(
                    EvictionCandidate(
                        path_hash=self._path_hash(relative),
                        size_bytes=size_bytes,
                        last_accessed_ns=last_accessed_ns,
                    )
                )
                selected_bytes += size_bytes

            return EvictionPlan(
                policy_hash=self._policy_hash,
                quota_bytes=self.quota_bytes,
                current_bytes=current_bytes,
                additional_bytes=additional_bytes,
                bytes_to_free=bytes_to_free,
                pinned_bytes=pinned_bytes,
                owned_artifact_count=len(records),
                candidates=tuple(selected),
            )

    def apply_eviction(
        self,
        plan: EvictionPlan,
        *,
        dry_run: bool = False,
    ) -> EvictionResult:
        """Apply a plan, deleting only still-owned and unchanged artifacts."""

        if not isinstance(plan, EvictionPlan):
            raise TypeError("plan must be an EvictionPlan")
        if plan.policy_hash != self._policy_hash:
            raise ModelCachePolicyError("eviction plan belongs to another cache policy")
        if not isinstance(dry_run, bool):
            raise ValueError("dry_run must be a boolean")

        with self._lock:
            records = self._read_records()
            if dry_run:
                remaining_bytes = self._total_bytes(records)
                return EvictionResult(
                    plan=plan,
                    evicted_path_hashes=(),
                    skipped_path_hashes=tuple(
                        candidate.path_hash for candidate in plan.candidates
                    ),
                    bytes_freed=0,
                    remaining_bytes=remaining_bytes,
                    dry_run=True,
                )

            evicted: list[str] = []
            skipped: list[str] = []
            bytes_freed = 0
            changed = False
            for candidate in plan.candidates:
                match = next(
                    (
                        (relative, record)
                        for relative, record in records.items()
                        if self._path_hash(relative) == candidate.path_hash
                    ),
                    None,
                )
                if match is None:
                    skipped.append(candidate.path_hash)
                    continue
                relative, record = match
                if record.pinned or relative in self._configured_pins:
                    skipped.append(candidate.path_hash)
                    continue
                try:
                    resolved = self._path_for_relative(relative)
                    if not resolved.exists():
                        del records[relative]
                        evicted.append(candidate.path_hash)
                        changed = True
                        continue
                    current_size, current_digest = _measure_path(resolved)
                    if current_digest != record.sha256:
                        skipped.append(candidate.path_hash)
                        continue
                    if resolved.is_dir():
                        shutil.rmtree(resolved)
                    else:
                        resolved.unlink()
                except (OSError, ModelCachePolicyError):
                    logger.warning(
                        "Unable to evict cache entry path_hash=%s",
                        candidate.path_hash,
                    )
                    skipped.append(candidate.path_hash)
                    continue
                del records[relative]
                evicted.append(candidate.path_hash)
                bytes_freed += current_size
                changed = True

            if changed:
                self._write_records(records)
            remaining_bytes = self._total_bytes(records)
            return EvictionResult(
                plan=plan,
                evicted_path_hashes=tuple(evicted),
                skipped_path_hashes=tuple(skipped),
                bytes_freed=bytes_freed,
                remaining_bytes=remaining_bytes,
            )

    def enforce_quota(
        self,
        *,
        additional_bytes: int = 0,
    ) -> EvictionResult:
        """Plan and apply LRU eviction for the current quota."""

        plan = self.plan_eviction(additional_bytes=additional_bytes)
        return self.apply_eviction(plan)

    def evict(self, plan: EvictionPlan, *, dry_run: bool = False) -> EvictionResult:
        """Alias for :meth:`apply_eviction`."""

        return self.apply_eviction(plan, dry_run=dry_run)

    def _set_pinned(self, path: str | Path, pinned: bool) -> CacheArtifact:
        with self._lock:
            relative, resolved = self._relative_path_for_input(path)
            records = self._read_records()
            record = records.get(relative)
            if record is None:
                raise CacheOwnershipError("cache artifact is not owned by this policy")
            self._configured_pins.discard(relative)
            records[relative] = replace(record, pinned=pinned)
            self._write_records(records)
            exists = resolved.exists() and not resolved.is_symlink()
            return self._summary(records[relative], exists=exists, verified=False)

    def _verify_record(
        self,
        relative: str,
        resolved: Path,
        record: _OwnedArtifact,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        expected = (
            _normalise_sha256(expected_sha256, name="expected_sha256")
            if expected_sha256 is not None
            else record.sha256
        )
        try:
            _, actual = _measure_path(resolved)
        except (OSError, ModelCachePolicyError) as exc:
            raise CacheIntegrityError(
                self._path_hash(relative),
                expected_sha256=expected,
                actual_sha256=_MISSING_DIGEST,
                reason="artifact is unavailable",
            ) from exc
        if actual != expected:
            raise CacheIntegrityError(
                self._path_hash(relative),
                expected_sha256=expected,
                actual_sha256=actual,
            )
        if expected_sha256 is not None and record.sha256 != expected:
            raise CacheIntegrityError(
                self._path_hash(relative),
                expected_sha256=record.sha256,
                actual_sha256=actual,
                reason="requested checksum differs from the owned checksum",
            )

    def _summary(
        self,
        record: _OwnedArtifact,
        *,
        exists: bool,
        verified: bool,
    ) -> CacheArtifact:
        return CacheArtifact(
            path_hash=self._path_hash(record.relative_path),
            size_bytes=record.size_bytes,
            sha256=record.sha256,
            last_accessed_ns=record.last_accessed_ns,
            pinned=record.pinned or record.relative_path in self._configured_pins,
            exists=exists,
            verified=verified,
        )

    def _path_hash(self, relative: str) -> str:
        return _hash_path_text(relative)

    def _relative_path_for_input(self, path: str | Path) -> tuple[str, Path]:
        try:
            raw = Path(path).expanduser()
        except (TypeError, ValueError) as exc:
            raise ValueError("cache artifact path is invalid") from exc
        if not raw.is_absolute():
            raw = self.cache_dir / raw
        if raw.is_symlink():
            raise CacheOwnershipError("symbolic-link artifacts are not managed")
        resolved = raw.resolve(strict=False)
        try:
            relative = resolved.relative_to(self.cache_dir)
        except ValueError as exc:
            raise CacheOwnershipError(
                "cache artifact must be inside the configured cache directory"
            ) from exc
        if relative == Path("."):
            raise CacheOwnershipError("the cache directory itself is not an artifact")
        relative_text = relative.as_posix()
        self._validate_relative_path(relative_text)
        if relative_text == self.manifest_path.name:
            raise CacheOwnershipError("the cache policy manifest is not an artifact")
        return relative_text, resolved

    def _path_for_relative(self, relative: str) -> Path:
        self._validate_relative_path(relative)
        candidate = self.cache_dir.joinpath(*PurePosixPath(relative).parts)
        if candidate.is_symlink():
            raise CacheOwnershipError("symbolic-link artifacts are not managed")
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.cache_dir)
        except ValueError as exc:
            raise CacheOwnershipError(
                "cache artifact must be inside the configured cache directory"
            ) from exc
        return resolved

    @staticmethod
    def _validate_relative_path(relative: str) -> None:
        parsed = PurePosixPath(relative)
        if (
            not relative
            or parsed.is_absolute()
            or "\\" in relative
            or parsed.as_posix() != relative
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise ModelCachePolicyError(
                "cache artifact path is not a safe relative path"
            )

    def _is_manifest_relative_path(self, relative: str) -> bool:
        return relative == self.manifest_path.name

    def _read_records(self) -> dict[str, _OwnedArtifact]:
        if not self.manifest_path.exists():
            return {}
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelCachePolicyError("cache policy manifest is invalid") from exc
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != CACHE_POLICY_SCHEMA
        ):
            raise ModelCachePolicyError("cache policy manifest is invalid")
        raw_artifacts = payload.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise ModelCachePolicyError("cache policy manifest is invalid")
        records: dict[str, _OwnedArtifact] = {}
        for raw in raw_artifacts:
            if not isinstance(raw, Mapping):
                raise ModelCachePolicyError("cache policy manifest is invalid")
            relative = raw.get("relative_path")
            if not isinstance(relative, str):
                raise ModelCachePolicyError("cache policy manifest is invalid")
            try:
                self._validate_relative_path(relative)
                if self._is_manifest_relative_path(relative):
                    raise ModelCachePolicyError(
                        "cache policy manifest is not an artifact"
                    )
                size_bytes = _validate_nonnegative_int(
                    raw.get("size_bytes"),
                    name="size_bytes",
                )
                last_accessed_ns = _validate_nonnegative_int(
                    raw.get("last_accessed_ns"),
                    name="last_accessed_ns",
                )
                digest = _normalise_sha256(str(raw.get("sha256")), name="sha256")
            except (TypeError, ValueError, ModelCachePolicyError) as exc:
                raise ModelCachePolicyError("cache policy manifest is invalid") from exc
            pinned = raw.get("pinned")
            if not isinstance(pinned, bool) or relative in records:
                raise ModelCachePolicyError("cache policy manifest is invalid")
            records[relative] = _OwnedArtifact(
                relative_path=relative,
                size_bytes=size_bytes,
                sha256=digest,
                last_accessed_ns=last_accessed_ns,
                pinned=pinned,
            )
        return records

    def _write_records(self, records: Mapping[str, _OwnedArtifact]) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": CACHE_POLICY_SCHEMA,
                "artifacts": [
                    {
                        "relative_path": relative,
                        "size_bytes": record.size_bytes,
                        "sha256": record.sha256,
                        "last_accessed_ns": record.last_accessed_ns,
                        "pinned": record.pinned,
                    }
                    for relative, record in sorted(records.items())
                ],
            }
            temporary = self.manifest_path.with_name(f".{self.manifest_path.name}.tmp")
            temporary.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.manifest_path)
        except OSError as exc:
            raise ModelCachePolicyError(
                "unable to update cache policy manifest"
            ) from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except (NameError, OSError):
                pass

    def _total_bytes(self, records: Mapping[str, _OwnedArtifact]) -> int:
        total = 0
        for relative, record in records.items():
            try:
                path = self._path_for_relative(relative)
                size, _ = _measure_path(path)
            except (OSError, ModelCachePolicyError):
                continue
            total += size
        return total


__all__ = [
    "CACHE_POLICY_MANIFEST",
    "CACHE_POLICY_SCHEMA",
    "CacheArtifact",
    "CacheIntegrityError",
    "CacheOwnershipError",
    "EvictionCandidate",
    "EvictionPlan",
    "EvictionResult",
    "ModelCachePolicy",
    "ModelCachePolicyError",
    "sha256_path",
    "verify_artifact_checksum",
]
