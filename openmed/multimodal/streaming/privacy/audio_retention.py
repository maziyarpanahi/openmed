"""Own transient audio buffers and account for local artifact destruction.

No audio is written to disk. Optional file enrollment only accepts completed,
application-owned artifacts in a private dedicated POSIX directory. Receipts
prove the stated operation, not physical erasure or discovery of unknown copies.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

_SCHEMA = "openmed.audio.retention.v1"
_REF = re.compile(r"[0-9a-f]{32}\Z")
_T = TypeVar("_T")
_VIOLATIONS = frozenset(
    {
        "operation_limit",
        "unexpected_artifact",
        "byte_limit",
        "file_enrollment_failed",
        "closed_with_unresolved_artifacts",
    }
)


class AudioRetentionError(ValueError):
    """Fixed-code refusal with no audio, source paths or provider exceptions."""


class ArtifactKind(str, Enum):
    """Storage ownership; external artifacts never count as verified cleanup."""

    BUFFER = "buffer"
    FILE = "file"
    EXTERNAL = "external"


class AudioStage(str, Enum):
    """Closed vocabulary for an artifact's bounded lifecycle history."""

    CAPTURE = "capture"
    BUFFERING = "buffering"
    INFERENCE = "inference"
    CACHE = "cache"
    CRASH = "crash"


class SessionEnd(str, Enum):
    """Terminal reasons; every reason destroys all enrolled artifacts."""

    FINALIZED = "finalized"
    CANCELLED = "cancelled"
    FAILURE = "failure"
    EXPIRED = "expired"
    CLOCK_INVALID = "clock_invalid"
    RECOVERED_CRASH = "recovered_crash"


class DestructionAction(str, Enum):
    """Actual verified action, without implying secure physical erasure."""

    BUFFER_OVERWRITTEN = "owned_buffer_overwritten"
    FILE_UNLINKED = "owned_file_unlinked"
    UNACCOUNTED = "unaccounted"
    FAILED = "failed"


def _require(condition: bool) -> None:
    if not condition:
        raise AudioRetentionError("invalid_audio_retention_contract")


def _ref(value: object) -> None:
    _require(type(value) is str and _REF.fullmatch(value) is not None)


def _integer(value: object, maximum: int, minimum: int = 0) -> None:
    _require(type(value) is int and minimum <= value <= maximum)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioRetentionPolicy:
    """Resource limits and a non-renewable session lifetime.

    Args:
        max_artifacts: Maximum total expected artifacts, including external ones.
        max_bytes: Maximum simultaneously owned buffer and enrolled file bytes.
        max_lifetime_ms: Deadline measured from session creation, never renewed.
        max_operations: Maximum captures, enrollments, stage changes and uses.
    """

    max_artifacts: int = 64
    max_bytes: int = 16 * 1024 * 1024
    max_lifetime_ms: int = 300_000
    max_operations: int = 4096

    def __post_init__(self) -> None:
        _integer(self.max_artifacts, 256, 1)
        _integer(self.max_bytes, 64 * 1024 * 1024, 1)
        _integer(self.max_lifetime_ms, 3_600_000, 1)
        _integer(self.max_operations, 16_384, 1)


@dataclass(frozen=True, slots=True)
class ExpectedAudioArtifact:
    """One opaque identity in the caller's complete expected inventory."""

    artifact_ref: str
    kind: ArtifactKind
    stage: AudioStage = AudioStage.CAPTURE

    def __post_init__(self) -> None:
        _ref(self.artifact_ref)
        _require(type(self.kind) is ArtifactKind and type(self.stage) is AudioStage)


@dataclass(frozen=True, slots=True)
class DestructionReceipt:
    """One content-free attempt, bound to policy, inventory and terminal reason."""

    sequence: int
    artifact_ref: str
    action: DestructionAction
    stage_history: tuple[AudioStage, ...]
    session_end: SessionEnd
    binding_digest: str


@dataclass(frozen=True, slots=True)
class AudioRetentionAudit:
    """Closed-session cleanup accounting, not a compliance certification."""

    session_ref: str
    binding_digest: str
    session_end: SessionEnd | None
    passed: bool
    blockers: tuple[str, ...]
    receipts: tuple[DestructionReceipt, ...]

    def to_json(self) -> str:
        """Serialize only opaque references, controlled codes and policy digests."""
        return _json({"schema": _SCHEMA, **asdict(self)})


@dataclass(slots=True)
class _Artifact:
    expected: ExpectedAudioArtifact
    stages: list[AudioStage]
    buffer: bytearray | None = field(default=None, repr=False)
    file_identity: tuple[int, ...] | None = None
    size: int = 0
    used: bool = False
    destroyed: bool = False
    quarantined: bool = False


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _overwrite(buffer: bytearray) -> bool:
    try:
        for offset in range(0, len(buffer), 4096):
            end = min(offset + 4096, len(buffer))
            buffer[offset:end] = b"\x00" * (end - offset)
        return not any(buffer)
    except Exception:
        return False


def _private_file(info: os.stat_result) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_nlink == 1
        and info.st_mode & 0o077 == 0
        and info.st_uid == os.getuid()
    )


class AudioRetentionSession:
    """Own bounded buffers and destroy enrolled artifacts at every terminal edge.

    Args:
        session_ref: Opaque identifier, never a patient identifier.
        expected: Complete inventory declared before capture; omissions cannot be
            discovered by this library. Unknown external copies must be declared
            EXTERNAL and will prevent a passing audit.
        policy: Non-renewable local retention and resource limits.
        artifact_directory: Optional dedicated private directory of completed
            application-owned files named ``<artifact_ref>.audio``. No raw audio
            is written here by this library. Requires POSIX descriptor-relative
            operations; file mode fails closed on other platforms.
        clock: Monotonic clock; injection supports deterministic host event loops.
        automatic_expiry: Default daemon timer enforces the deadline while this
            process runs. If False, the host MUST schedule ``expire`` even when
            idle. Process death requires a host watchdog and recovery checkpoint.

    The host must serialize producers before enrolling completed files and must
    not permit other processes with the same OS identity to modify the private
    directory. This is not a sandbox against such processes. Use a context
    manager or explicitly end the session; garbage collection is not cleanup.
    """

    def __init__(
        self,
        session_ref: str,
        expected: tuple[ExpectedAudioArtifact, ...],
        *,
        policy: AudioRetentionPolicy = AudioRetentionPolicy(),
        artifact_directory: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        automatic_expiry: bool = True,
    ) -> None:
        _ref(session_ref)
        _require(type(policy) is AudioRetentionPolicy)
        _require(type(expected) is tuple and 1 <= len(expected) <= policy.max_artifacts)
        _require(all(type(item) is ExpectedAudioArtifact for item in expected))
        _require(len({item.artifact_ref for item in expected}) == len(expected))
        _require(type(automatic_expiry) is bool and callable(clock))
        self._session_ref = session_ref
        self._policy = policy
        self._expected = tuple(sorted(expected, key=lambda item: item.artifact_ref))
        self._binding = _digest(
            [session_ref, asdict(policy), [asdict(item) for item in self._expected]]
        )
        self._artifacts = {
            item.artifact_ref: _Artifact(item, [item.stage]) for item in self._expected
        }
        self._clock = clock
        self._lock = threading.RLock()
        self._end: SessionEnd | None = None
        self._receipts: list[DestructionReceipt] = []
        self._violations: set[str] = set()
        self._operations = 0
        self._owned_bytes = 0
        self._timer: threading.Timer | None = None
        self._directory_fd: int | None = None
        self._directory_identity: tuple[int, int] | None = None
        self._started = self._read_clock()
        _require(self._started is not None)
        self._last_time = self._started
        self._automatic_expiry = automatic_expiry
        if artifact_directory is not None:
            self._open_directory(artifact_directory)
        if automatic_expiry:
            self._schedule(policy.max_lifetime_ms / 1000)

    def __repr__(self) -> str:
        return "AudioRetentionSession(protected_artifacts)"

    def __enter__(self) -> AudioRetentionSession:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.finish(
            SessionEnd.FAILURE if exc_type is not None else SessionEnd.FINALIZED
        )
        self.close()

    def _read_clock(self) -> float | None:
        result = None
        try:
            value = self._clock()
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                result = float(value)
        except Exception:
            pass
        return result

    def _open_directory(self, directory: Path) -> None:
        supported = (
            os.name == "posix"
            and os.open in os.supports_dir_fd
            and os.stat in os.supports_dir_fd
            and os.rename in os.supports_dir_fd
            and os.unlink in os.supports_dir_fd
        )
        _require(supported and isinstance(directory, Path))
        descriptor = None
        try:
            descriptor = os.open(
                directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
            info = os.fstat(descriptor)
            _require(info.st_uid == os.getuid() and info.st_mode & 0o077 == 0)
            self._directory_identity = (info.st_dev, info.st_ino)
            self._directory_fd = descriptor
            return
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
        raise AudioRetentionError("audio_directory_unavailable")

    def _schedule(self, delay: float) -> None:
        timer = threading.Timer(delay, self._deadline_callback)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _deadline_callback(self) -> None:
        with self._lock:
            self.expire()
            if self._end is None:
                # Timer delivery may be early; a clock rollback never renews it.
                assert self._last_time is not None and self._started is not None
                remaining = self._policy.max_lifetime_ms / 1000 - (
                    self._last_time - self._started
                )
                self._schedule(max(remaining, 0.001))

    def expire(self) -> bool:
        """Enforce clock validity and the fixed deadline; return whether closed."""
        with self._lock:
            if self._end is None:
                now = self._read_clock()
                if now is None or self._last_time is None or now < self._last_time:
                    self.finish(SessionEnd.CLOCK_INVALID)
                else:
                    self._last_time = now
                    assert self._started is not None
                    if (now - self._started) * 1000 >= self._policy.max_lifetime_ms:
                        self.finish(SessionEnd.EXPIRED)
            return self._end is not None

    def _active(self) -> None:
        if self.expire():
            raise AudioRetentionError("audio_session_closed")
        self._operations += 1
        if self._operations > self._policy.max_operations:
            self._violations.add("operation_limit")
            self.finish(SessionEnd.FAILURE)
            raise AudioRetentionError("audio_operation_limit")

    def _artifact(self, artifact_ref: str) -> _Artifact:
        _ref(artifact_ref)
        if artifact_ref not in self._artifacts:
            self._violations.add("unexpected_artifact")
            self.finish(SessionEnd.FAILURE)
            raise AudioRetentionError("unexpected_audio_artifact")
        return self._artifacts[artifact_ref]

    def capture(self, artifact_ref: str, audio: bytes | bytearray) -> None:
        """Copy capture data into owned mutable storage; caller copies remain theirs."""
        with self._lock:
            self._active()
            item = self._artifact(artifact_ref)
            _require(item.expected.kind is ArtifactKind.BUFFER and not item.used)
            _require(type(audio) in (bytes, bytearray) and len(audio) > 0)
            self._reserve(len(audio))
            failed = False
            try:
                item.buffer = bytearray(audio)
                item.size = len(audio)
                item.used = True
            except Exception:
                failed = True
            if failed:
                self.finish(SessionEnd.FAILURE)
                raise AudioRetentionError("audio_capture_failed")

    def _derive_buffer(
        self,
        session_ref: str,
        source_ref: str,
        destination_ref: str,
        expected_bytes: int,
        render: Callable[[memoryview, memoryview], _T],
    ) -> _T:
        """Render into a new owned buffer; never expose an unverified derivative.

        Only trusted package renderers may use this primitive. They run under
        the inventory lock, may not re-enter the session, and must not retain
        views. Check the non-renewable deadline again before making it usable.
        """
        with self._lock:
            self._active()
            _require(session_ref == self._session_ref and source_ref != destination_ref)
            source = self._artifact(source_ref)
            destination = self._artifact(destination_ref)
            _require(source.buffer is not None and len(source.buffer) == expected_bytes)
            _require(
                destination.expected.kind is ArtifactKind.BUFFER
                and not destination.used
            )
            self._reserve(expected_bytes)
            assert source.buffer is not None
            try:
                destination.buffer = bytearray(expected_bytes)
                destination.size = expected_bytes
                with (
                    memoryview(source.buffer).toreadonly() as original,
                    memoryview(destination.buffer) as output,
                ):
                    report = render(original, output)
                destination.used = True
            except BaseException:
                self.finish(SessionEnd.FAILURE)
            else:
                if self.expire():
                    raise AudioRetentionError("audio_result_discarded")
                return report
            raise AudioRetentionError("audio_derivation_failed")

    def _reserve(self, size: int) -> None:
        if size > self._policy.max_bytes - self._owned_bytes:
            self._violations.add("byte_limit")
            self.finish(SessionEnd.FAILURE)
            raise AudioRetentionError("audio_byte_limit")
        self._owned_bytes += size

    def enroll_file(self, artifact_ref: str) -> None:
        """Enroll a completed opaque-named file without reading its audio content."""
        with self._lock:
            self._active()
            item = self._artifact(artifact_ref)
            _require(item.expected.kind is ArtifactKind.FILE and not item.used)
            _require(self._directory_fd is not None)
            failed = False
            try:
                info = os.stat(
                    artifact_ref + ".audio",
                    dir_fd=self._directory_fd,
                    follow_symlinks=False,
                )
                _require(_private_file(info) and info.st_size > 0)
                self._reserve(info.st_size)
                item.file_identity = _identity(info)
                item.size = info.st_size
                item.used = True
            except Exception:
                failed = True
            if failed:
                self._violations.add("file_enrollment_failed")
                self.finish(SessionEnd.FAILURE)
                raise AudioRetentionError("audio_file_enrollment_failed")

    def advance(self, artifact_ref: str, stage: AudioStage) -> None:
        """Record an actual host lifecycle transition without renewing retention."""
        with self._lock:
            self._active()
            _require(type(stage) is AudioStage)
            item = self._artifact(artifact_ref)
            _require(item.used and not item.destroyed)
            if item.stages[-1] is not stage:
                item.stages.append(stage)

    def infer(self, artifact_ref: str, provider: Callable[[memoryview], _T]) -> _T:
        """Run a trusted local provider over a read-only owned buffer view.

        Cancelling or expiring overwrites the underlying buffer even during the
        call. Such calls discard their result. Provider exceptions are replaced
        outside the exception handler, so protected messages are not chained.
        Providers must not retain or copy the view without declaring that copy
        in the host's inventory. Returned values are protected, never audited.
        """
        with self._lock:
            self._active()
            item = self._artifact(artifact_ref)
            _require(
                item.buffer is not None and not item.destroyed and callable(provider)
            )
            self.advance(artifact_ref, AudioStage.INFERENCE)
            assert item.buffer is not None
            view = memoryview(item.buffer).toreadonly()
        failed = False
        try:
            result = provider(view)
        except BaseException:
            failed = True
        finally:
            view.release()
        with self._lock:
            if failed:
                self.finish(SessionEnd.FAILURE)
                raise AudioRetentionError("audio_provider_failed")
            if self.expire():
                raise AudioRetentionError("audio_result_discarded")
            return result

    def _receipt(self, item: _Artifact, action: DestructionAction) -> None:
        assert self._end is not None
        self._receipts.append(
            DestructionReceipt(
                len(self._receipts) + 1,
                item.expected.artifact_ref,
                action,
                tuple(item.stages),
                self._end,
                self._binding,
            )
        )

    def _destroy_file(self, item: _Artifact) -> bool:
        if self._directory_fd is None or item.file_identity is None:
            return False
        source = item.expected.artifact_ref + ".audio"
        quarantine = item.expected.artifact_ref + ".retiring"
        try:
            directory = os.fstat(self._directory_fd)
            if (
                (directory.st_dev, directory.st_ino) != self._directory_identity
                or directory.st_mode & 0o077
                or directory.st_uid != os.getuid()
            ):
                return False
            name = quarantine if item.quarantined else source
            info = os.stat(name, dir_fd=self._directory_fd, follow_symlinks=False)
            if not _private_file(info) or _identity(info) != item.file_identity:
                return False
            if not item.quarantined:
                # Refuse a preexisting quarantine entry; never overwrite it.
                if quarantine in os.listdir(self._directory_fd):
                    return False
                os.rename(
                    source,
                    quarantine,
                    src_dir_fd=self._directory_fd,
                    dst_dir_fd=self._directory_fd,
                )
                item.quarantined = True
            info = os.stat(quarantine, dir_fd=self._directory_fd, follow_symlinks=False)
            if not _private_file(info) or _identity(info) != item.file_identity:
                return False
            os.unlink(quarantine, dir_fd=self._directory_fd)
            os.fsync(self._directory_fd)
            names = os.listdir(self._directory_fd)
            return source not in names and quarantine not in names
        except Exception:
            return False

    def finish(self, reason: SessionEnd = SessionEnd.FINALIZED) -> AudioRetentionAudit:
        """Close admission and attempt destruction of every expected artifact.

        Failed artifacts remain unresolved. Repeated calls retry only failures
        and preserve earlier receipts and the original terminal reason. Missing
        files are not evidence of this session having destroyed them.
        """
        _require(type(reason) is SessionEnd)
        with self._lock:
            if self._end is None:
                now = self._read_clock()
                if now is None or self._last_time is None or now < self._last_time:
                    reason = SessionEnd.CLOCK_INVALID
                elif self._started is not None and (
                    (now - self._started) * 1000 >= self._policy.max_lifetime_ms
                ):
                    reason = SessionEnd.EXPIRED
                self._end = reason
                if self._timer is not None:
                    self._timer.cancel()
            for item in self._artifacts.values():
                if item.destroyed:
                    continue
                action = DestructionAction.UNACCOUNTED
                if item.buffer is not None:
                    # Slice replacement of equal length also overwrites borrowed
                    # views, without resizing or releasing an unmodified buffer.
                    if _overwrite(item.buffer):
                        item.buffer = None
                        action = DestructionAction.BUFFER_OVERWRITTEN
                    else:
                        action = DestructionAction.FAILED
                elif item.file_identity is not None:
                    action = (
                        DestructionAction.FILE_UNLINKED
                        if self._destroy_file(item)
                        else DestructionAction.FAILED
                    )
                if action in {
                    DestructionAction.BUFFER_OVERWRITTEN,
                    DestructionAction.FILE_UNLINKED,
                }:
                    item.destroyed = True
                    self._owned_bytes -= item.size
                # Bound retry receipts: retain first failure and changes of result.
                previous = next(
                    (
                        row
                        for row in reversed(self._receipts)
                        if row.artifact_ref == item.expected.artifact_ref
                    ),
                    None,
                )
                if previous is None or previous.action is not action:
                    self._receipt(item, action)
            return self.audit()

    def audit(self) -> AudioRetentionAudit:
        """Fail until closed and every expected artifact has verified cleanup."""
        with self._lock:
            self.expire()
            blockers = set(self._violations)
            if self._end is None:
                blockers.add("session_open")
            if any(not item.destroyed for item in self._artifacts.values()):
                blockers.add("unaccounted_artifacts")
            if self._directory_fd is not None:
                try:
                    if os.listdir(self._directory_fd):
                        blockers.add("remaining_directory_entries")
                except Exception:
                    blockers.add("directory_unavailable")
            return AudioRetentionAudit(
                self._session_ref,
                self._binding,
                self._end,
                not blockers,
                tuple(sorted(blockers)),
                tuple(self._receipts),
            )

    def close(self) -> AudioRetentionAudit:
        """Cancel if still open, retry cleanup, then release the directory handle.

        The returned audit remains failed if anything is unresolved. A closed
        directory cannot be retried through this object; preserve a checkpoint
        and resolve it through a separately authorized recovery session.
        """
        with self._lock:
            report = self.finish(SessionEnd.CANCELLED)
            if self._directory_fd is not None:
                if not report.passed:
                    self._violations.add("closed_with_unresolved_artifacts")
                os.close(self._directory_fd)
                self._directory_fd = None
            return self.audit()

    def checkpoint(self) -> str:
        """Return value-free restart metadata for host-controlled durable storage.

        Persist after enrollment and before starting inference. This function
        does not write files. A checkpoint is not authenticated; the host must
        integrity-protect it and its private directory. It contains no audio,
        user path, payload hash or encryption key.
        """
        with self._lock:
            return _json(
                {
                    "schema": _SCHEMA,
                    "session_ref": self._session_ref,
                    "policy": asdict(self._policy),
                    "expected": [asdict(item) for item in self._expected],
                    "directory_identity": self._directory_identity,
                    "violations": sorted(self._violations),
                    "histories": {
                        ref: item.stages for ref, item in self._artifacts.items()
                    },
                    "files": {
                        ref: {
                            "identity": item.file_identity,
                            "quarantined": item.quarantined,
                        }
                        for ref, item in self._artifacts.items()
                        if item.file_identity is not None and not item.destroyed
                    },
                }
            )

    @classmethod
    def recover(
        cls, checkpoint: str, *, artifact_directory: Path | None = None
    ) -> AudioRetentionSession:
        """Destroy checkpointed files after restart; never resume expired audio.

        Crashed memory, external artifacts, missing files and artifacts omitted
        from the checkpoint remain unaccounted. Recovery cannot prove their
        erasure. Call ``audit`` then ``close`` on the returned terminal session.
        """
        target = None
        failed = False
        try:
            _require(type(checkpoint) is str and len(checkpoint) <= 1_048_576)
            data = json.loads(checkpoint)
            _require(type(data) is dict and data.get("schema") == _SCHEMA)
            _require(type(data["expected"]) is list and len(data["expected"]) <= 256)
            expected = tuple(
                ExpectedAudioArtifact(
                    item["artifact_ref"],
                    ArtifactKind(item["kind"]),
                    AudioStage(item["stage"]),
                )
                for item in data["expected"]
            )
            target = cls(
                data["session_ref"],
                expected,
                policy=AudioRetentionPolicy(**data["policy"]),
                artifact_directory=artifact_directory,
                automatic_expiry=False,
            )
            directory_identity = data["directory_identity"]
            _require(
                directory_identity is None
                or (
                    type(directory_identity) is list
                    and len(directory_identity) == 2
                    and all(type(value) is int for value in directory_identity)
                )
            )
            _require(
                target._directory_identity
                == (
                    tuple(directory_identity)
                    if directory_identity is not None
                    else None
                )
            )
            violations = data["violations"]
            _require(type(violations) is list and len(violations) <= len(_VIOLATIONS))
            _require(
                all(type(code) is str and code in _VIOLATIONS for code in violations)
            )
            target._violations.update(violations)
            histories = data["histories"]
            _require(
                type(histories) is dict and set(histories) == set(target._artifacts)
            )
            total_stages = 0
            for ref, history in histories.items():
                _require(type(history) is list and len(history) > 0)
                total_stages += len(history)
                _require(
                    total_stages <= target._policy.max_operations + 2 * len(expected)
                )
                stages = [AudioStage(stage) for stage in history]
                _require(stages[0] is target._artifacts[ref].expected.stage)
                target._artifacts[ref].stages = stages
                if stages[-1] is not AudioStage.CRASH:
                    stages.append(AudioStage.CRASH)
            files = data["files"]
            _require(type(files) is dict and len(files) <= len(expected))
            for ref, metadata in files.items():
                _ref(ref)
                _require(ref in target._artifacts)
                item = target._artifacts[ref]
                _require(item.expected.kind is ArtifactKind.FILE)
                identity = metadata["identity"]
                _require(type(identity) is list and len(identity) == 4)
                _require(all(type(value) is int and value >= 0 for value in identity))
                _require(type(metadata["quarantined"]) is bool)
                _require(identity[2] <= target._policy.max_bytes - target._owned_bytes)
                target._owned_bytes += identity[2]
                item.file_identity = tuple(identity)
                item.quarantined = metadata["quarantined"]
                item.size = identity[2]
                item.used = True
        except Exception:
            failed = True
        if failed:
            # Parsing failures must not delete anything from a partially parsed
            # inventory. Only close the descriptor, never call finish here.
            if target is not None and target._directory_fd is not None:
                os.close(target._directory_fd)
                target._directory_fd = None
            raise AudioRetentionError("invalid_audio_recovery_checkpoint")
        assert target is not None
        target.finish(SessionEnd.RECOVERED_CRASH)
        return target
