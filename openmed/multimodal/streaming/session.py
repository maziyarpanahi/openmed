"""Consent-bound ambient capture and owned-memory lifetime, without device IO."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar

_T = TypeVar("_T")
_REF = re.compile(r"[0-9a-f]{32}\Z")


class AmbientSessionError(ValueError):
    """Controlled refusal without participant names or retained content."""


class CaptureState(str, Enum):
    """Visible capture states; interruptions never resume automatically."""

    AWAITING_CONSENT = "awaiting_consent"
    READY = "ready"
    CAPTURING = "capturing"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    STOPPED = "stopped"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    DELETED = "deleted"
    DELETION_INCOMPLETE = "deletion_incomplete"


class AmbientArtifactKind(str, Enum):
    """Content categories sharing the same consent and destruction boundary."""

    AUDIO = "audio"
    TRANSCRIPT = "transcript"
    CACHE = "cache"
    DRAFT = "draft"


_TERMINAL = {
    CaptureState.WITHDRAWN,
    CaptureState.EXPIRED,
    CaptureState.DELETED,
    CaptureState.DELETION_INCOMPLETE,
}
_READABLE = {CaptureState.CAPTURING, CaptureState.PAUSED, CaptureState.STOPPED}


def _require(condition: bool, reason: str = "invalid_session_contract") -> None:
    if not condition:
        raise AmbientSessionError(reason)


def _reference(value: object) -> None:
    _require(type(value) is str and _REF.fullmatch(value) is not None)


def _participants(values: tuple[str, ...]) -> None:
    _require(type(values) is tuple and 1 <= len(values) <= 16)
    for value in values:
        _reference(value)
    _require(len(set(values)) == len(values))


@dataclass(frozen=True, slots=True, kw_only=True)
class AmbientSessionPolicy:
    """Caller-approved policy identity and bounded local retention.

    Args:
        revision: SHA-256 of the host's complete approved policy document.
        max_lifetime_ms: Non-renewable deadline, including consent acquisition.
        max_artifacts: Total references admitted over the entire session.
        max_bytes: Maximum owned bytes and, separately, exported review bytes.
        max_operations: Maximum state or content operations before refusal.
        allow_review_export: Explicit opt-in to copies outside session ownership.
    """

    revision: str
    max_lifetime_ms: int = 300_000
    max_artifacts: int = 64
    max_bytes: int = 4 * 1024 * 1024
    max_operations: int = 4096
    allow_review_export: bool = False

    def __post_init__(self) -> None:
        _require(
            type(self.revision) is str
            and re.fullmatch(r"[0-9a-f]{64}", self.revision) is not None
        )
        for value, maximum in (
            (self.max_lifetime_ms, 3_600_000),
            (self.max_artifacts, 256),
            (self.max_bytes, 64 * 1024 * 1024),
            (self.max_operations, 16_384),
        ):
            _require(type(value) is int and 1 <= value <= maximum)
        _require(type(self.allow_review_export) is bool)


@dataclass(frozen=True, slots=True)
class AmbientReference:
    """Opaque capability; only the original object is accepted by its session."""

    artifact_ref: str
    kind: AmbientArtifactKind
    generation: int


@dataclass(frozen=True, slots=True)
class AmbientReviewCopy:
    """Explicit local-review copy, outside the session's deletion guarantee."""

    reference: AmbientReference
    content: bytes = field(repr=False)
    draft_only: bool = field(default=True, init=False)
    requires_review: bool = field(default=True, init=False)


@dataclass(slots=True)
class _Owned:
    reference: AmbientReference
    content: bytearray = field(repr=False)


class AmbientSession:
    """Gate capture and references with unanimous consent and bounded retention.

    Args:
        session_ref: Random opaque 32-hex identifier, never a patient identifier.
        participants: Complete set of random participant references.
        policy: Host-approved local policy; not a recording-law determination.
        clock: Monotonic seconds. Invalid or backwards clocks destroy owned data.
        automatic_expiry: Enforce idle expiry with a daemon timer by default. If
            disabled, the host must call ``expire`` on a reliable idle scheduler.

    This class does not open a microphone, cancel an external provider, discover
    copies or erase storage outside its owned mutable buffers. The host must stop
    producers on every capture-state change and register external copies before
    admitting data. Use ``delete`` or a context manager for deterministic cleanup.
    """

    def __init__(
        self,
        session_ref: str,
        participants: tuple[str, ...],
        *,
        policy: AmbientSessionPolicy,
        clock: Callable[[], float] = time.monotonic,
        automatic_expiry: bool = True,
    ) -> None:
        _reference(session_ref)
        _participants(participants)
        _require(type(policy) is AmbientSessionPolicy and callable(clock))
        _require(type(automatic_expiry) is bool)
        self._lock = threading.RLock()
        self._clock = clock
        self._policy = policy
        self._session_ref = session_ref
        self._participants = set(participants)
        self._consents: set[str] = set()
        self._state = CaptureState.AWAITING_CONSENT
        self._generation = 0
        self._owned: dict[str, _Owned] = {}
        self._seen: set[str] = set()
        self._external = 0
        self._export_bytes = 0
        self._operations = 0
        self._cleanup_failed = False
        self._reason = "none"
        self._timer: threading.Timer | None = None
        now = self._now()
        _require(now is not None)
        assert now is not None
        self._last_time = self._started = now
        if automatic_expiry:
            self._schedule(policy.max_lifetime_ms / 1000)

    def __repr__(self) -> str:
        return "AmbientSession(content=<private>)"

    def __enter__(self) -> AmbientSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.delete()

    def _now(self) -> float | None:
        try:
            value = self._clock()
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                return float(value)
        except Exception:
            pass
        return None

    def _schedule(self, delay: float) -> None:
        self._timer = threading.Timer(delay, self._on_deadline)
        self._timer.daemon = True
        self._timer.start()

    def _on_deadline(self) -> None:
        with self._lock:
            self.expire()
            if self._state not in _TERMINAL:
                remaining = self._policy.max_lifetime_ms / 1000 - (
                    self._last_time - self._started
                )
                self._schedule(max(remaining, 0.001))

    def _invalidate(self, state: CaptureState, reason: str) -> None:
        self._generation += 1
        for owned in self._owned.values():
            try:
                for offset in range(0, len(owned.content), 4096):
                    end = min(offset + 4096, len(owned.content))
                    owned.content[offset:end] = b"\0" * (end - offset)
                if any(owned.content):
                    self._cleanup_failed = True
            except Exception:
                self._cleanup_failed = True
        self._owned.clear()
        self._consents.clear()
        self._reason = reason
        self._state = state
        if self._cleanup_failed:
            self._state = CaptureState.DELETION_INCOMPLETE
        if self._state in _TERMINAL and self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def expire(self) -> bool:
        """Apply the non-renewable deadline; useful for host lifecycle callbacks."""
        with self._lock:
            if self._state in _TERMINAL:
                return True
            now = self._now()
            if now is None or now < self._last_time:
                self._invalidate(CaptureState.EXPIRED, "clock_invalid")
                return True
            self._last_time = now
            if (now - self._started) * 1000 >= self._policy.max_lifetime_ms:
                self._invalidate(CaptureState.EXPIRED, "expired")
                return True
            return False

    def _active(self) -> None:
        _require(not self.expire(), "session_closed")
        self._operations += 1
        if self._operations > self._policy.max_operations:
            self._invalidate(CaptureState.EXPIRED, "operation_limit")
            raise AmbientSessionError("operation_limit")

    def consent(self, participant_ref: str) -> None:
        """Record host-acquired consent for this policy, cohort and generation."""
        with self._lock:
            self._active()
            _reference(participant_ref)
            _require(self._state is CaptureState.AWAITING_CONSENT, "invalid_transition")
            _require(participant_ref in self._participants, "unknown_participant")
            self._consents.add(participant_ref)
            if self._consents == self._participants:
                self._state = CaptureState.READY

    def transition(self, event: str) -> None:
        """Apply explicit start, pause, resume, interrupt or stop host actions."""
        transitions = {
            "start": ({CaptureState.READY}, CaptureState.CAPTURING),
            "pause": ({CaptureState.CAPTURING}, CaptureState.PAUSED),
            "resume": (
                {CaptureState.PAUSED, CaptureState.INTERRUPTED},
                CaptureState.CAPTURING,
            ),
            "interrupt": (
                {CaptureState.READY, CaptureState.CAPTURING, CaptureState.PAUSED},
                CaptureState.INTERRUPTED,
            ),
            "stop": (
                {
                    CaptureState.READY,
                    CaptureState.CAPTURING,
                    CaptureState.PAUSED,
                    CaptureState.INTERRUPTED,
                },
                CaptureState.STOPPED,
            ),
        }
        with self._lock:
            self._active()
            _require(type(event) is str and event in transitions, "invalid_transition")
            allowed, target = transitions[event]
            _require(self._state in allowed, "invalid_transition")
            _require(self._consents == self._participants, "consent_required")
            self._state = target

    def change_participants(self, participants: tuple[str, ...]) -> None:
        """Destroy the previous cohort's data and require fresh unanimous consent."""
        with self._lock:
            self._active()
            _participants(participants)
            _require(self._state is not CaptureState.STOPPED, "invalid_transition")
            target = (
                CaptureState.DELETION_INCOMPLETE
                if self._external
                else CaptureState.AWAITING_CONSENT
            )
            self._invalidate(target, "participants_changed")
            self._participants = set(participants)

    def withdraw(self) -> None:
        """Immediately revoke all references and overwrite every owned buffer."""
        with self._lock:
            if self._state not in _TERMINAL:
                self._invalidate(CaptureState.WITHDRAWN, "consent_withdrawn")

    def delete(self) -> None:
        """Invalidate and destroy owned data; external copies prevent verification."""
        with self._lock:
            if self._state in {CaptureState.DELETED, CaptureState.DELETION_INCOMPLETE}:
                return
            target = (
                CaptureState.DELETION_INCOMPLETE
                if self._external
                else CaptureState.DELETED
            )
            self._invalidate(target, "deleted")

    def retain(
        self, artifact_ref: str, kind: AmbientArtifactKind, content: bytes | bytearray
    ) -> AmbientReference:
        """Copy content into owned memory; the caller still owns its input copy.

        Callers must clear their input and declare retained external copies. Audio
        admission is possible only during visible capture. Other kinds may be
        admitted while capturing, paused or stopped, under the same deadline.
        """
        with self._lock:
            self._active()
            _reference(artifact_ref)
            _require(type(kind) is AmbientArtifactKind)
            _require(type(content) in (bytes, bytearray) and len(content) > 0)
            allowed = (
                {CaptureState.CAPTURING}
                if kind is AmbientArtifactKind.AUDIO
                else _READABLE
            )
            _require(self._state in allowed, "capture_not_admitted")
            _require(artifact_ref not in self._seen, "duplicate_reference")
            size = sum(len(item.content) for item in self._owned.values()) + len(
                content
            )
            if (
                len(self._seen) >= self._policy.max_artifacts
                or size > self._policy.max_bytes
            ):
                self._invalidate(CaptureState.EXPIRED, "retention_limit")
                raise AmbientSessionError("retention_limit")
            reference = AmbientReference(artifact_ref, kind, self._generation)
            self._owned[artifact_ref] = _Owned(reference, bytearray(content))
            self._seen.add(artifact_ref)
            _require(not self.expire(), "session_closed")
            return reference

    def _resolve(self, reference: AmbientReference) -> _Owned:
        _require(type(reference) is AmbientReference, "stale_reference")
        owned = self._owned.get(reference.artifact_ref)
        _require(owned is not None and owned.reference is reference, "stale_reference")
        _require(self._state in _READABLE, "capture_not_admitted")
        assert owned is not None
        return owned

    def use(
        self, reference: AmbientReference, reader: Callable[[memoryview], _T]
    ) -> _T:
        """Borrow a read-only view for a bounded local callback, without a copy.

        The host must not retain or copy the view, block indefinitely, log content
        or start network work. Declare any deliberately escaped copy as external.
        The callback runs under the session lock, so it must be short and local.
        """
        with self._lock:
            self._active()
            owned = self._resolve(reference)
            _require(callable(reader))
            try:
                with memoryview(owned.content).toreadonly() as view:
                    result = reader(view)
            except Exception:
                self._invalidate(CaptureState.EXPIRED, "reader_failed")
                raise AmbientSessionError("reader_failed") from None
            _require(not self.expire(), "session_closed")
            self._resolve(reference)
            return result

    def declare_external_copy(self) -> None:
        """Record an unowned copy; this session can never attest to its deletion."""
        with self._lock:
            self._active()
            if self._external >= self._policy.max_artifacts:
                self._invalidate(CaptureState.DELETION_INCOMPLETE, "external_limit")
                raise AmbientSessionError("external_limit")
            self._external += 1

    def export_for_review(self, reference: AmbientReference) -> AmbientReviewCopy:
        """Create one explicit local-review copy, always draft-only and unreviewed.

        Requires stopped capture and policy opt-in. Export is irreversible from
        this object's perspective and makes later deletion accounting incomplete.
        No clinical write or automatic note-finalization capability is provided.
        """
        with self._lock:
            self._active()
            _require(self._policy.allow_review_export, "export_not_permitted")
            _require(self._state is CaptureState.STOPPED, "invalid_transition")
            owned = self._resolve(reference)
            _require(
                reference.kind is not AmbientArtifactKind.AUDIO, "export_not_permitted"
            )
            _require(
                self._export_bytes + len(owned.content) <= self._policy.max_bytes,
                "export_limit",
            )
            self.declare_external_copy()
            self._export_bytes += len(owned.content)
            copy = AmbientReviewCopy(reference, bytes(owned.content))
            _require(not self.expire(), "session_closed")
            return copy

    def snapshot(self) -> dict[str, object]:
        """Return bounded, content-free state; snapshots cannot restore consent."""
        with self._lock:
            self.expire()
            return {
                "schema": "openmed.ambient.session.v1",
                "session_ref": self._session_ref,
                "policy_revision": self._policy.revision,
                "state": self._state.value,
                "generation": self._generation,
                "participants": len(self._participants),
                "consents": len(self._consents),
                "owned_artifacts": len(self._owned),
                "owned_bytes": sum(len(item.content) for item in self._owned.values()),
                "unaccounted_external_copies": self._external,
                "deletion_verified": self._state in _TERMINAL
                and not self._owned
                and not self._external
                and not self._cleanup_failed,
                "reason": self._reason,
                "draft_only": True,
                "requires_review": True,
            }

    def to_json(self) -> str:
        """Serialize controlled state, never retained bytes or participant names."""
        return json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"))
