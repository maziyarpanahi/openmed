"""Value-free model lifecycle events (``openmed.model.lifecycle.v1``).

Model acquisition, verification, activation and eviction report what happened
to a model through one closed event vocabulary that an operator, the service
or a kit can subscribe to. Events carry a Hub repository id, a revision commit
hash, file and byte counts, a content digest and a controlled reason code. They
never carry a URL, a token, a filename, a filesystem path or exception text.

Emitters take an injectable :class:`ModelLifecycleSink`. The default sink is
:data:`NULL_MODEL_LIFECYCLE_SINK`, which drops every event, so nothing is
recorded or sent anywhere unless a caller passes its own sink. A sink that
raises never changes the outcome of the operation that emitted the event.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Protocol, runtime_checkable

from .errors import LegacyRootedError

MODEL_LIFECYCLE_SCHEMA_VERSION: Final[str] = "openmed.model.lifecycle.v1"


class ModelLifecycleEventType(str, Enum):
    """Closed lifecycle event vocabulary.

    Values:
        REQUESTED: A model snapshot was requested.
        FETCHING: One file download is starting.
        VERIFIED: One file matched its Hub size and digest.
        INTEGRITY_FAILED: One file did not match its Hub size or digest.
        READY: Every requested file is present and verified, or the snapshot
            was served from the local cache in offline mode.
        ACTIVATED: A verified model generation became the active one.
        ROLLED_BACK: Activation was rolled back to the previous generation.
        EVICTED: One cache artifact was removed under the cache quota.
        FAILED: The operation stopped; ``reason`` says why.
    """

    REQUESTED = "requested"
    FETCHING = "fetching"
    VERIFIED = "verified"
    INTEGRITY_FAILED = "integrity_failed"
    READY = "ready"
    ACTIVATED = "activated"
    ROLLED_BACK = "rolled_back"
    EVICTED = "evicted"
    FAILED = "failed"


MODEL_LIFECYCLE_EVENT_TYPES: Final[tuple[str, ...]] = tuple(
    item.value for item in ModelLifecycleEventType
)

#: Reason codes the Python emitters in this package produce. Activation and
#: rollback callers may pass their own controlled codes of the same shape.
MODEL_LIFECYCLE_REASON_CODES: Final[tuple[str, ...]] = (
    "artifact_absent",
    "commit_unverifiable",
    "corrupt_file_unremovable",
    "digest_mismatch",
    "download_error",
    "file_missing",
    "integrity_retry_exhausted",
    "integrity_unverifiable",
    "metadata_invalid_filename",
    "metadata_missing_digest",
    "metadata_missing_size",
    "no_matching_files",
    "offline_cache_miss",
    "revision_name_unsafe",
    "size_mismatch",
)

EVENT_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "event",
    "repo_id",
    "revision_sha",
    "file_count",
    "byte_count",
    "digest",
    "reason",
)

_REPO_ID_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
)
_REVISION_SHA_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}|git-sha1:[0-9a-f]{40}")
_REASON_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_MAX_COUNT: Final[int] = (1 << 63) - 1
_REASON_REQUIRED: Final[frozenset[ModelLifecycleEventType]] = frozenset(
    {
        ModelLifecycleEventType.INTEGRITY_FAILED,
        ModelLifecycleEventType.ROLLED_BACK,
        ModelLifecycleEventType.FAILED,
    }
)
_FILE_EVENTS: Final[frozenset[ModelLifecycleEventType]] = frozenset(
    {
        ModelLifecycleEventType.FETCHING,
        ModelLifecycleEventType.VERIFIED,
        ModelLifecycleEventType.INTEGRITY_FAILED,
    }
)


class ModelLifecycleEventError(ValueError, LegacyRootedError):
    """Raised with a value-free message when an event field is invalid."""


def _optional_match(value: object, pattern: re.Pattern[str], name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ModelLifecycleEventError(f"invalid model lifecycle {name}")


def _optional_count(value: object, name: str) -> None:
    if value is None:
        return
    if type(value) is not int or not 0 <= value <= _MAX_COUNT:
        raise ModelLifecycleEventError(f"invalid model lifecycle {name}")


@dataclass(frozen=True, slots=True)
class ModelLifecycleEvent:
    """One value-free model lifecycle event.

    Attributes:
        event: Event type.
        repo_id: Hub repository id, or ``None`` when the emitter does not know
            it (cache eviction tracks digests, not repositories).
        revision_sha: Resolved 40- or 64-hex commit hash, when known.
        file_count: Number of files the event covers, when known.
        byte_count: Number of bytes the event covers, when known.
        digest: ``sha256:<hex>`` or ``git-sha1:<hex>`` content digest.
        reason: Controlled reason code for failures, rollbacks and
            unusual evictions.
    """

    event: ModelLifecycleEventType
    repo_id: str | None = None
    revision_sha: str | None = None
    file_count: int | None = None
    byte_count: int | None = None
    digest: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        try:
            event = ModelLifecycleEventType(self.event)
        except ValueError:
            raise ModelLifecycleEventError("invalid model lifecycle event") from None
        object.__setattr__(self, "event", event)
        _optional_match(self.repo_id, _REPO_ID_RE, "repo_id")
        _optional_match(self.revision_sha, _REVISION_SHA_RE, "revision_sha")
        _optional_count(self.file_count, "file_count")
        _optional_count(self.byte_count, "byte_count")
        _optional_match(self.digest, _DIGEST_RE, "digest")
        _optional_match(self.reason, _REASON_RE, "reason")
        if event in _REASON_REQUIRED and self.reason is None:
            raise ModelLifecycleEventError("model lifecycle event needs a reason")
        if event in _FILE_EVENTS and (
            self.file_count != 1 or self.byte_count is None or self.digest is None
        ):
            raise ModelLifecycleEventError(
                "model lifecycle file event needs one file, bytes and a digest"
            )
        if event is ModelLifecycleEventType.EVICTED and self.byte_count is None:
            raise ModelLifecycleEventError("model lifecycle eviction needs bytes")

    def to_dict(self) -> dict[str, Any]:
        """Return the event in :data:`EVENT_FIELDS` order."""

        return {
            "schema_version": MODEL_LIFECYCLE_SCHEMA_VERSION,
            "event": self.event.value,
            "repo_id": self.repo_id,
            "revision_sha": self.revision_sha,
            "file_count": self.file_count,
            "byte_count": self.byte_count,
            "digest": self.digest,
            "reason": self.reason,
        }


@runtime_checkable
class ModelLifecycleSink(Protocol):
    """Receives model lifecycle events. Implementations must not block long."""

    def emit(self, event: ModelLifecycleEvent) -> None:
        """Receive one event."""


class NullModelLifecycleSink:
    """Default sink: drops every event and keeps no state."""

    __slots__ = ()

    def emit(self, event: ModelLifecycleEvent) -> None:
        """Drop ``event``."""

        del event


NULL_MODEL_LIFECYCLE_SINK: Final[NullModelLifecycleSink] = NullModelLifecycleSink()


class RecordingModelLifecycleSink:
    """Keep events in memory, in emission order, for tests and local tools."""

    def __init__(self) -> None:
        self._events: list[ModelLifecycleEvent] = []

    def emit(self, event: ModelLifecycleEvent) -> None:
        """Record ``event``."""

        self._events.append(event)

    @property
    def events(self) -> tuple[ModelLifecycleEvent, ...]:
        """Return the recorded events in order."""

        return tuple(self._events)

    def to_list(self) -> list[dict[str, Any]]:
        """Return the recorded events as dictionaries."""

        return [event.to_dict() for event in self._events]


def resolve_sink(sink: ModelLifecycleSink | None) -> ModelLifecycleSink:
    """Return ``sink`` or the no-op default.

    Raises:
        TypeError: If ``sink`` has no callable ``emit``.
    """

    if sink is None:
        return NULL_MODEL_LIFECYCLE_SINK
    if not callable(getattr(sink, "emit", None)):
        raise TypeError("lifecycle_sink must provide an emit(event) method")
    return sink


def emit_event(sink: ModelLifecycleSink | None, event: ModelLifecycleEvent) -> None:
    """Deliver ``event`` to ``sink``, ignoring sink failures.

    Observability must not change the operation it observes, so an exception
    raised by the sink is dropped without logging its message.
    """

    target = resolve_sink(sink)
    if target is NULL_MODEL_LIFECYCLE_SINK:
        return
    try:
        target.emit(event)
    except Exception:  # noqa: BLE001 - sinks must not break model operations
        return


def emit_model_activated(
    sink: ModelLifecycleSink | None,
    *,
    repo_id: str | None,
    revision_sha: str | None,
    digest: str | None = None,
) -> ModelLifecycleEvent:
    """Emit ``activated`` for a model generation that became active.

    This is the hook a transactional activation path calls after commit.

    Args:
        sink: Event sink; ``None`` drops the event.
        repo_id: Repository of the generation that became active.
        revision_sha: Commit hash of the generation that became active.
        digest: Optional artifact digest of that generation.

    Returns:
        The emitted event.

    Raises:
        ModelLifecycleEventError: If a value does not match its value-free
            shape.
    """

    event = ModelLifecycleEvent(
        ModelLifecycleEventType.ACTIVATED,
        repo_id=repo_id,
        revision_sha=revision_sha,
        digest=digest,
    )
    emit_event(sink, event)
    return event


def emit_model_rolled_back(
    sink: ModelLifecycleSink | None,
    *,
    repo_id: str | None,
    revision_sha: str | None,
    reason: str,
    digest: str | None = None,
) -> ModelLifecycleEvent:
    """Emit ``rolled_back`` when activation returns to the previous generation.

    Args:
        sink: Event sink; ``None`` drops the event.
        repo_id: Repository of the generation that was rolled back.
        revision_sha: Commit hash of the generation that was rolled back.
        reason: Controlled reason code, such as ``rollback_requested``.
        digest: Optional artifact digest of that generation.

    Returns:
        The emitted event.

    Raises:
        ModelLifecycleEventError: If a value does not match its value-free
            shape.
    """

    event = ModelLifecycleEvent(
        ModelLifecycleEventType.ROLLED_BACK,
        repo_id=repo_id,
        revision_sha=revision_sha,
        digest=digest,
        reason=reason,
    )
    emit_event(sink, event)
    return event


def events_to_list(events: Sequence[ModelLifecycleEvent]) -> list[dict[str, Any]]:
    """Return events as dictionaries in order."""

    return [event.to_dict() for event in events]


__all__ = [
    "EVENT_FIELDS",
    "MODEL_LIFECYCLE_EVENT_TYPES",
    "MODEL_LIFECYCLE_REASON_CODES",
    "MODEL_LIFECYCLE_SCHEMA_VERSION",
    "NULL_MODEL_LIFECYCLE_SINK",
    "ModelLifecycleEvent",
    "ModelLifecycleEventError",
    "ModelLifecycleEventType",
    "ModelLifecycleSink",
    "NullModelLifecycleSink",
    "RecordingModelLifecycleSink",
    "emit_event",
    "emit_model_activated",
    "emit_model_rolled_back",
    "events_to_list",
    "resolve_sink",
]
