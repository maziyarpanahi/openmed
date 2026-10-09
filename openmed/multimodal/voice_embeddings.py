"""Session-only biometric handles; never serialize or hash voice vector values."""

from __future__ import annotations

import hashlib
import math
import secrets
import weakref
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Literal, NoReturn, SupportsIndex

__all__ = [
    "EmbeddingDestructionReceipt",
    "VoiceEmbeddingError",
    "VoiceEmbeddingHandle",
    "VoiceEmbeddingSession",
    "VoiceSimilarity",
]

Boundary = Literal["pause", "withdrawal", "cancellation", "finalization"]


class VoiceEmbeddingError(ValueError):
    """Value-free refusal with a stable reason code."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class EmbeddingDestructionReceipt:
    """Evidence for erased owned buffers, without vector values or dimensions."""

    reason_code: str
    destroyed_count: int
    handle_digests: tuple[str, ...]


@dataclass(frozen=True)
class VoiceSimilarity:
    """Non-diagnostic cosine result, never a speaker identity or clinical role."""

    score: float
    notice: str = "non_diagnostic_voice_similarity"
    reviewer_confirmation_required: bool = True


class _Opaque:
    __slots__ = ()

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise VoiceEmbeddingError("embedding_serialization_refused")

    def __getstate__(self) -> NoReturn:
        raise VoiceEmbeddingError("embedding_serialization_refused")


class VoiceEmbeddingHandle(_Opaque):
    """Opaque reference that keeps neither an embedding nor its session alive.

    Obtain handles through ``VoiceEmbeddingSession.add``. Evidence and repr
    contain a digest of random reference bytes, never a digest of voice values.
    """

    __slots__ = ("_owner", "_digest", "_session_token")

    def __init__(self, owner: VoiceEmbeddingSession, digest: str) -> None:
        if (
            type(owner) is not VoiceEmbeddingSession
            or type(digest) is not str
            or digest not in owner._buffers
        ):
            raise VoiceEmbeddingError("embedding_handle_invalid")
        self._owner = weakref.ref(owner)
        self._digest = digest
        self._session_token = owner._session_token

    def __repr__(self) -> str:
        return f"VoiceEmbeddingHandle({self._digest})"

    def to_evidence(self) -> dict[str, str]:
        """Return only an opaque handle digest for evidence export."""
        return {"handle_digest": self._digest}

    def similarity(self, other: VoiceEmbeddingHandle) -> VoiceSimilarity:
        """Compare within the owning live session, refusing cross-session use."""
        owner = self._owner()
        if (
            type(other) is not VoiceEmbeddingHandle
            or other._session_token is not self._session_token
        ):
            raise VoiceEmbeddingError("embedding_cross_session_refused")
        if owner is None:
            raise VoiceEmbeddingError("embedding_destroyed")
        return owner._similarity(self, other)

    def persist(self) -> None:
        """Refuse persistence; a separately reviewed policy is required elsewhere."""
        raise VoiceEmbeddingError("embedding_persistence_refused")


class VoiceEmbeddingSession(_Opaque):
    """Own bounded mutable buffers and erase them at a session boundary.

    Args:
        allocator: Trusted local allocator returning a fresh, zero-filled
            ``array('d')`` of the requested size. Ownership transfers here.
            It receives a count only, never source values.
        register: Optional host retention hook, called with an opaque digest and
            an idempotent erasure callback returning a value-free receipt. The
            callback retains neither buffers nor this session. This is an
            embedding-owned seam, not an audio-retention or consent API.
        max_handles: Maximum live handles per session.
        max_dimensions: Maximum values per embedding.

    The host must invoke ``destroy`` on pause, withdrawal, cancellation and
    finalization. Registered callbacks erase individual buffers. Any boundary
    permanently invalidates this owner; resumption needs a new owner and fresh embeddings.
    """

    __slots__ = (
        "_buffers",
        "_lock",
        "_closed",
        "_allocator",
        "_register",
        "_max_handles",
        "_max_dimensions",
        "_session_token",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        allocator: Callable[[int], array] | None = None,
        register: Callable[[str, Callable[[], EmbeddingDestructionReceipt]], None]
        | None = None,
        max_handles: int = 256,
        max_dimensions: int = 4096,
    ) -> None:
        if (
            type(max_handles) is not int
            or type(max_dimensions) is not int
            or not 1 <= max_handles <= 65536
            or not 1 <= max_dimensions <= 65536
        ):
            raise VoiceEmbeddingError("embedding_limits_invalid")
        self._session_token = object()
        self._buffers: dict[str, array] = {}
        self._lock = RLock()
        self._closed = False
        self._allocator = allocator or (lambda size: array("d", [0.0]) * size)
        self._register = register
        self._max_handles = max_handles
        self._max_dimensions = max_dimensions

    def __repr__(self) -> str:
        return f"VoiceEmbeddingSession(handle_count={len(self._buffers)})"

    def add(self, values: Sequence[float]) -> VoiceEmbeddingHandle:
        """Copy a finite nonzero synthetic/provider vector into owned storage.

        Caller-owned source values are not erased; the caller/provider must
        independently manage those allocations. No model or transport is used.
        """
        with self._lock:
            if self._closed:
                raise VoiceEmbeddingError("embedding_session_closed")
            if len(self._buffers) >= self._max_handles:
                raise VoiceEmbeddingError("embedding_capacity_exceeded")
            # Do not coerce arbitrary objects whose conversion can leak payloads.
            if type(values) not in (list, tuple, array):
                raise VoiceEmbeddingError("embedding_vector_invalid")
            if not 1 <= len(values) <= self._max_dimensions or any(
                type(value) not in (int, float)
                or abs(value) > 1.7976931348623157e308
                or not math.isfinite(value)
                for value in values
            ):
                raise VoiceEmbeddingError("embedding_vector_invalid")
            if not any(value != 0 for value in values):
                raise VoiceEmbeddingError("embedding_vector_invalid")
            buffer = None
            failed = False
            try:
                digest = hashlib.sha256(secrets.token_bytes(32)).hexdigest()
                buffer = self._allocator(len(values))
                if (
                    type(buffer) is not array
                    or buffer.typecode != "d"
                    or len(buffer) != len(values)
                    or any(buffer)
                    or any(buffer is existing for existing in self._buffers.values())
                ):
                    raise VoiceEmbeddingError("embedding_allocator_failed")
                for index, value in enumerate(values):
                    buffer[index] = value
            except Exception:
                failed = True
            if failed:
                if type(buffer) is array and not any(
                    buffer is existing for existing in self._buffers.values()
                ):
                    _erase(buffer)
                raise VoiceEmbeddingError("embedding_allocator_failed")
            assert buffer is not None
            self._buffers[digest] = buffer
            owner_ref = weakref.ref(self)

            def erase() -> EmbeddingDestructionReceipt:
                owner = owner_ref()
                if owner is None:
                    return EmbeddingDestructionReceipt("embedding_destroyed", 0, ())
                return owner._erase_handle(digest)

            failed = False
            try:
                if self._register is not None:
                    self._register(digest, erase)
            except Exception:
                failed = True
            if failed:
                self._erase_handle(digest)
                raise VoiceEmbeddingError("embedding_registration_failed")
            if self._closed or digest not in self._buffers:
                self._erase_handle(digest)
                raise VoiceEmbeddingError("embedding_destroyed")
            return VoiceEmbeddingHandle(self, digest)

    def diagnostics(self) -> dict[str, object]:
        """Return counts and random handle digests only."""
        with self._lock:
            return {
                "handle_count": len(self._buffers),
                "handle_digests": tuple(sorted(self._buffers)),
            }

    def destroy(
        self, boundary: Boundary = "finalization"
    ) -> EmbeddingDestructionReceipt:
        """Erase and release every owned buffer, permanently closing this owner."""
        if boundary not in ("pause", "withdrawal", "cancellation", "finalization"):
            raise VoiceEmbeddingError("embedding_boundary_invalid")
        with self._lock:
            self._closed = True
            digests = tuple(sorted(self._buffers))
            for buffer in self._buffers.values():
                _erase(buffer)
            self._buffers.clear()
            return EmbeddingDestructionReceipt(boundary, len(digests), digests)

    def _erase_handle(self, digest: str) -> EmbeddingDestructionReceipt:
        with self._lock:
            buffer = self._buffers.pop(digest, None)
            if buffer is None:
                return EmbeddingDestructionReceipt("embedding_destroyed", 0, ())
            _erase(buffer)
            return EmbeddingDestructionReceipt("embedding_destroyed", 1, (digest,))

    def _similarity(
        self, left: VoiceEmbeddingHandle, right: VoiceEmbeddingHandle
    ) -> VoiceSimilarity:
        with self._lock:
            a = self._buffers.get(left._digest)
            b = self._buffers.get(right._digest)
            if a is None or b is None:
                raise VoiceEmbeddingError("embedding_destroyed")
            if len(a) != len(b):
                raise VoiceEmbeddingError("embedding_dimensions_mismatch")
            # Scaling avoids overflow for finite but very large vectors.
            scale_a, scale_b = max(map(abs, a)), max(map(abs, b))
            norm_a = math.sqrt(math.fsum((value / scale_a) ** 2 for value in a))
            norm_b = math.sqrt(math.fsum((value / scale_b) ** 2 for value in b))
            score = math.fsum(
                (x / scale_a / norm_a) * (y / scale_b / norm_b) for x, y in zip(a, b)
            )
            return VoiceSimilarity(max(-1.0, min(1.0, score)))

    def __del__(self) -> None:
        if hasattr(self, "_buffers"):
            self.destroy()


def _erase(buffer: array) -> None:
    # Overwrite before releasing backing storage, including injected aliases.
    for index in range(len(buffer)):
        buffer[index] = "\0" if buffer.typecode in ("u", "w") else 0
    del buffer[:]
