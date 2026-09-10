"""Bounded SHA-256 digests for in-memory and streamed multimodal assets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import BinaryIO, Callable

__all__ = [
    "AssetDigest",
    "DigestLimitExceededError",
    "DigestStreamError",
    "digest_asset",
]

_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class AssetDigest:
    """SHA-256 hex digest and the number of bytes hashed."""

    sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("sha256 must be a lowercase hexadecimal digest")
        if type(self.byte_count) is not int or self.byte_count < 0:
            raise ValueError("byte_count must be a non-negative integer")


class DigestLimitExceededError(ValueError):
    """Raised when an asset exceeds the configured byte limit."""

    category = "digest_size_limit"

    def __init__(self, *, maximum_bytes: int, bytes_read: int) -> None:
        self.maximum_bytes = maximum_bytes
        self.bytes_read = bytes_read
        super().__init__(
            f"{self.category}: maximum_bytes={maximum_bytes}, bytes_read={bytes_read}"
        )


class DigestStreamError(RuntimeError):
    """Value-free error raised when a binary stream cannot be hashed safely."""


def digest_asset(
    source: bytes | BinaryIO,
    *,
    max_bytes: int | None = None,
) -> AssetDigest:
    """Return the SHA-256 hex digest and byte count for ``source``.

    Binary streams are read from their current position in bounded chunks. A
    seekable stream is restored to that position before this function returns
    or raises, and caller-owned streams are never closed.
    """

    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise ValueError("max_bytes must be a non-negative integer or None")

    if isinstance(source, bytes):
        byte_count = len(source)
        _check_limit(byte_count, max_bytes)
        return AssetDigest(hashlib.sha256(source).hexdigest(), byte_count)

    read = getattr(source, "read", None)
    if not callable(read):
        raise TypeError("source must be bytes or a binary stream")

    initial_position = _stream_position(source)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        while True:
            request_bytes = _CHUNK_BYTES
            if max_bytes is not None:
                request_bytes = min(request_bytes, max_bytes - byte_count + 1)
            chunk = _read_chunk(read, request_bytes)
            if not chunk:
                break
            byte_count += len(chunk)
            _check_limit(byte_count, max_bytes)
            digest.update(chunk)
    finally:
        if initial_position is not None:
            _restore_position(source, initial_position)

    return AssetDigest(digest.hexdigest(), byte_count)


def _stream_position(stream: BinaryIO) -> int | None:
    seekable = getattr(stream, "seekable", None)
    if not callable(seekable):
        return None
    try:
        if not seekable():
            return None
        position = stream.tell()
    except Exception:
        pass
    else:
        if type(position) is int and position >= 0:
            return position
    raise DigestStreamError("digest_stream_position_error")


def _read_chunk(read: Callable[[int], bytes], request_bytes: int) -> bytes:
    try:
        chunk = read(request_bytes)
    except Exception:
        pass
    else:
        if isinstance(chunk, bytes) and len(chunk) <= request_bytes:
            return chunk
        raise DigestStreamError("digest_stream_contract_error")
    raise DigestStreamError("digest_stream_read_error")


def _restore_position(stream: BinaryIO, position: int) -> None:
    try:
        stream.seek(position)
    except Exception:
        pass
    else:
        return
    raise DigestStreamError("digest_stream_restore_error")


def _check_limit(byte_count: int, max_bytes: int | None) -> None:
    if max_bytes is not None and byte_count > max_bytes:
        raise DigestLimitExceededError(
            maximum_bytes=max_bytes,
            bytes_read=byte_count,
        )
