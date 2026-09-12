"""Bounded, dependency-free GIF logical-screen header preflight."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO, Callable, Final, Literal

__all__ = [
    "DEFAULT_MAX_GIF_HEADER_BYTES",
    "DEFAULT_MAX_GIF_PIXELS",
    "GifDimensions",
    "GifDimensionsError",
    "read_gif_dimensions",
]

DEFAULT_MAX_GIF_HEADER_BYTES: Final[int] = 13 + 3 * 256
DEFAULT_MAX_GIF_PIXELS: Final[int] = 100_000_000


class GifDimensionsError(ValueError):
    """Value-free failure for malformed, truncated, or over-limit GIF headers."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class GifDimensions:
    """Logical-screen geometry and the declared global color-table size."""

    width: int
    height: int
    version: Literal["87a", "89a"]
    global_color_table_entries: int

    @property
    def global_color_table_bytes(self) -> int:
        """Return the declared palette size without exposing any palette entries."""
        return self.global_color_table_entries * 3


def read_gif_dimensions(
    source: bytes | BinaryIO,
    *,
    max_header_bytes: int = DEFAULT_MAX_GIF_HEADER_BYTES,
    max_pixels: int = DEFAULT_MAX_GIF_PIXELS,
) -> GifDimensions:
    """Read a GIF87a/GIF89a header and discard its bounded global color table.

    Parsing stops before the first image descriptor or extension. This is not
    image decoding, animation inspection, sanitization, or full-file validation.
    Streams start at their current position; seekable streams are restored on
    success or failure. Caller-owned streams are never closed.
    """
    if type(max_header_bytes) is not int or max_header_bytes <= 0:
        raise ValueError("max_header_bytes must be a positive integer")
    if type(max_pixels) is not int or max_pixels <= 0:
        raise ValueError("max_pixels must be a positive integer")
    if isinstance(source, bytes):
        return _parse_gif(_HeaderReader(source, max_header_bytes), max_pixels)
    read = getattr(source, "read", None)
    if not callable(read):
        raise TypeError("source must be bytes or a binary stream")
    position = _stream_position(source)
    try:
        return _parse_gif(_HeaderReader(read, max_header_bytes), max_pixels)
    finally:
        if position is not None:
            _restore_position(source, position)


def _parse_gif(reader: _HeaderReader, max_pixels: int) -> GifDimensions:
    header = reader.read_exact(13)
    signature = header[:6]
    if signature not in (b"GIF87a", b"GIF89a"):
        raise GifDimensionsError("gif_signature_invalid")
    width, height = struct.unpack_from("<HH", header, 6)
    if width == 0 or height == 0:
        raise GifDimensionsError("gif_dimensions_invalid")
    # Division checks the area without relying on fixed-width multiplication.
    if width > max_pixels // height:
        raise GifDimensionsError("gif_pixel_limit_exceeded")
    packed = header[10]
    entries = 1 << ((packed & 7) + 1) if packed & 0x80 else 0
    if entries and header[11] >= entries:
        raise GifDimensionsError("gif_background_index_invalid")
    # At most 768 bytes; no image descriptors, extensions, or pixels are read.
    reader.read_exact(entries * 3)
    return GifDimensions(
        width=width,
        height=height,
        version="87a" if signature == b"GIF87a" else "89a",
        global_color_table_entries=entries,
    )


class _HeaderReader:
    def __init__(self, source: bytes | Callable[[int], bytes], limit: int) -> None:
        self._buffer = memoryview(source) if isinstance(source, bytes) else None
        self._read = source if callable(source) else None
        self._limit = limit
        self.offset = 0

    def read_exact(self, size: int) -> bytes:
        if size > self._limit - self.offset:
            raise GifDimensionsError("gif_header_limit_exceeded")
        if self._buffer is not None:
            end = self.offset + size
            if end > len(self._buffer):
                raise GifDimensionsError("gif_header_truncated")
            result = bytes(self._buffer[self.offset : end])
            self.offset = end
            return result
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = _read_chunk(self._read, remaining)
            if not chunk:
                raise GifDimensionsError("gif_header_truncated")
            chunks.append(chunk)
            self.offset += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


def _read_chunk(read: Callable[[int], bytes] | None, size: int) -> bytes:
    if read is None:
        raise GifDimensionsError("gif_stream_contract_error")
    try:
        chunk = read(size)
    except Exception:
        pass
    else:
        if isinstance(chunk, bytes) and len(chunk) <= size:
            return chunk
        raise GifDimensionsError("gif_stream_contract_error")
    # Raise outside the handler so an underlying I/O message is not retained.
    raise GifDimensionsError("gif_stream_read_error")


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
    raise GifDimensionsError("gif_stream_position_error")


def _restore_position(stream: BinaryIO, position: int) -> None:
    try:
        stream.seek(position)
    except Exception:
        pass
    else:
        return
    raise GifDimensionsError("gif_stream_restore_error")
