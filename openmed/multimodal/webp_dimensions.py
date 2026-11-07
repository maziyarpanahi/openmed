"""Bounded, dependency-free WebP geometry preflight."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO, Callable, Final, Literal

__all__ = [
    "DEFAULT_MAX_WEBP_HEADER_BYTES",
    "DEFAULT_MAX_WEBP_PIXELS",
    "WebpDimensions",
    "WebpDimensionsError",
    "read_webp_dimensions",
]

DEFAULT_MAX_WEBP_HEADER_BYTES: Final[int] = 30
DEFAULT_MAX_WEBP_PIXELS: Final[int] = 100_000_000
_MAX_RIFF_SIZE: Final[int] = (1 << 32) - 10
_MAX_CANVAS_PIXELS: Final[int] = (1 << 32) - 1


class WebpDimensionsError(ValueError):
    """Value-free failure for malformed, unsupported, or over-limit WebP headers."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class WebpDimensions:
    """Declared canvas geometry and flags, not decoded-image validation."""

    width: int
    height: int
    chunk_type: Literal["VP8", "VP8L", "VP8X"]
    has_alpha: bool
    is_animated: bool


def read_webp_dimensions(
    source: bytes | BinaryIO,
    *,
    max_header_bytes: int = DEFAULT_MAX_WEBP_HEADER_BYTES,
    max_pixels: int = DEFAULT_MAX_WEBP_PIXELS,
) -> WebpDimensions:
    """Read geometry from the first VP8, VP8L, or VP8X chunk of a RIFF/WEBP file.

    Reads at most 30 bytes. The declared first-chunk extent must fit the RIFF
    envelope, but pixel payloads, padding, later chunks, and actual full-file
    length are not read or validated. Unknown first-chunk layouts are refused.
    For VP8X, the result is the declared canvas, including its animation flag;
    it is not a promise that frames or metadata are valid or even present.
    Seekable streams are restored on success/failure; streams are never closed.
    """
    if type(max_header_bytes) is not int or max_header_bytes <= 0:
        raise ValueError("max_header_bytes must be a positive integer")
    if type(max_pixels) is not int or max_pixels <= 0:
        raise ValueError("max_pixels must be a positive integer")
    if isinstance(source, bytes):
        return _parse_webp(_HeaderReader(source, max_header_bytes), max_pixels)
    read = getattr(source, "read", None)
    if not callable(read):
        raise TypeError("source must be bytes or a binary stream")
    position = _stream_position(source)
    try:
        return _parse_webp(_HeaderReader(read, max_header_bytes), max_pixels)
    finally:
        if position is not None:
            _restore_position(source, position)


def _parse_webp(reader: _HeaderReader, max_pixels: int) -> WebpDimensions:
    header = reader.read_exact(12)
    if header[:4] != b"RIFF" or header[8:] != b"WEBP":
        raise WebpDimensionsError("webp_signature_invalid")
    riff_size = struct.unpack_from("<I", header, 4)[0]
    if riff_size < 12 or riff_size > _MAX_RIFF_SIZE or riff_size & 1:
        raise WebpDimensionsError("webp_riff_size_invalid")
    chunk_header = reader.read_exact(8)
    kind = chunk_header[:4]
    chunk_size = struct.unpack_from("<I", chunk_header, 4)[0]
    if chunk_size + (chunk_size & 1) > riff_size - 12:
        raise WebpDimensionsError("webp_chunk_size_invalid")
    if kind == b"VP8 ":
        if chunk_size < 10:
            raise WebpDimensionsError("webp_chunk_size_invalid")
        payload = reader.read_exact(10)
        if payload[0] & 1 or payload[3:6] != b"\x9d\x01\x2a":
            raise WebpDimensionsError("webp_vp8_header_invalid")
        if (payload[0] >> 1) & 7 > 3:
            raise WebpDimensionsError("webp_vp8_version_unsupported")
        raw_width, raw_height = struct.unpack_from("<HH", payload, 6)
        result = WebpDimensions(
            raw_width & 0x3FFF, raw_height & 0x3FFF, "VP8", False, False
        )
    elif kind == b"VP8L":
        if chunk_size < 5:
            raise WebpDimensionsError("webp_chunk_size_invalid")
        payload = reader.read_exact(5)
        if payload[0] != 0x2F:
            raise WebpDimensionsError("webp_vp8l_header_invalid")
        bits = struct.unpack_from("<I", payload, 1)[0]
        if bits >> 29:
            raise WebpDimensionsError("webp_vp8l_version_unsupported")
        result = WebpDimensions(
            (bits & 0x3FFF) + 1,
            ((bits >> 14) & 0x3FFF) + 1,
            "VP8L",
            bool(bits & (1 << 28)),
            False,
        )
    elif kind == b"VP8X":
        if chunk_size != 10:
            raise WebpDimensionsError("webp_chunk_size_invalid")
        payload = reader.read_exact(10)
        if payload[0] & 0xC1 or payload[1:4] != b"\x00\x00\x00":
            raise WebpDimensionsError("webp_vp8x_reserved_bits_invalid")
        result = WebpDimensions(
            int.from_bytes(payload[4:7], "little") + 1,
            int.from_bytes(payload[7:10], "little") + 1,
            "VP8X",
            bool(payload[0] & 0x10),
            bool(payload[0] & 0x02),
        )
    else:
        raise WebpDimensionsError("webp_layout_unsupported")
    if (
        result.width == 0
        or result.height == 0
        or result.width > _MAX_CANVAS_PIXELS // result.height
    ):
        raise WebpDimensionsError("webp_dimensions_invalid")
    if result.width > max_pixels // result.height:
        raise WebpDimensionsError("webp_pixel_limit_exceeded")
    return result


class _HeaderReader:
    def __init__(self, source: bytes | Callable[[int], bytes], limit: int) -> None:
        self._buffer = memoryview(source) if isinstance(source, bytes) else None
        self._read = source if callable(source) else None
        self._limit = limit
        self.offset = 0

    def read_exact(self, size: int) -> bytes:
        if size > self._limit - self.offset:
            raise WebpDimensionsError("webp_header_limit_exceeded")
        if self._buffer is not None:
            end = self.offset + size
            if end > len(self._buffer):
                raise WebpDimensionsError("webp_header_truncated")
            result = bytes(self._buffer[self.offset : end])
            self.offset = end
            return result
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = _read_chunk(self._read, remaining)
            if not chunk:
                raise WebpDimensionsError("webp_header_truncated")
            chunks.append(chunk)
            self.offset += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


def _read_chunk(read: Callable[[int], bytes] | None, size: int) -> bytes:
    if read is None:
        raise WebpDimensionsError("webp_stream_contract_error")
    try:
        chunk = read(size)
    except Exception:
        pass
    else:
        if isinstance(chunk, bytes) and len(chunk) <= size:
            return chunk
        raise WebpDimensionsError("webp_stream_contract_error")
    # Raise outside the handler so an underlying I/O message is not retained.
    raise WebpDimensionsError("webp_stream_read_error")


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
    raise WebpDimensionsError("webp_stream_position_error")


def _restore_position(stream: BinaryIO, position: int) -> None:
    try:
        stream.seek(position)
    except Exception:
        pass
    else:
        return
    raise WebpDimensionsError("webp_stream_restore_error")
