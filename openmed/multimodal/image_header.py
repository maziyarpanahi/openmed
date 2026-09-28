"""Bounded, dependency-free PNG and JPEG header preflight.

Dimensions and categorical encoding metadata are read from declared header
bytes so an orchestrator can size work before importing an imaging library or
a vision backend. No pixel, palette, or entropy-coded data is decoded or
returned, and caller-owned streams are never closed.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, BinaryIO, Callable, Final

DEFAULT_MAX_IMAGE_HEADER_BYTES: Final[int] = 64 * 1024
DEFAULT_MAX_IMAGE_PIXELS: Final[int] = 100_000_000
DEFAULT_MAX_JPEG_MARKERS: Final[int] = 256

_PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"
_PNG_IHDR_LENGTH: Final[int] = 13
_PNG_MAX_DIMENSION: Final[int] = (1 << 31) - 1
_PNG_ALLOWED_DEPTHS: Final[dict[int, tuple[int, ...]]] = {
    0: (1, 2, 4, 8, 16),
    2: (8, 16),
    3: (1, 2, 4, 8),
    4: (8, 16),
    6: (8, 16),
}
_PNG_COMPONENTS: Final[dict[int, int]] = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}

_JPEG_SOI: Final[bytes] = b"\xff\xd8"
_JPEG_EOI: Final[int] = 0xD9
_JPEG_SOS: Final[int] = 0xDA
_JPEG_START_OF_IMAGE: Final[int] = 0xD8
_JPEG_STANDALONE: Final[frozenset[int]] = frozenset({0x01, *range(0xD0, 0xD8)})
_JPEG_SOF_MARKERS: Final[frozenset[int]] = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)
_JPEG_PROGRESSIVE_SOF: Final[frozenset[int]] = frozenset({0xC2, 0xC6, 0xCA, 0xCE})
_JPEG_MAX_FILL_BYTES: Final[int] = 64
_JPEG_MIN_SOF_LENGTH: Final[int] = 8
_JPEG_MAX_PRECISION: Final[int] = 16

_SKIP_CHUNK_BYTES: Final[int] = 8192


class ImageFormat(str, Enum):
    """Closed set of formats this preflight helper understands.

    Values:
        PNG: A PNG datastream introduced by the eight-byte signature.
        JPEG: A JFIF/EXIF-style JPEG datastream introduced by SOI.
    """

    PNG = "png"
    JPEG = "jpeg"


class ImageHeaderError(ValueError):
    """Value-free failure for malformed or unsupported image headers."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class ImageHeader:
    """Declared image geometry without pixels, palettes, or text metadata.

    Attributes:
        image_format: Which header grammar produced this record.
        width: Declared width in pixels.
        height: Declared height in pixels.
        bit_depth: PNG IHDR bit depth, or JPEG sample precision.
        component_count: Channels implied by the PNG color type, or the JPEG
            frame component count.
        color_type: PNG color type, or ``None`` for JPEG.
        progressive: PNG Adam7 interlace, or a progressive JPEG frame.
    """

    image_format: ImageFormat
    width: int
    height: int
    bit_depth: int
    component_count: int
    color_type: int | None
    progressive: bool


def read_image_header(
    source: bytes | BinaryIO,
    *,
    max_header_bytes: int = DEFAULT_MAX_IMAGE_HEADER_BYTES,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    max_jpeg_markers: int = DEFAULT_MAX_JPEG_MARKERS,
) -> ImageHeader:
    """Read PNG or JPEG geometry, choosing the grammar from the signature.

    Seekable streams are restored to their original position on return and on
    failure; caller-owned streams are never closed. Only header bytes are read,
    never pixel or entropy-coded data.
    """

    return _read(
        source,
        _dispatch,
        max_header_bytes=max_header_bytes,
        max_pixels=max_pixels,
        max_jpeg_markers=max_jpeg_markers,
    )


def read_png_header(
    source: bytes | BinaryIO,
    *,
    max_header_bytes: int = DEFAULT_MAX_IMAGE_HEADER_BYTES,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> ImageHeader:
    """Read PNG IHDR geometry, rejecting any other datastream.

    The signature, IHDR length, IHDR CRC, declared methods, and the bit
    depth allowed for the color type are all checked. No later chunk is read.
    """

    return _read(
        source,
        _png_entry,
        max_header_bytes=max_header_bytes,
        max_pixels=max_pixels,
        max_jpeg_markers=DEFAULT_MAX_JPEG_MARKERS,
    )


def read_jpeg_header(
    source: bytes | BinaryIO,
    *,
    max_header_bytes: int = DEFAULT_MAX_IMAGE_HEADER_BYTES,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    max_jpeg_markers: int = DEFAULT_MAX_JPEG_MARKERS,
) -> ImageHeader:
    """Read the first JPEG frame header through a bounded marker scan.

    Segments before the frame header are skipped without being retained. The
    scan stops at the frame header and never reaches entropy-coded data.
    """

    return _read(
        source,
        _jpeg_entry,
        max_header_bytes=max_header_bytes,
        max_pixels=max_pixels,
        max_jpeg_markers=max_jpeg_markers,
    )


def _read(
    source: bytes | BinaryIO,
    parse: Callable[[_HeaderReader, int, int], ImageHeader],
    *,
    max_header_bytes: int,
    max_pixels: int,
    max_jpeg_markers: int,
) -> ImageHeader:
    if type(max_header_bytes) is not int or max_header_bytes <= 0:
        raise ValueError("max_header_bytes must be a positive integer")
    if type(max_pixels) is not int or max_pixels <= 0:
        raise ValueError("max_pixels must be a positive integer")
    if type(max_jpeg_markers) is not int or max_jpeg_markers <= 0:
        raise ValueError("max_jpeg_markers must be a positive integer")
    if isinstance(source, bytes):
        reader = _HeaderReader(source, max_header_bytes)
        return parse(reader, max_pixels, max_jpeg_markers)
    read = getattr(source, "read", None)
    if not callable(read):
        raise TypeError("source must be bytes or a binary stream")
    position = _stream_position(source)
    try:
        return parse(
            _HeaderReader(read, max_header_bytes), max_pixels, max_jpeg_markers
        )
    finally:
        if position is not None:
            _restore_position(source, position)


def _dispatch(
    reader: _HeaderReader, max_pixels: int, max_jpeg_markers: int
) -> ImageHeader:
    prefix = reader.read_exact(2)
    if prefix == _PNG_SIGNATURE[:2]:
        return _parse_png(reader, prefix, max_pixels)
    if prefix == _JPEG_SOI:
        return _parse_jpeg(reader, max_pixels, max_jpeg_markers)
    raise ImageHeaderError("image_signature_unsupported")


def _png_entry(
    reader: _HeaderReader, max_pixels: int, max_jpeg_markers: int
) -> ImageHeader:
    return _parse_png(reader, reader.read_exact(2), max_pixels)


def _jpeg_entry(
    reader: _HeaderReader, max_pixels: int, max_jpeg_markers: int
) -> ImageHeader:
    if reader.read_exact(2) != _JPEG_SOI:
        raise ImageHeaderError("jpeg_signature_invalid")
    return _parse_jpeg(reader, max_pixels, max_jpeg_markers)


def _parse_png(reader: _HeaderReader, prefix: bytes, max_pixels: int) -> ImageHeader:
    if prefix + reader.read_exact(6) != _PNG_SIGNATURE:
        raise ImageHeaderError("png_signature_invalid")
    length = struct.unpack(">I", reader.read_exact(4))[0]
    chunk_type = reader.read_exact(4)
    if chunk_type != b"IHDR":
        raise ImageHeaderError("png_ihdr_missing")
    if length != _PNG_IHDR_LENGTH:
        raise ImageHeaderError("png_ihdr_length_invalid")
    payload = reader.read_exact(_PNG_IHDR_LENGTH)
    checksum = struct.unpack(">I", reader.read_exact(4))[0]
    if zlib.crc32(chunk_type + payload) != checksum:
        raise ImageHeaderError("png_ihdr_crc_mismatch")

    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", payload)
    )
    if not 1 <= width <= _PNG_MAX_DIMENSION or not 1 <= height <= _PNG_MAX_DIMENSION:
        raise ImageHeaderError("png_dimensions_invalid")
    if color_type not in _PNG_ALLOWED_DEPTHS:
        raise ImageHeaderError("png_color_type_unsupported")
    if bit_depth not in _PNG_ALLOWED_DEPTHS[color_type]:
        raise ImageHeaderError("png_bit_depth_unsupported")
    if compression != 0 or filtering != 0:
        raise ImageHeaderError("png_method_unsupported")
    if interlace not in (0, 1):
        raise ImageHeaderError("png_interlace_unsupported")
    if width > max_pixels // height:
        raise ImageHeaderError("image_pixel_limit_exceeded")
    return ImageHeader(
        image_format=ImageFormat.PNG,
        width=width,
        height=height,
        bit_depth=bit_depth,
        component_count=_PNG_COMPONENTS[color_type],
        color_type=color_type,
        progressive=interlace == 1,
    )


def _parse_jpeg(
    reader: _HeaderReader, max_pixels: int, max_jpeg_markers: int
) -> ImageHeader:
    scanned = 0
    while True:
        marker = _next_jpeg_marker(reader)
        scanned += 1
        if scanned > max_jpeg_markers:
            raise ImageHeaderError("jpeg_marker_limit_exceeded")
        if marker in _JPEG_STANDALONE:
            continue
        if marker == _JPEG_START_OF_IMAGE:
            raise ImageHeaderError("jpeg_marker_unexpected")
        if marker in (_JPEG_EOI, _JPEG_SOS):
            raise ImageHeaderError("jpeg_frame_header_missing")
        length = struct.unpack(">H", reader.read_exact(2))[0]
        if length < 2:
            raise ImageHeaderError("jpeg_segment_length_invalid")
        if marker in _JPEG_SOF_MARKERS:
            return _parse_jpeg_frame(reader, marker, length, max_pixels)
        reader.skip_exact(length - 2)


def _parse_jpeg_frame(
    reader: _HeaderReader, marker: int, length: int, max_pixels: int
) -> ImageHeader:
    if length < _JPEG_MIN_SOF_LENGTH:
        raise ImageHeaderError("jpeg_frame_length_invalid")
    precision, height, width, components = struct.unpack(">BHHB", reader.read_exact(6))
    if components < 1:
        raise ImageHeaderError("jpeg_component_count_invalid")
    if length != _JPEG_MIN_SOF_LENGTH + 3 * components:
        raise ImageHeaderError("jpeg_frame_length_invalid")
    if not 1 <= precision <= _JPEG_MAX_PRECISION:
        raise ImageHeaderError("jpeg_precision_unsupported")
    if width < 1 or height < 1:
        raise ImageHeaderError("jpeg_dimensions_invalid")
    if width > max_pixels // height:
        raise ImageHeaderError("image_pixel_limit_exceeded")
    reader.skip_exact(3 * components)
    return ImageHeader(
        image_format=ImageFormat.JPEG,
        width=width,
        height=height,
        bit_depth=precision,
        component_count=components,
        color_type=None,
        progressive=marker in _JPEG_PROGRESSIVE_SOF,
    )


def _next_jpeg_marker(reader: _HeaderReader) -> int:
    if reader.read_exact(1) != b"\xff":
        raise ImageHeaderError("jpeg_marker_invalid")
    for _ in range(_JPEG_MAX_FILL_BYTES + 1):
        marker = reader.read_exact(1)[0]
        if marker == 0x00:
            raise ImageHeaderError("jpeg_marker_invalid")
        if marker != 0xFF:
            return marker
    raise ImageHeaderError("jpeg_marker_fill_limit_exceeded")


class _HeaderReader:
    def __init__(self, source: bytes | Callable[[int], bytes], limit: int) -> None:
        self._buffer = memoryview(source) if isinstance(source, bytes) else None
        self._read = source if callable(source) else None
        self._limit = limit
        self.offset = 0

    def read_exact(self, size: int) -> bytes:
        if size > self._limit - self.offset:
            raise ImageHeaderError("image_header_limit_exceeded")
        if self._buffer is not None:
            end = self.offset + size
            if end > len(self._buffer):
                raise ImageHeaderError("image_header_truncated")
            result = bytes(self._buffer[self.offset : end])
            self.offset = end
            return result
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = _read_chunk(self._read, remaining)
            if not chunk:
                raise ImageHeaderError("image_header_truncated")
            chunks.append(chunk)
            self.offset += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def skip_exact(self, size: int) -> None:
        remaining = size
        while remaining:
            step = min(remaining, _SKIP_CHUNK_BYTES)
            self.read_exact(step)
            remaining -= step


def _read_chunk(read: Callable[[int], bytes] | None, size: int) -> bytes:
    if read is None:
        raise ImageHeaderError("image_stream_contract_error")
    try:
        chunk = read(size)
    except Exception:
        pass
    else:
        if isinstance(chunk, bytes) and len(chunk) <= size:
            return chunk
        raise ImageHeaderError("image_stream_contract_error")
    # Raise outside the handler so an underlying I/O message is not retained.
    raise ImageHeaderError("image_stream_read_error")


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
    raise ImageHeaderError("image_stream_position_error")


def _restore_position(stream: BinaryIO, position: Any) -> None:
    try:
        stream.seek(position)
    except Exception:
        pass
    else:
        return
    raise ImageHeaderError("image_stream_restore_error")


__all__ = [
    "DEFAULT_MAX_IMAGE_HEADER_BYTES",
    "DEFAULT_MAX_IMAGE_PIXELS",
    "DEFAULT_MAX_JPEG_MARKERS",
    "ImageFormat",
    "ImageHeader",
    "ImageHeaderError",
    "read_image_header",
    "read_jpeg_header",
    "read_png_header",
]
