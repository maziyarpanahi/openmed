"""Bounded, dependency-free classic TIFF header preflight.

TIFF is offset addressed, so a bounded prefix is loaded once and every
declared offset is resolved inside it. Only an allowlisted set of geometry and
sample tags is read; arbitrary tags are skipped without following their
offsets, and no strip, tile, or pixel payload is ever decoded or returned.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum
from typing import BinaryIO, Final

DEFAULT_MAX_TIFF_HEADER_BYTES: Final[int] = 64 * 1024
DEFAULT_MAX_TIFF_PIXELS: Final[int] = 100_000_000
DEFAULT_MAX_IFD_ENTRIES: Final[int] = 512

TAG_IMAGE_WIDTH: Final[int] = 256
TAG_IMAGE_LENGTH: Final[int] = 257
TAG_BITS_PER_SAMPLE: Final[int] = 258
TAG_COMPRESSION: Final[int] = 259
TAG_PHOTOMETRIC: Final[int] = 262
TAG_SAMPLES_PER_PIXEL: Final[int] = 277
TAG_PLANAR_CONFIGURATION: Final[int] = 284

_TIFF_MAGIC: Final[int] = 42
_TIFF_HEADER_BYTES: Final[int] = 8
_IFD_ENTRY_BYTES: Final[int] = 12
_INLINE_VALUE_BYTES: Final[int] = 4
_UINT32_MAX: Final[int] = (1 << 32) - 1
_MAX_SAMPLES_PER_PIXEL: Final[int] = 64
_MAX_BITS_PER_SAMPLE: Final[int] = 64
_READ_CHUNK_BYTES: Final[int] = 8192

_TYPE_BYTE: Final[int] = 1
_TYPE_SHORT: Final[int] = 3
_TYPE_LONG: Final[int] = 4
_TYPE_SIZES: Final[dict[int, int]] = {
    _TYPE_BYTE: 1,
    _TYPE_SHORT: 2,
    _TYPE_LONG: 4,
}
_TYPE_FORMATS: Final[dict[int, str]] = {
    _TYPE_BYTE: "B",
    _TYPE_SHORT: "H",
    _TYPE_LONG: "I",
}

_ALLOWED_TAGS: Final[frozenset[int]] = frozenset(
    {
        TAG_IMAGE_WIDTH,
        TAG_IMAGE_LENGTH,
        TAG_BITS_PER_SAMPLE,
        TAG_COMPRESSION,
        TAG_PHOTOMETRIC,
        TAG_SAMPLES_PER_PIXEL,
        TAG_PLANAR_CONFIGURATION,
    }
)
_SCALAR_SHORT_TAGS: Final[frozenset[int]] = frozenset(
    {
        TAG_COMPRESSION,
        TAG_PHOTOMETRIC,
        TAG_SAMPLES_PER_PIXEL,
        TAG_PLANAR_CONFIGURATION,
    }
)


class TiffByteOrder(str, Enum):
    """Closed set of classic TIFF byte orders.

    Values:
        LITTLE: An ``II`` datastream.
        BIG: An ``MM`` datastream.
    """

    LITTLE = "little"
    BIG = "big"


class TiffMetadataError(ValueError):
    """Value-free failure for malformed or unsupported TIFF headers."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class TiffMetadata:
    """Declared TIFF geometry and sample layout, without pixels or tags.

    Attributes:
        byte_order: Byte order declared by the header.
        width: ``ImageWidth`` in pixels.
        height: ``ImageLength`` in pixels.
        samples_per_pixel: ``SamplesPerPixel``, defaulting to one.
        bits_per_sample: One entry per sample, defaulting to one bit each.
        photometric: ``PhotometricInterpretation``.
        compression: ``Compression``, defaulting to one (uncompressed).
        planar_configuration: ``PlanarConfiguration``, defaulting to one.
        ifd_entry_count: Number of entries in the first IFD.
    """

    byte_order: TiffByteOrder
    width: int
    height: int
    samples_per_pixel: int
    bits_per_sample: tuple[int, ...]
    photometric: int
    compression: int
    planar_configuration: int
    ifd_entry_count: int


def read_tiff_metadata(
    source: bytes | BinaryIO,
    *,
    max_header_bytes: int = DEFAULT_MAX_TIFF_HEADER_BYTES,
    max_pixels: int = DEFAULT_MAX_TIFF_PIXELS,
    max_ifd_entries: int = DEFAULT_MAX_IFD_ENTRIES,
) -> TiffMetadata:
    """Read classic TIFF geometry from a bounded prefix and the first IFD.

    Because TIFF resolves values through file offsets, the helper loads up to
    ``max_header_bytes`` once and answers every offset inside that prefix. A
    nonseekable stream is therefore consumed up to that bound or to end of
    file. Seekable streams are restored to their original position on return
    and on failure, and caller-owned streams are never closed.

    BigTIFF, later IFDs, EXIF sub-IFDs, and decoding are out of scope.
    """

    if type(max_header_bytes) is not int or max_header_bytes <= 0:
        raise ValueError("max_header_bytes must be a positive integer")
    if type(max_pixels) is not int or max_pixels <= 0:
        raise ValueError("max_pixels must be a positive integer")
    if type(max_ifd_entries) is not int or max_ifd_entries <= 0:
        raise ValueError("max_ifd_entries must be a positive integer")
    buffer = _load(source, max_header_bytes)
    return _parse(buffer, max_pixels, max_ifd_entries)


def _load(source: bytes | BinaryIO, max_header_bytes: int) -> bytes:
    if isinstance(source, bytes):
        return source[:max_header_bytes]
    read = getattr(source, "read", None)
    if not callable(read):
        raise TypeError("source must be bytes or a binary stream")
    position = _stream_position(source)
    try:
        chunks: list[bytes] = []
        loaded = 0
        while loaded < max_header_bytes:
            chunk = _read_chunk(read, min(_READ_CHUNK_BYTES, max_header_bytes - loaded))
            if not chunk:
                break
            chunks.append(chunk)
            loaded += len(chunk)
        return b"".join(chunks)
    finally:
        if position is not None:
            _restore_position(source, position)


def _parse(buffer: bytes, max_pixels: int, max_ifd_entries: int) -> TiffMetadata:
    if len(buffer) < _TIFF_HEADER_BYTES:
        raise TiffMetadataError("tiff_header_truncated")
    marker = buffer[:2]
    if marker == b"II":
        order, prefix = TiffByteOrder.LITTLE, "<"
    elif marker == b"MM":
        order, prefix = TiffByteOrder.BIG, ">"
    else:
        raise TiffMetadataError("tiff_byte_order_invalid")
    magic, ifd_offset = struct.unpack(f"{prefix}HI", buffer[2:_TIFF_HEADER_BYTES])
    if magic != _TIFF_MAGIC:
        raise TiffMetadataError("tiff_magic_invalid")
    if ifd_offset < _TIFF_HEADER_BYTES:
        raise TiffMetadataError("tiff_ifd_offset_invalid")

    entry_count = _unpack_at(buffer, prefix, "H", ifd_offset, "tiff_ifd_truncated")
    if entry_count == 0:
        raise TiffMetadataError("tiff_ifd_empty")
    if entry_count > max_ifd_entries:
        raise TiffMetadataError("tiff_ifd_entry_limit_exceeded")
    directory_end = ifd_offset + 2 + entry_count * _IFD_ENTRY_BYTES
    if directory_end + 4 > len(buffer):
        raise TiffMetadataError("tiff_ifd_truncated")

    values: dict[int, tuple[int, ...]] = {}
    seen: set[int] = set()
    followed: set[int] = set()
    for index in range(entry_count):
        start = ifd_offset + 2 + index * _IFD_ENTRY_BYTES
        tag, value_type, count, raw = struct.unpack(
            f"{prefix}HHI4s", buffer[start : start + _IFD_ENTRY_BYTES]
        )
        if tag in seen:
            raise TiffMetadataError("tiff_tag_duplicate")
        seen.add(tag)
        if tag not in _ALLOWED_TAGS:
            continue
        values[tag] = _read_values(
            buffer, prefix, value_type, count, raw, followed, ifd_offset
        )
    return _build(order, values, entry_count, max_pixels)


def _read_values(
    buffer: bytes,
    prefix: str,
    value_type: int,
    count: int,
    raw: bytes,
    followed: set[int],
    ifd_offset: int,
) -> tuple[int, ...]:
    size = _TYPE_SIZES.get(value_type)
    if size is None:
        raise TiffMetadataError("tiff_value_type_unsupported")
    if count == 0:
        raise TiffMetadataError("tiff_value_count_invalid")
    if count > _UINT32_MAX // size:
        raise TiffMetadataError("tiff_value_length_overflow")
    byte_count = count * size
    if byte_count <= _INLINE_VALUE_BYTES:
        payload = raw[:byte_count]
    else:
        offset = struct.unpack(f"{prefix}I", raw)[0]
        if offset < _TIFF_HEADER_BYTES or offset == ifd_offset:
            raise TiffMetadataError("tiff_value_offset_invalid")
        if offset in followed:
            raise TiffMetadataError("tiff_value_offset_cycle")
        followed.add(offset)
        if offset > len(buffer) or byte_count > len(buffer) - offset:
            raise TiffMetadataError("tiff_value_truncated")
        payload = buffer[offset : offset + byte_count]
    return struct.unpack(f"{prefix}{count}{_TYPE_FORMATS[value_type]}", payload)


def _build(
    order: TiffByteOrder,
    values: dict[int, tuple[int, ...]],
    entry_count: int,
    max_pixels: int,
) -> TiffMetadata:
    width = _scalar(values, TAG_IMAGE_WIDTH)
    height = _scalar(values, TAG_IMAGE_LENGTH)
    if width is None or height is None:
        raise TiffMetadataError("tiff_dimensions_missing")
    if width < 1 or height < 1:
        raise TiffMetadataError("tiff_dimensions_invalid")
    if width > max_pixels // height:
        raise TiffMetadataError("tiff_pixel_limit_exceeded")

    for tag in _SCALAR_SHORT_TAGS:
        if tag in values and len(values[tag]) != 1:
            raise TiffMetadataError("tiff_value_count_invalid")
    photometric = _scalar(values, TAG_PHOTOMETRIC)
    if photometric is None:
        raise TiffMetadataError("tiff_photometric_missing")

    samples = _scalar(values, TAG_SAMPLES_PER_PIXEL)
    samples = 1 if samples is None else samples
    if not 1 <= samples <= _MAX_SAMPLES_PER_PIXEL:
        raise TiffMetadataError("tiff_samples_per_pixel_invalid")
    bits = values.get(TAG_BITS_PER_SAMPLE, (1,) * samples)
    if len(bits) != samples:
        raise TiffMetadataError("tiff_bits_per_sample_count_invalid")
    if any(not 1 <= value <= _MAX_BITS_PER_SAMPLE for value in bits):
        raise TiffMetadataError("tiff_bits_per_sample_invalid")

    compression = _scalar(values, TAG_COMPRESSION)
    planar = _scalar(values, TAG_PLANAR_CONFIGURATION)
    return TiffMetadata(
        byte_order=order,
        width=width,
        height=height,
        samples_per_pixel=samples,
        bits_per_sample=tuple(bits),
        photometric=photometric,
        compression=1 if compression is None else compression,
        planar_configuration=1 if planar is None else planar,
        ifd_entry_count=entry_count,
    )


def _scalar(values: dict[int, tuple[int, ...]], tag: int) -> int | None:
    entry = values.get(tag)
    if entry is None:
        return None
    if len(entry) != 1:
        raise TiffMetadataError("tiff_value_count_invalid")
    return entry[0]


def _unpack_at(buffer: bytes, prefix: str, fmt: str, offset: int, category: str) -> int:
    size = struct.calcsize(f"{prefix}{fmt}")
    if offset > len(buffer) - size:
        raise TiffMetadataError(category)
    return int(struct.unpack_from(f"{prefix}{fmt}", buffer, offset)[0])


def _read_chunk(read: object, size: int) -> bytes:
    try:
        chunk = read(size)  # type: ignore[operator]
    except Exception:
        pass
    else:
        if isinstance(chunk, bytes) and len(chunk) <= size:
            return chunk
        raise TiffMetadataError("tiff_stream_contract_error")
    # Raise outside the handler so an underlying I/O message is not retained.
    raise TiffMetadataError("tiff_stream_read_error")


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
    raise TiffMetadataError("tiff_stream_position_error")


def _restore_position(stream: BinaryIO, position: int) -> None:
    try:
        stream.seek(position)
    except Exception:
        pass
    else:
        return
    raise TiffMetadataError("tiff_stream_restore_error")


__all__ = [
    "DEFAULT_MAX_IFD_ENTRIES",
    "DEFAULT_MAX_TIFF_HEADER_BYTES",
    "DEFAULT_MAX_TIFF_PIXELS",
    "TAG_BITS_PER_SAMPLE",
    "TAG_COMPRESSION",
    "TAG_IMAGE_LENGTH",
    "TAG_IMAGE_WIDTH",
    "TAG_PHOTOMETRIC",
    "TAG_PLANAR_CONFIGURATION",
    "TAG_SAMPLES_PER_PIXEL",
    "TiffByteOrder",
    "TiffMetadata",
    "TiffMetadataError",
    "read_tiff_metadata",
]
