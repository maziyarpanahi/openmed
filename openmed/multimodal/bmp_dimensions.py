"""Bounded, dependency-free BMP CORE/INFO header preflight."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO, Callable, Final

__all__ = [
    "DEFAULT_MAX_BMP_HEADER_BYTES",
    "DEFAULT_MAX_BMP_PIXELS",
    "BmpDimensions",
    "BmpDimensionsError",
    "read_bmp_dimensions",
]

DEFAULT_MAX_BMP_HEADER_BYTES: Final[int] = 54
DEFAULT_MAX_BMP_PIXELS: Final[int] = 100_000_000
_UINT32_MAX: Final[int] = (1 << 32) - 1


class BmpDimensionsError(ValueError):
    """Value-free failure for malformed or unsupported BMP headers."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class BmpDimensions:
    """BMP geometry without pixels, color tables, paths, or text metadata."""

    width: int
    height: int
    planes: int
    bit_depth: int
    top_down: bool
    dib_header_bytes: int


def read_bmp_dimensions(
    source: bytes | BinaryIO,
    *,
    max_header_bytes: int = DEFAULT_MAX_BMP_HEADER_BYTES,
    max_pixels: int = DEFAULT_MAX_BMP_PIXELS,
) -> BmpDimensions:
    """Read Windows BMP CORE (12-byte) or uncompressed INFO (40-byte) geometry.

    Other DIB layouts and compression modes are explicitly unsupported. Only
    the file and DIB headers are read, never palette entries or pixels. Declared
    offsets and pixel sizes are checked, but payload presence is not validated.
    Seekable streams are restored to their original position on return/failure;
    caller-owned streams are never closed.
    """
    if type(max_header_bytes) is not int or max_header_bytes <= 0:
        raise ValueError("max_header_bytes must be a positive integer")
    if type(max_pixels) is not int or max_pixels <= 0:
        raise ValueError("max_pixels must be a positive integer")
    if isinstance(source, bytes):
        return _parse_bmp(_HeaderReader(source, max_header_bytes), max_pixels)
    read = getattr(source, "read", None)
    if not callable(read):
        raise TypeError("source must be bytes or a binary stream")
    position = _stream_position(source)
    try:
        return _parse_bmp(_HeaderReader(read, max_header_bytes), max_pixels)
    finally:
        if position is not None:
            _restore_position(source, position)


def _parse_bmp(reader: _HeaderReader, max_pixels: int) -> BmpDimensions:
    file_header = reader.read_exact(14)
    signature, file_size, reserved1, reserved2, pixel_offset = struct.unpack(
        "<2sIHHI", file_header
    )
    if signature != b"BM":
        raise BmpDimensionsError("bmp_signature_invalid")
    if reserved1 or reserved2:
        raise BmpDimensionsError("bmp_reserved_fields_invalid")
    dib_size = struct.unpack("<I", reader.read_exact(4))[0]
    width: int
    signed_height: int
    planes: int
    bit_depth: int
    compression: int
    image_size: int
    allowed_depths: tuple[int, ...]
    if dib_size == 12:
        dib_fields_12 = struct.unpack("<HHHH", reader.read_exact(8))
        width = dib_fields_12[0]
        signed_height = dib_fields_12[1]
        planes = dib_fields_12[2]
        bit_depth = dib_fields_12[3]
        allowed_depths = (1, 4, 8, 24)
        palette_entries = 1 << bit_depth if bit_depth in (1, 4, 8) else 0
        palette_bytes = palette_entries * 3
        image_size = 0
    elif dib_size == 40:
        dib_fields_40 = struct.unpack("<iiHHIIiiII", reader.read_exact(36))
        width = dib_fields_40[0]
        signed_height = dib_fields_40[1]
        planes = dib_fields_40[2]
        bit_depth = dib_fields_40[3]
        compression = dib_fields_40[4]
        image_size = dib_fields_40[5]
        colors_used = dib_fields_40[8]
        if compression != 0:
            raise BmpDimensionsError("bmp_compression_unsupported")
        allowed_depths = (1, 4, 8, 16, 24, 32)
        palette_entries = colors_used
        if bit_depth in (1, 4, 8):
            if colors_used > 1 << bit_depth:
                raise BmpDimensionsError("bmp_color_table_size_invalid")
            palette_entries = colors_used or (1 << bit_depth)
        palette_bytes = palette_entries * 4
    else:
        raise BmpDimensionsError("bmp_dib_header_unsupported")
    if planes != 1:
        raise BmpDimensionsError("bmp_planes_invalid")
    if bit_depth not in allowed_depths:
        raise BmpDimensionsError("bmp_bit_depth_unsupported")
    if width <= 0 or signed_height == 0:
        raise BmpDimensionsError("bmp_dimensions_invalid")
    height = abs(signed_height)
    if width > max_pixels // height:
        raise BmpDimensionsError("bmp_pixel_limit_exceeded")
    minimum_offset = 14 + dib_size + palette_bytes
    if pixel_offset < minimum_offset:
        raise BmpDimensionsError("bmp_pixel_offset_invalid")
    row_bytes = ((width * bit_depth + 31) // 32) * 4
    pixel_bytes = row_bytes * height
    if (
        pixel_bytes > _UINT32_MAX - pixel_offset
        or file_size < pixel_offset + pixel_bytes
    ):
        raise BmpDimensionsError("bmp_file_size_invalid")
    if image_size not in (0, pixel_bytes):
        raise BmpDimensionsError("bmp_image_size_invalid")
    return BmpDimensions(
        width=width,
        height=height,
        planes=planes,
        bit_depth=bit_depth,
        top_down=signed_height < 0,
        dib_header_bytes=dib_size,
    )


class _HeaderReader:
    def __init__(self, source: bytes | Callable[[int], bytes], limit: int) -> None:
        self._buffer = memoryview(source) if isinstance(source, bytes) else None
        self._read = source if callable(source) else None
        self._limit = limit
        self.offset = 0

    def read_exact(self, size: int) -> bytes:
        if size > self._limit - self.offset:
            raise BmpDimensionsError("bmp_header_limit_exceeded")
        if self._buffer is not None:
            end = self.offset + size
            if end > len(self._buffer):
                raise BmpDimensionsError("bmp_header_truncated")
            result = bytes(self._buffer[self.offset : end])
            self.offset = end
            return result
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = _read_chunk(self._read, remaining)
            if not chunk:
                raise BmpDimensionsError("bmp_header_truncated")
            chunks.append(chunk)
            self.offset += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


def _read_chunk(read: Callable[[int], bytes] | None, size: int) -> bytes:
    if read is None:
        raise BmpDimensionsError("bmp_stream_contract_error")
    try:
        chunk = read(size)
    except Exception:
        pass
    else:
        if isinstance(chunk, bytes) and len(chunk) <= size:
            return chunk
        raise BmpDimensionsError("bmp_stream_contract_error")
    # Raise outside the handler so an underlying I/O message is not retained.
    raise BmpDimensionsError("bmp_stream_read_error")


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
    raise BmpDimensionsError("bmp_stream_position_error")


def _restore_position(stream: BinaryIO, position: int) -> None:
    try:
        stream.seek(position)
    except Exception:
        pass
    else:
        return
    raise BmpDimensionsError("bmp_stream_restore_error")
