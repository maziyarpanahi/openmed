"""Opt-in, bounded BigTIFF pyramid metadata preflight without pixel decoding."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum
from typing import BinaryIO

from . import tiff_metadata as tiff

_UINT64_MAX = (1 << 64) - 1
_TYPES = {3: (2, "H"), 4: (4, "I"), 16: (8, "Q"), 18: (8, "Q")}
_SCALARS = {254, 256, 257, 259, 277, 284, 322, 323}
_TABLES = {324, 325, 330}
_COMPRESSION = {1: "uncompressed", 5: "lzw", 7: "jpeg", 8: "deflate", 32946: "deflate"}
_UNSUPPORTED_LAYOUT = {273, 278, 279, 32997, 32998}


class BigTiffProfile(str, Enum):
    """Caller-declared layouts; these are not vendor compatibility claims."""

    IFD_CHAIN = "bigtiff-ifd-chain-v1"
    SUBIFDS = "bigtiff-subifds-v1"


@dataclass(frozen=True, slots=True)
class BigTiffLimits:
    """Resource ceilings checked before traversal or count-based allocation.

    Attributes:
        max_header_bytes: Maximum prefix loaded using the classic TIFF helper.
        max_pixels: Maximum declared pixels per level.
        max_ifd_entries: Maximum entries per directory.
        max_directories: Maximum directories visited.
        max_levels: Maximum pyramid levels, including the base.
        max_tiles: Maximum tiles across all levels.
        max_offset: Maximum exclusive end of any referenced file range.
    """

    max_header_bytes: int = tiff.DEFAULT_MAX_TIFF_HEADER_BYTES
    max_pixels: int = tiff.DEFAULT_MAX_TIFF_PIXELS
    max_ifd_entries: int = tiff.DEFAULT_MAX_IFD_ENTRIES
    max_directories: int = 32
    max_levels: int = 32
    max_tiles: int = 1_000_000
    max_offset: int = (1 << 63) - 1

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= _UINT64_MAX:
                raise ValueError("bigtiff_limit_invalid")


@dataclass(frozen=True, slots=True)
class BigTiffLevel:
    """Geometry and compression category only; no tile or identifying payload."""

    index: int
    width: int
    height: int
    tile_width: int
    tile_height: int
    columns: int
    rows: int
    tile_count: int
    compression: str


@dataclass(frozen=True, slots=True)
class BigTiffPyramid:
    """Deterministic metadata manifest, not diagnostic or decoder approval."""

    profile: BigTiffProfile
    byte_order: tiff.TiffByteOrder
    levels: tuple[BigTiffLevel, ...]
    total_tiles: int


def read_bigtiff_metadata(
    source: bytes | BinaryIO,
    *,
    profile: BigTiffProfile | str,
    source_size: int,
    limits: BigTiffLimits = BigTiffLimits(),
) -> BigTiffPyramid:
    """Preflight a declared tiled BigTIFF pyramid using a bounded prefix.

    Args:
        source: Bytes or a caller-owned binary stream at the TIFF start.
        profile: Explicit supported layout; no automatic vendor detection.
        source_size: Caller-verified byte length from the TIFF start. For bytes,
            this must match their length. Streams are never scanned for length.
        limits: Positive resource ceilings. Metadata outside the prefix fails.

    Returns:
        Immutable level geometry, in declared base-to-reduced order.

    Raises:
        tiff.TiffMetadataError: A controlled category without source content.
        ValueError: Invalid limits, profile or source-size declaration.
        TypeError: Invalid source or limits type.

    Uses the classic TIFF bounded loader, including short-read support and
    stream restoration. It validates tile ranges but never seeks to, decodes,
    or returns pixels. It adds no planning, inference or clinical action.
    """
    if not isinstance(limits, BigTiffLimits):
        raise TypeError("bigtiff_limits_invalid")
    # Validate outside exception handlers to avoid retaining caller values.
    if not isinstance(profile, str) or profile not in tuple(BigTiffProfile):
        raise ValueError("bigtiff_profile_unsupported")
    if type(source_size) is not int or not 16 <= source_size <= _UINT64_MAX:
        raise ValueError("bigtiff_source_size_invalid")
    if isinstance(source, bytes) and source_size != len(source):
        raise ValueError("bigtiff_source_size_invalid")
    buffer = tiff._load(source, limits.max_header_bytes)
    return _Reader(buffer, source_size, limits, BigTiffProfile(profile)).parse()


class _Reader:
    def __init__(self, buffer, source_size, limits, profile):
        self.buffer = buffer
        self.source_size = source_size
        self.limits = limits
        self.profile = profile
        self.prefix = "<"
        self.metadata_ranges = [(0, 16)]
        self.tile_ranges = []
        self.visited = set()
        self.levels = []
        self.total_tiles = 0

    def fail(self, code):
        raise tiff.TiffMetadataError("bigtiff_" + code)

    def range(self, offset, size, *, metadata=False):
        if offset > _UINT64_MAX - size:
            self.fail("offset_overflow")
        if offset + size > self.limits.max_offset:
            self.fail("offset_limit_exceeded")
        if offset + size > self.source_size:
            self.fail("range_out_of_source")
        if metadata and offset + size > len(self.buffer):
            self.fail("header_limit_or_truncated")
        return (offset, offset + size)

    def uint(self, offset, fmt):
        self.range(offset, struct.calcsize(fmt), metadata=True)
        return struct.unpack_from(self.prefix + fmt, self.buffer, offset)[0]

    def parse(self):
        if len(self.buffer) < 16:
            self.fail("header_limit_or_truncated")
        marker = self.buffer[:2]
        if marker not in (b"II", b"MM"):
            self.fail("byte_order_invalid")
        self.prefix = "<" if marker == b"II" else ">"
        if (self.uint(2, "H"), self.uint(4, "H"), self.uint(6, "H")) != (43, 8, 0):
            self.fail("header_unsupported")
        pending = [self.uint(8, "Q")]
        while pending:
            offset = pending.pop(0)
            if offset in self.visited:
                self.fail("directory_cycle")
            if len(self.visited) >= self.limits.max_directories:
                self.fail("directory_limit_exceeded")
            if len(self.levels) >= self.limits.max_levels:
                self.fail("level_limit_exceeded")
            self.visited.add(offset)
            values, next_ifd = self.directory(offset)
            self.level(values)
            children = values.get(330, ())
            if self.profile == BigTiffProfile.IFD_CHAIN:
                if children:
                    self.fail("layout_unsupported")
                if next_ifd:
                    pending.append(next_ifd)
            else:
                if next_ifd or (len(self.levels) > 1 and children):
                    self.fail("layout_unsupported")
                pending.extend(children)
            if len(self.levels) + len(pending) > self.limits.max_levels:
                self.fail("level_limit_exceeded")
            if len(self.visited) + len(pending) > self.limits.max_directories:
                self.fail("directory_limit_exceeded")
        # No metadata aliases or tile ranges overlapping metadata or each other.
        ranges = sorted(self.metadata_ranges + self.tile_ranges)
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            self.fail("range_overlap")
        return BigTiffPyramid(
            self.profile,
            tiff.TiffByteOrder.LITTLE if self.prefix == "<" else tiff.TiffByteOrder.BIG,
            tuple(self.levels),
            self.total_tiles,
        )

    def directory(self, offset):
        if offset < 16 or offset % 8:
            self.fail("directory_offset_invalid")
        count = self.uint(offset, "Q")
        if count < 1 or count > self.limits.max_ifd_entries:
            self.fail("entry_limit_exceeded")
        if count > (_UINT64_MAX - 16) // 20:
            self.fail("offset_overflow")
        self.metadata_ranges.append(self.range(offset, 16 + count * 20, metadata=True))
        values = {}
        seen = set()
        for i in range(count):
            start = offset + 8 + i * 20
            tag, kind, length, raw = struct.unpack_from(
                self.prefix + "HHQ8s", self.buffer, start
            )
            if tag in seen:
                self.fail("tag_duplicate")
            seen.add(tag)
            if tag in _UNSUPPORTED_LAYOUT or tag >= 32768:
                self.fail("layout_unsupported")
            if tag not in _SCALARS | _TABLES:
                continue  # Never follow arbitrary identifying metadata.
            if kind not in _TYPES or (kind == 18 and tag != 330):
                self.fail("value_type_unsupported")
            size, fmt = _TYPES[kind]
            if not length:
                self.fail("value_count_invalid")
            if length > _UINT64_MAX // size:
                self.fail("offset_overflow")
            ceiling = (
                1
                if tag in _SCALARS
                else self.limits.max_levels
                if tag == 330
                else self.limits.max_tiles
            )
            if length > ceiling:
                self.fail("value_count_limit_exceeded")
            byte_count = length * size
            if byte_count <= 8:
                payload = raw[:byte_count]
            else:
                value_offset = struct.unpack(self.prefix + "Q", raw)[0]
                if value_offset < 16 or value_offset % 8:
                    self.fail("value_offset_invalid")
                span = self.range(value_offset, byte_count, metadata=True)
                self.metadata_ranges.append(span)
                payload = self.buffer[span[0] : span[1]]
            values[tag] = struct.unpack(self.prefix + str(length) + fmt, payload)
        return values, self.uint(offset + 8 + count * 20, "Q")

    def level(self, values):
        width, height, tile_width, tile_height = (
            tiff._scalar(values, tag) for tag in (256, 257, 322, 323)
        )
        if any(
            value is None or value < 1
            for value in (width, height, tile_width, tile_height)
        ):
            self.fail("geometry_invalid")
        if width > self.limits.max_pixels // height:
            self.fail("pixel_limit_exceeded")
        index = len(self.levels)
        if values.get(254, (0,)) != ((0,) if index == 0 else (1,)):
            self.fail("layout_unsupported")
        if values.get(284, (1,)) != (1,):
            self.fail("layout_unsupported")
        if not 1 <= values.get(277, (1,))[0] <= 64:
            self.fail("layout_unsupported")
        if index:
            previous = self.levels[-1]
            if (
                width > previous.width
                or height > previous.height
                or (width == previous.width and height == previous.height)
            ):
                self.fail("pyramid_geometry_inconsistent")
        compression = _COMPRESSION.get(values.get(259, (1,))[0])
        if compression is None:
            self.fail("compression_unsupported")
        columns = (width - 1) // tile_width + 1
        rows = (height - 1) // tile_height + 1
        remaining = self.limits.max_tiles - self.total_tiles
        if columns > remaining // rows:
            self.fail("tile_limit_exceeded")
        count = columns * rows
        offsets, sizes = values.get(324, ()), values.get(325, ())
        if len(offsets) != count or len(sizes) != count:
            self.fail("tile_table_inconsistent")
        for offset, size in zip(offsets, sizes):
            if offset < 16 or not size:
                self.fail("tile_range_invalid")
            self.tile_ranges.append(self.range(offset, size))
        self.total_tiles += count
        self.levels.append(
            BigTiffLevel(
                index,
                width,
                height,
                tile_width,
                tile_height,
                columns,
                rows,
                count,
                compression,
            )
        )


__all__ = [
    "BigTiffLevel",
    "BigTiffLimits",
    "BigTiffProfile",
    "BigTiffPyramid",
    "read_bigtiff_metadata",
]
