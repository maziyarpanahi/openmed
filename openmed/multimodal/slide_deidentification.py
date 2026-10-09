"""Bounded, local TIFF-family slide de-identification without pixel decoding."""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass, field

from .tiff_metadata import TiffMetadataError

# TIFF numeric imaging tags only. Text, EXIF, XMP, ICC and vendor payloads are
# never copied. SubIFDs are refused, rather than silently hiding extra images.
_TECHNICAL = {
    254,
    256,
    257,
    258,
    259,
    262,
    266,
    273,
    274,
    277,
    278,
    279,
    282,
    283,
    284,
    296,
    317,
    322,
    323,
    324,
    325,
    338,
    339,
    347,
    529,
    530,
    531,
    532,
}
_TYPES = {
    1: (1, "B"),
    2: (1, "B"),
    3: (2, "H"),
    4: (4, "I"),
    5: (8, "II"),
    7: (1, "B"),
    16: (8, "Q"),
}


@dataclass(frozen=True)
class SlideDeidLimits:
    """Positive bounds for in-memory traversal and rewriting.

    Whole-slide files exceeding these bounds are refused; callers must explicitly
    select budgets appropriate for their device. No allocation follows dimensions.
    """

    max_file_bytes: int = 256 * 1024 * 1024
    max_directories: int = 64
    max_ifd_entries: int = 128
    max_value_bytes: int = 1024 * 1024
    max_blocks: int = 1_000_000

    def __post_init__(self) -> None:
        for value in vars(self).values():
            if type(value) is not int or value <= 0:
                raise ValueError("slide_limits_invalid")


@dataclass(frozen=True)
class SlideDirectoryReport:
    """Directory role and payload digests, never descriptions or identifiers."""

    role: str
    block_count: int
    block_sha256: tuple[str, ...]


@dataclass(frozen=True)
class SlideDeidResult:
    """Protected output bytes and a value-free report.

    Output pixels remain sensitive; this structural operation is non-diagnostic
    and does not establish that pyramid pixels are identifier-free.
    """

    data: bytes = field(repr=False)
    outcome: str
    directories: tuple[SlideDirectoryReport, ...]
    output_sha256: str

    def to_audit_report(self) -> dict:
        """Return controlled roles, counts, digests and an outcome code only."""
        return {
            "outcome": self.outcome,
            "directory_count": len(self.directories),
            "removed_directory_count": sum(d.role != "level" for d in self.directories),
            "output_sha256": self.output_sha256,
            "directories": [
                {
                    "role": d.role,
                    "block_count": d.block_count,
                    "block_sha256": list(d.block_sha256),
                }
                for d in self.directories
            ],
        }


@dataclass
class _Entry:
    kind: int
    count: int
    raw: bytes


@dataclass
class _Directory:
    entries: dict[int, _Entry]
    blocks: list[bytes]
    role: str = ""


class _Reader:
    def __init__(self, data: bytes, limits: SlideDeidLimits):
        self.data, self.limits = data, limits
        self.ranges: list[tuple[int, int]] = []
        self.block_count = 0
        self.claimed_bytes = 0
        self.prefix = "<" if data[:2] == b"II" else ">"
        if data[:2] not in (b"II", b"MM"):
            self.fail("byte_order_invalid")
        magic = self.number(2, 2)
        self.big = magic == 43
        if magic not in (42, 43):
            self.fail("magic_invalid")
        self.word, self.count_size, self.entry_size = (
            (8, 8, 20) if self.big else (4, 2, 12)
        )
        self.header_size = 16 if self.big else 8
        if self.big and (self.number(4, 2) != 8 or self.number(6, 2) != 0):
            self.fail("header_invalid")
        self.claim(0, self.header_size)

    @staticmethod
    def fail(code: str) -> None:
        raise TiffMetadataError(f"slide_{code}")

    def number(self, offset: int, size: int) -> int:
        return int.from_bytes(
            self.slice(offset, size), "little" if self.prefix == "<" else "big"
        )

    def slice(self, offset: int, size: int) -> bytes:
        if offset < 0 or size <= 0 or offset > len(self.data) - size:
            self.fail("range_invalid")
        return self.data[offset : offset + size]

    def claim(self, offset: int, size: int) -> bytes:
        if size > len(self.data) - self.claimed_bytes:
            self.fail("range_overlap")
        raw = self.slice(offset, size)
        self.claimed_bytes += size
        self.ranges.append((offset, offset + size))
        return raw

    def numbers(self, entry: _Entry) -> tuple[int, ...]:
        if entry.kind not in (1, 3, 4, 16):
            self.fail("type_unsupported")
        fmt = _TYPES[entry.kind][1]
        return struct.unpack(f"{self.prefix}{entry.count}{fmt}", entry.raw)

    def scalar(
        self, entries: dict[int, _Entry], tag: int, default: int | None = None
    ) -> int:
        if tag not in entries:
            if default is None:
                self.fail("tag_missing")
            return default
        value = self.numbers(entries[tag])
        if len(value) != 1:
            self.fail("count_invalid")
        return value[0]

    def read(self) -> list[_Directory]:
        offset = self.number(8 if self.big else 4, self.word)
        directories: list[_Directory] = []
        seen: set[int] = set()
        while offset:
            if offset in seen:
                self.fail("directory_cycle")
            if len(directories) >= self.limits.max_directories:
                self.fail("directory_limit")
            seen.add(offset)
            count = self.number(offset, self.count_size)
            if not 0 < count <= self.limits.max_ifd_entries:
                self.fail("entry_limit")
            self.claim(offset, self.count_size + count * self.entry_size + self.word)
            entries: dict[int, _Entry] = {}
            tags: set[int] = set()
            for i in range(count):
                start = offset + self.count_size + i * self.entry_size
                tag, kind = self.number(start, 2), self.number(start + 2, 2)
                if tag in tags:
                    self.fail("tag_duplicate")
                tags.add(tag)
                if tag in (330, 34665, 34853):
                    self.fail("nested_directory_unsupported")
                if tag not in _TECHNICAL and tag != 270:
                    continue
                n = self.number(start + 4, self.word)
                if kind not in _TYPES or (kind == 16 and not self.big):
                    self.fail("type_unsupported")
                size = n * _TYPES[kind][0]
                if not 0 < size <= self.limits.max_value_bytes:
                    self.fail("value_limit")
                value_at = start + 4 + self.word
                raw = (
                    self.slice(value_at, size)
                    if size <= self.word
                    else self.claim(self.number(value_at, self.word), size)
                )
                if (tag == 270 and kind != 2) or (tag != 270 and kind == 2):
                    self.fail("type_unsupported")
                if tag == 347:
                    _validate_jpeg_tables(raw)
                entries[tag] = _Entry(kind, n, raw)
            tiled = 324 in entries
            offset_tag, count_tag = (324, 325) if tiled else (273, 279)
            if offset_tag not in entries or count_tag not in entries:
                self.fail("layout_unknown")
            if tiled and any(t in entries for t in (273, 278, 279)):
                self.fail("layout_unknown")
            offsets, sizes = (
                self.numbers(entries[offset_tag]),
                self.numbers(entries[count_tag]),
            )
            self.block_count += len(offsets)
            if len(offsets) != len(sizes) or self.block_count > self.limits.max_blocks:
                self.fail("block_limit")
            blocks = [self.claim(o, n) for o, n in zip(offsets, sizes)]
            directories.append(_Directory(entries, blocks))
            offset = self.number(
                offset + self.count_size + count * self.entry_size, self.word
            )
        if not directories:
            self.fail("layout_unknown")
        # Reject aliases into descriptions, IFDs, header or other payloads. Sorting
        # keeps this check O(n log n), even under the configured tile budget.
        ordered = sorted(self.ranges)
        if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
            self.fail("range_overlap")
        self.classify(directories)
        return directories

    def classify(self, directories: list[_Directory]) -> None:
        first = directories[0].entries
        description = first.get(270, _Entry(2, 1, b"\0")).raw
        aperio = description.startswith(b"Aperio")
        previous = None
        associated = False
        roles: set[str] = set()
        for index, directory in enumerate(directories):
            e = directory.entries
            width, height = self.scalar(e, 256), self.scalar(e, 257)
            if width == 0 or height == 0:
                self.fail("dimensions_invalid")
            samples = self.scalar(e, 277, 1)
            photo = self.scalar(e, 262)
            if not 0 < samples <= 64:
                self.fail("sample_layout_unsupported")
            bits = self.numbers(e[258]) if 258 in e else (1,) * samples
            if (
                photo not in (0, 1, 2, 6)
                or (photo in (0, 1) and samples != 1)
                or (photo in (2, 6) and samples < 3)
                or len(bits) != samples
                or any(not 0 < bit <= 64 for bit in bits)
            ):
                self.fail("sample_layout_unsupported")
            subtype = self.scalar(e, 254, 0)
            if 324 in e:
                if associated or subtype not in (0, 1) or (index == 0 and subtype != 0):
                    self.fail("layout_unknown")
                if not aperio and index > 0 and subtype != 1:
                    self.fail("layout_unknown")
                tw, th = self.scalar(e, 322), self.scalar(e, 323)
                if tw == 0 or th == 0 or self.scalar(e, 284, 1) != 1:
                    self.fail("layout_unknown")
                expected = ((width - 1) // tw + 1) * ((height - 1) // th + 1)
                if expected != len(directory.blocks):
                    self.fail("block_count_invalid")
                if previous and (
                    width > previous[0]
                    or height > previous[1]
                    or (width, height) == previous
                ):
                    self.fail("layout_unknown")
                if aperio and any(
                    self.scalar(e, tag, 1) != self.scalar(first, tag, 1)
                    for tag in (259, 322, 323)
                ):
                    self.fail("layout_unknown")
                previous = (width, height)
                directory.role = "level"
            elif aperio and index == 1 and subtype == 0:
                directory.role = "thumbnail"
            elif aperio and index > 0 and subtype in (1, 9):
                associated = True
                directory.role = "label" if subtype == 1 else "macro"
            else:
                self.fail("layout_unknown")
            if directory.role != "level":
                if directory.role in roles:
                    self.fail("associated_image_unexpected")
                roles.add(directory.role)
            if self.scalar(e, 259, 1) not in (1, 5, 7, 8, 32946, 33003, 33005):
                self.fail("compression_unsupported")


def _validate_jpeg_tables(raw: bytes) -> None:
    # Only Huffman/quantization tables and delimiters; no comment/APP segments
    # that could carry identifiers. Tile codestreams themselves are unchanged.
    pos = 0
    while pos < len(raw):
        if pos + 2 > len(raw) or raw[pos] != 255:
            _Reader.fail("jpeg_tables_invalid")
        marker = raw[pos + 1]
        pos += 2
        if marker in (0xD8, 0xD9):
            continue
        if marker not in (0xDB, 0xC4) or pos + 2 > len(raw):
            _Reader.fail("jpeg_tables_invalid")
        size = int.from_bytes(raw[pos : pos + 2], "big")
        if size < 2 or size > len(raw) - pos:
            _Reader.fail("jpeg_tables_invalid")
        pos += size


def _description(raw: bytes, aperio: bool) -> bytes:
    prefix = b"Aperio\n" if aperio else b"OpenMed deidentified slide\n"
    # Numeric-only values, bounded spelling and range; names, dates, paths and
    # unrecognized key/value pairs never propagate into the rewritten copy.
    values: dict[bytes, bytes] = {}
    for field_value in raw.rstrip(b"\0").split(b"|")[1:]:
        key, sep, value = field_value.partition(b"=")
        key, value = key.strip(), value.strip()
        if (
            sep
            and key in (b"MPP", b"AppMag")
            and re.fullmatch(rb"[0-9]{1,6}(?:\.[0-9]{1,6})?", value)
            and 0 < float(value) <= 100000
        ):
            if key in values:
                _Reader.fail("description_duplicate_key")
            values[key] = value
    return (
        prefix
        + b"".join(b"|" + key + b"=" + values[key] for key in sorted(values))
        + b"\0"
    )


def _write(reader: _Reader, directories: list[_Directory]) -> bytes:
    kept = [d for d in directories if d.role == "level"]
    output = bytearray(reader.data[: reader.header_size])
    word_format = "Q" if reader.big else "I"
    count_format = "Q" if reader.big else "H"

    def pack(value: int, fmt: str = word_format) -> bytes:
        if value < 0 or value >= 1 << (8 * struct.calcsize(fmt)):
            reader.fail("file_limit")
        return struct.pack(reader.prefix + fmt, value)

    def append(raw: bytes) -> int:
        # BigTIFF requires 8-byte alignment; classic TIFF requires word alignment.
        alignment = 8 if reader.big else 2
        output.extend(b"\0" * (-len(output) % alignment))
        offset = len(output)
        if offset + len(raw) > reader.limits.max_file_bytes:
            reader.fail("file_limit")
        output.extend(raw)
        return offset

    previous_pointer = 8 if reader.big else 4
    aperio = (
        directories[0].entries.get(270, _Entry(2, 1, b"\0")).raw.startswith(b"Aperio")
    )
    for d in kept:
        entries = dict(d.entries)
        description = _description(entries.get(270, _Entry(2, 1, b"\0")).raw, aperio)
        entries[270] = _Entry(2, len(description), description)
        offsets = [append(block) for block in d.blocks]
        entries[324] = _Entry(
            16 if reader.big else 4, len(offsets), b"".join(pack(o) for o in offsets)
        )
        raw_entries = bytearray()
        for tag, entry in sorted(entries.items()):
            value = entry.raw
            if len(value) > reader.word:
                value = pack(append(value))
            raw_entries.extend(
                pack(tag, "H")
                + pack(entry.kind, "H")
                + pack(entry.count)
                + value.ljust(reader.word, b"\0")
            )
        offset = append(pack(len(entries), count_format) + raw_entries + pack(0))
        output[previous_pointer : previous_pointer + reader.word] = pack(offset)
        previous_pointer = offset + reader.count_size + len(entries) * reader.entry_size
    return bytes(output)


def deidentify_tiff_slide(
    source: bytes, *, limits: SlideDeidLimits | None = None
) -> SlideDeidResult:
    """Drop associated images and rewrite TIFF/BigTIFF metadata locally.

    Args:
        source: Protected file bytes; never logged or written to temporary files.
        limits: Explicit device budgets for file, IFD, value and block counts.

    Returns:
        A new compact TIFF with byte-identical pyramid payloads and digest reports.
        Thumbnail, label and macro directories are removed, including orphan bytes.

    Raises:
        TiffMetadataError: A controlled ``slide_*`` code for refused input.
        ValueError: Invalid limits. No partial copy is returned on refusal.

    This is structural de-identification, not pixel PHI detection or a clinical
    decision. Human review and pixel redaction remain necessary before release.
    """
    if not isinstance(source, bytes):
        raise TypeError("slide_source_must_be_bytes")
    limits = limits or SlideDeidLimits()
    if len(source) > limits.max_file_bytes:
        _Reader.fail("file_limit")
    reader = _Reader(source, limits)
    directories = reader.read()
    output = _write(reader, directories)
    reports = tuple(
        SlideDirectoryReport(
            d.role,
            len(d.blocks),
            tuple(hashlib.sha256(b).hexdigest() for b in d.blocks),
        )
        for d in directories
    )
    return SlideDeidResult(
        output,
        "deidentified"
        if any(d.role != "level" for d in directories)
        else "no_associated_images",
        reports,
        hashlib.sha256(output).hexdigest(),
    )


__all__ = [
    "SlideDeidLimits",
    "SlideDirectoryReport",
    "SlideDeidResult",
    "deidentify_tiff_slide",
]
