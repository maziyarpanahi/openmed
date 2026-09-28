"""Bounded, dependency-free PDF page geometry preflight.

OCR and rasterization need page counts and page boxes before any rendering
starts. This module reads the PDF version, the page tree, and each page's
inherited ``MediaBox``, ``CropBox``, and ``Rotate`` values from a bounded byte
budget. It never decodes content streams, text, images, fonts, or document
metadata, and a report carries numbers and stable reason codes only.

The reader is intentionally partial: it scans indirect objects, follows the
catalog's page tree, and expands uncompressed or FlateDecode object streams.
It is not a conformance validator, and it refuses encrypted documents.
"""

from __future__ import annotations

import json
import math
import re
import zlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, BinaryIO, Final

__all__ = [
    "DEFAULT_MAX_PDF_BYTES",
    "DEFAULT_MAX_PDF_DECOMPRESSED_BYTES",
    "DEFAULT_MAX_PDF_OBJECTS",
    "DEFAULT_MAX_PDF_PAGES",
    "PDF_GEOMETRY_SCHEMA_VERSION",
    "PDF_REASON_CODES",
    "PdfGeometryError",
    "PdfGeometryReport",
    "PdfGeometryStatus",
    "PdfPageGeometry",
    "read_pdf_geometry",
]

PDF_GEOMETRY_SCHEMA_VERSION: Final[str] = "openmed.multimodal.pdf_geometry.v1"
DEFAULT_MAX_PDF_BYTES: Final[int] = 64 * 1024 * 1024
DEFAULT_MAX_PDF_PAGES: Final[int] = 10_000
DEFAULT_MAX_PDF_OBJECTS: Final[int] = 250_000
DEFAULT_MAX_PDF_DECOMPRESSED_BYTES: Final[int] = 32 * 1024 * 1024

PDF_REASON_CODES: Final[tuple[str, ...]] = (
    "pdf_size_limit",
    "pdf_header_missing",
    "pdf_encrypted",
    "pdf_object_limit",
    "pdf_decompression_limit",
    "pdf_object_stream_unsupported",
    "pdf_catalog_missing",
    "pdf_page_tree_invalid",
    "pdf_page_limit",
    "page_count_mismatch",
    "media_box_missing",
    "media_box_invalid",
    "crop_box_invalid",
    "crop_box_outside_media_box",
    "rotation_invalid",
)

_PAGE_REASONS: Final[tuple[str, ...]] = PDF_REASON_CODES[10:]
_HEADER_SEARCH_BYTES: Final[int] = 1024
_MAX_NESTING: Final[int] = 64
_MAX_TREE_DEPTH: Final[int] = 64
_MAX_REFERENCE_HOPS: Final[int] = 32
_MAX_BOX_COORDINATE: Final[float] = 10_000_000.0

_WHITESPACE: Final[bytes] = b"\x00\t\n\x0c\r "
_DELIMITERS: Final[bytes] = b"()<>[]{}/%"
_HEADER_RE = re.compile(rb"%PDF-(\d)\.(\d)")
_OBJECT_RE = re.compile(
    rb"(?<![0-9])(\d{1,10})[\x00\t\n\x0c\r ]+(\d{1,5})[\x00\t\n\x0c\r ]+obj(?![A-Za-z])"
)
_TRAILER_RE = re.compile(rb"trailer(?![A-Za-z])")
_NUMBER_RE = re.compile(rb"[+-]?(?:\d+\.?\d*|\.\d+)")
_REFERENCE_TAIL_RE = re.compile(
    rb"[\x00\t\n\x0c\r ]+(\d{1,5})[\x00\t\n\x0c\r ]+R(?![A-Za-z0-9])"
)
_VERSION_NAME_RE = re.compile(r"^(\d)\.(\d)$")


class PdfGeometryError(ValueError):
    """Value-free failure raised for unusable arguments."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


class PdfGeometryStatus(str, Enum):
    """Closed verdict vocabulary for one PDF geometry read.

    Values:
        READABLE: Every page has valid geometry.
        REVIEW: Pages were read, but some geometry needs attention.
        REJECTED: The document was not read; ``pages`` is empty.
    """

    READABLE = "readable"
    REVIEW = "review"
    REJECTED = "rejected"


Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PdfPageGeometry:
    """Numeric geometry for one page in page-tree order.

    Attributes:
        page_index: Zero-based position in the page tree.
        media_box: Normalized ``(x0, y0, x1, y1)`` media box, or ``None`` when
            missing or invalid.
        crop_box: Effective crop box clipped to the media box. It equals the
            media box when no valid crop box is declared.
        rotation: Clockwise display rotation, one of 0, 90, 180, or 270.
        width: Displayed width of the crop box after rotation.
        height: Displayed height of the crop box after rotation.
        reason_codes: Page findings in :data:`PDF_REASON_CODES` order.
    """

    page_index: int
    media_box: Box | None
    crop_box: Box | None
    rotation: int
    width: float | None
    height: float | None
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible mapping."""
        return {
            "page_index": self.page_index,
            "media_box": None if self.media_box is None else list(self.media_box),
            "crop_box": None if self.crop_box is None else list(self.crop_box),
            "rotation": self.rotation,
            "width": self.width,
            "height": self.height,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class PdfGeometryReport:
    """Numeric PDF geometry plus stable reason codes.

    Attributes:
        status: Most severe verdict implied by ``reason_codes``.
        reason_codes: Document and page findings in :data:`PDF_REASON_CODES`
            order, without duplicates.
        version_major: Effective PDF major version, when a header was found.
        version_minor: Effective PDF minor version, when a header was found.
        page_count: Number of page-tree leaves read.
        declared_page_count: ``Count`` declared by the root page-tree node.
        pages: Per-page geometry. Empty when the document was rejected.
    """

    status: PdfGeometryStatus
    reason_codes: tuple[str, ...]
    version_major: int | None
    version_minor: int | None
    page_count: int | None
    declared_page_count: int | None
    pages: tuple[PdfPageGeometry, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic mapping in declared field order."""
        return {
            "schema_version": PDF_GEOMETRY_SCHEMA_VERSION,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "version_major": self.version_major,
            "version_minor": self.version_minor,
            "page_count": self.page_count,
            "declared_page_count": self.declared_page_count,
            "pages": [page.to_dict() for page in self.pages],
        }

    def to_json(self) -> str:
        """Return compact JSON with sorted keys."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def read_pdf_geometry(
    source: bytes | bytearray | memoryview | BinaryIO,
    *,
    max_bytes: int = DEFAULT_MAX_PDF_BYTES,
    max_pages: int = DEFAULT_MAX_PDF_PAGES,
    max_objects: int = DEFAULT_MAX_PDF_OBJECTS,
    max_decompressed_bytes: int = DEFAULT_MAX_PDF_DECOMPRESSED_BYTES,
) -> PdfGeometryReport:
    """Read version, page count, and page boxes from a bounded PDF.

    Args:
        source: PDF bytes or a readable binary stream. Seekable streams are
            restored to their original position and never closed.
        max_bytes: Largest document accepted; larger input is rejected.
        max_pages: Largest page count accepted.
        max_objects: Largest number of indirect objects scanned.
        max_decompressed_bytes: Total budget for expanding object streams.

    Returns:
        A :class:`PdfGeometryReport`.

    Raises:
        PdfGeometryError: If a limit is not a positive integer or ``source`` is
            neither bytes-like nor a binary stream.
    """
    for category, limit in (
        ("max_bytes_invalid", max_bytes),
        ("max_pages_invalid", max_pages),
        ("max_objects_invalid", max_objects),
        ("max_decompressed_bytes_invalid", max_decompressed_bytes),
    ):
        if type(limit) is not int or limit <= 0:
            raise PdfGeometryError(category)

    data = _load(source, max_bytes)
    if data is None:
        return _rejected("pdf_size_limit")
    header = _HEADER_RE.search(data, 0, _HEADER_SEARCH_BYTES + 8)
    if header is None or header.start() > _HEADER_SEARCH_BYTES:
        return _rejected("pdf_header_missing")
    major, minor = int(header.group(1)), int(header.group(2))

    document = _Document(data, max_objects, max_decompressed_bytes)
    failure = document.scan()
    if failure is not None:
        return _rejected(failure, major, minor)
    if any("Encrypt" in trailer for trailer in document.trailers):
        return _rejected("pdf_encrypted", major, minor)

    catalog = None
    for trailer in reversed(document.trailers):
        if "Root" in trailer:
            catalog = document.resolve(trailer["Root"])
            break
    if not isinstance(catalog, dict):
        return _rejected(
            document.structure_failure("pdf_catalog_missing"), major, minor
        )

    version = catalog.get("Version")
    if isinstance(version, _Name):
        match = _VERSION_NAME_RE.fullmatch(version)
        if match is not None and (int(match.group(1)), int(match.group(2))) > (
            major,
            minor,
        ):
            major, minor = int(match.group(1)), int(match.group(2))

    root = document.resolve(catalog.get("Pages"))
    if not isinstance(root, dict):
        return _rejected(
            document.structure_failure("pdf_page_tree_invalid"), major, minor
        )
    walked = _walk_page_tree(document, catalog.get("Pages"), max_pages)
    if isinstance(walked, str):
        return _rejected(document.structure_failure(walked), major, minor)

    reasons: set[str] = set()
    declared = document.resolve(root.get("Count"))
    declared_count = declared if type(declared) is int and declared >= 0 else None
    if declared_count != len(walked):
        reasons.add("page_count_mismatch")

    pages = tuple(
        _page_geometry(document, index, attributes)
        for index, attributes in enumerate(walked)
    )
    for page in pages:
        reasons.update(page.reason_codes)
    reason_codes = tuple(code for code in PDF_REASON_CODES if code in reasons)
    return PdfGeometryReport(
        status=PdfGeometryStatus.REVIEW if reason_codes else PdfGeometryStatus.READABLE,
        reason_codes=reason_codes,
        version_major=major,
        version_minor=minor,
        page_count=len(pages),
        declared_page_count=declared_count,
        pages=pages,
    )


def _rejected(
    reason: str, major: int | None = None, minor: int | None = None
) -> PdfGeometryReport:
    return PdfGeometryReport(
        status=PdfGeometryStatus.REJECTED,
        reason_codes=(reason,),
        version_major=major,
        version_minor=minor,
        page_count=None,
        declared_page_count=None,
        pages=(),
    )


def _load(source: Any, max_bytes: int) -> bytes | None:
    if isinstance(source, (bytes, bytearray, memoryview)):
        size = source.nbytes if isinstance(source, memoryview) else len(source)
        return bytes(source) if size <= max_bytes else None
    read = getattr(source, "read", None)
    if not callable(read):
        raise PdfGeometryError("source_invalid")
    position = _stream_position(source)
    chunks: list[bytes] = []
    total = 0
    try:
        while total <= max_bytes:
            chunk = read(min(1 << 20, max_bytes + 1 - total))
            if not isinstance(chunk, (bytes, bytearray)):
                raise PdfGeometryError("source_invalid")
            if not chunk:
                break
            chunks.append(bytes(chunk))
            total += len(chunk)
    finally:
        if position is not None:
            try:
                source.seek(position)
            except (AttributeError, OSError, ValueError):
                pass
    if total > max_bytes:
        return None
    return b"".join(chunks)


def _stream_position(stream: Any) -> int | None:
    seekable = getattr(stream, "seekable", None)
    try:
        if callable(seekable) and not seekable():
            return None
        position = stream.tell()
    except (AttributeError, OSError, ValueError):
        return None
    return position if type(position) is int else None


class _Name(str):
    """A decoded PDF name. Never serialized."""


@dataclass(frozen=True, slots=True)
class _Ref:
    number: int


class _Keyword(str):
    """A bare PDF keyword such as ``stream`` or ``endobj``."""


_STRING: Final[object] = object()


class _SyntaxFailure(Exception):
    pass


class _Parser:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def skip_space(self, position: int) -> int:
        data = self.data
        length = len(data)
        while position < length:
            byte = data[position]
            if byte in _WHITESPACE:
                position += 1
            elif byte == 0x25:  # %
                while position < length and data[position] not in b"\r\n":
                    position += 1
            else:
                break
        return position

    def parse(self, position: int, depth: int = 0) -> tuple[Any, int]:
        if depth > _MAX_NESTING:
            raise _SyntaxFailure
        data = self.data
        position = self.skip_space(position)
        if position >= len(data):
            raise _SyntaxFailure
        if data.startswith(b"<<", position):
            return self._dictionary(position + 2, depth)
        byte = data[position]
        if byte == 0x5B:  # [
            items: list[Any] = []
            position += 1
            while True:
                position = self.skip_space(position)
                if position >= len(data):
                    raise _SyntaxFailure
                if data[position] == 0x5D:  # ]
                    return items, position + 1
                item, position = self.parse(position, depth + 1)
                items.append(item)
        if byte == 0x28:  # (
            return _STRING, self._literal_string_end(position + 1)
        if byte == 0x3C:  # <
            end = data.find(b">", position + 1)
            if end < 0:
                raise _SyntaxFailure
            return _STRING, end + 1
        if byte == 0x2F:  # /
            return self._name(position + 1)
        number = _NUMBER_RE.match(data, position)
        if number is not None:
            return self._number_or_reference(number)
        end = position
        while (
            end < len(data)
            and data[end] not in _WHITESPACE
            and data[end] not in _DELIMITERS
        ):
            end += 1
        if end == position:
            raise _SyntaxFailure
        word = data[position:end]
        if word == b"true":
            return True, end
        if word == b"false":
            return False, end
        if word == b"null":
            return None, end
        return _Keyword(word.decode("latin-1")), end

    def _dictionary(self, position: int, depth: int) -> tuple[dict[str, Any], int]:
        data = self.data
        result: dict[str, Any] = {}
        while True:
            position = self.skip_space(position)
            if data.startswith(b">>", position):
                return result, position + 2
            if position >= len(data) or data[position] != 0x2F:
                raise _SyntaxFailure
            key, position = self._name(position + 1)
            value, position = self.parse(position, depth + 1)
            result[key] = value

    def _name(self, position: int) -> tuple[_Name, int]:
        data = self.data
        end = position
        while (
            end < len(data)
            and data[end] not in _WHITESPACE
            and data[end] not in _DELIMITERS
        ):
            end += 1
        raw = data[position:end]
        decoded = re.sub(
            rb"#([0-9A-Fa-f]{2})", lambda match: bytes([int(match.group(1), 16)]), raw
        )
        return _Name(decoded.decode("latin-1")), end

    def _literal_string_end(self, position: int) -> int:
        data = self.data
        balance = 1
        while position < len(data):
            byte = data[position]
            if byte == 0x5C:  # backslash
                position += 2
                continue
            if byte == 0x28:
                balance += 1
            elif byte == 0x29:
                balance -= 1
                if balance == 0:
                    return position + 1
            position += 1
        raise _SyntaxFailure

    def _number_or_reference(self, match: re.Match[bytes]) -> tuple[Any, int]:
        text = match.group(0)
        end = match.end()
        if b"." not in text:
            value = int(text)
            reference = _REFERENCE_TAIL_RE.match(self.data, end)
            if (
                reference is not None
                and value >= 0
                and not text.startswith((b"+", b"-"))
            ):
                return _Ref(value), reference.end()
            return value, end
        return float(text), end


class _Document:
    def __init__(self, data: bytes, max_objects: int, max_decompressed: int) -> None:
        self.data = data
        self.parser = _Parser(data)
        self.max_objects = max_objects
        self.decompression_budget = max_decompressed
        self.objects: dict[int, Any] = {}
        self.trailers: list[dict[str, Any]] = []
        self.object_stream_failed = False
        self.syntax_failed = False

    def scan(self) -> str | None:
        """Index indirect objects, trailers, and object streams."""
        data = self.data
        direct: dict[int, Any] = {}
        located: list[tuple[int, dict[str, Any]]] = []
        object_streams: list[tuple[dict[str, Any], bytes]] = []
        count = 0
        position = 0
        while True:
            match = _OBJECT_RE.search(data, position)
            if match is None:
                break
            count += 1
            if count > self.max_objects:
                return "pdf_object_limit"
            try:
                value, end = self.parser.parse(match.end())
            except _SyntaxFailure:
                self.syntax_failed = True
                position = match.end()
                continue
            stream: bytes | None = None
            after = self.parser.skip_space(end)
            if isinstance(value, dict) and data.startswith(b"stream", after):
                stream, end = self._stream(value, after + len(b"stream"))
            direct[int(match.group(1))] = value
            if isinstance(value, dict):
                kind = value.get("Type")
                if kind == "XRef":
                    located.append((match.start(), value))
                elif kind == "ObjStm" and stream is not None:
                    object_streams.append((value, stream))
            position = max(end, match.end())

        for trailer in _TRAILER_RE.finditer(data):
            try:
                value, _ = self.parser.parse(trailer.end())
            except _SyntaxFailure:
                continue
            if isinstance(value, dict):
                located.append((trailer.start(), value))
        located.sort(key=lambda item: item[0])
        self.trailers = [value for _, value in located]

        if any("Encrypt" in trailer for trailer in self.trailers):
            return None
        for header, stream in object_streams:
            failure = self._expand_object_stream(header, stream, direct)
            if failure is not None:
                return failure
            if len(self.objects) + len(direct) > self.max_objects:
                return "pdf_object_limit"
        self.objects.update(direct)
        return None

    def _stream(
        self, header: dict[str, Any], position: int
    ) -> tuple[bytes | None, int]:
        data = self.data
        if data.startswith(b"\r\n", position):
            position += 2
        elif data.startswith(b"\n", position) or data.startswith(b"\r", position):
            position += 1
        length = header.get("Length")
        if type(length) is int and 0 <= length <= len(data) - position:
            tail = self.parser.skip_space(position + length)
            if data.startswith(b"endstream", tail):
                return data[position : position + length], tail + len(b"endstream")
        end = data.find(b"endstream", position)
        if end < 0:
            return None, len(data)
        return data[position:end].rstrip(b"\r\n"), end + len(b"endstream")

    def _expand_object_stream(
        self, header: dict[str, Any], stream: bytes, direct: dict[int, Any]
    ) -> str | None:
        filters = header.get("Filter")
        if isinstance(filters, list) and len(filters) == 1:
            filters = filters[0]
        if filters is None:
            payload = stream
        elif filters == "FlateDecode" and "DecodeParms" not in header:
            decompressor = zlib.decompressobj()
            try:
                payload = decompressor.decompress(stream, self.decompression_budget + 1)
            except zlib.error:
                self.object_stream_failed = True
                return None
            if len(payload) > self.decompression_budget:
                return "pdf_decompression_limit"
        else:
            self.object_stream_failed = True
            return None
        self.decompression_budget -= len(payload)

        count, first = header.get("N"), header.get("First")
        if (
            type(count) is not int
            or type(first) is not int
            or count < 0
            or not 0 <= first <= len(payload)
        ):
            self.object_stream_failed = True
            return None
        if count > self.max_objects:
            return "pdf_object_limit"
        parser = _Parser(payload)
        offsets: list[tuple[int, int]] = []
        position = 0
        try:
            for _ in range(count):
                number, position = parser.parse(position)
                offset, position = parser.parse(position)
                if type(number) is not int or type(offset) is not int or offset < 0:
                    raise _SyntaxFailure
                offsets.append((number, offset))
            for number, offset in offsets:
                if number in direct:
                    continue
                value, _ = parser.parse(first + offset)
                self.objects[number] = value
        except _SyntaxFailure:
            self.object_stream_failed = True
        return None

    def resolve(self, value: Any) -> Any:
        for _ in range(_MAX_REFERENCE_HOPS):
            if not isinstance(value, _Ref):
                return value
            value = self.objects.get(value.number)
        return None

    def structure_failure(self, default: str) -> str:
        if self.object_stream_failed:
            return "pdf_object_stream_unsupported"
        return default


_INHERITED: Final[tuple[str, ...]] = ("MediaBox", "CropBox", "Rotate")


def _walk_page_tree(
    document: _Document, root: Any, max_pages: int
) -> list[dict[str, Any]] | str:
    pages: list[dict[str, Any]] = []
    visited: set[int] = set()
    stack: list[tuple[Any, dict[str, Any], int]] = [(root, {}, 0)]
    while stack:
        reference, inherited, depth = stack.pop()
        if depth > _MAX_TREE_DEPTH:
            return "pdf_page_tree_invalid"
        if isinstance(reference, _Ref):
            if reference.number in visited:
                return "pdf_page_tree_invalid"
            visited.add(reference.number)
        node = document.resolve(reference)
        if not isinstance(node, dict):
            return "pdf_page_tree_invalid"
        attributes = dict(inherited)
        for key in _INHERITED:
            if key in node:
                attributes[key] = node[key]
        kind = node.get("Type")
        kids = document.resolve(node.get("Kids"))
        if kind == "Pages" or (kind is None and "Kids" in node):
            if not isinstance(kids, list):
                return "pdf_page_tree_invalid"
            for kid in reversed(kids):
                stack.append((kid, attributes, depth + 1))
        elif kind == "Page" or kind is None:
            if len(pages) >= max_pages:
                return "pdf_page_limit"
            pages.append(attributes)
        else:
            return "pdf_page_tree_invalid"
    return pages


def _page_geometry(
    document: _Document, index: int, attributes: dict[str, Any]
) -> PdfPageGeometry:
    reasons: set[str] = set()
    media: Box | None = None
    if "MediaBox" not in attributes:
        reasons.add("media_box_missing")
    else:
        media = _box(document, attributes["MediaBox"])
        if media is None:
            reasons.add("media_box_invalid")

    crop = media
    if media is not None and "CropBox" in attributes:
        declared = _box(document, attributes["CropBox"])
        if declared is None:
            reasons.add("crop_box_invalid")
        else:
            clipped = (
                max(declared[0], media[0]),
                max(declared[1], media[1]),
                min(declared[2], media[2]),
                min(declared[3], media[3]),
            )
            if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                reasons.add("crop_box_invalid")
            else:
                if clipped != declared:
                    reasons.add("crop_box_outside_media_box")
                crop = clipped

    rotation = 0
    if "Rotate" in attributes:
        raw = document.resolve(attributes["Rotate"])
        if type(raw) is int and raw % 90 == 0 and abs(raw) <= 360_000:
            rotation = raw % 360
        else:
            reasons.add("rotation_invalid")

    width: float | None = None
    height: float | None = None
    if crop is not None:
        width, height = crop[2] - crop[0], crop[3] - crop[1]
        if rotation in (90, 270):
            width, height = height, width
    return PdfPageGeometry(
        page_index=index,
        media_box=media,
        crop_box=crop,
        rotation=rotation,
        width=width,
        height=height,
        reason_codes=tuple(code for code in _PAGE_REASONS if code in reasons),
    )


def _box(document: _Document, value: Any) -> Box | None:
    items = document.resolve(value)
    if not isinstance(items, list) or len(items) != 4:
        return None
    numbers: list[float] = []
    for item in items:
        number = document.resolve(item)
        if type(number) not in (int, float):
            return None
        converted = float(number)
        if not math.isfinite(converted) or abs(converted) > _MAX_BOX_COORDINATE:
            return None
        numbers.append(converted)
    x0, x1 = sorted((numbers[0], numbers[2]))
    y0, y1 = sorted((numbers[1], numbers[3]))
    if x0 == x1 or y0 == y1:
        return None
    return (x0 + 0.0, y0 + 0.0, x1 + 0.0, y1 + 0.0)
