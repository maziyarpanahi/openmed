"""EPUB text extraction with source character offsets.

The ingester uses only the Python standard library so EPUB dispatch remains
available without pulling extra dependencies into the multimodal install path.
It reads the EPUB package manifest and spine, extracts visible XHTML text in
reading order, and records source offsets back into each content document.
"""

from __future__ import annotations

import html as html_lib
import posixpath
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote
from xml.etree import ElementTree as ET

from .base import ExtractedDocument, SourceSpan, register_handler
from .exceptions import UnsupportedDocumentError

_CONTAINER_PATH = "META-INF/container.xml"
_SUPPORTED_CONTENT_TYPES = frozenset({"application/xhtml+xml", "text/html"})
_IGNORED_TAGS = frozenset({"head", "script", "style"})
_BREAK_TAGS = frozenset({"br"})
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "caption",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)


@dataclass(frozen=True)
class _ManifestItem:
    item_id: str
    href: str
    path: str
    media_type: str


@dataclass(frozen=True)
class _ParsedPackage:
    package_path: str
    manifest: dict[str, _ManifestItem]
    spine: tuple[str, ...]


@dataclass(frozen=True)
class _ParsedXhtml:
    text: str
    spans: tuple[SourceSpan, ...]


def extract_epub(path: str | Path) -> ExtractedDocument:
    """Extract EPUB spine text and source-offset metadata.

    Args:
        path: EPUB file path.

    Returns:
        An :class:`ExtractedDocument` with normalized text in spine reading
        order. Each mapped span carries source offsets into the XHTML content
        document named by ``section_href``.

    Raises:
        UnsupportedDocumentError: If the file is not a readable EPUB, declares
            encrypted content, or has no supported XHTML/HTML spine text.
    """
    source_path = Path(path)
    try:
        with zipfile.ZipFile(source_path) as archive:
            _ensure_supported_archive(archive)
            package = _read_package(archive)
            return _extract_spine_text(archive, package, source_path)
    except zipfile.BadZipFile as exc:
        raise UnsupportedDocumentError("EPUB must be a valid ZIP archive") from exc


def _ensure_supported_archive(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        if info.flag_bits & 0x1:
            raise UnsupportedDocumentError(
                "Encrypted EPUB ZIP entries are not supported"
            )
    if any(name.lower() == "meta-inf/encryption.xml" for name in archive.namelist()):
        raise UnsupportedDocumentError("DRM-protected EPUB files are not supported")


def _read_package(archive: zipfile.ZipFile) -> _ParsedPackage:
    container = _parse_xml(
        _read_required(archive, _CONTAINER_PATH),
        "container",
    )
    package_path = _package_path(container)
    package_root = _parse_xml(
        _read_required(archive, package_path),
        "package",
    )
    package_dir = posixpath.dirname(package_path)
    manifest = _manifest_items(package_root, package_dir)
    spine = _spine_itemrefs(package_root)
    if not spine:
        raise UnsupportedDocumentError("EPUB package does not define a spine")
    return _ParsedPackage(
        package_path=package_path,
        manifest=manifest,
        spine=spine,
    )


def _extract_spine_text(
    archive: zipfile.ZipFile,
    package: _ParsedPackage,
    source_path: Path,
) -> ExtractedDocument:
    parts: list[str] = []
    spans: list[SourceSpan] = []
    sections: list[dict[str, Any]] = []
    cursor = 0

    for idref in package.spine:
        item = package.manifest.get(idref)
        if item is None or item.media_type not in _SUPPORTED_CONTENT_TYPES:
            continue

        section_source = _decode_text(_read_required(archive, item.path))
        section = _parse_xhtml(section_source)
        if not section.text:
            continue

        if parts:
            parts.append("\n")
            cursor += 1

        section_index = len(sections)
        section_start = cursor
        parts.append(section.text)
        cursor += len(section.text)
        section_end = cursor

        for span in section.spans:
            metadata = dict(span.metadata)
            metadata.update(
                {
                    "format": "epub",
                    "section_index": section_index,
                    "section_id": item.item_id,
                    "section_href": item.path,
                }
            )
            spans.append(
                SourceSpan(
                    start=section_start + span.start,
                    end=section_start + span.end,
                    metadata=metadata,
                )
            )

        sections.append(
            {
                "index": section_index,
                "id": item.item_id,
                "href": item.path,
                "start": section_start,
                "end": section_end,
            }
        )

    if not sections:
        raise UnsupportedDocumentError(
            "EPUB spine does not contain supported XHTML or HTML text"
        )

    return ExtractedDocument(
        text="".join(parts),
        spans=tuple(spans),
        metadata={
            "format": "epub",
            "source_path": str(source_path),
            "package_path": package.package_path,
            "section_count": len(sections),
            "sections": sections,
        },
    )


def _package_path(container: ET.Element) -> str:
    rootfiles = [
        element
        for element in container.iter()
        if _local_name(element.tag) == "rootfile"
    ]
    for rootfile in rootfiles:
        full_path = rootfile.get("full-path")
        if full_path and rootfile.get("media-type") == "application/oebps-package+xml":
            return _clean_zip_path(full_path)
    for rootfile in rootfiles:
        full_path = rootfile.get("full-path")
        if full_path:
            return _clean_zip_path(full_path)
    raise UnsupportedDocumentError("EPUB container does not declare a package file")


def _manifest_items(root: ET.Element, package_dir: str) -> dict[str, _ManifestItem]:
    manifest = _required_child(root, "manifest")
    items: dict[str, _ManifestItem] = {}
    for element in _children(manifest, "item"):
        item_id = element.get("id")
        href = element.get("href")
        media_type = (element.get("media-type") or "").lower()
        if not item_id or not href:
            continue
        items[item_id] = _ManifestItem(
            item_id=item_id,
            href=href,
            path=_resolve_href(package_dir, href),
            media_type=media_type,
        )
    return items


def _spine_itemrefs(root: ET.Element) -> tuple[str, ...]:
    spine = _required_child(root, "spine")
    itemrefs: list[str] = []
    for element in _children(spine, "itemref"):
        idref = element.get("idref")
        if idref:
            itemrefs.append(idref)
    return tuple(itemrefs)


def _parse_xhtml(source: str) -> _ParsedXhtml:
    parser = _XhtmlTextParser(source)
    parser.feed(source)
    parser.close()
    return parser.document()


class _XhtmlTextParser(HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=False)
        self._source = source
        self._line_offsets = _line_offsets(source)
        self._parts: list[str] = []
        self._spans: list[SourceSpan] = []
        self._cursor = 0
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized in _IGNORED_TAGS:
            self._ignore_depth += 1
            return
        if self._ignore_depth:
            return
        if normalized in _BREAK_TAGS or normalized in _BLOCK_TAGS:
            self._append_break()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if self._ignore_depth or normalized in _IGNORED_TAGS:
            return
        if normalized in _BREAK_TAGS or normalized in _BLOCK_TAGS:
            self._append_break()

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in _IGNORED_TAGS and self._ignore_depth:
            self._ignore_depth -= 1
            return
        if self._ignore_depth:
            return
        if normalized in _BLOCK_TAGS:
            self._append_break()

    def handle_data(self, data: str) -> None:
        if self._ignore_depth or not data:
            return
        source_start = self._source_offset()
        if not data.strip():
            self._append_whitespace(source_start, source_start + len(data))
            return
        self._append_mapped(data, source_start, source_start + len(data))

    def handle_entityref(self, name: str) -> None:
        if self._ignore_depth:
            return
        source_start = self._source_offset()
        source_end = source_start + 1 + len(name)
        if source_end < len(self._source) and self._source[source_end] == ";":
            source_end += 1
        self._append_mapped(
            html_lib.unescape(self._source[source_start:source_end]),
            source_start,
            source_end,
        )

    def handle_charref(self, name: str) -> None:
        if self._ignore_depth:
            return
        source_start = self._source_offset()
        source_end = source_start + 2 + len(name)
        if source_end < len(self._source) and self._source[source_end] == ";":
            source_end += 1
        self._append_mapped(
            html_lib.unescape(self._source[source_start:source_end]),
            source_start,
            source_end,
        )

    def document(self) -> _ParsedXhtml:
        while self._parts and self._parts[-1] == "\n":
            self._parts.pop()
            self._cursor -= 1
        return _ParsedXhtml(text="".join(self._parts), spans=tuple(self._spans))

    def _append_break(self) -> None:
        if not self._parts or self._parts[-1].endswith("\n"):
            return
        self._parts.append("\n")
        self._cursor += 1

    def _append_whitespace(self, source_start: int, source_end: int) -> None:
        if not self._parts or self._parts[-1].endswith((" ", "\n")):
            return
        self._append_mapped(" ", source_start, source_end)

    def _append_mapped(self, text: str, source_start: int, source_end: int) -> None:
        if not text:
            return
        start = self._cursor
        self._parts.append(text)
        self._cursor += len(text)
        self._spans.append(
            SourceSpan(
                start=start,
                end=self._cursor,
                metadata={
                    "source_start": source_start,
                    "source_end": source_end,
                },
            )
        )

    def _source_offset(self) -> int:
        line, column = self.getpos()
        if line <= 0:
            return min(max(column, 0), len(self._source))
        index = line - 1
        if index >= len(self._line_offsets):
            return len(self._source)
        return min(self._line_offsets[index] + column, len(self._source))


def _line_offsets(text: str) -> tuple[int, ...]:
    starts = [0]
    for index, character in enumerate(text):
        if character == "\n":
            starts.append(index + 1)
    return tuple(starts)


def _read_required(archive: zipfile.ZipFile, path: str) -> bytes:
    try:
        return archive.read(path)
    except KeyError as exc:
        raise UnsupportedDocumentError(f"EPUB archive is missing {path}") from exc


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnsupportedDocumentError(
        "EPUB content documents must be UTF-8 or UTF-16 encoded"
    )


def _parse_xml(data: bytes, name: str) -> ET.Element:
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise UnsupportedDocumentError(f"EPUB {name} XML is invalid") from exc


def _required_child(element: ET.Element, name: str) -> ET.Element:
    for child in element:
        if _local_name(child.tag) == name:
            return child
    raise UnsupportedDocumentError(f"EPUB package is missing {name}")


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _resolve_href(package_dir: str, href: str) -> str:
    href_path = unquote(href.split("#", 1)[0])
    if package_dir:
        href_path = posixpath.join(package_dir, href_path)
    return _clean_zip_path(href_path)


def _clean_zip_path(path: str) -> str:
    normalized = posixpath.normpath(unquote(path).lstrip("/"))
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        raise UnsupportedDocumentError("EPUB contains an invalid archive path")
    return normalized


def _epub_handler(
    path: str | Path,
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    return extract_epub(path)


register_handler(".epub", _epub_handler, requires_multimodal=False)


__all__ = ["extract_epub"]
