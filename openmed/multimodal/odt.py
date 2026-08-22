"""ODT text extraction with character-offset source maps.

OpenDocument Text files are ZIP archives whose visible document content lives
in ``content.xml``. This module uses only the Python standard library, keeps
paragraph/list reading order, and linearizes tables as tab-separated cells and
newline-separated rows. Each non-empty XML text fragment becomes a
:class:`~openmed.multimodal.base.SourceSpan` in the shared multimodal contract.

Writing redactions back into ODT archives is intentionally out of scope.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree as ET

from .base import ExtractedDocument, SourceSpan, register_handler
from .exceptions import UnsupportedDocumentError

_CONTENT_PATH = "content.xml"
_ODT_MIMETYPE = "application/vnd.oasis.opendocument.text"
_MAX_REPEAT = 10_000
_UNSAFE_XML_DECLARATION = re.compile(rb"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)

_OFFICE_NS = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"

_OFFICE_BODY = f"{{{_OFFICE_NS}}}body"
_OFFICE_TEXT = f"{{{_OFFICE_NS}}}text"
_TEXT_HEADING = f"{{{_TEXT_NS}}}h"
_TEXT_PARAGRAPH = f"{{{_TEXT_NS}}}p"
_TEXT_LIST = f"{{{_TEXT_NS}}}list"
_TEXT_LIST_HEADER = f"{{{_TEXT_NS}}}list-header"
_TEXT_LIST_ITEM = f"{{{_TEXT_NS}}}list-item"
_TEXT_SPACE = f"{{{_TEXT_NS}}}s"
_TEXT_TAB = f"{{{_TEXT_NS}}}tab"
_TEXT_LINE_BREAK = f"{{{_TEXT_NS}}}line-break"
_TEXT_SPACE_COUNT = f"{{{_TEXT_NS}}}c"
_TABLE = f"{{{_TABLE_NS}}}table"
_TABLE_ROW = f"{{{_TABLE_NS}}}table-row"
_TABLE_CELL = f"{{{_TABLE_NS}}}table-cell"
_COVERED_TABLE_CELL = f"{{{_TABLE_NS}}}covered-table-cell"
_TABLE_ROWS_REPEATED = f"{{{_TABLE_NS}}}number-rows-repeated"
_TABLE_COLUMNS_REPEATED = f"{{{_TABLE_NS}}}number-columns-repeated"
_PARAGRAPH_TAGS = frozenset({_TEXT_HEADING, _TEXT_PARAGRAPH})
_TABLE_CELL_TAGS = frozenset({_TABLE_CELL, _COVERED_TABLE_CELL})


@dataclass(frozen=True)
class _Fragment:
    text: str
    metadata: Mapping[str, Any] | None


@dataclass
class _ExtractionState:
    parts: list[str] = field(default_factory=list)
    spans: list[SourceSpan] = field(default_factory=list)
    cursor: int = 0
    block_count: int = 0
    paragraph_count: int = 0
    list_item_count: int = 0
    table_count: int = 0
    table_row_count: int = 0
    table_cell_count: int = 0

    def append_block(self, fragments: list[_Fragment]) -> None:
        """Append one non-empty logical block and map its source fragments."""
        if not any(fragment.text for fragment in fragments):
            return
        if self.parts:
            self.parts.append("\n")
            self.cursor += 1

        block_index = self.block_count
        for fragment in fragments:
            if not fragment.text:
                continue
            start = self.cursor
            self.parts.append(fragment.text)
            self.cursor += len(fragment.text)
            if fragment.metadata is None:
                continue
            metadata = dict(fragment.metadata)
            metadata["block_index"] = block_index
            metadata["document_text_node_index"] = len(self.spans)
            self.spans.append(
                SourceSpan(start=start, end=self.cursor, metadata=metadata)
            )
        self.block_count += 1


class _OdtReader:
    def __init__(self) -> None:
        self.state = _ExtractionState()
        self._source_paragraph_index = 0
        self._next_list_item_index = 0

    def document(self, root: ET.Element, source_path: Path) -> ExtractedDocument:
        """Extract an ODT XML tree into the shared document contract."""
        text_body = _find_text_body(root)
        self._walk_container(text_body, path="office:text")
        return ExtractedDocument(
            text="".join(self.state.parts),
            spans=tuple(self.state.spans),
            metadata={
                "format": "odt",
                "source_path": str(source_path),
                "content_path": _CONTENT_PATH,
                "paragraph_count": self.state.paragraph_count,
                "text_node_count": len(self.state.spans),
                "block_count": self.state.block_count,
                "list_item_count": self.state.list_item_count,
                "table_count": self.state.table_count,
                "table_row_count": self.state.table_row_count,
                "table_cell_count": self.state.table_cell_count,
            },
        )

    def _walk_container(
        self,
        container: ET.Element,
        *,
        path: str,
        list_level: int = 0,
        list_item_index: int | None = None,
    ) -> None:
        for child_index, child in enumerate(container):
            child_path = f"{path}/{_display_tag(child.tag)}[{child_index}]"
            if child.tag in _PARAGRAPH_TAGS:
                block_type = (
                    "list_item"
                    if list_item_index is not None
                    else "heading"
                    if child.tag == _TEXT_HEADING
                    else "paragraph"
                )
                fragments = self._paragraph_fragments(
                    child,
                    path=child_path,
                    block_type=block_type,
                    list_level=list_level,
                    list_item_index=list_item_index,
                )
                self.state.append_block(fragments)
            elif child.tag == _TEXT_LIST:
                self._walk_list(child, path=child_path, list_level=list_level + 1)
            elif child.tag == _TABLE:
                self._append_table(child, path=child_path)
            else:
                self._walk_container(
                    child,
                    path=child_path,
                    list_level=list_level,
                    list_item_index=list_item_index,
                )

    def _walk_list(self, element: ET.Element, *, path: str, list_level: int) -> None:
        for child_index, child in enumerate(element):
            if child.tag not in {_TEXT_LIST_ITEM, _TEXT_LIST_HEADER}:
                continue
            item_index = self._next_list_item_index
            self._next_list_item_index += 1
            self.state.list_item_count += 1
            item_path = f"{path}/{_display_tag(child.tag)}[{child_index}]"
            self._walk_container(
                child,
                path=item_path,
                list_level=list_level,
                list_item_index=item_index,
            )

    def _append_table(self, element: ET.Element, *, path: str) -> None:
        table_index = self.state.table_count
        self.state.table_count += 1
        row_index = 0

        for source_row_index, row in enumerate(_iter_table_rows(element)):
            row_path = f"{path}/table-row[{source_row_index}]"
            template, cell_count = self._table_row_fragments(
                row,
                path=row_path,
                table_index=table_index,
                row_index=row_index,
            )
            row_repeat = _repeat_count(row, _TABLE_ROWS_REPEATED, "table row")
            for repeat_index in range(row_repeat):
                logical_row_index = row_index + repeat_index
                fragments = _updated_fragments(template, row_index=logical_row_index)
                self.state.append_block(fragments)
                self.state.table_row_count += 1
                self.state.table_cell_count += cell_count
            row_index += row_repeat

    def _table_row_fragments(
        self,
        row: ET.Element,
        *,
        path: str,
        table_index: int,
        row_index: int,
    ) -> tuple[list[_Fragment], int]:
        cells: list[list[_Fragment]] = []
        cell_index = 0
        source_cell_index = 0
        for cell in row:
            if cell.tag not in _TABLE_CELL_TAGS:
                continue
            cell_path = f"{path}/table-cell[{source_cell_index}]"
            source_cell_index += 1
            template = self._table_cell_fragments(
                cell,
                path=cell_path,
                table_index=table_index,
                row_index=row_index,
                cell_index=cell_index,
            )
            cell_repeat = _repeat_count(cell, _TABLE_COLUMNS_REPEATED, "table cell")
            for repeat_index in range(cell_repeat):
                logical_cell_index = cell_index + repeat_index
                cells.append(
                    _updated_fragments(template, cell_index=logical_cell_index)
                )
            cell_index += cell_repeat

        row_fragments: list[_Fragment] = []
        for index, cell_fragments in enumerate(cells):
            if index:
                row_fragments.append(_Fragment("\t", None))
            row_fragments.extend(cell_fragments)
        return row_fragments, len(cells)

    def _table_cell_fragments(
        self,
        cell: ET.Element,
        *,
        path: str,
        table_index: int,
        row_index: int,
        cell_index: int,
    ) -> list[_Fragment]:
        paragraphs: list[list[_Fragment]] = []

        def collect(container: ET.Element, container_path: str) -> None:
            for child_index, child in enumerate(container):
                child_path = (
                    f"{container_path}/{_display_tag(child.tag)}[{child_index}]"
                )
                if child.tag in _PARAGRAPH_TAGS:
                    fragments = self._paragraph_fragments(
                        child,
                        path=child_path,
                        block_type="table_cell",
                        table_index=table_index,
                        row_index=row_index,
                        cell_index=cell_index,
                    )
                    if any(fragment.text for fragment in fragments):
                        paragraphs.append(fragments)
                else:
                    collect(child, child_path)

        collect(cell, path)
        fragments: list[_Fragment] = []
        for index, paragraph in enumerate(paragraphs):
            if index:
                fragments.append(_Fragment("\n", None))
            fragments.extend(paragraph)
        return fragments

    def _paragraph_fragments(
        self,
        element: ET.Element,
        *,
        path: str,
        block_type: str,
        list_level: int = 0,
        list_item_index: int | None = None,
        table_index: int | None = None,
        row_index: int | None = None,
        cell_index: int | None = None,
    ) -> list[_Fragment]:
        paragraph_index = self._source_paragraph_index
        self._source_paragraph_index += 1
        metadata: dict[str, Any] = {
            "format": "odt",
            "part": "body",
            "block_type": block_type,
            "paragraph_index": paragraph_index,
        }
        optional_values = {
            "list_level": list_level or None,
            "list_item_index": list_item_index,
            "table_index": table_index,
            "row_index": row_index,
            "cell_index": cell_index,
        }
        metadata.update(
            {key: value for key, value in optional_values.items() if value is not None}
        )
        fragments = _inline_fragments(element, path=path, metadata=metadata)
        if any(fragment.text for fragment in fragments):
            self.state.paragraph_count += 1
        return fragments


def extract_odt(path: str | Path) -> ExtractedDocument:
    """Extract normalized ODT text plus char-offset source spans.

    Paragraphs and list items retain XML reading order. Table cells are joined
    with tabs and rows with newlines. Every mapped span records its ODT block,
    paragraph, list, and table location without retaining raw document text in
    metadata.

    Args:
        path: OpenDocument Text (``.odt``) file path.

    Returns:
        An :class:`ExtractedDocument` containing normalized text and structural
        source spans.

    Raises:
        UnsupportedDocumentError: If the ODT archive or ``content.xml`` is
            missing, encrypted, mislabeled, or malformed.
    """
    source_path = Path(path)
    try:
        with zipfile.ZipFile(source_path) as archive:
            _ensure_supported_archive(archive)
            content = _read_required(archive, _CONTENT_PATH)
    except zipfile.BadZipFile as exc:
        raise UnsupportedDocumentError("ODT must be a valid ZIP archive") from exc

    root = _parse_content_xml(content)
    return _OdtReader().document(root, source_path)


def _ensure_supported_archive(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        if info.flag_bits & 0x1:
            raise UnsupportedDocumentError("Encrypted ODT ZIP entries are unsupported")
    try:
        mimetype = archive.read("mimetype").decode("ascii").strip()
    except KeyError:
        return
    except UnicodeDecodeError as exc:
        raise UnsupportedDocumentError("ODT mimetype entry must be ASCII") from exc
    if mimetype != _ODT_MIMETYPE:
        raise UnsupportedDocumentError(
            f"ODT mimetype must be {_ODT_MIMETYPE!r}, got {mimetype!r}"
        )


def _read_required(archive: zipfile.ZipFile, path: str) -> bytes:
    try:
        return archive.read(path)
    except KeyError as exc:
        raise UnsupportedDocumentError(f"ODT archive is missing {path}") from exc


def _parse_content_xml(data: bytes) -> ET.Element:
    if _UNSAFE_XML_DECLARATION.search(data):
        raise UnsupportedDocumentError(
            "ODT content.xml with DOCTYPE or ENTITY declarations is unsupported"
        )
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise UnsupportedDocumentError("ODT content.xml is invalid") from exc


def _find_text_body(root: ET.Element) -> ET.Element:
    for body in root.iter(_OFFICE_BODY):
        for child in body:
            if child.tag == _OFFICE_TEXT:
                return child
    raise UnsupportedDocumentError("ODT content.xml is missing office:text")


def _inline_fragments(
    element: ET.Element,
    *,
    path: str,
    metadata: Mapping[str, Any],
) -> list[_Fragment]:
    fragments: list[_Fragment] = []
    text_node_index = 0

    def append(text: str | None, *, node_type: str, node_path: str) -> None:
        nonlocal text_node_index
        if not text:
            return
        node_metadata = dict(metadata)
        node_metadata.update(
            {
                "element_path": node_path,
                "node_type": node_type,
                "text_node_index": text_node_index,
            }
        )
        fragments.append(_Fragment(text, node_metadata))
        text_node_index += 1

    def walk(node: ET.Element, node_path: str) -> None:
        append(node.text, node_type="text", node_path=node_path)
        for child_index, child in enumerate(node):
            child_path = f"{node_path}/{_display_tag(child.tag)}[{child_index}]"
            if child.tag == _TEXT_SPACE:
                count = _repeat_count(child, _TEXT_SPACE_COUNT, "text space")
                append(" " * count, node_type="space", node_path=child_path)
            elif child.tag == _TEXT_TAB:
                append("\t", node_type="tab", node_path=child_path)
            elif child.tag == _TEXT_LINE_BREAK:
                append("\n", node_type="line_break", node_path=child_path)
            else:
                walk(child, child_path)
            append(child.tail, node_type="tail", node_path=child_path)

    walk(element, path)
    return fragments


def _iter_table_rows(table: ET.Element):
    for child in table:
        if child.tag == _TABLE_ROW:
            yield child
        elif child.tag != _TABLE:
            yield from _iter_table_rows(child)


def _repeat_count(element: ET.Element, attribute: str, description: str) -> int:
    raw = element.get(attribute)
    if raw is None:
        return 1
    try:
        count = int(raw)
    except ValueError as exc:
        raise UnsupportedDocumentError(
            f"ODT {description} repeat count must be an integer"
        ) from exc
    if count < 1 or count > _MAX_REPEAT:
        raise UnsupportedDocumentError(
            f"ODT {description} repeat count must be between 1 and {_MAX_REPEAT}"
        )
    return count


def _updated_fragments(fragments: list[_Fragment], **values: int) -> list[_Fragment]:
    updated: list[_Fragment] = []
    for fragment in fragments:
        if fragment.metadata is None:
            updated.append(fragment)
            continue
        metadata = dict(fragment.metadata)
        metadata.update(values)
        updated.append(_Fragment(fragment.text, metadata))
    return updated


def _display_tag(tag: str) -> str:
    namespace, _, local_name = tag.removeprefix("{").partition("}")
    prefixes = {_OFFICE_NS: "office", _TEXT_NS: "text", _TABLE_NS: "table"}
    prefix = prefixes.get(namespace)
    return f"{prefix}:{local_name}" if prefix else local_name or tag


def _odt_handler(
    path: str | Path,
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    return extract_odt(path)


register_handler(".odt", _odt_handler, requires_multimodal=False)


__all__ = ["extract_odt"]
