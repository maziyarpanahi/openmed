"""Markdown and AsciiDoc text extraction with source character maps.

The extractors normalize lightweight clinical handover markup into plain text
while preserving exact source character ranges for every emitted text segment.
That lets downstream redaction work on normalized text and then rewrite only
the corresponding raw markup characters.
"""

from __future__ import annotations

import importlib.util
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import ExtractedDocument, SourceSpan, register_handler
from .exceptions import MissingDependencyError

_MULTIMODAL_INSTALL_HINT = 'Install with: pip install "openmed[multimodal]".'

_MARKUP_DEPENDENCIES = {
    "markdown": ("markdown_it", "markdown-it-py"),
}

_MARKDOWN_EXTENSIONS = (".md", ".markdown")
_ASCIIDOC_EXTENSIONS = (".adoc", ".asciidoc")


@dataclass(frozen=True)
class _Segment:
    text: str
    source_start: int | None
    source_end: int | None
    kind: str

    @property
    def is_mapped(self) -> bool:
        return self.source_start is not None and self.source_end is not None


TextReplacement = tuple[int, int, str]


def extract_markdown(source: str | os.PathLike[str] | Any) -> ExtractedDocument:
    """Extract normalized text and source offsets from Markdown/GFM content.

    Args:
        source: A filesystem path, raw text string, or text file-like object.

    Returns:
        An :class:`ExtractedDocument` with ``source_start``/``source_end`` in
        each span's metadata.

    Raises:
        MissingDependencyError: If the ``multimodal`` extra's Markdown parser
            dependency is absent.
    """
    _ensure_markup_parser_available("markdown")
    text, path = _read_source(source)
    lines = _extract_markdown_lines(text)
    return _document_from_lines(
        text,
        lines,
        format_name="markdown",
        source_path=path,
    )


def extract_asciidoc(source: str | os.PathLike[str] | Any) -> ExtractedDocument:
    """Extract normalized text and source offsets from AsciiDoc content."""
    text, path = _read_source(source)
    lines = _extract_asciidoc_lines(text)
    return _document_from_lines(
        text,
        lines,
        format_name="asciidoc",
        source_path=path,
    )


def redact_source_text(
    document: ExtractedDocument,
    replacements: Iterable[TextReplacement],
) -> str:
    """Apply normalized-text replacements back to the original markup source.

    Each replacement is ``(start, end, replacement_text)`` where ``start`` and
    ``end`` are character offsets into ``document.text``. The function rewrites
    only mapped source ranges and leaves every surrounding markup byte
    unchanged for ASCII-compatible sources.
    """
    source_text = document.metadata.get("source_text")
    if not isinstance(source_text, str):
        raise ValueError("document metadata must include source_text")

    source_replacements: list[tuple[int, int, str]] = []
    for start, end, replacement in replacements:
        ranges = _source_ranges_for_text_span(document, int(start), int(end))
        if not ranges:
            raise ValueError(f"no source range maps normalized span {start}:{end}")
        for index, (source_start, source_end) in enumerate(ranges):
            source_replacements.append(
                (source_start, source_end, str(replacement) if index == 0 else "")
            )

    _validate_source_replacements(source_replacements)
    redacted = source_text
    for source_start, source_end, replacement in sorted(
        source_replacements, reverse=True
    ):
        redacted = redacted[:source_start] + replacement + redacted[source_end:]
    return redacted


def _ensure_markup_parser_available(flavor: str) -> None:
    module, distribution = _MARKUP_DEPENDENCIES[flavor]
    if importlib.util.find_spec(module) is None:
        raise MissingDependencyError(
            dependency=distribution,
            instruction=_MULTIMODAL_INSTALL_HINT,
        )


def _read_source(source: str | os.PathLike[str] | Any) -> tuple[str, str | None]:
    if hasattr(source, "read"):
        return str(source.read()), None
    if isinstance(source, os.PathLike):
        path = Path(source)
        return path.read_text(encoding="utf-8"), str(path)
    if isinstance(source, str):
        if "\n" in source or "\r" in source:
            return source, None
        path = Path(source)
        if path.exists():
            return path.read_text(encoding="utf-8"), str(path)
        return source, None
    raise TypeError("source must be a path, text content, or text file-like object")


def _iter_lines(text: str) -> Iterable[tuple[str, int]]:
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        yield line, offset
        offset += len(raw_line)


def _extract_markdown_lines(text: str) -> list[list[_Segment]]:
    lines = list(_iter_lines(text))
    extracted: list[list[_Segment]] = []
    in_fence = False
    fence_marker = ""

    for line, line_start in lines:
        if in_fence:
            if _is_markdown_fence_close(line, fence_marker):
                in_fence = False
                fence_marker = ""
                continue
            extracted.append(_line_as_segment(line, line_start, "code"))
            continue

        fence = _markdown_fence_marker(line)
        if fence is not None:
            in_fence = True
            fence_marker = fence
            continue

        if _is_markdown_table_separator(line) or _is_markdown_thematic_rule(line):
            continue

        segments = _markdown_line_segments(line, line_start)
        if _has_text(segments):
            extracted.append(segments)

    return extracted


def _markdown_line_segments(line: str, line_start: int) -> list[_Segment]:
    admonition = re.match(r"\s*!!!\s+\S+(?:\s+(?P<title>.+?))?\s*$", line)
    if admonition is not None:
        title = admonition.group("title")
        if not title:
            return []
        title_start = line_start + admonition.start("title")
        title_text, source_start, source_end = _strip_wrapping_quotes(
            title, title_start
        )
        return _parse_markdown_inline(title_text, source_start)

    if _looks_like_markdown_table_row(line):
        return _parse_delimited_row(
            line,
            line_start,
            delimiter="|",
            inline_parser=_parse_markdown_inline,
            kind="table_cell",
        )

    content_start = _markdown_content_start(line)
    if content_start is None:
        return []
    return _parse_markdown_inline(line[content_start:], line_start + content_start)


def _markdown_content_start(line: str) -> int | None:
    if not line.strip():
        return None

    index = len(line) - len(line.lstrip(" \t"))
    remainder = line[index:]

    for _ in range(6):
        blockquote = re.match(r">\s?", remainder)
        if blockquote is None:
            break
        index += blockquote.end()
        remainder = line[index:]

    heading = re.match(r"#{1,6}\s+", remainder)
    if heading is not None:
        return index + heading.end()

    task_or_list = re.match(r"(?:[-+*]|\d+[.)])\s+(?:\[[ xX]\]\s+)?", remainder)
    if task_or_list is not None:
        return index + task_or_list.end()

    return index


def _markdown_fence_marker(line: str) -> str | None:
    match = re.match(r" {0,3}(`{3,}|~{3,})", line)
    return match.group(1) if match is not None else None


def _is_markdown_fence_close(line: str, marker: str) -> bool:
    char = marker[0]
    length = len(marker)
    return re.match(rf" {{0,3}}{re.escape(char)}{{{length},}}\s*$", line) is not None


def _is_markdown_table_separator(line: str) -> bool:
    stripped = line.strip()
    return (
        "|" in stripped
        and "-" in stripped
        and re.fullmatch(r"\|?[\s|:-]+\|?", stripped) is not None
    )


def _is_markdown_thematic_rule(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and re.fullmatch(r"(?:[-*_]\s*){3,}", stripped) is not None


def _looks_like_markdown_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2


def _parse_markdown_inline(fragment: str, source_base: int) -> list[_Segment]:
    segments: list[_Segment] = []
    index = 0
    while index < len(fragment):
        link = _parse_markdown_link_or_image(fragment, source_base, index)
        if link is not None:
            link_segments, index = link
            segments.extend(link_segments)
            continue

        code = _parse_backtick_span(fragment, source_base, index, "inline_code")
        if code is not None:
            code_segments, index = code
            segments.extend(code_segments)
            continue

        autolink = _parse_markdown_autolink(fragment, source_base, index)
        if autolink is not None:
            autolink_segments, index = autolink
            segments.extend(autolink_segments)
            continue

        formatted = _parse_wrapped_inline(
            fragment,
            source_base,
            index,
            markers=("**", "__", "~~", "*", "_"),
            parser=_parse_markdown_inline,
        )
        if formatted is not None:
            formatted_segments, index = formatted
            segments.extend(formatted_segments)
            continue

        if fragment[index] == "\\" and index + 1 < len(fragment):
            escaped_start = source_base + index + 1
            segments.append(
                _mapped(fragment[index + 1], escaped_start, escaped_start + 1, "text")
            )
            index += 2
            continue

        next_index = _next_markdown_special(fragment, index + 1)
        segments.append(
            _mapped(
                fragment[index:next_index],
                source_base + index,
                source_base + next_index,
                "text",
            )
        )
        index = next_index
    return segments


def _parse_markdown_link_or_image(
    fragment: str, source_base: int, index: int
) -> tuple[list[_Segment], int] | None:
    is_image = fragment.startswith("![", index)
    if not is_image and not fragment.startswith("[", index):
        return None

    label_open = index + (1 if is_image else 0)
    label_close = _find_matching_delimiter(fragment, label_open, "[", "]")
    if label_close is None:
        return None

    paren_open = _skip_spaces(fragment, label_close + 1)
    if paren_open >= len(fragment) or fragment[paren_open] != "(":
        return None
    paren_close = _find_matching_delimiter(fragment, paren_open, "(", ")")
    if paren_close is None:
        return None

    label_start = label_open + 1
    label_text = fragment[label_start:label_close]
    label_segments = _parse_markdown_inline(label_text, source_base + label_start)
    if is_image:
        label_segments = [
            _Segment(
                segment.text,
                segment.source_start,
                segment.source_end,
                "image_alt" if segment.is_mapped else segment.kind,
            )
            for segment in label_segments
        ]
    destination_segments = _parse_markdown_destination(
        fragment[paren_open + 1 : paren_close],
        source_base + paren_open + 1,
        image=is_image,
    )
    return _join_components(label_segments, destination_segments), paren_close + 1


def _parse_markdown_destination(
    inner: str,
    source_base: int,
    *,
    image: bool,
) -> list[_Segment]:
    segments: list[_Segment] = []
    stripped, start, end = _trimmed_range(inner, source_base)
    if not stripped:
        return segments

    if stripped.startswith("<"):
        close = stripped.find(">")
        if close != -1:
            target_start = start + 1
            target_end = start + close
            target_text = stripped[1:close]
            title_text = stripped[close + 1 :]
            title_base = start + close + 1
        else:
            target_text, target_start, target_end = stripped, start, end
            title_text, title_base = "", end
    else:
        split = _first_unquoted_space(stripped)
        if split is None:
            target_text, target_start, target_end = stripped, start, end
            title_text, title_base = "", end
        else:
            target_text = stripped[:split]
            target_start = start
            target_end = start + split
            title_text = stripped[split:]
            title_base = start + split

    target_text, target_start, target_end = _trimmed_range_text(
        target_text, target_start
    )
    if target_text:
        segments.append(
            _mapped(
                target_text,
                target_start,
                target_end,
                "image_target" if image else "link_target",
            )
        )

    title, title_start, title_end = _strip_wrapping_quotes(title_text, title_base)
    if title:
        segments = _join_components(
            segments,
            [
                _mapped(
                    title,
                    title_start,
                    title_end,
                    "image_title" if image else "link_title",
                )
            ],
        )
    return segments


def _parse_markdown_autolink(
    fragment: str, source_base: int, index: int
) -> tuple[list[_Segment], int] | None:
    if fragment[index] != "<":
        return None
    close = fragment.find(">", index + 1)
    if close == -1:
        return None
    value = fragment[index + 1 : close]
    if not _looks_like_uri_or_email(value):
        if re.fullmatch(r"/?[A-Za-z][^>]*", value):
            return [], close + 1
        return None
    return (
        [_mapped(value, source_base + index + 1, source_base + close, "link_target")],
        close + 1,
    )


def _next_markdown_special(fragment: str, index: int) -> int:
    while index < len(fragment) and fragment[index] not in r"![`<\*_~":
        index += 1
    return index


def _extract_asciidoc_lines(text: str) -> list[list[_Segment]]:
    extracted: list[list[_Segment]] = []
    in_literal_block = False
    literal_delimiter = ""
    in_table = False

    for line, line_start in _iter_lines(text):
        stripped = line.strip()

        if stripped == "|===":
            in_table = not in_table
            continue

        if in_table:
            if line.lstrip().startswith("|"):
                segments = _parse_delimited_row(
                    line,
                    line_start,
                    delimiter="|",
                    inline_parser=_parse_asciidoc_inline,
                    kind="table_cell",
                )
                if _has_text(segments):
                    extracted.append(segments)
            continue

        if in_literal_block:
            if stripped == literal_delimiter:
                in_literal_block = False
                literal_delimiter = ""
                continue
            extracted.append(_line_as_segment(line, line_start, "literal"))
            continue

        if stripped in {"----", "....", "____"}:
            in_literal_block = True
            literal_delimiter = stripped
            continue

        segments = _asciidoc_line_segments(line, line_start)
        if _has_text(segments):
            extracted.append(segments)

    return extracted


def _asciidoc_line_segments(line: str, line_start: int) -> list[_Segment]:
    if not line.strip() or line.lstrip().startswith("//"):
        return []

    attribute = re.match(r"\s*:[^:]+:\s*(?P<value>.*?)\s*$", line)
    if attribute is not None:
        value = attribute.group("value")
        value_start = line_start + attribute.start("value")
        return _parse_asciidoc_inline(value, value_start)

    title = re.match(r"\s*\.(?P<title>\S.*)$", line)
    if title is not None:
        title_text = title.group("title")
        title_start = line_start + title.start("title")
        return _parse_asciidoc_inline(title_text, title_start)

    content_start = _asciidoc_content_start(line)
    if content_start is None:
        return []
    return _parse_asciidoc_inline(line[content_start:], line_start + content_start)


def _asciidoc_content_start(line: str) -> int | None:
    if not line.strip():
        return None
    index = len(line) - len(line.lstrip(" \t"))
    remainder = line[index:]

    heading = re.match(r"=+\s+", remainder)
    if heading is not None:
        return index + heading.end()

    admonition = re.match(r"(?:NOTE|TIP|IMPORTANT|WARNING|CAUTION):\s+", remainder)
    if admonition is not None:
        return index + admonition.end()

    list_marker = re.match(r"(?:[*.-]+|\d+[.)])\s+(?:\[[ xX]\]\s+)?", remainder)
    if list_marker is not None:
        return index + list_marker.end()

    quote = re.match(r">\s?", remainder)
    if quote is not None:
        return index + quote.end()

    return index


def _parse_asciidoc_inline(fragment: str, source_base: int) -> list[_Segment]:
    segments: list[_Segment] = []
    index = 0
    while index < len(fragment):
        macro = _parse_asciidoc_macro(fragment, source_base, index)
        if macro is not None:
            macro_segments, index = macro
            segments.extend(macro_segments)
            continue

        url = _parse_asciidoc_url_macro(fragment, source_base, index)
        if url is not None:
            url_segments, index = url
            segments.extend(url_segments)
            continue

        code = _parse_backtick_span(fragment, source_base, index, "inline_code")
        if code is not None:
            code_segments, index = code
            segments.extend(code_segments)
            continue

        formatted = _parse_wrapped_inline(
            fragment,
            source_base,
            index,
            markers=("*", "_", "#"),
            parser=_parse_asciidoc_inline,
        )
        if formatted is not None:
            formatted_segments, index = formatted
            segments.extend(formatted_segments)
            continue

        next_index = _next_asciidoc_special(fragment, index + 1)
        segments.append(
            _mapped(
                fragment[index:next_index],
                source_base + index,
                source_base + next_index,
                "text",
            )
        )
        index = next_index
    return segments


def _parse_asciidoc_macro(
    fragment: str, source_base: int, index: int
) -> tuple[list[_Segment], int] | None:
    for prefix, kind in (
        ("image::", "image"),
        ("image:", "image"),
        ("link:", "link"),
        ("xref:", "link"),
    ):
        if not fragment.startswith(prefix, index):
            continue
        target_start = index + len(prefix)
        bracket_open = fragment.find("[", target_start)
        if bracket_open == -1:
            return None
        bracket_close = _find_matching_delimiter(fragment, bracket_open, "[", "]")
        if bracket_close is None:
            return None

        target_text, source_start, source_end = _trimmed_range(
            fragment[target_start:bracket_open],
            source_base + target_start,
        )
        target_kind = "image_target" if kind == "image" else "link_target"
        target = [_mapped(target_text, source_start, source_end, target_kind)]
        attrs = fragment[bracket_open + 1 : bracket_close]
        attr_segments = _parse_asciidoc_attributes(
            attrs,
            source_base + bracket_open + 1,
            image=kind == "image",
        )
        if kind == "image":
            return _join_components(target, attr_segments), bracket_close + 1
        return _join_components(attr_segments, target), bracket_close + 1
    return None


def _parse_asciidoc_url_macro(
    fragment: str, source_base: int, index: int
) -> tuple[list[_Segment], int] | None:
    match = re.match(r"(?:https?://|mailto:)[^\s\[]+", fragment[index:])
    if match is None:
        return None
    target_end = index + match.end()
    if target_end >= len(fragment) or fragment[target_end] != "[":
        return None
    label_end = _find_matching_delimiter(fragment, target_end, "[", "]")
    if label_end is None:
        return None

    target_text = fragment[index:target_end]
    label_text = fragment[target_end + 1 : label_end]
    label_segments = _parse_asciidoc_inline(label_text, source_base + target_end + 1)
    target_segment = _mapped(
        target_text,
        source_base + index,
        source_base + target_end,
        "link_target",
    )
    return _join_components(label_segments, [target_segment]), label_end + 1


def _parse_asciidoc_attributes(
    attrs: str,
    source_base: int,
    *,
    image: bool,
) -> list[_Segment]:
    segments: list[_Segment] = []
    for index, (item, item_start) in enumerate(_split_csv_like(attrs, source_base)):
        stripped, start, end = _trimmed_range(item, item_start)
        if not stripped:
            continue
        key, value, value_start = _split_attribute(stripped, start)
        if key is None and index == 0:
            text, text_start, text_end = _strip_wrapping_quotes(value, value_start)
            segments.append(
                _mapped(
                    text, text_start, text_end, "image_alt" if image else "link_text"
                )
            )
        elif key == "title":
            text, text_start, text_end = _strip_wrapping_quotes(value, value_start)
            segments.append(
                _mapped(
                    text,
                    text_start,
                    text_end,
                    "image_title" if image else "link_title",
                )
            )
    return segments


def _next_asciidoc_special(fragment: str, index: int) -> int:
    while index < len(fragment):
        if fragment[index] in "`*_#":
            return index
        if any(
            fragment.startswith(prefix, index)
            for prefix in (
                "image::",
                "image:",
                "link:",
                "xref:",
                "http://",
                "https://",
                "mailto:",
            )
        ):
            return index
        index += 1
    return index


def _parse_backtick_span(
    fragment: str,
    source_base: int,
    index: int,
    kind: str,
) -> tuple[list[_Segment], int] | None:
    if fragment[index] != "`":
        return None
    marker_end = index
    while marker_end < len(fragment) and fragment[marker_end] == "`":
        marker_end += 1
    marker = fragment[index:marker_end]
    close = fragment.find(marker, marker_end)
    if close == -1:
        return None
    content = fragment[marker_end:close]
    return (
        [_mapped(content, source_base + marker_end, source_base + close, kind)],
        close + len(marker),
    )


def _parse_wrapped_inline(
    fragment: str,
    source_base: int,
    index: int,
    *,
    markers: Sequence[str],
    parser: Any,
) -> tuple[list[_Segment], int] | None:
    for marker in markers:
        if not fragment.startswith(marker, index):
            continue
        close = fragment.find(marker, index + len(marker))
        if close == -1:
            continue
        content_start = index + len(marker)
        content = fragment[content_start:close]
        if not content:
            continue
        return parser(content, source_base + content_start), close + len(marker)
    return None


def _parse_delimited_row(
    line: str,
    line_start: int,
    *,
    delimiter: str,
    inline_parser: Any,
    kind: str,
) -> list[_Segment]:
    cells = _split_delimited(line, line_start, delimiter)
    components: list[list[_Segment]] = []
    for cell, cell_start in cells:
        stripped, source_start, _ = _trimmed_range(cell, cell_start)
        if stripped:
            parsed = inline_parser(stripped, source_start)
            components.append(
                [
                    _Segment(
                        segment.text,
                        segment.source_start,
                        segment.source_end,
                        kind if segment.kind == "text" else segment.kind,
                    )
                    for segment in parsed
                ]
            )
    return _join_components(*components)


def _split_delimited(
    line: str,
    line_start: int,
    delimiter: str,
) -> list[tuple[str, int]]:
    cells: list[tuple[str, int]] = []
    cell_start = 1 if line.startswith(delimiter) else 0
    index = cell_start
    escaped = False
    while index < len(line):
        char = line[index]
        if char == "\\" and not escaped:
            escaped = True
            index += 1
            continue
        if char == delimiter and not escaped:
            cells.append((line[cell_start:index], line_start + cell_start))
            cell_start = index + 1
        else:
            escaped = False
        index += 1
    if cell_start < len(line):
        cells.append((line[cell_start:], line_start + cell_start))
    return cells


def _split_csv_like(text: str, source_base: int) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    start = 0
    quote: str | None = None
    for index, char in enumerate(text):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == ",":
            items.append((text[start:index], source_base + start))
            start = index + 1
    items.append((text[start:], source_base + start))
    return items


def _split_attribute(text: str, source_start: int) -> tuple[str | None, str, int]:
    equals = text.find("=")
    if equals == -1:
        return None, text, source_start
    key = text[:equals].strip().lower()
    value = text[equals + 1 :]
    value_start = source_start + equals + 1
    return key, value, value_start


def _line_as_segment(line: str, line_start: int, kind: str) -> list[_Segment]:
    if not line:
        return []
    return [_mapped(line, line_start, line_start + len(line), kind)]


def _document_from_lines(
    source_text: str,
    lines: Sequence[Sequence[_Segment]],
    *,
    format_name: str,
    source_path: str | None,
) -> ExtractedDocument:
    parts: list[str] = []
    spans: list[SourceSpan] = []
    cursor = 0

    for line_index, line_segments in enumerate(lines):
        if line_index:
            parts.append("\n")
            cursor += 1
        for segment in line_segments:
            if not segment.text:
                continue
            start = cursor
            parts.append(segment.text)
            cursor += len(segment.text)
            if not segment.is_mapped:
                continue
            spans.append(
                SourceSpan(
                    start=start,
                    end=cursor,
                    metadata={
                        "format": format_name,
                        "kind": segment.kind,
                        "source_start": segment.source_start,
                        "source_end": segment.source_end,
                    },
                )
            )

    metadata: dict[str, Any] = {"format": format_name, "source_text": source_text}
    if source_path is not None:
        metadata["source_path"] = source_path
    return ExtractedDocument(text="".join(parts), spans=tuple(spans), metadata=metadata)


def _source_ranges_for_text_span(
    document: ExtractedDocument,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    if start < 0 or end < start or end > len(document.text):
        raise ValueError("replacement offsets must fall within document.text")

    ranges: list[tuple[int, int]] = []
    for span in document.spans:
        overlap_start = max(start, span.start)
        overlap_end = min(end, span.end)
        if overlap_start >= overlap_end:
            continue
        source_start = span.metadata.get("source_start")
        source_end = span.metadata.get("source_end")
        if not isinstance(source_start, int) or not isinstance(source_end, int):
            continue
        ranges.append(
            (
                source_start + (overlap_start - span.start),
                source_start + (overlap_end - span.start),
            )
        )
    return ranges


def _validate_source_replacements(replacements: Sequence[tuple[int, int, str]]) -> None:
    ordered = sorted(replacements)
    previous_end = -1
    for source_start, source_end, _ in ordered:
        if source_start < previous_end:
            raise ValueError("source replacements must not overlap")
        if source_end < source_start:
            raise ValueError("source replacement end must be >= start")
        previous_end = source_end


def _mapped(text: str, source_start: int, source_end: int, kind: str) -> _Segment:
    return _Segment(text, source_start, source_end, kind)


def _synthetic(text: str) -> _Segment:
    return _Segment(text, None, None, "separator")


def _join_components(*components: Sequence[_Segment]) -> list[_Segment]:
    joined: list[_Segment] = []
    for component in components:
        clean = [segment for segment in component if segment.text]
        if not clean:
            continue
        if joined and not joined[-1].text.endswith((" ", "\t", "\n")):
            joined.append(_synthetic(" "))
        joined.extend(clean)
    return joined


def _has_text(segments: Sequence[_Segment]) -> bool:
    return any(segment.text.strip() for segment in segments)


def _find_matching_delimiter(
    text: str,
    open_index: int,
    open_char: str,
    close_char: str,
) -> int | None:
    depth = 0
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
    return None


def _skip_spaces(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t":
        index += 1
    return index


def _trimmed_range(text: str, source_base: int) -> tuple[str, int, int]:
    return _trimmed_range_text(text, source_base)


def _trimmed_range_text(text: str, source_base: int) -> tuple[str, int, int]:
    left = 0
    right = len(text)
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    return text[left:right], source_base + left, source_base + right


def _strip_wrapping_quotes(text: str, source_base: int) -> tuple[str, int, int]:
    stripped, start, end = _trimmed_range(text, source_base)
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1], start + 1, end - 1
    if len(stripped) >= 2 and stripped[0] == "(" and stripped[-1] == ")":
        return stripped[1:-1], start + 1, end - 1
    return stripped, start, end


def _first_unquoted_space(text: str) -> int | None:
    quote: str | None = None
    for index, char in enumerate(text):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char.isspace():
            return index
    return None


def _looks_like_uri_or_email(value: str) -> bool:
    return (
        value.startswith(("http://", "https://", "mailto:"))
        or re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value) is not None
    )


def _markdown_handler(
    path: str | os.PathLike[str],
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    return extract_markdown(path)


def _asciidoc_handler(
    path: str | os.PathLike[str],
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    return extract_asciidoc(path)


register_handler(_MARKDOWN_EXTENSIONS, _markdown_handler)
register_handler(_ASCIIDOC_EXTENSIONS, _asciidoc_handler)


__all__ = [
    "TextReplacement",
    "extract_asciidoc",
    "extract_markdown",
    "redact_source_text",
]
