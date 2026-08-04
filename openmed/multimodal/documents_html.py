"""Static HTML text extraction with source-preserving character offsets."""

from __future__ import annotations

import html as html_lib
import os
from dataclasses import dataclass
from html.entities import html5
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .base import ExtractedDocument, SourceSpan

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
_HARD_SUPPRESSION_TAGS = frozenset({"script", "style"})


@dataclass(frozen=True)
class _ParsedHtml:
    source: str
    document: ExtractedDocument


def extract_html(source: str | os.PathLike[str] | Any) -> ExtractedDocument:
    """Extract visible HTML text with raw-source character offsets.

    Args:
        source: A UTF-8 path, raw HTML string, or text file-like object.

    Returns:
        Visible normalized text and a complete source-offset map. Raw HTML is
        retained only while parsing and is never exposed in public metadata.
    """
    return _parse_source(source).document


def _parse_source(source: str | os.PathLike[str] | Any) -> _ParsedHtml:
    raw, path = _read_source(source)
    parser = _HtmlTextParser(raw, source_path=path)
    parser.feed(raw)
    parser.close()
    return _ParsedHtml(source=raw, document=parser.document())


def _read_source(source: str | os.PathLike[str] | Any) -> tuple[str, str | None]:
    if hasattr(source, "read"):
        return str(source.read()), None
    if isinstance(source, os.PathLike):
        path = Path(source)
        with path.open("r", encoding="utf-8", newline="") as handle:
            return handle.read(), str(path)
    if isinstance(source, str):
        if "\n" not in source and "\r" not in source:
            path = Path(source)
            try:
                exists = path.exists()
            except OSError:
                exists = False
            if exists and path.is_file():
                with path.open("r", encoding="utf-8", newline="") as handle:
                    return handle.read(), str(path)
        return source, None
    raise TypeError("source must be a path, HTML text, or text file-like object")


class _HtmlTextParser(HTMLParser):
    def __init__(self, source: str, *, source_path: str | None) -> None:
        super().__init__(convert_charrefs=False)
        self._source = source
        self._source_path = source_path
        self._line_starts = _line_starts(source)
        self._parts: list[str] = []
        self._spans: list[SourceSpan] = []
        self._cursor = 0
        self._in_head = False
        self._body_started = False
        self._hard_suppression = {tag: 0 for tag in _HARD_SUPPRESSION_TAGS}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized == "head" and not self._body_started:
            self._in_head = True
        elif normalized == "body":
            self._body_started = True
            self._in_head = False
        if normalized in _HARD_SUPPRESSION_TAGS:
            self._hard_suppression[normalized] += 1
            return
        if self._suppressed:
            return
        if normalized in _BREAK_TAGS or normalized in _BLOCK_TAGS:
            start = self._source_offset()
            raw_tag = self.get_starttag_text() or ""
            self._append_break(start, start + len(raw_tag))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if self._suppressed or normalized in _HARD_SUPPRESSION_TAGS:
            return
        if normalized in _BREAK_TAGS or normalized in _BLOCK_TAGS:
            start = self._source_offset()
            raw_tag = self.get_starttag_text() or ""
            self._append_break(start, start + len(raw_tag))

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in _HARD_SUPPRESSION_TAGS:
            if self._hard_suppression[normalized]:
                self._hard_suppression[normalized] -= 1
            return
        if normalized == "head":
            self._in_head = False
            return
        if self._suppressed:
            return
        if normalized in _BLOCK_TAGS:
            start = self._source_offset()
            end = self._tag_end(start)
            self._append_break(start, end)

    def handle_data(self, data: str) -> None:
        if self._suppressed or not data:
            return
        start = self._source_offset()
        if not data.strip():
            self._append_whitespace(start, start + len(data))
            return
        self._append(data, start, start + len(data), kind="text", mode="linear")

    def handle_entityref(self, name: str) -> None:
        if self._suppressed:
            return
        start = self._source_offset()
        callback_end = start + 1 + len(name)
        terminated = (
            callback_end < len(self._source) and self._source[callback_end] == ";"
        )
        if terminated:
            token = f"&{name};"
            if f"{name};" in html5:
                self._append(
                    html_lib.unescape(token),
                    start,
                    callback_end + 1,
                    kind="reference",
                    mode="atomic",
                )
            else:
                self._append(
                    token,
                    start,
                    callback_end + 1,
                    kind="literal_reference",
                    mode="linear",
                )
            return

        prefix = _longest_legacy_entity_prefix(name)
        if prefix is None:
            self._append(
                f"&{name}",
                start,
                callback_end,
                kind="literal_reference",
                mode="linear",
            )
            return
        atomic_end = start + 1 + len(prefix)
        self._append(
            html_lib.unescape(f"&{prefix}"),
            start,
            atomic_end,
            kind="reference",
            mode="atomic",
        )
        suffix = name[len(prefix) :]
        if suffix:
            self._append(
                suffix,
                atomic_end,
                callback_end,
                kind="literal_reference",
                mode="linear",
            )

    def handle_charref(self, name: str) -> None:
        if self._suppressed:
            return
        start = self._source_offset()
        end = start + 2 + len(name)
        if end < len(self._source) and self._source[end] == ";":
            end += 1
        token = self._source[start:end]
        decoded = html_lib.unescape(token)
        mode = "atomic" if decoded != token else "linear"
        self._append(
            decoded,
            start,
            end,
            kind="reference" if mode == "atomic" else "literal_reference",
            mode=mode,
        )

    @property
    def _suppressed(self) -> bool:
        return self._in_head or any(self._hard_suppression.values())

    def document(self) -> ExtractedDocument:
        while self._parts and self._parts[-1] == "\n":
            self._parts.pop()
            span = self._spans.pop()
            self._cursor = span.start
        metadata = {"format": "html"}
        if self._source_path is not None:
            metadata["source_path"] = self._source_path
        return ExtractedDocument(
            text="".join(self._parts),
            spans=tuple(self._spans),
            metadata=metadata,
        )

    def _append_whitespace(self, source_start: int, source_end: int) -> None:
        if not self._parts or self._parts[-1].endswith((" ", "\n")):
            return
        self._append(
            " ",
            source_start,
            source_end,
            kind="whitespace",
            mode="atomic",
        )

    def _append_break(self, source_start: int, source_end: int) -> None:
        if not self._parts or self._parts[-1].endswith("\n"):
            return
        self._append(
            "\n",
            source_start,
            source_end,
            kind="structural_separator",
            mode="atomic",
            replaceable=False,
        )

    def _append(
        self,
        text: str,
        source_start: int,
        source_end: int,
        *,
        kind: str,
        mode: str,
        replaceable: bool = True,
    ) -> None:
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
                    "format": "html",
                    "kind": kind,
                    "source_start": source_start,
                    "source_end": source_end,
                    "source_map_mode": mode,
                    "replaceable": replaceable,
                },
            )
        )

    def _source_offset(self) -> int:
        line, column = self.getpos()
        index = max(line - 1, 0)
        if index >= len(self._line_starts):
            return len(self._source)
        return min(self._line_starts[index] + column, len(self._source))

    def _tag_end(self, start: int) -> int:
        end = self._source.find(">", start)
        return len(self._source) if end < 0 else end + 1


def _line_starts(text: str) -> tuple[int, ...]:
    starts = [0]
    starts.extend(
        index + 1 for index, character in enumerate(text) if character == "\n"
    )
    return tuple(starts)


def _longest_legacy_entity_prefix(name: str) -> str | None:
    for end in range(len(name), 0, -1):
        prefix = name[:end]
        if prefix in html5:
            return prefix
    return None


__all__ = ["extract_html"]
