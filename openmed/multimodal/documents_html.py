"""Static HTML text extraction with source-preserving character offsets."""

from __future__ import annotations

import html as html_lib
import inspect
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from html.entities import html5
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .base import ExtractedDocument, SourceSpan, register_handler

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


def write_redacted_html(
    source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    replacements: Iterable[tuple[int, int, str]],
) -> Path:
    """Write normalized-text replacements into a distinct HTML source copy.

    Args:
        source_path: UTF-8 HTML source path.
        output_path: Distinct destination path for redacted HTML.
        replacements: ``(start, end, text)`` ranges in extracted visible text.

    Returns:
        The destination :class:`~pathlib.Path`.

    Raises:
        ValueError: If paths alias, ranges are invalid, or projected edits
            collide.
    """
    source = Path(source_path)
    output = Path(output_path)
    _validate_distinct_paths(source, output)
    parsed = _parse_source(source)
    return _write_parsed_html(parsed, output, replacements)


def _write_parsed_html(
    parsed: _ParsedHtml,
    output: Path,
    replacements: Iterable[tuple[int, int, str]],
) -> Path:
    logical = _validate_logical_replacements(parsed.document, replacements)
    projected = [
        (_project_replacement(parsed.document, start, end), replacement)
        for start, end, replacement in logical
    ]
    _validate_projected_replacements(projected)
    redacted = _render_replacements(parsed.source, projected)
    with output.open("w", encoding="utf-8", newline="") as handle:
        handle.write(redacted)
    return output


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
        self._head_depth = 0
        self._body_started = False
        self._hard_suppression = {tag: 0 for tag in _HARD_SUPPRESSION_TAGS}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized == "head" and not self._body_started:
            self._head_depth += 1
        elif normalized == "body":
            self._body_started = True
            self._head_depth = 0
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
            if self._head_depth:
                self._head_depth -= 1
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
        return bool(self._head_depth) or any(self._hard_suppression.values())

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


def _validate_distinct_paths(source: Path, output: Path) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("source and output paths must be distinct")
    if source.exists() and output.exists() and os.path.samefile(source, output):
        raise ValueError("source and output paths must be distinct")


def _validate_logical_replacements(
    document: ExtractedDocument,
    replacements: Iterable[tuple[int, int, str]],
) -> tuple[tuple[int, int, str], ...]:
    unique: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for raw_start, raw_end, raw_replacement in replacements:
        item = (int(raw_start), int(raw_end), str(raw_replacement))
        if item in seen:
            continue
        seen.add(item)
        start, end, _ = item
        if start < 0 or end <= start or end > len(document.text):
            raise ValueError("replacement range is outside normalized HTML text")
        unique.append(item)
    ordered = sorted(unique, key=lambda item: (item[0], item[1], item[2]))
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            raise ValueError("normalized replacement ranges overlap")
    return tuple(ordered)


def _project_replacement(
    document: ExtractedDocument, start: int, end: int
) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for span in document.spans:
        overlap_start = max(start, span.start)
        overlap_end = min(end, span.end)
        if overlap_start >= overlap_end or not span.metadata.get("replaceable", False):
            continue
        source_start = int(span.metadata["source_start"])
        source_end = int(span.metadata["source_end"])
        if span.metadata.get("source_map_mode") == "linear":
            source_start += overlap_start - span.start
            source_end = source_start + (overlap_end - overlap_start)
        candidate = (source_start, source_end)
        if candidate not in ranges:
            ranges.append(candidate)
    if not ranges:
        raise ValueError("replacement range contains no replaceable HTML text")
    return tuple(sorted(ranges))


def _validate_projected_replacements(
    projected: Sequence[tuple[tuple[tuple[int, int], ...], str]],
) -> None:
    owned: list[tuple[int, int, int]] = []
    for request_index, (ranges, _) in enumerate(projected):
        for start, end in ranges:
            for other_start, other_end, other_request in owned:
                if (
                    request_index != other_request
                    and start < other_end
                    and other_start < end
                ):
                    raise ValueError("projected HTML replacement ranges overlap")
            owned.append((start, end, request_index))


def _render_replacements(
    source: str,
    projected: Sequence[tuple[tuple[tuple[int, int], ...], str]],
) -> str:
    edits: list[tuple[int, int, str]] = []
    for ranges, replacement in projected:
        escaped = html_lib.escape(replacement)
        for index, (start, end) in enumerate(ranges):
            edits.append((start, end, escaped if index == 0 else ""))
    redacted = source
    for start, end, replacement in sorted(
        edits, key=lambda item: item[0], reverse=True
    ):
        redacted = redacted[:start] + replacement + redacted[end:]
    return redacted


def _detect_entities(document: ExtractedDocument, models: Any, lang: str | None) -> Any:
    detector = _resolve_detector(models)
    if detector is None:
        return ()
    if _accepts_lang_keyword(detector, document.text, lang):
        return detector(document.text, lang=lang)
    return detector(document.text)


def _accepts_lang_keyword(detector: Any, text: str, lang: str | None) -> bool:
    try:
        signature = inspect.signature(detector)
    except (TypeError, ValueError):
        return True
    try:
        signature.bind(text, lang=lang)
    except TypeError:
        return False
    return True


def _resolve_detector(models: Any) -> Any:
    if models is None:
        return None
    if callable(models):
        return models
    if isinstance(models, Mapping):
        for key in ("detector", "extract_pii", "analyze_text", "predict_entities"):
            candidate = models.get(key)
            if callable(candidate):
                return candidate
        return None
    for name in (
        "detect",
        "extract_pii",
        "analyze_text",
        "predict_entities",
        "predict",
    ):
        candidate = getattr(models, name, None)
        if callable(candidate):
            return candidate
    return None


def _iter_entity_inputs(spans: Any) -> tuple[Any, ...]:
    if spans is None:
        return ()
    for name in ("entities", "pii_entities"):
        entities = getattr(spans, name, None)
        if entities is not None:
            return tuple(entities)
    if isinstance(spans, Mapping):
        for key in ("entities", "pii_entities", "spans"):
            entities = spans.get(key)
            if entities is not None:
                return tuple(entities)
        if "start" in spans and "end" in spans:
            return (spans,)
    if _looks_like_sequence_entity(spans):
        return (spans,)
    if isinstance(spans, Iterable) and not isinstance(spans, (str, bytes, bytearray)):
        return tuple(spans)
    return ()


def _coerce_entity(
    span: Any, *, default_replacement: str | None
) -> tuple[int, int, str] | None:
    if _looks_like_sequence_entity(span):
        label = str(span[2]) if len(span) >= 3 and span[2] is not None else None
        return int(span[0]), int(span[1]), default_replacement or _mask_for_label(label)
    if isinstance(span, Mapping):
        start = span.get("start")
        end = span.get("end")
        label = span.get("label", span.get("entity_type", span.get("entity_group")))
        replacement = span.get("replacement", span.get("redacted_text"))
    else:
        start = getattr(span, "start", None)
        end = getattr(span, "end", None)
        label = getattr(
            span,
            "label",
            getattr(span, "entity_type", getattr(span, "entity_group", None)),
        )
        replacement = getattr(span, "replacement", getattr(span, "redacted_text", None))
    if start is None or end is None:
        return None
    return (
        int(start),
        int(end),
        str(replacement)
        if replacement is not None
        else default_replacement or _mask_for_label(label),
    )


def _looks_like_sequence_entity(value: Any) -> bool:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) >= 2:
            try:
                int(value[0])
                int(value[1])
            except (TypeError, ValueError):
                return False
            return True
    return False


def _mask_for_label(label: Any) -> str:
    safe = "".join(
        character if character.isalnum() else "_"
        for character in str(label or "PHI").upper()
    ).strip("_")
    return f"[{safe or 'PHI'}]"


def _policy_value(policy: Any, *names: str) -> Any:
    if policy is None:
        return None
    if isinstance(policy, Mapping):
        for name in names:
            if name in policy:
                return policy[name]
        return None
    for name in names:
        value = getattr(policy, name, None)
        if value is not None:
            return value
    return None


def _html_handler(
    path: str | Path,
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    parsed = _parse_source(path)
    entity_inputs = _iter_entity_inputs(_detect_entities(parsed.document, models, lang))
    default_replacement = _policy_value(policy, "replacement")
    replacements = tuple(
        replacement
        for entity in entity_inputs
        if (
            replacement := _coerce_entity(
                entity, default_replacement=default_replacement
            )
        )
        is not None
    )
    output_path = _policy_value(
        policy, "output_path", "redacted_path", "destination_path"
    )
    if replacements and output_path is not None:
        output = Path(output_path)
        _validate_distinct_paths(Path(path), output)
        _write_parsed_html(parsed, output, replacements)
    metadata = {
        "format": "html",
        "detected_span_count": len(replacements),
    }
    source_path = parsed.document.metadata.get("source_path")
    if source_path is not None:
        metadata["source_path"] = source_path
    if replacements and output_path is not None:
        metadata["redacted_html_path"] = str(output_path)
    return ExtractedDocument(
        text=parsed.document.text,
        spans=parsed.document.spans,
        metadata=metadata,
    )


register_handler((".html", ".htm"), _html_handler, requires_multimodal=False)


__all__ = ["extract_html", "write_redacted_html"]
