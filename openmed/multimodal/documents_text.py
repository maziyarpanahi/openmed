"""Fixed-width plaintext extraction and layout-preserving write-back."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ._text_redaction import (
    TextReplacement,
    detect_replacements,
    policy_value,
    validate_distinct_paths,
    validate_replacements,
)
from .base import ExtractedDocument, SourceSpan, register_handler


def extract_text(
    path: str | Path,
    *,
    encoding: str = "utf-8",
) -> ExtractedDocument:
    """Read plaintext without newline normalization and record line geometry."""
    source_path = Path(path)
    with source_path.open("r", encoding=encoding, newline="") as handle:
        text = handle.read()

    spans: list[SourceSpan] = []
    lines: list[dict[str, Any]] = []
    cursor = 0
    for line_index, raw_line in enumerate(text.splitlines(keepends=True)):
        newline = _line_ending(raw_line)
        content_end = cursor + len(raw_line) - len(newline)
        end = cursor + len(raw_line)
        lines.append(
            {
                "line": line_index,
                "start": cursor,
                "content_end": content_end,
                "end": end,
                "columns": content_end - cursor,
                "newline": newline,
            }
        )
        if cursor < end:
            spans.append(
                SourceSpan(
                    start=cursor,
                    end=end,
                    metadata={
                        "format": "text",
                        "line": line_index,
                        "source_start": cursor,
                        "source_end": end,
                        "content_end": content_end,
                        "source_map_mode": "linear",
                    },
                )
            )
        cursor = end

    return ExtractedDocument(
        text=text,
        spans=tuple(spans),
        metadata={
            "format": "text",
            "source_path": str(source_path),
            "encoding": encoding,
            "line_count": len(lines),
            "max_columns": max((line["columns"] for line in lines), default=0),
            "lines": tuple(lines),
        },
    )


def write_redacted_text(
    source_path: str | Path,
    output_path: str | Path,
    replacements: Iterable[TextReplacement],
    *,
    encoding: str = "utf-8",
    preserve_columns: bool = True,
) -> Path:
    """Write redacted plaintext while retaining fixed-width column positions.

    When ``preserve_columns`` is true, replacement text is padded or truncated
    to the original span width. Replacements cannot cross a line ending.
    """
    source = Path(source_path)
    output = Path(output_path)
    validate_distinct_paths(source, output)
    document = extract_text(source, encoding=encoding)
    logical = validate_replacements(document, replacements)

    edits: list[TextReplacement] = []
    for start, end, replacement in logical:
        original = document.text[start:end]
        if any(character in original for character in "\r\n"):
            raise ValueError("plaintext replacement ranges cannot cross line endings")
        if any(character in replacement for character in "\r\n"):
            raise ValueError("plaintext replacements cannot contain line endings")
        if preserve_columns:
            width = end - start
            replacement = replacement[:width].ljust(width)
        edits.append((start, end, replacement))

    redacted = document.text
    for start, end, replacement in reversed(edits):
        redacted = redacted[:start] + replacement + redacted[end:]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding=encoding, newline="") as handle:
        handle.write(redacted)
    return output


def _line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith(("\r", "\n")):
        return line[-1]
    return ""


def _text_handler(
    path: str | Path,
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    encoding = str(policy_value(policy, "encoding") or "utf-8")
    document = extract_text(path, encoding=encoding)
    replacements = detect_replacements(document, models, lang, policy)
    output_path = policy_value(
        policy,
        "output_path",
        "redacted_path",
        "destination_path",
    )
    metadata = dict(document.metadata)
    metadata["detected_span_count"] = len(replacements)
    if replacements and output_path is not None:
        preserve_columns = policy_value(policy, "preserve_columns")
        write_redacted_text(
            path,
            output_path,
            replacements,
            encoding=encoding,
            preserve_columns=preserve_columns is not False,
        )
        metadata["redacted_text_path"] = str(output_path)
    return ExtractedDocument(
        text=document.text,
        spans=document.spans,
        metadata=metadata,
    )


register_handler(".txt", _text_handler, requires_multimodal=False)


__all__ = ["extract_text", "write_redacted_text"]
