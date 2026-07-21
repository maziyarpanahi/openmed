"""RTF text extraction with char-offset source spans for multimodal redaction.

The RTF ingester parses Rich Text Format documents produced by legacy clinical
and dictation systems.  It strips control words, groups, and unicode escapes
while preserving the reading order of visible text.  Each text fragment becomes
a :class:`SourceSpan` whose metadata records the byte offset range in the
original RTF source so that detected PII spans can be projected back for
write-back or audit.

This module is stdlib-only (no third-party dependencies) because RTF is a
plain-text format that can be parsed with regular expressions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import ExtractedDocument, SourceSpan, register_handler

_RTF_EXTENSION = ".rtf"

# --- RTF control word patterns ------------------------------------------------

# Control word: backslash + 1-32 alpha chars, optional numeric param, optional space.
_CONTROL_WORD_RE = re.compile(
    r"\\([a-zA-Z]{1,32})(-?\d+)?[ ]?"
)

# Escaped special characters: \\ \{ \}
_ESCAPED_CHAR_RE = re.compile(r"\\([-\\{}])")

# Unicode escape: \uN followed by optional fallback char (usually '?').
_UNICODE_RE = re.compile(r"\\u(\-?\d+)\??")

# Hex escape: \'XX
_HEX_RE = re.compile(r"\\'([0-9a-fA-F]{2})")

# Destination control words whose groups should be skipped entirely.
_SKIP_KEYWORDS = frozenset({
    "fonttbl", "colortbl", "stylesheet", "info", "pict", "object",
    "header", "footer", "headerf", "footerf", "footnote", "annotation",
    "comment", "atnauthor", "atnid", "revtbl", "rsidtbl", "latentstyles",
    "panose", "mmathPr", "generator", "template",
    "listtable", "listoverridetable",
})


@dataclass
class _RtfParseState:
    """Mutable state for a single RTF extraction pass."""

    parts: list[str] = field(default_factory=list)
    spans: list[SourceSpan] = field(default_factory=list)
    cursor: int = 0

    group_depth: int = 0
    skipping: bool = False
    skip_depth: int = 0

    def append_text(self, text: str, src_start: int, src_end: int) -> None:
        """Append visible text and record its source span."""
        if not text:
            return
        start = self.cursor
        self.parts.append(text)
        self.cursor += len(text)
        self.spans.append(
            SourceSpan(
                start=start,
                end=start + len(text),
                page=0,
                bbox=None,
                metadata={
                    "format": "rtf",
                    "src_byte_start": src_start,
                    "src_byte_end": src_end,
                },
            )
        )


def _parse_rtf(source: str) -> _RtfParseState:
    """Parse RTF source and extract visible text with char-offset spans."""
    state = _RtfParseState()
    pos = 0
    length = len(source)

    while pos < length:
        ch = source[pos]

        if ch == "{":
            state.group_depth += 1
            pos += 1
            continue

        if ch == "}":
            if state.skipping and state.group_depth <= state.skip_depth:
                state.skipping = False
            state.group_depth = max(0, state.group_depth - 1)
            pos += 1
            continue

        if ch == "\\":
            # Escaped characters: \\ \{ \}
            esc_match = _ESCAPED_CHAR_RE.match(source, pos)
            if esc_match:
                if not state.skipping:
                    state.append_text(esc_match.group(1), pos, esc_match.end())
                pos = esc_match.end()
                continue

            # Hex escape: \'XX
            hex_match = _HEX_RE.match(source, pos)
            if hex_match:
                if not state.skipping:
                    char_code = int(hex_match.group(1), 16)
                    state.append_text(chr(char_code), pos, hex_match.end())
                pos = hex_match.end()
                continue

            # Unicode escape: \uN?
            uni_match = _UNICODE_RE.match(source, pos)
            if uni_match:
                if not state.skipping:
                    code_point = int(uni_match.group(1))
                    if code_point < 0:
                        code_point += 65536
                    try:
                        char = chr(code_point)
                    except (ValueError, OverflowError):
                        char = "?"
                    state.append_text(char, pos, uni_match.end())
                pos = uni_match.end()
                continue

            # Control word
            cw_match = _CONTROL_WORD_RE.match(source, pos)
            if cw_match:
                keyword = cw_match.group(1)
                cw_end = cw_match.end()

                if keyword in _SKIP_KEYWORDS:
                    state.skipping = True
                    state.skip_depth = state.group_depth

                if not state.skipping:
                    if keyword in ("par", "pard", "line"):
                        state.append_text("\n", pos, cw_end)
                    elif keyword in ("tab", "cell"):
                        state.append_text("\t", pos, cw_end)
                    elif keyword == "bullet":
                        state.append_text("\u2022", pos, cw_end)
                    elif keyword == "emdash":
                        state.append_text("\u2014", pos, cw_end)
                    elif keyword == "endash":
                        state.append_text("\u2013", pos, cw_end)
                    elif keyword == "lquote":
                        state.append_text("\u2018", pos, cw_end)
                    elif keyword == "rquote":
                        state.append_text("\u2019", pos, cw_end)
                    elif keyword == "ldblquote":
                        state.append_text("\u201c", pos, cw_end)
                    elif keyword == "rdblquote":
                        state.append_text("\u201d", pos, cw_end)

                pos = cw_end
                continue

            # Lone backslash — skip
            pos += 1
            continue

        # Literal text character
        if not state.skipping:
            state.append_text(ch, pos, pos + 1)
        pos += 1

    return state


def _clean_text(text: str) -> str:
    """Normalize extracted text: collapse whitespace, fix newlines."""
    # Collapse runs of spaces/tabs (but not newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip each line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def extract_rtf(path: str | Path) -> ExtractedDocument:
    r"""Extract normalized RTF text plus char-offset source spans.

    Each text fragment in the RTF becomes a :class:`SourceSpan` whose metadata
    includes the byte range in the original RTF source.  Paragraph boundaries
    (``\par``) are mapped to newline characters.

    Args:
        path: Path to an RTF file.

    Returns:
        An :class:`ExtractedDocument` with normalized text and source spans.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not appear to be valid RTF.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RTF file not found: {path}")

    source = path.read_text(encoding="utf-8", errors="replace")

    # Basic RTF validation
    stripped = source.lstrip()
    if not stripped.startswith("{\\rtf"):
        raise ValueError(
            "File does not appear to be valid RTF (missing {\\rtf header}): " + str(path)
        )

    state = _parse_rtf(source)
    raw_text = "".join(state.parts)
    cleaned = _clean_text(raw_text)

    return ExtractedDocument(
        text=cleaned,
        spans=tuple(state.spans),
        metadata={
            "format": "rtf",
            "source_path": str(path),
            "source_size_bytes": len(source.encode("utf-8")),
            "span_count": len(state.spans),
        },
    )


def _handle_rtf(path: str | Path, **kwargs: Any) -> ExtractedDocument:
    """Handler registered with the multimodal dispatch system."""
    return extract_rtf(path)


# Register the RTF handler at import time.
register_handler(_RTF_EXTENSION, _handle_rtf, requires_multimodal=False)
