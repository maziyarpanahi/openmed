"""RTF text extraction with source character offsets.

The ingester uses only the Python standard library so RTF dispatch remains
available without pulling extra dependencies into the multimodal install path.
It walks the RTF control stream, skips non-content destination groups, decodes
control words, control symbols, and Unicode escapes, and records offsets back
into the original document.

RTF keeps its control stream in 7-bit ASCII and encodes everything else through
escapes, so the source is read as a single-byte stream: a ``source_start`` /
``source_end`` pair is both a character offset into the decoded source and a
byte offset into the file on disk.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .base import ExtractedDocument, SourceSpan, register_handler
from .exceptions import UnsupportedDocumentError

# Latin-1 round-trips every byte to the code point of the same value, which
# keeps source offsets identical to byte offsets in the file.
_SOURCE_ENCODING = "latin-1"
_DEFAULT_CODEPAGE = "cp1252"
_HEADER_PREFIX = "{\\rtf"
_HEADER_PADDING = "\ufeff \t\r\n"

# Characters that carry no document text outside of an escape sequence.
_IGNORED_SOURCE_CHARS = frozenset("\r\n\x00")
_RUN_DELIMITERS = frozenset("{}\\") | _IGNORED_SOURCE_CHARS

# Control-word parameters are bounded so a malformed document cannot force an
# unbounded integer conversion.
_MAX_PARAMETER_DIGITS = 10
_PARAMETER_LIMIT = 10**_MAX_PARAMETER_DIGITS

_CHARSET_CODEPAGES = {
    "ansi": "cp1252",
    "mac": "mac_roman",
    "pc": "cp437",
    "pca": "cp850",
}

# Destination groups whose contents are markup, metadata, or binary payloads
# rather than document text. Unknown destinations marked with ``\*`` are skipped
# too, which covers the long tail of writer-specific extensions.
_SKIPPED_DESTINATIONS = frozenset(
    {
        "annotation",
        "atnauthor",
        "atndate",
        "atnid",
        "atnparent",
        "atnref",
        "author",
        "background",
        "bkmkend",
        "bkmkstart",
        "buptim",
        "category",
        "colorschememapping",
        "colortbl",
        "comment",
        "company",
        "creatim",
        "datastore",
        "doccomm",
        "docvar",
        "falt",
        "fchars",
        "filetbl",
        "fldinst",
        "fname",
        "fontemb",
        "fontfile",
        "fonttbl",
        "footer",
        "footerf",
        "footerl",
        "footerr",
        "footnote",
        "generator",
        "header",
        "headerf",
        "headerl",
        "headerr",
        "hlinkbase",
        "htmltag",
        "info",
        "keywords",
        "latentstyles",
        "listtable",
        "listoverridetable",
        "listtext",
        "manager",
        "mhtmltag",
        "nonshppict",
        "objclass",
        "objdata",
        "objname",
        "operator",
        "panose",
        "password",
        "passwordhash",
        "pict",
        "pntext",
        "pntxta",
        "pntxtb",
        "printim",
        "private",
        "revtbl",
        "revtim",
        "rsidtbl",
        "shppict",
        "stylesheet",
        "subject",
        "themedata",
        "title",
        "upr",
        "userprops",
        "xe",
        "xmlnstbl",
    }
)

# Control words that end a text block. The emitted newline is structural, so it
# is not mapped back to a source range.
_BREAK_CONTROL_WORDS = frozenset(
    {
        "column",
        "line",
        "nestrow",
        "page",
        "par",
        "row",
        "sect",
        "softline",
    }
)

# Control words that stand in for a literal character.
_TEXT_CONTROL_WORDS = {
    "bullet": "\u2022",
    "cell": "\t",
    "emdash": "\u2014",
    "emspace": "\u2003",
    "endash": "\u2013",
    "enspace": "\u2002",
    "ldblquote": "\u201c",
    "lquote": "\u2018",
    "nestcell": "\t",
    "qmspace": "\u2005",
    "rdblquote": "\u201d",
    "rquote": "\u2019",
    "tab": "\t",
}

# An escaped line break is a paragraph break, the same as ``\par``.
_BREAK_CONTROL_SYMBOLS = frozenset("\r\n")

# Control symbols that stand in for a literal character. ``\-`` marks an
# optional hyphen, which contributes no visible text.
_TEXT_CONTROL_SYMBOLS = {
    "\\": "\\",
    "{": "{",
    "}": "}",
    "_": "\u2011",
    "~": "\u00a0",
}

_HIGH_SURROGATE_RANGE = range(0xD800, 0xDC00)
_LOW_SURROGATE_RANGE = range(0xDC00, 0xE000)


@dataclass(frozen=True)
class _Control:
    """A parsed control word or control symbol and the offset after it."""

    name: str
    parameter: int | None
    end: int
    is_word: bool


@dataclass(frozen=True)
class _GroupState:
    """Group-scoped RTF state inherited by nested groups."""

    ignore: bool = False
    unicode_skip: int = 1


def extract_rtf(path: str | Path) -> ExtractedDocument:
    """Extract RTF body text and source-offset metadata.

    Args:
        path: RTF file path.

    Returns:
        An :class:`ExtractedDocument` whose text holds the document body in
        reading order. Each mapped span carries ``source_start`` and
        ``source_end`` offsets into the RTF source.

    Raises:
        UnsupportedDocumentError: If the file does not start with an ``{\\rtf``
            header or contains no extractable body text.
    """
    source_path = Path(path)
    source = source_path.read_bytes().decode(_SOURCE_ENCODING)
    _ensure_rtf_header(source)
    return _RtfExtractor(source).document(source_path)


def _ensure_rtf_header(source: str) -> None:
    if not source.lstrip(_HEADER_PADDING).startswith(_HEADER_PREFIX):
        raise UnsupportedDocumentError("RTF documents must start with an {\\rtf header")


class _RtfExtractor:
    """Single-pass RTF reader producing normalized text and source spans."""

    def __init__(self, source: str) -> None:
        self._source = source
        self._length = len(source)
        self._parts: list[str] = []
        self._spans: list[SourceSpan] = []
        self._cursor = 0
        self._state = _GroupState()
        self._stack: list[_GroupState] = []
        self._codepage = _DEFAULT_CODEPAGE
        self._rtf_version: int | None = None
        self._pending_surrogate: tuple[int, int] | None = None

    def document(self, source_path: Path) -> ExtractedDocument:
        """Parse the source and assemble the extracted document."""
        self._parse()
        self._drop_pending_surrogate()
        text = _strip_trailing_breaks("".join(self._parts), self._mapped_end())
        if not text:
            raise UnsupportedDocumentError(
                "RTF document does not contain extractable text"
            )
        return ExtractedDocument(
            text=text,
            spans=tuple(self._spans),
            metadata={
                "format": "rtf",
                "source_path": str(source_path),
                "encoding": self._codepage,
                "rtf_version": self._rtf_version,
            },
        )

    def _mapped_end(self) -> int:
        return self._spans[-1].end if self._spans else 0

    def _parse(self) -> None:
        index = 0
        while index < self._length:
            character = self._source[index]
            if character == "{":
                self._stack.append(self._state)
                self._state = replace(self._state)
                index += 1
            elif character == "}":
                if self._stack:
                    self._state = self._stack.pop()
                index += 1
            elif character == "\\":
                index = self._handle_escape(index)
            elif character in _IGNORED_SOURCE_CHARS:
                index += 1
            else:
                index = self._handle_literal_run(index)

    def _handle_literal_run(self, index: int) -> int:
        end = index
        while end < self._length and self._source[end] not in _RUN_DELIMITERS:
            end += 1
        if not self._state.ignore:
            self._drop_pending_surrogate()
            self._append_mapped(self._decode(self._source[index:end]), index, end)
        return end

    def _handle_escape(self, index: int) -> int:
        if index + 1 < self._length and self._source[index + 1] == "'":
            return self._handle_hex_escapes(index)

        control = self._read_control(index)
        if control is None:
            return self._length
        if control.is_word:
            return self._handle_control_word(control, index)
        return self._handle_control_symbol(control, index)

    def _handle_control_word(self, control: _Control, index: int) -> int:
        name = control.name
        if name == "rtf":
            self._rtf_version = control.parameter
            return control.end
        if name == "ansicpg":
            self._codepage = _resolve_codepage(control.parameter)
            return control.end
        if name in _CHARSET_CODEPAGES:
            self._codepage = _CHARSET_CODEPAGES[name]
            return control.end
        if name == "uc":
            if control.parameter is not None and control.parameter >= 0:
                self._state = replace(self._state, unicode_skip=control.parameter)
            return control.end
        if name == "bin":
            return self._skip_binary(control)
        if name in _SKIPPED_DESTINATIONS:
            self._state = replace(self._state, ignore=True)
            return control.end
        if name == "u":
            return self._handle_unicode_escape(control, index)

        self._drop_pending_surrogate()
        if self._state.ignore:
            return control.end
        if name in _BREAK_CONTROL_WORDS:
            self._append_break()
        elif name in _TEXT_CONTROL_WORDS:
            self._append_mapped(_TEXT_CONTROL_WORDS[name], index, control.end)
        return control.end

    def _handle_control_symbol(self, control: _Control, index: int) -> int:
        if control.name == "*":
            self._state = replace(self._state, ignore=True)
            return control.end
        self._drop_pending_surrogate()
        if self._state.ignore:
            return control.end
        if control.name in _BREAK_CONTROL_SYMBOLS:
            self._append_break()
        elif control.name in _TEXT_CONTROL_SYMBOLS:
            self._append_mapped(
                _TEXT_CONTROL_SYMBOLS[control.name],
                index,
                control.end,
            )
        return control.end

    def _handle_hex_escapes(self, index: int) -> int:
        """Decode a run of adjacent ``\\'hh`` escapes as one encoded string."""
        self._drop_pending_surrogate()
        payload = bytearray()
        end = index
        while (
            end + 3 < self._length
            and self._source[end] == "\\"
            and self._source[end + 1] == "'"
        ):
            digits = self._source[end + 2 : end + 4]
            if not _is_hex_pair(digits):
                break
            payload.append(int(digits, 16))
            end += 4
        if not payload:
            # A malformed escape contributes no text; skip the ``\'`` marker.
            return min(index + 2, self._length)
        if not self._state.ignore:
            self._append_mapped(
                bytes(payload).decode(self._codepage, errors="replace"),
                index,
                end,
            )
        return end

    def _handle_unicode_escape(self, control: _Control, index: int) -> int:
        end = self._skip_unicode_alternates(control.end)
        value = control.parameter
        if value is None:
            self._drop_pending_surrogate()
            return end
        if value < 0:
            value += 0x10000

        pending = self._pending_surrogate
        if pending is not None and value in _LOW_SURROGATE_RANGE:
            high, source_start = pending
            self._pending_surrogate = None
            code_point = 0x10000 + ((high - 0xD800) << 10) + (value - 0xDC00)
            if not self._state.ignore:
                self._append_mapped(chr(code_point), source_start, end)
            return end

        self._drop_pending_surrogate()
        if value in _HIGH_SURROGATE_RANGE:
            self._pending_surrogate = (value, index)
        elif 0 <= value <= 0x10FFFF and value not in _LOW_SURROGATE_RANGE:
            if not self._state.ignore:
                self._append_mapped(chr(value), index, end)
        return end

    def _skip_unicode_alternates(self, index: int) -> int:
        """Skip the fallback characters that follow a ``\\u`` escape."""
        remaining = self._state.unicode_skip
        cursor = index
        while remaining > 0 and cursor < self._length:
            character = self._source[cursor]
            if character in "{}":
                break
            if character in _IGNORED_SOURCE_CHARS:
                cursor += 1
                continue
            if character == "\\":
                if (
                    cursor + 3 < self._length
                    and self._source[cursor + 1] == "'"
                    and _is_hex_pair(self._source[cursor + 2 : cursor + 4])
                ):
                    cursor += 4
                else:
                    control = self._read_control(cursor)
                    if control is None:
                        return self._length
                    cursor = control.end
            else:
                cursor += 1
            remaining -= 1
        return cursor

    def _skip_binary(self, control: _Control) -> int:
        length = control.parameter or 0
        if length <= 0:
            return control.end
        return min(control.end + length, self._length)

    def _read_control(self, index: int) -> _Control | None:
        """Parse the control word or symbol starting at the backslash."""
        cursor = index + 1
        if cursor >= self._length:
            return None
        if not _is_ascii_letter(self._source[cursor]):
            return _Control(
                name=self._source[cursor],
                parameter=None,
                end=cursor + 1,
                is_word=False,
            )

        start = cursor
        while cursor < self._length and _is_ascii_letter(self._source[cursor]):
            cursor += 1
        name = self._source[start:cursor]

        parameter = None
        digits_start = cursor
        if cursor < self._length and self._source[cursor] == "-":
            cursor += 1
        while cursor < self._length and _is_ascii_digit(self._source[cursor]):
            cursor += 1
        raw_parameter = self._source[digits_start:cursor]
        if raw_parameter in {"", "-"}:
            cursor = digits_start
        else:
            parameter = _bounded_int(raw_parameter)

        if cursor < self._length and self._source[cursor] == " ":
            cursor += 1
        return _Control(name=name, parameter=parameter, end=cursor, is_word=True)

    def _decode(self, raw: str) -> str:
        if raw.isascii():
            return raw
        return raw.encode(_SOURCE_ENCODING).decode(self._codepage, errors="replace")

    def _drop_pending_surrogate(self) -> None:
        """Discard an unpaired high surrogate; it carries no text on its own."""
        self._pending_surrogate = None

    def _append_break(self) -> None:
        if not self._parts or self._parts[-1].endswith("\n"):
            return
        self._parts.append("\n")
        self._cursor += 1

    def _append_mapped(self, text: str, source_start: int, source_end: int) -> None:
        if not text or source_start >= source_end:
            return
        start = self._cursor
        self._parts.append(text)
        self._cursor += len(text)
        self._spans.append(
            SourceSpan(
                start=start,
                end=self._cursor,
                metadata={
                    "format": "rtf",
                    "source_start": source_start,
                    "source_end": source_end,
                },
            )
        )


def _strip_trailing_breaks(text: str, mapped_end: int) -> str:
    """Drop unmapped trailing newlines emitted by document-final breaks."""
    end = len(text)
    while end > mapped_end and text[end - 1] == "\n":
        end -= 1
    return text[:end]


def _resolve_codepage(parameter: int | None) -> str:
    if parameter is None or parameter <= 0:
        return _DEFAULT_CODEPAGE
    try:
        return codecs.lookup(f"cp{parameter}").name
    except LookupError:
        return _DEFAULT_CODEPAGE


def _bounded_int(raw: str) -> int:
    sign = -1 if raw.startswith("-") else 1
    digits = raw.lstrip("-").lstrip("0")
    if not digits:
        return 0
    if len(digits) > _MAX_PARAMETER_DIGITS:
        return sign * _PARAMETER_LIMIT
    return sign * int(digits)


def _is_ascii_letter(character: str) -> bool:
    return "a" <= character <= "z" or "A" <= character <= "Z"


def _is_ascii_digit(character: str) -> bool:
    return "0" <= character <= "9"


def _is_hex_pair(digits: str) -> bool:
    return len(digits) == 2 and all(
        _is_ascii_digit(digit) or "a" <= digit.lower() <= "f" for digit in digits
    )


def _rtf_handler(
    path: str | Path,
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    return extract_rtf(path)


register_handler(".rtf", _rtf_handler, requires_multimodal=False)


__all__ = ["extract_rtf"]
