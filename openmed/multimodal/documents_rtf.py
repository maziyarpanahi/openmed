"""RTF text extraction with source character offsets.

The ingester uses only the Python standard library so RTF dispatch remains
available without pulling extra dependencies into the multimodal install path
(RTF is a plain-ASCII control-word format, so it does not need a binary
parser the way DOCX or PDF do).

The source file is decoded with ``latin-1`` rather than UTF-8. RTF's own
syntax (braces, control words, control symbols) is always plain ASCII;
anything outside that range is represented through escapes (``\\'hh`` hex
escapes or ``\\uN`` Unicode escapes), never as a raw multi-byte sequence. A
``latin-1`` decode is a lossless, one-byte-to-one-character mapping, so every
character offset produced while scanning is also the exact byte offset in the
original file -- which is what a redaction step needs to locate the matching
bytes to blank out or rewrite.

The scanner tracks group (``{`` / ``}``) nesting so that non-body
destinations -- font/color/style/list tables, document metadata, headers,
footers, annotations, and embedded picture/object payloads -- are walked (to
keep brace counting correct) but never contribute to the extracted text. It
also honors the generic ``\\*`` "ignorable destination" marker so unrecognized
vendor destinations are skipped safely rather than leaking raw control words
into the output, and it consumes ``\\binN`` raw binary payloads verbatim so
that arbitrary bytes embedded in an OLE object cannot be misread as RTF
syntax and desynchronize brace counting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import ExtractedDocument, SourceSpan, register_handler
from .exceptions import UnsupportedDocumentError

_RTF_EXTENSION = ".rtf"

# Destinations whose content is metadata, formatting definitions, or a
# non-body section rather than document text. A child group inherits its
# parent's skip state (see ``_Group``), so destinations that only ever appear
# nested inside one of these (e.g. "listlevel" under "listtable", or "author"
# under "info") do not need to be listed separately. "fldinst" is listed
# explicitly as a defense-in-depth measure: conformant writers also mark it
# ignorable with ``\*``, but not every RTF producer bothers.
_SKIP_DESTINATIONS = frozenset(
    {
        "fonttbl",
        "colortbl",
        "stylesheet",
        "listtable",
        "listoverridetable",
        "revtbl",
        "rsidtbl",
        "latentstyles",
        "panose",
        "generator",
        "info",
        "pict",
        "objdata",
        "objclass",
        "themedata",
        "colorschememapping",
        "datastore",
        "xmlnstbl",
        "filetbl",
        "template",
        "fldinst",
        "header",
        "headerf",
        "footer",
        "footerf",
        "footnote",
        "annotation",
        "atnauthor",
        "atnid",
        "atnref",
        "atntime",
        "comment",
    }
)

# Control words that stand in for a single visible character. "pard" (reset
# paragraph formatting) is deliberately absent: it commonly appears
# immediately after "par" at the start of the next paragraph, and mapping it
# to another newline would double up paragraph breaks in real documents.
_CHAR_CONTROL_WORDS = {
    "par": "\n",
    "line": "\n",
    "page": "\n",
    "sect": "\n",
    "row": "\n",
    "cell": "\t",
    "tab": "\t",
    "emdash": "—",
    "endash": "–",
    "emspace": " ",
    "enspace": " ",
    "qmspace": " ",
    "bullet": "•",
    "lquote": "‘",
    "rquote": "’",
    "ldblquote": "“",
    "rdblquote": "”",
}

# \ansicpgN -> Python codec name, for decoding \'hh hex escapes. Falls back to
# cp1252 (the default Windows "ANSI" code page and the overwhelmingly common
# case for clinical/dictation RTF) when the declared page is missing or
# unrecognized.
_CODEPAGE_CODECS = {
    437: "cp437",
    850: "cp850",
    852: "cp852",
    860: "cp860",
    863: "cp863",
    865: "cp865",
    874: "cp874",
    932: "cp932",
    936: "cp936",
    949: "cp949",
    950: "cp950",
    1250: "cp1250",
    1251: "cp1251",
    1252: "cp1252",
    1253: "cp1253",
    1254: "cp1254",
    1255: "cp1255",
    1256: "cp1256",
    1257: "cp1257",
    1258: "cp1258",
    10000: "mac_roman",
    20127: "ascii",
    65001: "utf-8",
}

_DEFAULT_CODEC = "cp1252"


@dataclass
class _Group:
    """Per-group (brace-scoped) state, inherited by nested groups.

    ``uc`` is the number of "alternative representation" characters that
    follow each ``\\uN`` Unicode escape (set by ``\\ucN``); like other RTF
    formatting properties it is scoped to the enclosing group and defaults to
    1 until a document sets it otherwise.
    """

    skip: bool = False
    uc: int = 1


@dataclass
class _ScanResult:
    text: str
    spans: tuple[SourceSpan, ...]


def _scan_rtf(source: str) -> _ScanResult:
    parts: list[str] = []
    spans: list[SourceSpan] = []
    cursor = 0
    length = len(source)
    i = 0

    groups: list[_Group] = [_Group()]
    codec = _DEFAULT_CODEC
    just_opened = False
    pending_skip = 0
    pending_hex: list[int] = []
    pending_hex_start = -1

    # A `\uN` escape that decoded to a UTF-16 high surrogate (0xD800-0xDBFF)
    # is held here rather than emitted immediately: astral-plane characters
    # (code points above U+FFFF) are represented in RTF as a *pair* of `\uN`
    # escapes -- a high surrogate followed by a low surrogate -- each of
    # which only carries half of the real code point on its own. Emitting
    # each half independently via chr() would leak two lone (invalid)
    # surrogate code points into the output text, which later crashes any
    # UTF-8 encode of that text. `pending_surrogate` is resolved as soon as
    # the next `\uN` escape (after this one's own `\ucN` fallback run) is
    # seen: if it is the matching low surrogate, the two combine into a
    # single real character; otherwise -- or if any other real content or
    # end-of-document arrives first -- the held high surrogate is malformed
    # and is dropped in favor of a replacement character.
    pending_surrogate: int | None = None
    pending_surrogate_start = -1
    pending_surrogate_end = -1

    def flush_pending_surrogate() -> None:
        nonlocal pending_surrogate, pending_surrogate_start, pending_surrogate_end
        if pending_surrogate is not None:
            # Unpaired high surrogate: malformed/truncated input. Emit the
            # Unicode replacement character rather than the raw, invalid
            # lone surrogate -- the same "replace" treatment already used
            # for undecodable `\'hh` byte sequences below.
            emit("\ufffd", pending_surrogate_start, pending_surrogate_end)
            pending_surrogate = None
            pending_surrogate_start = -1
            pending_surrogate_end = -1

    def flush_hex(end: int) -> None:
        nonlocal cursor, pending_hex, pending_hex_start
        if not pending_hex:
            return
        raw = bytes(pending_hex)
        try:
            decoded = raw.decode(codec, errors="replace")
        except LookupError:
            decoded = raw.decode(_DEFAULT_CODEC, errors="replace")
        if decoded and not groups[-1].skip:
            start = cursor
            parts.append(decoded)
            cursor += len(decoded)
            spans.append(
                SourceSpan(
                    start=start,
                    end=cursor,
                    metadata={
                        "format": "rtf",
                        "source_start": pending_hex_start,
                        "source_end": end,
                    },
                )
            )
        pending_hex = []
        pending_hex_start = -1

    def emit(text: str, source_start: int, source_end: int) -> None:
        nonlocal cursor
        if not text or groups[-1].skip:
            return
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append(
            SourceSpan(
                start=start,
                end=cursor,
                metadata={
                    "format": "rtf",
                    "source_start": source_start,
                    "source_end": source_end,
                },
            )
        )

    while i < length:
        ch = source[i]

        if ch == "\\" and i + 1 < length and source[i + 1] == "'":
            hex_digits = source[i + 2 : i + 4]
            if len(hex_digits) == 2 and all(
                c in "0123456789abcdefABCDEF" for c in hex_digits
            ):
                end = i + 4
                if pending_skip > 0:
                    flush_hex(i)
                    pending_skip -= 1
                else:
                    if not pending_hex:
                        pending_hex_start = i
                        # A real (non-fallback) hex escape is new content
                        # that isn't the low-surrogate `\uN` we might be
                        # waiting on -- resolve any pending high surrogate
                        # now, before this run's own text is emitted.
                        flush_pending_surrogate()
                    pending_hex.append(int(hex_digits, 16))
                i = end
                just_opened = False
                continue

        # Every other token type breaks a run of consecutive hex escapes.
        flush_hex(i)

        if ch == "{":
            flush_pending_surrogate()
            groups.append(_Group(skip=groups[-1].skip, uc=groups[-1].uc))
            just_opened = True
            pending_skip = 0
            i += 1
            continue

        if ch == "}":
            flush_pending_surrogate()
            if len(groups) > 1:
                groups.pop()
            just_opened = False
            pending_skip = 0
            i += 1
            continue

        # A `\uN` escape (word == "u") is deferred to the control-word
        # branch below, which needs to inspect its decoded value before
        # deciding whether to flush -- it may be the low surrogate this
        # pending high surrogate is waiting for. Raw/escaped CR-LF is also
        # excluded: like `pending_skip`, it is pure source formatting with
        # no effect on token flow (see the branches below), so it must not
        # count as content breaking a pending surrogate pair either.
        _next = source[i + 1] if i + 1 < length else ""
        is_alpha_control = ch == "\\" and _next.isalpha()
        is_noop_newline = ch in "\r\n" or (ch == "\\" and _next in "\r\n")
        if pending_skip <= 0 and not is_alpha_control and not is_noop_newline:
            flush_pending_surrogate()

        if ch in "\r\n":
            # Raw CR/LF between tokens is source formatting only; real
            # paragraph breaks are the explicit \par control word.
            i += 1
            continue

        if ch != "\\":
            if pending_skip > 0:
                pending_skip -= 1
            else:
                emit(ch, i, i + 1)
            just_opened = False
            i += 1
            continue

        # ch == "\\" and not a (valid) hex escape.
        if i + 1 >= length:
            i += 1
            continue
        nxt = source[i + 1]

        if nxt in "{}\\":
            if pending_skip > 0:
                pending_skip -= 1
            else:
                emit(nxt, i, i + 2)
            i += 2
            just_opened = False
            continue

        if nxt == "~":
            if pending_skip > 0:
                pending_skip -= 1
            else:
                emit(" ", i, i + 2)
            i += 2
            just_opened = False
            continue

        if nxt == "-":
            # Optional hyphen: an invisible line-break candidate, not text.
            if pending_skip > 0:
                pending_skip -= 1
            i += 2
            just_opened = False
            continue

        if nxt == "_":
            if pending_skip > 0:
                pending_skip -= 1
            else:
                emit("-", i, i + 2)
            i += 2
            just_opened = False
            continue

        if nxt == "*":
            if just_opened:
                groups[-1].skip = True
            # just_opened stays True: the control word that follows is the
            # destination's identifying keyword, not the "first token" of a
            # different group.
            i += 2
            continue

        if nxt in "\r\n":
            i += 2
            just_opened = False
            continue

        if nxt.isalpha():
            j = i + 1
            while j < length and source[j].isalpha():
                j += 1
            word = source[i + 1 : j].lower()

            param: int | None = None
            if j < length and (source[j] == "-" or source[j].isdigit()):
                k = j + 1 if source[j] == "-" else j
                while k < length and source[k].isdigit():
                    k += 1
                if k > j and not (source[j] == "-" and k == j + 1):
                    param = int(source[j:k])
                    j = k
            if j < length and source[j] == " ":
                j += 1
            control_end = j

            # Any control word other than "u" itself is new real content
            # once it's not being swallowed as someone else's \ucN fallback
            # (pending_skip <= 0), so it can't be the low surrogate a
            # pending high surrogate is waiting for -- resolve it now. The
            # "u" branch below handles its own resolution, since it needs
            # the decoded value to decide whether this *is* the pairing.
            if word != "u" and pending_skip <= 0:
                flush_pending_surrogate()

            if word in _SKIP_DESTINATIONS:
                groups[-1].skip = True

            if word == "bin":
                skip_bytes = max(param or 0, 0)
                i = min(control_end + skip_bytes, length)
                just_opened = False
                pending_skip = 0
                continue

            if word == "uc":
                if param is not None:
                    groups[-1].uc = max(param, 0)
                i = control_end
                just_opened = False
                continue

            if word == "u":
                if param is not None:
                    codepoint = param
                    if codepoint < 0:
                        codepoint += 65536
                    codepoint &= 0xFFFF
                    if pending_skip <= 0:
                        if (
                            0xDC00 <= codepoint <= 0xDFFF
                            and pending_surrogate is not None
                        ):
                            # Matches the held high surrogate: combine the
                            # UTF-16 surrogate pair into the single astral
                            # code point it represents.
                            combined = (
                                0x10000
                                + (pending_surrogate - 0xD800) * 0x400
                                + (codepoint - 0xDC00)
                            )
                            emit(chr(combined), pending_surrogate_start, control_end)
                            pending_surrogate = None
                            pending_surrogate_start = -1
                            pending_surrogate_end = -1
                        else:
                            # Not a low surrogate pairing with what we were
                            # holding (if anything) -- resolve/drop that
                            # first, then handle this token on its own.
                            flush_pending_surrogate()
                            if 0xD800 <= codepoint <= 0xDBFF:
                                # High surrogate: hold it rather than emit,
                                # pending the next \uN escape being its
                                # matching low surrogate.
                                pending_surrogate = codepoint
                                pending_surrogate_start = i
                                pending_surrogate_end = control_end
                            elif 0xDC00 <= codepoint <= 0xDFFF:
                                # Orphan low surrogate, no preceding high
                                # surrogate: malformed on its own.
                                emit("\ufffd", i, control_end)
                            else:
                                emit(chr(codepoint), i, control_end)
                    pending_skip = groups[-1].uc
                elif pending_skip <= 0:
                    # Bare "\u" with no parameter: not a pairing candidate.
                    flush_pending_surrogate()
                i = control_end
                just_opened = False
                continue

            if word == "ansicpg" and param is not None:
                codec = _CODEPAGE_CODECS.get(param, _DEFAULT_CODEC)
                i = control_end
                just_opened = False
                continue

            if word == "ansi":
                codec = "cp1252"
            elif word == "mac":
                codec = "mac_roman"
            elif word == "pc":
                codec = "cp437"
            elif word == "pca":
                codec = "cp850"

            if pending_skip > 0:
                pending_skip -= 1
            elif word in _CHAR_CONTROL_WORDS:
                emit(_CHAR_CONTROL_WORDS[word], i, control_end)

            i = control_end
            just_opened = False
            continue

        # Unrecognized control symbol (e.g. "\:" or "\|"): consume, no text.
        if pending_skip > 0:
            pending_skip -= 1
        i += 2
        just_opened = False

    # End of document with a high surrogate still awaiting its low-surrogate
    # pair: truncated/malformed input, resolved the same way as any other
    # unpaired high surrogate.
    flush_pending_surrogate()
    flush_hex(length)
    return _ScanResult(text="".join(parts), spans=tuple(spans))


def extract_rtf(path: str | Path) -> ExtractedDocument:
    r"""Extract normalized RTF text plus char-offset source spans.

    Args:
        path: RTF file path.

    Returns:
        An :class:`ExtractedDocument` with normalized text and one
        :class:`SourceSpan` per emitted character run. Each span's metadata
        carries ``source_start``/``source_end``, the byte range in the
        original RTF file (the file is read with a lossless ``latin-1``
        decode, so these offsets are exact byte offsets, not just character
        offsets in some intermediate re-encoding).

    Raises:
        UnsupportedDocumentError: If the file is not a readable RTF document
            (missing the ``{\rtf`` header).
    """
    source_path = Path(path)
    raw = source_path.read_bytes()
    source = raw.decode("latin-1")

    if source.lstrip()[:5].lower() != r"{\rtf":
        raise UnsupportedDocumentError(
            f"{source_path} does not look like an RTF document "
            r'(missing "{\rtf" header)'
        )

    result = _scan_rtf(source)
    return ExtractedDocument(
        text=result.text,
        spans=result.spans,
        metadata={
            "format": "rtf",
            "source_path": str(source_path),
        },
    )


def _rtf_handler(
    path: str | Path,
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    return extract_rtf(path)


register_handler(_RTF_EXTENSION, _rtf_handler, requires_multimodal=False)


__all__ = ["extract_rtf"]
