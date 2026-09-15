"""Deterministic formatting normalization for local clinical summaries.

The normalizer changes presentation-only syntax: whitespace, Markdown-style
headings and list markers, and inline citation token placement. It does not
rewrite words or require a model, clock, filesystem, or network service.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final

SUMMARY_OUTPUT_NORMALIZATION_SCHEMA_VERSION: Final[int] = 1
SUMMARY_OUTPUT_NORMALIZATION_OPERATION_CODES: Final[tuple[str, ...]] = (
    "whitespace",
    "heading",
    "list_marker",
    "citation_placement",
)

_HORIZONTAL_SPACES_RE = re.compile(r" {2,}")
_SETEXT_HEADING_RE = re.compile(r"^(?P<marker>=|-)(?P=marker)+$")
_ATX_HEADING_RE = re.compile(r"^(?P<indent> *)(?P<markers>#{1,6})(?: +(?P<body>.*))?$")
_LIST_ITEM_RE = re.compile(
    r"^(?P<indent> *)(?P<marker>[-+*]|\d+[.)])(?: +(?P<body>.*))?$"
)
_CITATION_TOKEN_RE = re.compile(
    r"(?P<open>\[|【)(?P<body>[^\]\】\r\n]+)(?P<close>\]|】)"
)
_NUMERIC_CITATION_RE = re.compile(r"^\^?\d+(?:\s*(?:,|-)\s*\^?\d+)*$")
_PREFIX_CITATION_RE = re.compile(
    r"^(?:cite|citation|ref|reference|source|evidence)"
    r"(?:[-_: ]+)?[A-Za-z0-9][A-Za-z0-9_.:/-]*$",
    re.IGNORECASE,
)


class SummaryOutputNormalizationError(ValueError):
    """Raised when a summary output normalization request is invalid."""


@dataclass(frozen=True, repr=False)
class SummaryOutputNormalization:
    """Normalized summary text plus a value-free reproducibility record.

    ``normalized_text`` is the caller-facing summary output. ``to_dict()``
    and ``to_json()`` intentionally omit it so an audit artifact contains
    only fixed operation codes and a schema version, never summary content.
    """

    normalized_text: str
    operations: tuple[str, ...] = ()
    schema_version: int = SUMMARY_OUTPUT_NORMALIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.normalized_text) is not str:
            raise SummaryOutputNormalizationError(
                "normalized summary output must be text"
            )
        if (
            type(self.schema_version) is not int
            or self.schema_version != SUMMARY_OUTPUT_NORMALIZATION_SCHEMA_VERSION
        ):
            raise SummaryOutputNormalizationError(
                "unsupported summary normalization schema"
            )
        if isinstance(self.operations, (str, bytes, bytearray)):
            raise SummaryOutputNormalizationError(
                "summary normalization operations are invalid"
            )
        try:
            operations = tuple(self.operations)
        except Exception:
            raise SummaryOutputNormalizationError(
                "summary normalization operations are invalid"
            ) from None
        if any(
            type(operation) is not str
            or operation not in SUMMARY_OUTPUT_NORMALIZATION_OPERATION_CODES
            for operation in operations
        ) or len(set(operations)) != len(operations):
            raise SummaryOutputNormalizationError(
                "summary normalization operations are invalid"
            )
        object.__setattr__(self, "operations", operations)

    @property
    def text(self) -> str:
        """Return the caller-facing normalized summary text."""

        return self.normalized_text

    @property
    def operation_codes(self) -> tuple[str, ...]:
        """Return stable, value-free operation codes."""

        return self.operations

    @property
    def changed(self) -> bool:
        """Return whether at least one presentation operation was applied."""

        return bool(self.operations)

    def to_dict(self) -> dict[str, object]:
        """Return a value-free audit record for the normalization run."""

        return {
            "schema_version": self.schema_version,
            "operation_codes": list(self.operations),
        }

    def to_json(self) -> str:
        """Return byte-stable JSON for a value-free audit artifact."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )

    def __repr__(self) -> str:
        """Return a log-safe representation without summary content."""

        return (
            "SummaryOutputNormalization("
            f"operation_codes={self.operations!r}, changed={self.changed!r})"
        )


class SummaryOutputNormalizer:
    """Stateless facade for deterministic summary output normalization."""

    __slots__ = ()

    def normalize(self, summary_output: str) -> SummaryOutputNormalization:
        """Normalize one summary output without making external calls."""

        return normalize_summary_output(summary_output)


def normalize_summary_output(summary_output: str) -> SummaryOutputNormalization:
    """Normalize presentation syntax while preserving summary claim words.

    The canonical form uses LF line endings, at most one blank line between
    blocks, single horizontal spaces, ATX headings with one separating space,
    ``-`` unordered markers, sequential ``1.`` ordered markers, and citation
    tokens at the end of each logical line. Citation tokens are limited to
    numeric forms such as ``[1]``/``[1, 2]`` and explicit forms such as
    ``[citation:1]``; ordinary Markdown links are left untouched.

    Args:
        summary_output: Local model output to normalize. The caller remains
            responsible for applying any PHI policy before displaying or
            storing this text.

    Returns:
        A :class:`SummaryOutputNormalization` containing the normalized text
        and a value-free audit record.

    Raises:
        SummaryOutputNormalizationError: If ``summary_output`` is not text.
    """

    if type(summary_output) is not str:
        raise SummaryOutputNormalizationError("summary output must be text")

    canonical_newlines = _canonicalize_line_endings(summary_output)
    raw_lines = canonical_newlines.split("\n")
    lines = [_canonicalize_line_surface(line) for line in raw_lines]
    whitespace_changed = canonical_newlines != summary_output or lines != raw_lines

    lines, heading_changed = _normalize_headings(lines)
    lines, list_changed = _normalize_list_markers(lines)
    lines, citation_changed = _normalize_citation_placement(lines)
    lines, final_whitespace_changed = _finalize_whitespace(lines)

    normalized_text = "\n".join(lines)
    if normalized_text != summary_output and not (
        whitespace_changed
        or heading_changed
        or list_changed
        or citation_changed
        or final_whitespace_changed
    ):
        whitespace_changed = True

    changed_codes = {
        code
        for code, changed in (
            ("whitespace", whitespace_changed or final_whitespace_changed),
            ("heading", heading_changed),
            ("list_marker", list_changed),
            ("citation_placement", citation_changed),
        )
        if changed
    }
    operations = tuple(
        code
        for code in SUMMARY_OUTPUT_NORMALIZATION_OPERATION_CODES
        if code in changed_codes
    )
    return SummaryOutputNormalization(
        normalized_text=normalized_text,
        operations=operations,
    )


def normalize_summary_text(summary_output: str) -> str:
    """Return only the normalized text for callers that need no audit record."""

    return normalize_summary_output(summary_output).normalized_text


def _canonicalize_line_endings(value: str) -> str:
    return (
        value.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
    )


def _canonicalize_line_surface(value: str) -> str:
    value = _canonicalize_horizontal_whitespace(value)
    value = value.rstrip()
    return "" if not value.strip() else value


def _canonicalize_horizontal_whitespace(value: str) -> str:
    value = "".join(" " if char.isspace() else char for char in value)
    return _HORIZONTAL_SPACES_RE.sub(" ", value)


def _normalize_headings(lines: list[str]) -> tuple[list[str], bool]:
    normalized: list[str] = []
    changed = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            index + 1 < len(lines)
            and line.strip()
            and not _is_list_line(line)
            and _SETEXT_HEADING_RE.fullmatch(lines[index + 1].strip())
        ):
            underline = lines[index + 1].strip()
            level = 1 if underline[0] == "=" else 2
            candidate = f"{'#' * level} {line.strip()}"
            changed = changed or candidate != line or lines[index + 1] != ""
            normalized.append(candidate)
            index += 2
            continue

        candidate = _canonicalize_atx_heading(line)
        changed = changed or candidate != line
        normalized.append(candidate)
        index += 1
    return normalized, changed


def _canonicalize_atx_heading(line: str) -> str:
    match = _ATX_HEADING_RE.fullmatch(line)
    if match is None:
        return line
    body = (match.group("body") or "").strip()
    body = re.sub(r" +#{1,6}$", "", body).rstrip()
    markers = match.group("markers")
    return f"{markers} {body}" if body else markers


def _normalize_list_markers(lines: list[str]) -> tuple[list[str], bool]:
    normalized: list[str] = []
    ordered_counts: dict[int, int] = {}
    list_kinds: dict[int, str] = {}
    changed = False

    for line in lines:
        match = _LIST_ITEM_RE.fullmatch(line)
        if match is None:
            if not line.strip():
                ordered_counts.clear()
                list_kinds.clear()
            else:
                ordered_counts.clear()
                list_kinds.clear()
            normalized.append(line)
            continue

        indent_width = len(match.group("indent"))
        level = (indent_width + 1) // 2
        indent = "  " * level
        marker = match.group("marker")
        body = (match.group("body") or "").strip()
        is_ordered = marker[0].isdigit()

        for nested_level in tuple(ordered_counts):
            if nested_level > level:
                ordered_counts.pop(nested_level, None)
                list_kinds.pop(nested_level, None)

        if is_ordered:
            if list_kinds.get(level) == "ordered":
                ordered_counts[level] = ordered_counts.get(level, 0) + 1
            else:
                ordered_counts[level] = 1
            list_kinds[level] = "ordered"
            canonical_marker = f"{ordered_counts[level]}."
        else:
            ordered_counts.pop(level, None)
            list_kinds[level] = "unordered"
            canonical_marker = "-"

        candidate = f"{indent}{canonical_marker}"
        if body:
            candidate += f" {body}"
        changed = changed or candidate != line
        normalized.append(candidate)

    return normalized, changed


def _normalize_citation_placement(lines: list[str]) -> tuple[list[str], bool]:
    normalized: list[str] = []
    changed = False
    for line in lines:
        candidate, line_changed = _canonicalize_citations_in_line(line)
        normalized.append(candidate)
        changed = changed or line_changed
    return normalized, changed


def _canonicalize_citations_in_line(line: str) -> tuple[str, bool]:
    matches: list[str] = []
    pieces: list[str] = []
    cursor = 0
    for match in _CITATION_TOKEN_RE.finditer(line):
        if _is_markdown_link_label(line, match.end()):
            continue
        token = _canonical_citation_token(match.group("body"))
        if token is None:
            continue
        pieces.append(line[cursor : match.start()])
        cursor = match.end()
        matches.append(token)

    if not matches:
        return line, False

    pieces.append(line[cursor:])
    content = _canonicalize_horizontal_whitespace("".join(pieces)).strip()
    content = re.sub(r" +([,.;:!?])", r"\1", content)
    citation_suffix = " ".join(matches)
    candidate = f"{content} {citation_suffix}" if content else citation_suffix
    return candidate, candidate != line


def _is_markdown_link_label(line: str, end: int) -> bool:
    return re.match(r"\s*\(", line[end:]) is not None


def _canonical_citation_token(body: str) -> str | None:
    body = body.strip()
    if _NUMERIC_CITATION_RE.fullmatch(body):
        compact = re.sub(r"\s+", "", body)
        return f"[{compact.replace(',', ', ')}]"
    if _PREFIX_CITATION_RE.fullmatch(body):
        compact = re.sub(r"\s+", "", body)
        return f"[{compact}]"
    return None


def _finalize_whitespace(lines: list[str]) -> tuple[list[str], bool]:
    normalized: list[str] = []
    changed = False
    for line in lines:
        candidate = _canonicalize_horizontal_whitespace(line).rstrip()
        if not candidate.strip():
            candidate = ""
        elif _is_list_line(candidate):
            pass
        elif _ATX_HEADING_RE.fullmatch(candidate) is not None:
            candidate = candidate.lstrip()
        else:
            candidate = candidate.strip()
        changed = changed or candidate != line
        normalized.append(candidate)

    collapsed: list[str] = []
    previous_blank = False
    for line in normalized:
        is_blank = not line
        if is_blank and previous_blank:
            changed = True
            continue
        collapsed.append(line)
        previous_blank = is_blank

    while collapsed and not collapsed[0]:
        collapsed.pop(0)
        changed = True
    while collapsed and not collapsed[-1]:
        collapsed.pop()
        changed = True
    return collapsed, changed


def _is_list_line(line: str) -> bool:
    return _LIST_ITEM_RE.fullmatch(line) is not None


__all__ = [
    "SUMMARY_OUTPUT_NORMALIZATION_OPERATION_CODES",
    "SUMMARY_OUTPUT_NORMALIZATION_SCHEMA_VERSION",
    "SummaryOutputNormalization",
    "SummaryOutputNormalizationError",
    "SummaryOutputNormalizer",
    "normalize_summary_output",
    "normalize_summary_text",
]
