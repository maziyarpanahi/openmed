"""Deterministic clinical list and enumeration parsing.

The parser is intentionally rules-first and operates only on source text.  It
does not load vocabularies, models, or clinical data.  Explicit enumeration
markers take precedence; otherwise, two or more unindented non-empty lines are
treated as a line-per-item list, with indented dosing or descriptive lines
kept inside the preceding logical item.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

ListMarkerStyle = Literal["numeric", "bullet", "lettered", "line"]

_MARKER_RE = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<marker>"
    r"(?:\d{1,3}|[A-Za-z])[.)]"
    r"|\((?:\d{1,3}|[A-Za-z])\)"
    r"|[-*+\u2022\u2023\u25e6\u25aa\u2013\u2014]"
    r")"
    r"(?P<spacing>[ \t]+)"
    r"(?P<body>\S.*)?$"
)
_CONTINUATION_RE = re.compile(
    r"^(?:"
    r"administer\b|apply\b|as\s+needed|at\s+(?:bedtime|night)|"
    r"by\s+(?:mouth|inhalation)|"
    r"daily|dose\b|every\b|for\b|frequency\b|"
    r"inhale\b|inject\b|instill\b|once\b|"
    r"route\b|sig\b|take\b|three\s+times\b|twice\b|use\b|with\b"
    r")",
    re.IGNORECASE,
)
_INCOMPLETE_ENDINGS = frozenset({",", ";", ":", "/", "(", "[", "{"})


@dataclass(frozen=True)
class ListItemSpan:
    """An offset-aligned logical list item, optionally containing descendants.

    Parent spans contain their nested descendants, so top-level spans form the
    non-overlapping units consumed by sentence segmentation.  Nested spans are
    also returned for callers that need enumeration structure.

    Attributes:
        text: Exact source substring from ``start`` through ``end``.
        start: Inclusive source character offset.
        end: Exclusive source character offset.
        nesting_level: Zero-based structural depth.
        style: Detected enumeration style.
        marker: Source marker such as ``"1."`` or ``"-"``, if present.
        marker_start: Inclusive marker offset, if present.
        marker_end: Exclusive marker offset, if present.
        content_start: Inclusive offset of content after indentation and marker.
        parent_index: Index of the containing item in the returned sequence.
    """

    text: str
    start: int
    end: int
    nesting_level: int
    style: ListMarkerStyle
    marker: str | None = None
    marker_start: int | None = None
    marker_end: int | None = None
    content_start: int | None = None
    parent_index: int | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("ListItemSpan requires 0 <= start < end")
        if self.nesting_level < 0:
            raise ValueError("ListItemSpan nesting_level must be non-negative")
        if (self.marker_start is None) != (self.marker_end is None):
            raise ValueError("ListItemSpan marker offsets must be provided together")
        if self.marker_start is not None and not (
            self.start <= self.marker_start < self.marker_end <= self.end
        ):
            raise ValueError("ListItemSpan marker offsets must fall inside the item")
        if self.content_start is not None and not (
            self.start <= self.content_start <= self.end
        ):
            raise ValueError("ListItemSpan content_start must fall inside the item")
        if self.parent_index is not None and self.parent_index < 0:
            raise ValueError("ListItemSpan parent_index must be non-negative")

    @property
    def level(self) -> int:
        """Return ``nesting_level`` as a compact compatibility alias."""

        return self.nesting_level

    @property
    def content_text(self) -> str:
        """Return item text after its marker while preserving descendant text."""

        if self.content_start is None:
            return self.text
        return self.text[self.content_start - self.start :]


@dataclass(frozen=True)
class _Line:
    text: str
    start: int
    end: int
    content_end: int
    indent: int


@dataclass(frozen=True)
class _MarkedLine:
    line: _Line
    marker: str
    marker_start: int
    marker_end: int
    content_start: int
    style: ListMarkerStyle


@dataclass(frozen=True)
class _ItemSeed:
    start: int
    indent: int
    nesting_level: int
    style: ListMarkerStyle
    marker: str | None
    marker_start: int | None
    marker_end: int | None
    content_start: int
    parent_index: int | None


def parse_lists(text: str) -> list[ListItemSpan]:
    """Parse enumerated or line-per-item structure with exact source offsets.

    Numeric markers (``1.``/``1)``), bullets, lettered markers, indentation,
    continuation lines, and unmarked line-per-item lists are supported.  A
    single unmarked line is not classified as a list, which avoids turning
    ordinary one-line prose into a false item.

    Args:
        text: Source text containing a possible list.

    Returns:
        Items in source order. Top-level item spans are non-overlapping; parent
        spans contain nested child spans.
    """

    if not text:
        return []

    lines = tuple(_iter_lines(text))
    marked = tuple(
        candidate for line in lines if (candidate := _marked_line(line)) is not None
    )
    seeds = _explicit_seeds(marked) if marked else _line_item_seeds(lines)
    if not seeds:
        return []

    items = _materialize_items(text, seeds)
    validate_list_items(text, items)
    return items


def validate_list_items(text: str, items: Sequence[ListItemSpan]) -> None:
    """Validate offsets, hierarchy, ordering, and exact source alignment.

    Args:
        text: Original source text.
        items: Candidate list items in source order.

    Raises:
        ValueError: If an item is malformed, misaligned, or attached to an
            invalid parent.
    """

    previous_start = -1
    previous_top_end = 0
    for index, item in enumerate(items):
        if not isinstance(item, ListItemSpan):
            raise ValueError(f"list item {index} is not a ListItemSpan")
        if item.start < previous_start:
            raise ValueError(f"list item {index} is out of source order")
        if item.end > len(text) or text[item.start : item.end] != item.text:
            raise ValueError(f"list item {index} does not match source text")

        if item.parent_index is None:
            if item.nesting_level != 0:
                raise ValueError(f"list item {index} has no parent at nested level")
            if item.start < previous_top_end:
                raise ValueError(f"top-level list item {index} overlaps its sibling")
            previous_top_end = item.end
        else:
            if item.parent_index >= index:
                raise ValueError(f"list item {index} parent must precede it")
            parent = items[item.parent_index]
            if item.nesting_level != parent.nesting_level + 1:
                raise ValueError(f"list item {index} skips a nesting level")
            if not (parent.start <= item.start and item.end <= parent.end):
                raise ValueError(f"list item {index} falls outside its parent")

        previous_start = item.start


def list_boundary_f1(
    gold_items: Sequence[ListItemSpan],
    predicted_items: Sequence[ListItemSpan],
) -> float:
    """Compute top-level item-boundary F1 from exact character offsets.

    The first top-level start is excluded because it marks the list origin,
    not a boundary between logical items.
    """

    gold = _interior_top_level_boundaries(gold_items)
    predicted = _interior_top_level_boundaries(predicted_items)
    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0
    true_positives = len(gold & predicted)
    return 2.0 * true_positives / (len(gold) + len(predicted))


def _iter_lines(text: str) -> Sequence[_Line]:
    lines: list[_Line] = []
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        end = cursor + len(raw_line)
        content_end = end - len(raw_line) + len(raw_line.rstrip("\r\n"))
        leading = raw_line[: len(raw_line) - len(raw_line.lstrip(" \t"))]
        lines.append(
            _Line(
                text=raw_line,
                start=cursor,
                end=end,
                content_end=content_end,
                indent=len(leading.expandtabs(4)),
            )
        )
        cursor = end
    if cursor < len(text):
        raw_line = text[cursor:]
        leading = raw_line[: len(raw_line) - len(raw_line.lstrip(" \t"))]
        lines.append(
            _Line(
                text=raw_line,
                start=cursor,
                end=len(text),
                content_end=len(text),
                indent=len(leading.expandtabs(4)),
            )
        )
    return lines


def _marked_line(line: _Line) -> _MarkedLine | None:
    line_text = line.text.rstrip("\r\n")
    match = _MARKER_RE.match(line_text)
    if match is None or not match.group("body"):
        return None
    marker = match.group("marker")
    marker_start = line.start + match.start("marker")
    marker_end = line.start + match.end("marker")
    return _MarkedLine(
        line=line,
        marker=marker,
        marker_start=marker_start,
        marker_end=marker_end,
        content_start=line.start + match.start("body"),
        style=_marker_style(marker),
    )


def _marker_style(marker: str) -> ListMarkerStyle:
    normalized = marker.strip("()")
    if normalized and normalized[0].isdigit():
        return "numeric"
    if normalized and normalized[0].isalpha():
        return "lettered"
    return "bullet"


def _explicit_seeds(marked: Sequence[_MarkedLine]) -> list[_ItemSeed]:
    seeds: list[_ItemSeed] = []
    stack: list[int] = []
    for candidate in marked:
        while stack and candidate.line.indent <= seeds[stack[-1]].indent:
            stack.pop()
        parent_index = stack[-1] if stack else None
        nesting_level = seeds[parent_index].nesting_level + 1 if stack else 0
        seeds.append(
            _ItemSeed(
                start=candidate.line.start,
                indent=candidate.line.indent,
                nesting_level=nesting_level,
                style=candidate.style,
                marker=candidate.marker,
                marker_start=candidate.marker_start,
                marker_end=candidate.marker_end,
                content_start=candidate.content_start,
                parent_index=parent_index,
            )
        )
        stack.append(len(seeds) - 1)
    return seeds


def _line_item_seeds(lines: Sequence[_Line]) -> list[_ItemSeed]:
    nonempty = [line for line in lines if line.text.strip()]
    if len(nonempty) < 2:
        return []
    base_indent = min(line.indent for line in nonempty)
    starts: list[_Line] = []
    previous: _Line | None = None
    for line in nonempty:
        if line.indent == base_indent and not _is_continuation(line, previous):
            starts.append(line)
        previous = line
    if len(starts) < 2:
        return []

    return [
        _ItemSeed(
            start=line.start,
            indent=line.indent,
            nesting_level=0,
            style="line",
            marker=None,
            marker_start=None,
            marker_end=None,
            content_start=line.start + len(line.text) - len(line.text.lstrip(" \t")),
            parent_index=None,
        )
        for line in starts
    ]


def _is_continuation(line: _Line, previous: _Line | None) -> bool:
    if previous is None:
        return False
    if line.indent > previous.indent:
        return True
    previous_text = previous.text.rstrip()
    if previous_text and previous_text[-1] in _INCOMPLETE_ENDINGS:
        return True
    return bool(_CONTINUATION_RE.match(line.text.strip()))


def _materialize_items(text: str, seeds: Sequence[_ItemSeed]) -> list[ListItemSpan]:
    items: list[ListItemSpan] = []
    for index, seed in enumerate(seeds):
        end = len(text)
        for following in seeds[index + 1 :]:
            if following.nesting_level <= seed.nesting_level:
                end = following.start
                break
        items.append(
            ListItemSpan(
                text=text[seed.start : end],
                start=seed.start,
                end=end,
                nesting_level=seed.nesting_level,
                style=seed.style,
                marker=seed.marker,
                marker_start=seed.marker_start,
                marker_end=seed.marker_end,
                content_start=seed.content_start,
                parent_index=seed.parent_index,
            )
        )
    return items


def _interior_top_level_boundaries(
    items: Sequence[ListItemSpan],
) -> frozenset[int]:
    starts = sorted(item.start for item in items if item.nesting_level == 0)
    return frozenset(starts[1:])


__all__ = [
    "ListItemSpan",
    "ListMarkerStyle",
    "list_boundary_f1",
    "parse_lists",
    "validate_list_items",
]
