"""Deterministic multi-column reading order for positioned PDF words.

The detector operates on the dictionaries returned by
``pdfplumber.Page.extract_words`` and has no dependency of its own.  It looks
for repeated, unusually wide horizontal whitespace gaps on aligned text lines,
validates that each resulting column has enough parallel content, and then
emits a column-major word order.  Pages without a confidently repeated gutter
retain their exact source order.

Only word indexes and geometry are retained in the layout result.  Source text
is never copied into diagnostics or document metadata.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from statistics import median
from typing import Any, Literal

PdfReadingOrder = Literal["auto", "source"]
PdfBBox = tuple[float, float, float, float]

_MAX_COLUMNS = 3
_MIN_COLUMN_WORDS = 2
_MIN_PARALLEL_LINES = 2
_MIN_COLUMN_FRACTION = 0.20


@dataclass(frozen=True)
class PdfColumn:
    """One inferred page column described without retaining source text."""

    index: int
    bbox: PdfBBox
    word_indices: tuple[int, ...]

    @property
    def word_count(self) -> int:
        """Return the number of positioned words assigned to this column."""

        return len(self.word_indices)


@dataclass(frozen=True)
class PdfPageLayout:
    """Detected page columns and the source-word permutation to read them."""

    columns: tuple[PdfColumn, ...]
    reading_order: tuple[int, ...]
    word_columns: tuple[int | None, ...]
    column_boundaries: tuple[float, ...] = ()
    line_count: int = 0

    @property
    def is_multicolumn(self) -> bool:
        """Return whether two or more independently readable columns exist."""

        return len(self.columns) > 1

    @property
    def column_count(self) -> int:
        """Return the number of detected columns."""

        return len(self.columns)

    def ordered_words(
        self, words: Sequence[Mapping[str, Any]]
    ) -> tuple[Mapping[str, Any], ...]:
        """Return ``words`` permuted into this layout's reading order.

        Args:
            words: The same positioned-word sequence used for detection.

        Raises:
            ValueError: If ``words`` does not match the detected page size.
        """

        if len(words) != len(self.word_columns):
            raise ValueError("words must match the detected PDF page layout")
        return tuple(words[index] for index in self.reading_order)


@dataclass(frozen=True)
class _WordRecord:
    index: int
    bbox: PdfBBox

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def top(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def bottom(self) -> float:
        return self.bbox[3]

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass(frozen=True)
class _Line:
    index: int
    words: tuple[_WordRecord, ...]

    @property
    def top(self) -> float:
        return min(word.top for word in self.words)

    @property
    def bottom(self) -> float:
        return max(word.bottom for word in self.words)

    @property
    def center_y(self) -> float:
        return median(word.center_y for word in self.words)

    @property
    def x0(self) -> float:
        return min(word.x0 for word in self.words)

    @property
    def x1(self) -> float:
        return max(word.x1 for word in self.words)


@dataclass(frozen=True)
class _Gap:
    line_index: int
    x0: float
    x1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def center(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass(frozen=True)
class _Boundary:
    position: float
    support: int
    width: float


def detect_pdf_columns(
    words: Sequence[Mapping[str, Any]],
    *,
    page_width: float | None = None,
    line_tolerance: float | None = None,
    min_column_words: int = _MIN_COLUMN_WORDS,
    min_parallel_lines: int = _MIN_PARALLEL_LINES,
    max_columns: int = _MAX_COLUMNS,
) -> PdfPageLayout:
    """Detect repeated whitespace gutters and reconstruct a page's word order.

    The returned ``reading_order`` contains source indexes, so callers can
    reorder text while keeping each original word dictionary and bbox intact.
    Detection is conservative: when a gutter is not repeated across enough
    parallel lines, the result contains one column and the input order is
    returned unchanged.

    Args:
        words: Non-empty positioned words with ``x0``, ``top``, ``x1``, and
            ``bottom`` values, such as the output of ``pdfplumber``.
        page_width: Optional page width in the same coordinate system as the
            word bboxes. It helps scale the minimum gutter size.
        line_tolerance: Optional vertical-center tolerance for grouping words
            into lines. By default it is inferred from median word height.
        min_column_words: Minimum words required in every detected column.
        min_parallel_lines: Minimum aligned lines supporting every gutter.
        max_columns: Maximum supported columns. Values above three are allowed
            but three remains the default because clinical PDFs are normally
            one-, two-, or three-column documents.

    Returns:
        A :class:`PdfPageLayout` with a source-word permutation and column map.

    Raises:
        TypeError: If an option has the wrong type.
        ValueError: If an option or positioned bbox is invalid.
    """

    _validate_options(
        page_width=page_width,
        line_tolerance=line_tolerance,
        min_column_words=min_column_words,
        min_parallel_lines=min_parallel_lines,
        max_columns=max_columns,
    )
    records = tuple(_coerce_word(word, index) for index, word in enumerate(words))
    if not records:
        return PdfPageLayout(columns=(), reading_order=(), word_columns=())

    tolerance = (
        float(line_tolerance)
        if line_tolerance is not None
        else _default_line_tolerance(records)
    )
    lines = _group_lines(records, tolerance)
    if len(records) < min_column_words * 2 or len(lines) < min_parallel_lines:
        return _single_column_layout(records, line_count=len(lines))

    content_width = max(word.x1 for word in records) - min(word.x0 for word in records)
    scale_width = max(content_width, float(page_width or 0.0), 1.0)
    gap_threshold = _gutter_threshold(records, lines, scale_width)
    gaps = _wide_line_gaps(lines, gap_threshold)
    boundaries = _boundary_candidates(
        gaps,
        min_parallel_lines=min_parallel_lines,
        merge_tolerance=max(gap_threshold * 0.5, 0.5),
    )
    selected = _select_boundaries(
        records,
        lines,
        gaps,
        boundaries,
        min_column_words=min_column_words,
        min_parallel_lines=min_parallel_lines,
        max_columns=max_columns,
    )
    if not selected:
        return _single_column_layout(records, line_count=len(lines))

    positions = tuple(boundary.position for boundary in selected)
    assignments = tuple(_column_for_x(word.center_x, positions) for word in records)
    spanning_lines = _spanning_line_indexes(
        lines,
        positions,
        assignments,
        content_width=max(content_width, 1.0),
        gap_threshold=gap_threshold,
    )
    word_columns: list[int | None] = list(assignments)
    for line in lines:
        if line.index in spanning_lines:
            for word in line.words:
                word_columns[word.index] = None

    reading_order = _column_major_order(
        lines,
        column_count=len(positions) + 1,
        word_columns=word_columns,
        spanning_lines=spanning_lines,
    )
    if len(reading_order) != len(records) or len(set(reading_order)) != len(records):
        raise RuntimeError("PDF reading-order reconstruction lost positioned words")

    columns: list[PdfColumn] = []
    for column_index in range(len(positions) + 1):
        indexes = tuple(
            index for index in reading_order if word_columns[index] == column_index
        )
        boxes = tuple(records[index].bbox for index in indexes)
        if (
            len(indexes) < min_column_words
            or len(indexes) / len(records) <= _MIN_COLUMN_FRACTION
        ):
            return _single_column_layout(records, line_count=len(lines))
        columns.append(
            PdfColumn(
                index=column_index,
                bbox=_union_bbox(boxes),
                word_indices=indexes,
            )
        )

    return PdfPageLayout(
        columns=tuple(columns),
        reading_order=reading_order,
        word_columns=tuple(word_columns),
        column_boundaries=positions,
        line_count=len(lines),
    )


def reconstruct_pdf_reading_order(
    words: Sequence[Mapping[str, Any]],
    *,
    page_width: float | None = None,
    line_tolerance: float | None = None,
    min_column_words: int = _MIN_COLUMN_WORDS,
    min_parallel_lines: int = _MIN_PARALLEL_LINES,
    max_columns: int = _MAX_COLUMNS,
) -> PdfPageLayout:
    """Return the detected column-major layout for positioned PDF ``words``.

    This descriptive alias makes reconstruction discoverable to callers that
    already obtained words from ``pdfplumber``. See :func:`detect_pdf_columns`
    for supported options and fallback behavior.
    """

    return detect_pdf_columns(
        words,
        page_width=page_width,
        line_tolerance=line_tolerance,
        min_column_words=min_column_words,
        min_parallel_lines=min_parallel_lines,
        max_columns=max_columns,
    )


def validate_pdf_reading_order(value: str) -> PdfReadingOrder:
    """Validate and normalize a public PDF reading-order mode."""

    if value not in {"auto", "source"}:
        raise ValueError("reading_order must be 'auto' or 'source'")
    return value  # type: ignore[return-value]


def _validate_options(
    *,
    page_width: float | None,
    line_tolerance: float | None,
    min_column_words: int,
    min_parallel_lines: int,
    max_columns: int,
) -> None:
    if page_width is not None:
        if not isinstance(page_width, (int, float)):
            raise TypeError("page_width must be a number or None")
        if not isfinite(float(page_width)) or page_width <= 0:
            raise ValueError("page_width must be positive and finite")
    if line_tolerance is not None:
        if not isinstance(line_tolerance, (int, float)):
            raise TypeError("line_tolerance must be a number or None")
        if not isfinite(float(line_tolerance)) or line_tolerance < 0:
            raise ValueError("line_tolerance must be non-negative and finite")
    for name, value in (
        ("min_column_words", min_column_words),
        ("min_parallel_lines", min_parallel_lines),
        ("max_columns", max_columns),
    ):
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 2:
            raise ValueError(f"{name} must be at least 2")


def _coerce_word(word: Mapping[str, Any], index: int) -> _WordRecord:
    if not isinstance(word, Mapping):
        raise TypeError(f"PDF word {index} must be a mapping")
    try:
        bbox = tuple(float(word[field]) for field in ("x0", "top", "x1", "bottom"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"PDF word {index} has an invalid bbox") from exc
    if len(bbox) != 4 or not all(isfinite(value) for value in bbox):
        raise ValueError(f"PDF word {index} has an invalid bbox")
    x0, top, x1, bottom = bbox
    if x1 <= x0 or bottom <= top:
        raise ValueError(f"PDF word {index} has an invalid bbox")
    return _WordRecord(index=index, bbox=bbox)  # type: ignore[arg-type]


def _default_line_tolerance(records: Sequence[_WordRecord]) -> float:
    heights = [record.bottom - record.top for record in records]
    return max(median(heights) * 0.55, 0.5)


def _group_lines(records: Sequence[_WordRecord], tolerance: float) -> tuple[_Line, ...]:
    groups: list[list[_WordRecord]] = []
    for record in sorted(records, key=lambda item: (item.top, item.x0, item.index)):
        matches = [
            (index, group)
            for index, group in enumerate(groups)
            if _same_line(record, group, tolerance)
        ]
        if not matches:
            groups.append([record])
            continue
        group_index, _ = min(
            matches,
            key=lambda item: abs(
                record.center_y - median(word.center_y for word in item[1])
            ),
        )
        groups[group_index].append(record)

    ordered = sorted(
        groups,
        key=lambda group: (
            min(word.top for word in group),
            min(word.x0 for word in group),
            min(word.index for word in group),
        ),
    )
    return tuple(
        _Line(
            index=line_index,
            words=tuple(
                sorted(group, key=lambda word: (word.x0, word.top, word.index))
            ),
        )
        for line_index, group in enumerate(ordered)
    )


def _same_line(
    record: _WordRecord, line: Sequence[_WordRecord], tolerance: float
) -> bool:
    top = min(word.top for word in line)
    bottom = max(word.bottom for word in line)
    overlap = min(record.bottom, bottom) - max(record.top, top)
    center = median(word.center_y for word in line)
    return overlap > 0 or abs(record.center_y - center) <= tolerance


def _all_line_gaps(lines: Sequence[_Line]) -> tuple[_Gap, ...]:
    return tuple(
        _Gap(line.index, previous.x1, current.x0)
        for line in lines
        for previous, current in zip(line.words, line.words[1:])
        if current.x0 > previous.x1
    )


def _gutter_threshold(
    records: Sequence[_WordRecord], lines: Sequence[_Line], page_width: float
) -> float:
    heights = [record.bottom - record.top for record in records]
    median_height = median(heights)
    gaps = [gap.width for gap in _all_line_gaps(lines)]
    ordinary = [gap for gap in gaps if gap <= median_height * 2.5]
    typical_gap = median(ordinary) if ordinary else median_height * 0.5
    return max(median_height * 1.25, typical_gap * 3.0, page_width * 0.03)


def _wide_line_gaps(lines: Sequence[_Line], threshold: float) -> tuple[_Gap, ...]:
    return tuple(gap for gap in _all_line_gaps(lines) if gap.width >= threshold)


def _boundary_candidates(
    gaps: Sequence[_Gap], *, min_parallel_lines: int, merge_tolerance: float
) -> tuple[_Boundary, ...]:
    proposals: list[_Boundary] = []
    for candidate in gaps:
        supporting = tuple(gap for gap in gaps if gap.x0 <= candidate.center <= gap.x1)
        line_count = len({gap.line_index for gap in supporting})
        if line_count < min_parallel_lines:
            continue
        intersection_left = max(gap.x0 for gap in supporting)
        intersection_right = min(gap.x1 for gap in supporting)
        if intersection_right < intersection_left:
            continue
        proposals.append(
            _Boundary(
                position=(intersection_left + intersection_right) / 2.0,
                support=line_count,
                width=median(gap.width for gap in supporting),
            )
        )

    selected: list[_Boundary] = []
    for proposal in sorted(
        proposals,
        key=lambda item: (-item.support, -item.width, item.position),
    ):
        if any(
            abs(proposal.position - existing.position) <= merge_tolerance
            for existing in selected
        ):
            continue
        selected.append(proposal)
    return tuple(sorted(selected, key=lambda item: item.position))


def _select_boundaries(
    records: Sequence[_WordRecord],
    lines: Sequence[_Line],
    gaps: Sequence[_Gap],
    candidates: Sequence[_Boundary],
    *,
    min_column_words: int,
    min_parallel_lines: int,
    max_columns: int,
) -> tuple[_Boundary, ...]:
    if not candidates:
        return ()
    pool = tuple(sorted(candidates, key=lambda item: (-item.support, -item.width))[:8])
    best: tuple[tuple[int, int, float], tuple[_Boundary, ...]] | None = None
    max_boundaries = min(max_columns - 1, len(pool))
    for count in range(1, max_boundaries + 1):
        for choice in combinations(pool, count):
            ordered = tuple(sorted(choice, key=lambda item: item.position))
            if not _valid_columns(
                records,
                lines,
                gaps,
                ordered,
                min_column_words=min_column_words,
                min_parallel_lines=min_parallel_lines,
            ):
                continue
            score = (
                count,
                min(boundary.support for boundary in ordered),
                sum(boundary.width for boundary in ordered),
            )
            if best is None or score > best[0]:
                best = score, ordered
    return () if best is None else best[1]


def _valid_columns(
    records: Sequence[_WordRecord],
    lines: Sequence[_Line],
    gaps: Sequence[_Gap],
    boundaries: Sequence[_Boundary],
    *,
    min_column_words: int,
    min_parallel_lines: int,
) -> bool:
    positions = tuple(boundary.position for boundary in boundaries)
    assignments = {
        record.index: _column_for_x(record.center_x, positions) for record in records
    }
    for column_index in range(len(positions) + 1):
        members = [
            record for record in records if assignments[record.index] == column_index
        ]
        if len(members) < min_column_words or (
            len(members) / len(records) <= _MIN_COLUMN_FRACTION
        ):
            return False
        member_indexes = {member.index for member in members}
        member_lines = {
            line.index
            for line in lines
            if any(word.index in member_indexes for word in line.words)
        }
        if len(member_lines) < min_parallel_lines:
            return False

    for position in positions:
        supporting_lines = {
            gap.line_index for gap in gaps if gap.x0 <= position <= gap.x1
        }
        if len(supporting_lines) < min_parallel_lines:
            return False
    return True


def _column_for_x(value: float, boundaries: Sequence[float]) -> int:
    return sum(value > boundary for boundary in boundaries)


def _spanning_line_indexes(
    lines: Sequence[_Line],
    boundaries: Sequence[float],
    assignments: Sequence[int],
    *,
    content_width: float,
    gap_threshold: float,
) -> frozenset[int]:
    spanning: set[int] = set()
    for line in lines:
        populated = {assignments[word.index] for word in line.words}
        if len(populated) < 2 or line.x1 - line.x0 < content_width * 0.55:
            continue
        gaps = tuple(
            _Gap(line.index, previous.x1, current.x0)
            for previous, current in zip(line.words, line.words[1:])
            if current.x0 > previous.x1
        )
        crossed = tuple(
            boundary
            for boundary in boundaries
            if min(populated)
            < _column_for_x(boundary + 0.001, boundaries)
            <= max(populated)
        )
        if crossed and any(
            not any(
                gap.x0 <= boundary <= gap.x1 and gap.width >= gap_threshold * 0.75
                for gap in gaps
            )
            for boundary in crossed
        ):
            spanning.add(line.index)
    return frozenset(spanning)


def _column_major_order(
    lines: Sequence[_Line],
    *,
    column_count: int,
    word_columns: Sequence[int | None],
    spanning_lines: frozenset[int],
) -> tuple[int, ...]:
    ordered: list[int] = []
    band: list[_Line] = []

    def flush_band() -> None:
        for column_index in range(column_count):
            for line in band:
                ordered.extend(
                    word.index
                    for word in line.words
                    if word_columns[word.index] == column_index
                )

    for line in lines:
        if line.index not in spanning_lines:
            band.append(line)
            continue
        flush_band()
        band.clear()
        ordered.extend(word.index for word in line.words)
    flush_band()
    return tuple(ordered)


def _single_column_layout(
    records: Sequence[_WordRecord], *, line_count: int
) -> PdfPageLayout:
    indexes = tuple(record.index for record in records)
    column = PdfColumn(
        index=0,
        bbox=_union_bbox(record.bbox for record in records),
        word_indices=indexes,
    )
    return PdfPageLayout(
        columns=(column,),
        reading_order=indexes,
        word_columns=tuple(0 for _ in records),
        line_count=line_count,
    )


def _union_bbox(boxes: Iterable[PdfBBox]) -> PdfBBox:
    values = tuple(boxes)
    if not values:
        raise ValueError("cannot compute a bbox for an empty PDF column")
    return (
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    )


__all__ = [
    "PdfBBox",
    "PdfColumn",
    "PdfPageLayout",
    "PdfReadingOrder",
    "detect_pdf_columns",
    "reconstruct_pdf_reading_order",
]
