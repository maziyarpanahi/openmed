"""Flowsheet / vitals time-series structurer (roadmap section 4.2).

Vitals and nursing flowsheets are inherently time-series: the same parameters
(HR, BP, Temp, ...) are measured repeatedly and recorded as a grid of parameter
rows against timestamp columns. The single-point vital-signs extractor cannot
represent that shape, so this module assembles a delimited flowsheet grid into
per-parameter, ordered, timestamped series for trend analysis.

Alignment is by column position, so a blank cell becomes a gap in that
parameter's series rather than shifting later values onto the wrong timestamp.
Every emitted point carries the value's character offsets, which round-trip to
the source text. Structuring is deterministic and offline.
"""

from __future__ import annotations

import re
from typing import TypedDict

FLOWSHEET_ADVISORY = (
    "Flowsheet structuring aligns a delimited grid into per-parameter time "
    "series deterministically; blank cells are gaps, not interpolated values. "
    "It is support tooling, not a clinical trend interpretation."
)

_DEFAULT_DELIMITER = "|"
# A cell that is a bare number with an optional alphabetic unit ("37.0 C",
# "98 %"). Composite values such as a blood pressure ("120/80") do not match and
# keep ``number``/``unit`` unset while the raw value is preserved.
_NUMBER_UNIT_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*([A-Za-z%°µ]+)?$")


class TimeSeriesPoint(TypedDict):
    """One measurement of a parameter at a timestamp."""

    timestamp: str
    value: str
    number: float | None
    unit: str | None
    start: int
    end: int


class ParameterSeries(TypedDict):
    """A parameter's ordered, timestamped series."""

    parameter: str
    points: list[TimeSeriesPoint]


class Flowsheet(TypedDict):
    """A structured flowsheet: the timestamp axis and per-parameter series."""

    timestamps: list[str]
    series: list[ParameterSeries]


def _parse_number_unit(value: str) -> tuple[float | None, str | None]:
    match = _NUMBER_UNIT_RE.match(value)
    if match is None:
        return None, None
    return float(match.group(1)), (match.group(2) or None)


def _split_cells(line: str, line_start: int, delimiter: str) -> list[tuple[int, str]]:
    """Return ``(cell_start, cell_text)`` for each delimited cell in a line."""

    cells: list[tuple[int, str]] = []
    cursor = line_start
    for part in line.split(delimiter):
        cells.append((cursor, part))
        cursor += len(part) + len(delimiter)
    return cells


def structure_flowsheet(
    text: str,
    *,
    delimiter: str = _DEFAULT_DELIMITER,
) -> Flowsheet:
    """Structure a delimited flowsheet grid into per-parameter time series.

    The first non-empty line is the header: its first cell is a label heading
    (ignored) and the remaining cells are timestamps. Each later line is a
    parameter row whose value cells align to the timestamp columns by position;
    a blank cell is a gap. Every point`s ``start``/``end`` offsets index the
    value in ``text``.
    """

    rows: list[tuple[int, str]] = []
    offset = 0
    for line in text.split("\n"):
        rows.append((offset, line))
        offset += len(line) + 1  # account for the newline separator

    header_index = next((i for i, (_, line) in enumerate(rows) if line.strip()), None)
    if header_index is None:
        return Flowsheet(timestamps=[], series=[])

    header_start, header_line = rows[header_index]
    header_cells = _split_cells(header_line, header_start, delimiter)
    timestamps = [cell.strip() for _, cell in header_cells[1:]]

    series: list[ParameterSeries] = []
    for line_start, line in rows[header_index + 1 :]:
        if not line.strip():
            continue
        cells = _split_cells(line, line_start, delimiter)
        parameter = cells[0][1].strip()
        if not parameter:
            continue

        points: list[TimeSeriesPoint] = []
        for column, (cell_start, cell_text) in enumerate(cells[1:]):
            if column >= len(timestamps):
                break
            stripped = cell_text.strip()
            if not stripped:
                continue  # blank cell -> gap for this timestamp
            value_start = cell_start + cell_text.index(stripped)
            number, unit = _parse_number_unit(stripped)
            points.append(
                TimeSeriesPoint(
                    timestamp=timestamps[column],
                    value=stripped,
                    number=number,
                    unit=unit,
                    start=value_start,
                    end=value_start + len(stripped),
                )
            )
        series.append(ParameterSeries(parameter=parameter, points=points))

    return Flowsheet(timestamps=timestamps, series=series)


__all__ = [
    "FLOWSHEET_ADVISORY",
    "TimeSeriesPoint",
    "ParameterSeries",
    "Flowsheet",
    "structure_flowsheet",
]
