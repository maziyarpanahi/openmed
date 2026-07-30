"""Table and grid extraction engine (roadmap section 4.2).

Clinical documents carry critical data in tables (medication lists, lab panels,
problem lists), but that structure is lost once a layout-aware extractor flattens
the page into text. This module reconstructs the row/column grid from
coordinate-bearing tokens: each token knows its bounding box on the page and its
character span in the source text, and the engine clusters those tokens back into
cells, detects header rows and columns, and preserves per-cell character offsets
so downstream redaction and extraction map straight back to the source.

Rows are recovered by clustering token vertical centres and columns by clustering
token left edges, so a row that is missing a cell (ragged) leaves a gap instead of
shifting later cells left, and a cell whose box spans several columns (merged) is
reported with a ``colspan`` rather than crashing the reconstruction. Every cell's
``start``/``end`` offsets round-trip to ``text``. Reconstruction is deterministic
and offline.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from typing import TypedDict

TABLE_ADVISORY = (
    "Table structuring clusters coordinate-bearing tokens into a row/column grid "
    "deterministically; ragged rows leave gaps and merged cells are reported with "
    "a span, never inferred away. It is layout support tooling, not a semantic "
    "typing of the cell contents."
)

# A cell is treated as a header candidate when it carries no digit; numeric cells
# are data. Header rows/columns are the leading label band of a table.
_DIGIT_RE = re.compile(r"\d")


class TableToken(TypedDict):
    """A coordinate-bearing token: its page box and its span in the source text."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    start: int
    end: int


class TableCell(TypedDict):
    """One reconstructed grid cell with its source char offsets."""

    row: int
    column: int
    colspan: int
    text: str
    start: int
    end: int
    is_header: bool


class Table(TypedDict):
    """A reconstructed table: grid shape, cells, and detected header bands."""

    n_rows: int
    n_columns: int
    cells: list[TableCell]
    header_rows: list[int]
    header_columns: list[int]


def _cluster_index(values: Sequence[float], tolerance: float) -> tuple[list[int], int]:
    """Assign each value a 0-based cluster index in ascending order of value.

    A new cluster starts whenever the gap to the previous (sorted) value exceeds
    ``tolerance``. Cluster 0 holds the smallest values, so for vertical centres it
    is the top row and for left edges it is the leftmost column.
    """

    order = sorted(range(len(values)), key=lambda i: values[i])
    index_of = [0] * len(values)
    cluster = 0
    previous: float | None = None
    for i in order:
        if previous is not None and values[i] - previous > tolerance:
            cluster += 1
        index_of[i] = cluster
        previous = values[i]
    return index_of, (cluster + 1 if values else 0)


def _is_numeric(text: str) -> bool:
    return bool(_DIGIT_RE.search(text))


def structure_table(
    text: str,
    tokens: Sequence[TableToken],
    *,
    row_tolerance: float | None = None,
    column_tolerance: float | None = None,
) -> Table:
    """Reconstruct a table grid from coordinate-bearing tokens.

    ``tokens`` carry a page bounding box (``x0``/``y0``/``x1``/``y1``) and a
    character span (``start``/``end``) into ``text``. Tokens are clustered into
    rows by vertical centre and columns by left edge; when the tolerances are left
    as ``None`` they default to half the median token height, which keeps the
    clustering scale-independent. Each cell's ``start``/``end`` index its content
    in ``text`` so ``text[start:end] == cell["text"]``. A cell whose box overhangs
    later columns reports a ``colspan`` greater than one (merged), and a row that
    omits a column simply has no cell there (ragged).
    """

    if not tokens:
        return Table(
            n_rows=0,
            n_columns=0,
            cells=[],
            header_rows=[],
            header_columns=[],
        )

    heights = [token["y1"] - token["y0"] for token in tokens]
    median_height = statistics.median(heights)
    default_tolerance = max(1.0, median_height * 0.5)
    row_tol = default_tolerance if row_tolerance is None else row_tolerance
    column_tol = default_tolerance if column_tolerance is None else column_tolerance

    centres = [(token["y0"] + token["y1"]) / 2 for token in tokens]
    lefts = [token["x0"] for token in tokens]
    row_of, n_rows = _cluster_index(centres, row_tol)
    column_of, n_columns = _cluster_index(lefts, column_tol)

    # Left boundary of each column band, used to measure how far a merged cell's
    # box overhangs into the columns to its right.
    boundaries = [float("inf")] * n_columns
    for index, column in enumerate(column_of):
        boundaries[column] = min(boundaries[column], lefts[index])

    groups: dict[tuple[int, int], list[int]] = {}
    for index in range(len(tokens)):
        groups.setdefault((row_of[index], column_of[index]), []).append(index)

    cell_by_position: dict[tuple[int, int], TableCell] = {}
    for (row, column), members in groups.items():
        start = min(tokens[i]["start"] for i in members)
        end = max(tokens[i]["end"] for i in members)
        cell_x1 = max(tokens[i]["x1"] for i in members)
        colspan = max(
            1,
            sum(1 for c in range(column, n_columns) if boundaries[c] < cell_x1),
        )
        cell_by_position[(row, column)] = TableCell(
            row=row,
            column=column,
            colspan=colspan,
            text=text[start:end],
            start=start,
            end=end,
            is_header=False,
        )

    header_rows = _detect_header_band(cell_by_position, n_rows, n_columns, axis="row")
    header_columns = _detect_header_band(
        cell_by_position, n_rows, n_columns, axis="column", skip_rows=header_rows
    )

    for (row, column), cell in cell_by_position.items():
        cell["is_header"] = row in header_rows or column in header_columns

    cells = [cell_by_position[key] for key in sorted(cell_by_position, key=lambda k: k)]
    return Table(
        n_rows=n_rows,
        n_columns=n_columns,
        cells=cells,
        header_rows=header_rows,
        header_columns=header_columns,
    )


def _detect_header_band(
    cell_by_position: dict[tuple[int, int], TableCell],
    n_rows: int,
    n_columns: int,
    *,
    axis: str,
    skip_rows: Sequence[int] = (),
) -> list[int]:
    """Return the leading rows (or columns) that are entirely non-numeric labels.

    A header band is the contiguous run of leading rows/columns whose every cell
    lacks a digit. If that run would cover the whole axis the band collapses to
    just the first index, so a table that happens to be all text is not reported
    as pure header.
    """

    band: list[int] = []
    if axis == "row":
        outer, inner = n_rows, n_columns
        cell_of = lambda i, j: cell_by_position.get((i, j))  # noqa: E731
        skip = set()
    else:
        outer, inner = n_columns, n_rows
        cell_of = lambda i, j: cell_by_position.get((j, i))  # noqa: E731
        skip = set(skip_rows)

    for i in range(outer):
        cells = [cell_of(i, j) for j in range(inner) if j not in skip and cell_of(i, j)]
        if cells and all(not _is_numeric(cell["text"]) for cell in cells):
            band.append(i)
        else:
            break

    if outer > 1 and len(band) == outer:
        band = [0]
    return band


def cell_at(table: Table, row: int, column: int) -> TableCell | None:
    """Return the cell at ``(row, column)`` or ``None`` if that position is empty."""

    for cell in table["cells"]:
        if cell["row"] == row and cell["column"] == column:
            return cell
    return None


__all__ = [
    "TABLE_ADVISORY",
    "Table",
    "TableCell",
    "TableToken",
    "cell_at",
    "structure_table",
]
