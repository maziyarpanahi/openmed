"""Tests for the coordinate-bearing table / grid extraction engine."""

from __future__ import annotations

from openmed.structured.tables import (
    TABLE_ADVISORY,
    Table,
    TableToken,
    cell_at,
    structure_table,
)

_COL_WIDTH = 100.0
_ROW_HEIGHT = 20.0


def _build(grid: list[list[str | None]]) -> tuple[str, list[TableToken]]:
    """Lay a grid of cell strings into a source text and coordinate-bearing tokens.

    Each cell occupies its own column band and row line; a ``None`` entry is a
    missing (ragged) cell. Token character offsets are computed from the source
    text as it is assembled, so ``text[token.start:token.end] == token.text`` by
    construction — the fixtures never hard-code an offset.
    """

    parts: list[str] = []
    tokens: list[TableToken] = []
    cursor = 0
    for row_index, row in enumerate(grid):
        for column_index, value in enumerate(row):
            if value is None:
                continue
            start = cursor
            parts.append(value)
            cursor += len(value)
            tokens.append(
                TableToken(
                    text=value,
                    x0=column_index * _COL_WIDTH,
                    y0=row_index * _ROW_HEIGHT,
                    x1=column_index * _COL_WIDTH + float(len(value)),
                    y1=row_index * _ROW_HEIGHT + 10.0,
                    start=start,
                    end=cursor,
                )
            )
            parts.append(" ")
            cursor += 1
    return "".join(parts), tokens


# A synthetic medication table: a header row of column titles, a header column of
# drug names, and a numeric dose column.
_MED_GRID: list[list[str | None]] = [
    ["Medication", "Dose", "Frequency"],
    ["Aspirin", "81", "Daily"],
    ["Metformin", "500", "BID"],
]


# --------------------------------------------------------------------------
# Grid reconstruction
# --------------------------------------------------------------------------


def test_grid_shape_is_reconstructed():
    text, tokens = _build(_MED_GRID)

    table = structure_table(text, tokens)

    assert (table["n_rows"], table["n_columns"]) == (3, 3)
    assert len(table["cells"]) == 9


def test_cells_land_in_the_right_row_and_column():
    text, tokens = _build(_MED_GRID)

    table = structure_table(text, tokens)

    assert cell_at(table, 0, 0)["text"] == "Medication"
    assert cell_at(table, 1, 1)["text"] == "81"
    assert cell_at(table, 2, 2)["text"] == "BID"


def test_multiple_tokens_in_one_cell_are_merged_into_one_span():
    # Two tokens share the same column band and row -> a single cell whose text is
    # the contiguous source slice spanning both.
    text = "Aspirin 81mg Daily "
    tokens = [
        TableToken(text="Aspirin", x0=0.0, y0=0.0, x1=7.0, y1=10.0, start=0, end=7),
        TableToken(text="81mg", x0=0.0, y0=0.0, x1=4.0, y1=10.0, start=8, end=12),
        TableToken(text="Daily", x0=100.0, y0=0.0, x1=5.0, y1=10.0, start=13, end=18),
    ]

    table = structure_table(text, tokens)

    assert cell_at(table, 0, 0)["text"] == "Aspirin 81mg"
    assert cell_at(table, 0, 1)["text"] == "Daily"


# --------------------------------------------------------------------------
# Header detection
# --------------------------------------------------------------------------


def test_leading_non_numeric_row_is_detected_as_header():
    text, tokens = _build(_MED_GRID)

    table = structure_table(text, tokens)

    assert table["header_rows"] == [0]
    assert all(cell_at(table, 0, c)["is_header"] for c in range(3))
    assert not cell_at(table, 1, 1)["is_header"]


def test_leading_non_numeric_column_is_detected_as_header():
    text, tokens = _build(_MED_GRID)

    table = structure_table(text, tokens)

    assert table["header_columns"] == [0]
    assert cell_at(table, 1, 0)["is_header"]


def test_all_text_table_is_not_reported_as_entirely_header():
    text, tokens = _build([["Name", "City"], ["Ada", "Paris"]])

    table = structure_table(text, tokens)

    assert table["header_rows"] == [0]


# --------------------------------------------------------------------------
# Ragged rows
# --------------------------------------------------------------------------


def test_ragged_row_leaves_a_gap_without_shifting_cells():
    grid: list[list[str | None]] = [
        ["Medication", "Dose", "Frequency"],
        ["Aspirin", None, "Daily"],  # dose missing
    ]
    text, tokens = _build(grid)

    table = structure_table(text, tokens)

    assert table["n_columns"] == 3
    assert cell_at(table, 1, 1) is None
    # "Daily" must stay in column 2, not slide left into the empty dose column.
    assert cell_at(table, 1, 2)["text"] == "Daily"


# --------------------------------------------------------------------------
# Merged cells
# --------------------------------------------------------------------------


def test_merged_cell_reports_colspan_without_crashing():
    text = "Patient Vitals BP 120 80 "
    tokens = [
        # Row 0: a label column and a "Vitals" header whose box overhangs cols 1-2.
        TableToken(text="Patient", x0=0.0, y0=0.0, x1=7.0, y1=10.0, start=0, end=7),
        TableToken(text="Vitals", x0=100.0, y0=0.0, x1=290.0, y1=10.0, start=8, end=14),
        # Row 1: three aligned columns establish the three column bands.
        TableToken(text="BP", x0=0.0, y0=20.0, x1=2.0, y1=30.0, start=15, end=17),
        TableToken(text="120", x0=100.0, y0=20.0, x1=103.0, y1=30.0, start=18, end=21),
        TableToken(text="80", x0=200.0, y0=20.0, x1=202.0, y1=30.0, start=22, end=24),
    ]

    table = structure_table(text, tokens)

    assert table["n_columns"] == 3
    merged = cell_at(table, 0, 1)
    assert merged["text"] == "Vitals"
    assert merged["colspan"] == 2
    # The overhung column has no cell of its own in the merged row.
    assert cell_at(table, 0, 2) is None
    assert cell_at(table, 1, 1)["colspan"] == 1


# --------------------------------------------------------------------------
# Offset round-trip
# --------------------------------------------------------------------------


def test_offsets_round_trip_to_source_text():
    text, tokens = _build(_MED_GRID)

    table = structure_table(text, tokens)

    for cell in table["cells"]:
        assert text[cell["start"] : cell["end"]] == cell["text"]


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_deterministic():
    text, tokens = _build(_MED_GRID)

    assert structure_table(text, tokens) == structure_table(text, tokens)


def test_empty_input_returns_empty_table():
    empty = structure_table("", [])

    assert empty == Table(
        n_rows=0,
        n_columns=0,
        cells=[],
        header_rows=[],
        header_columns=[],
    )


def test_advisory_exposed():
    assert isinstance(TABLE_ADVISORY, str) and TABLE_ADVISORY
