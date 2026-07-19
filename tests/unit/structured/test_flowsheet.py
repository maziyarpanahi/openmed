"""Tests for the flowsheet / vitals time-series structurer."""

from __future__ import annotations

from openmed.structured.flowsheet import (
    FLOWSHEET_ADVISORY,
    Flowsheet,
    ParameterSeries,
    structure_flowsheet,
)

# A synthetic pipe-delimited flowsheet with a ragged row (BP missing at 16:00)
# and a gap (Temp missing at 12:00).
FLOWSHEET = (
    "Parameter|08:00|12:00|16:00\nHR|72|75|80\nBP|120/80|118/78|\nTemp|37.0 C||37.2 C\n"
)


def _series(flowsheet: Flowsheet, name: str) -> ParameterSeries:
    return next(s for s in flowsheet["series"] if s["parameter"] == name)


# --------------------------------------------------------------------------
# Parsing into per-parameter time series
# --------------------------------------------------------------------------


def test_timestamps_and_parameters_are_parsed():
    flowsheet = structure_flowsheet(FLOWSHEET)

    assert flowsheet["timestamps"] == ["08:00", "12:00", "16:00"]
    assert {s["parameter"] for s in flowsheet["series"]} == {"HR", "BP", "Temp"}


def test_each_parameter_yields_ordered_timestamped_series():
    hr = _series(structure_flowsheet(FLOWSHEET), "HR")

    assert [(p["timestamp"], p["value"]) for p in hr["points"]] == [
        ("08:00", "72"),
        ("12:00", "75"),
        ("16:00", "80"),
    ]


def test_number_and_unit_are_extracted():
    temp = _series(structure_flowsheet(FLOWSHEET), "Temp")
    first = temp["points"][0]

    assert first["number"] == 37.0
    assert first["unit"] == "C"


def test_vital_sign_units_are_preserved():
    text = "Parameter|08:00\nTemp|37.0 °C\nResp|16 breaths/min\n"

    flowsheet = structure_flowsheet(text)

    temperature = _series(flowsheet, "Temp")["points"][0]
    respiration = _series(flowsheet, "Resp")["points"][0]
    assert (temperature["number"], temperature["unit"]) == (37.0, "°C")
    assert (respiration["number"], respiration["unit"]) == (16, "breaths/min")


def test_non_numeric_value_is_kept_verbatim():
    bp = _series(structure_flowsheet(FLOWSHEET), "BP")
    assert bp["points"][0]["value"] == "120/80"
    assert bp["points"][0]["number"] is None


# --------------------------------------------------------------------------
# Ragged columns / gaps
# --------------------------------------------------------------------------


def test_missing_cells_produce_gaps_not_misaligned_points():
    flowsheet = structure_flowsheet(FLOWSHEET)

    bp = _series(flowsheet, "BP")
    # BP is missing at 16:00 -> two points, still aligned to the right columns.
    assert [p["timestamp"] for p in bp["points"]] == ["08:00", "12:00"]

    temp = _series(flowsheet, "Temp")
    # Temp is missing at 12:00 -> the 16:00 value must not shift to 12:00.
    assert [p["timestamp"] for p in temp["points"]] == ["08:00", "16:00"]


def test_continuation_rows_carry_forward_parameter_identity_and_units():
    text = "Parameter|08:00|12:00\nBP|120/80 mmHg|\n||118/78 mmHg\n"

    flowsheet = structure_flowsheet(text)

    assert [series["parameter"] for series in flowsheet["series"]] == ["BP"]
    assert [
        (point["timestamp"], point["value"], point["number"], point["unit"])
        for point in flowsheet["series"][0]["points"]
    ] == [
        ("08:00", "120/80 mmHg", None, "mmHg"),
        ("12:00", "118/78 mmHg", None, "mmHg"),
    ]
    for point in flowsheet["series"][0]["points"]:
        assert text[point["start"] : point["end"]] == point["value"]


def test_repeated_parameter_rows_are_coalesced():
    text = "Parameter|08:00|12:00\nHR|72|\nHR||75\n"

    flowsheet = structure_flowsheet(text)

    assert [series["parameter"] for series in flowsheet["series"]] == ["HR"]
    assert [
        (point["timestamp"], point["value"])
        for point in flowsheet["series"][0]["points"]
    ] == [("08:00", "72"), ("12:00", "75")]


# --------------------------------------------------------------------------
# Offset round-trip
# --------------------------------------------------------------------------


def test_offsets_round_trip_to_source_text():
    flowsheet = structure_flowsheet(FLOWSHEET)
    for series in flowsheet["series"]:
        for point in series["points"]:
            assert FLOWSHEET[point["start"] : point["end"]] == point["value"]


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_deterministic():
    assert structure_flowsheet(FLOWSHEET) == structure_flowsheet(FLOWSHEET)


def test_empty_input_returns_empty():
    empty = structure_flowsheet("")
    assert empty["timestamps"] == []
    assert empty["series"] == []


def test_advisory_exposed():
    assert isinstance(FLOWSHEET_ADVISORY, str) and FLOWSHEET_ADVISORY
