"""Tests for deterministic serial-measurement grouping and trend derivation."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from openmed.clinical import (
    TREND_ADVISORY,
    build_measurement_trends,
    extract_measurement_trends,
    serialize_measurement_trends,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "measurement_trend.jsonl"
)

DIRECTIONS = {"increasing", "decreasing", "stable", "mixed", "unknown"}


def _load_fixture() -> list[dict]:
    with FIXTURE.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _by_entity(trends: list[dict]) -> dict[str, dict]:
    return {trend["entity"].casefold(): trend for trend in trends}


# ---------------------------------------------------------------------------
# Fixture-driven gold rows
# ---------------------------------------------------------------------------


def test_fixture_rows_are_synthetic():
    rows = _load_fixture()
    assert rows
    assert all(row["metadata"]["synthetic"] is True for row in rows)


@pytest.mark.parametrize("row", _load_fixture())
def test_extractor_matches_gold_row(row):
    produced = extract_measurement_trends(
        row["points"], reference_date=row.get("reference_date")
    )
    got = _by_entity(produced)
    assert len(produced) == len(row["gold"]["trends"])
    for gold in row["gold"]["trends"]:
        trend = got[gold["entity"].casefold()]
        assert trend["direction"] == gold["direction"], gold["entity"]
        assert trend["comparable_count"] == gold["comparable_count"], gold["entity"]
        assert len(trend["incomparable_points"]) == gold["incomparable_count"], gold[
            "entity"
        ]
        assert trend["direction"] in DIRECTIONS


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def test_groups_same_entity_case_and_whitespace_insensitively():
    trends = extract_measurement_trends(
        [
            {
                "entity": "Hemoglobin",
                "value": 9,
                "unit": "g/dL",
                "timepoint": "2026-01-01",
            },
            {
                "entity": "  hemoglobin ",
                "value": 11,
                "unit": "g/dL",
                "timepoint": "2026-02-01",
            },
            {
                "entity": "HEMOGLOBIN",
                "value": 13,
                "unit": "g/dL",
                "timepoint": "2026-03-01",
            },
        ]
    )
    assert len(trends) == 1
    assert trends[0]["comparable_count"] == 3
    # Display entity preserves the first occurrence's surface form.
    assert trends[0]["entity"] == "Hemoglobin"


def test_distinct_entities_yield_distinct_trends_in_first_seen_order():
    trends = extract_measurement_trends(
        [
            {
                "entity": "hemoglobin",
                "value": 9,
                "unit": "g/dL",
                "timepoint": "2026-01-01",
            },
            {
                "entity": "sodium",
                "value": 140,
                "unit": "mmol/L",
                "timepoint": "2026-01-01",
            },
            {
                "entity": "hemoglobin",
                "value": 12,
                "unit": "g/dL",
                "timepoint": "2026-02-01",
            },
            {
                "entity": "sodium",
                "value": 141,
                "unit": "mmol/L",
                "timepoint": "2026-02-01",
            },
        ]
    )
    assert [trend["entity"] for trend in trends] == ["hemoglobin", "sodium"]


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_points_are_ordered_by_resolved_timepoint_not_input_order():
    trends = extract_measurement_trends(
        [
            {"entity": "mass", "value": 19, "unit": "mm", "timepoint": "2026-03-01"},
            {"entity": "mass", "value": 12, "unit": "mm", "timepoint": "2026-01-05"},
            {"entity": "mass", "value": 15, "unit": "mm", "timepoint": "2026-02-05"},
        ]
    )
    (trend,) = trends
    assert [point["value"] for point in trend["points"]] == [12, 15, 19]
    assert trend["direction"] == "increasing"


def test_issue_defined_builder_preserves_span_and_direction_name():
    span = {"start": 12, "end": 18}
    (trend,) = build_measurement_trends(
        [
            {
                "entity": "mass",
                "value": 12,
                "unit": "mm",
                "timepoint": "2026-01-05",
                "span": span,
            },
            {
                "entity": "mass",
                "value": 15,
                "unit": "mm",
                "timepoint": "2026-02-05",
                "span": {"start": 30, "end": 36},
            },
        ]
    )

    assert trend["trend_direction"] == "increasing"
    assert trend["direction"] == trend["trend_direction"]
    assert trend["points"][0]["span"] == span


def test_conflicting_values_at_same_timepoint_are_unknown():
    (trend,) = build_measurement_trends(
        [
            {
                "entity": "mass",
                "value": 12,
                "unit": "mm",
                "timepoint": "2026-01-05",
            },
            {
                "entity": "mass",
                "value": 15,
                "unit": "mm",
                "timepoint": "2026-01-05",
            },
            {
                "entity": "mass",
                "value": 18,
                "unit": "mm",
                "timepoint": "2026-02-05",
            },
        ]
    )

    assert trend["trend_direction"] == "unknown"
    assert trend["ordered"] is False
    assert trend["delta"] is None


def test_relative_timepoints_order_by_offset_and_reference_date_adds_iso():
    points = [
        {"entity": "crp", "value": 120, "unit": "mg/L", "timepoint": "3 weeks ago"},
        {"entity": "crp", "value": 60, "unit": "mg/L", "timepoint": "2 weeks ago"},
        {"entity": "crp", "value": 20, "unit": "mg/L", "timepoint": "1 week ago"},
    ]
    # Relative expressions carry signed day offsets, so the series orders even
    # without a reference date -- but no absolute ISO timepoint is available.
    (relative,) = extract_measurement_trends(points)
    assert relative["direction"] == "decreasing"
    assert relative["ordered"] is True
    assert all(point["normalized_timepoint"] is None for point in relative["points"])

    # A reference date resolves each timepoint to an absolute ISO interval.
    (resolved,) = extract_measurement_trends(points, reference_date="2026-03-01")
    assert resolved["direction"] == "decreasing"
    assert resolved["ordered"] is True
    assert all(
        point["normalized_timepoint"] is not None for point in resolved["points"]
    )


# ---------------------------------------------------------------------------
# Unit conversion and comparability
# ---------------------------------------------------------------------------


def test_mixed_but_commensurable_units_are_converted_and_compared():
    (trend,) = extract_measurement_trends(
        [
            {
                "entity": "liver lesion",
                "value": 1.2,
                "unit": "cm",
                "timepoint": "2026-01-15",
            },
            {
                "entity": "liver lesion",
                "value": 15,
                "unit": "mm",
                "timepoint": "2026-02-15",
            },
            {
                "entity": "liver lesion",
                "value": 1.8,
                "unit": "cm",
                "timepoint": "2026-03-15",
            },
        ]
    )
    assert trend["canonical_unit"] == "m"
    assert trend["comparable_count"] == 3
    assert trend["direction"] == "increasing"
    assert trend["first_value"] == pytest.approx(0.012)
    assert trend["last_value"] == pytest.approx(0.018)
    assert trend["delta"] == pytest.approx(0.006)


def test_incomparable_unit_is_flagged_not_dropped():
    (trend,) = extract_measurement_trends(
        [
            {
                "entity": "glucose",
                "value": 90,
                "unit": "mg/dL",
                "timepoint": "2026-01-03",
            },
            {
                "entity": "glucose",
                "value": 96,
                "unit": "mg/dL",
                "timepoint": "2026-01-17",
            },
            {
                "entity": "glucose",
                "value": 5,
                "unit": "mmol/L",
                "timepoint": "2026-01-31",
            },
        ]
    )
    assert trend["comparable_count"] == 2
    assert trend["direction"] == "increasing"
    assert len(trend["incomparable_points"]) == 1
    flagged = trend["incomparable_points"][0]
    assert flagged["unit"] == "mmol/L"
    assert flagged["comparable"] is False
    # The reading parses fine; it is incomparable because its dimension differs
    # from the group's (mass/volume vs amount/volume), so it keeps its own
    # normalized value rather than being forced into an analyte-specific mole
    # conversion.
    assert flagged["canonical_unit"] != trend["canonical_unit"]


def test_ambiguous_unit_is_flagged_and_direction_unknown():
    (trend,) = extract_measurement_trends(
        [
            {"entity": "wbc", "value": 5, "unit": "units", "timepoint": "2026-01-04"},
            {"entity": "wbc", "value": 7, "unit": "units", "timepoint": "2026-01-18"},
        ]
    )
    assert trend["comparable_count"] == 0
    assert len(trend["incomparable_points"]) == 2
    assert trend["direction"] == "unknown"
    assert trend["canonical_unit"] is None


# ---------------------------------------------------------------------------
# Direction derivation
# ---------------------------------------------------------------------------


def test_stable_series_when_values_unchanged_after_normalization():
    (trend,) = extract_measurement_trends(
        [
            {"entity": "nodule", "value": 8, "unit": "mm", "timepoint": "2026-01-01"},
            {"entity": "nodule", "value": 0.8, "unit": "cm", "timepoint": "2026-02-01"},
            {"entity": "nodule", "value": 8, "unit": "mm", "timepoint": "2026-03-01"},
        ]
    )
    assert trend["direction"] == "stable"
    assert trend["delta"] == pytest.approx(0.0, abs=1e-12)


def test_mixed_series_when_both_directions_occur():
    (trend,) = extract_measurement_trends(
        [
            {
                "entity": "potassium",
                "value": 4.0,
                "unit": "mmol/L",
                "timepoint": "2026-01-02",
            },
            {
                "entity": "potassium",
                "value": 5.2,
                "unit": "mmol/L",
                "timepoint": "2026-01-09",
            },
            {
                "entity": "potassium",
                "value": 4.4,
                "unit": "mmol/L",
                "timepoint": "2026-01-16",
            },
        ]
    )
    assert trend["direction"] == "mixed"


def test_series_without_timepoints_is_unknown():
    (trend,) = extract_measurement_trends(
        [
            {"entity": "hematocrit", "value": 30, "unit": "%"},
            {"entity": "hematocrit", "value": 34, "unit": "%"},
        ]
    )
    assert trend["direction"] == "unknown"
    assert trend["ordered"] is False
    assert trend["comparable_count"] == 2
    assert trend["delta"] is None


def test_single_point_series_is_unknown():
    (trend,) = extract_measurement_trends(
        [
            {
                "entity": "thyroid nodule",
                "value": 10,
                "unit": "mm",
                "timepoint": "2026-01-05",
            }
        ]
    )
    assert trend["direction"] == "unknown"
    assert trend["comparable_count"] == 1
    # An unknown trend reports no summary values, even for a single timed point.
    assert trend["first_value"] is None
    assert trend["last_value"] is None
    assert trend["delta"] is None


def test_primary_unit_selection_is_input_order_invariant():
    # Same entity measured equally often in two incommensurable units: which
    # subset is comparable -- and therefore the reported unit and direction --
    # must not depend on the order the caller happens to pass the points in.
    points = [
        {"entity": "glucose", "value": 10, "unit": "mmol/L", "timepoint": "2026-01-01"},
        {"entity": "glucose", "value": 12, "unit": "mmol/L", "timepoint": "2026-02-01"},
        {"entity": "glucose", "value": 180, "unit": "mg/dL", "timepoint": "2026-01-01"},
        {"entity": "glucose", "value": 100, "unit": "mg/dL", "timepoint": "2026-02-01"},
    ]
    (forward,) = extract_measurement_trends(points)
    (reverse,) = extract_measurement_trends(list(reversed(points)))
    assert forward["canonical_unit"] == reverse["canonical_unit"]
    assert forward["direction"] == reverse["direction"]
    assert forward["comparable_count"] == reverse["comparable_count"]


def test_partially_timed_series_is_unknown():
    # One point cannot be placed on the timeline -> the whole series is unknown
    # rather than ordered from the points that happen to carry a timepoint.
    (trend,) = extract_measurement_trends(
        [
            {"entity": "ldl", "value": 160, "unit": "mg/dL", "timepoint": "2026-01-01"},
            {"entity": "ldl", "value": 120, "unit": "mg/dL"},
            {"entity": "ldl", "value": 90, "unit": "mg/dL", "timepoint": "2026-03-01"},
        ]
    )
    assert trend["direction"] == "unknown"
    assert trend["ordered"] is False


def test_non_temporal_timepoint_string_is_unknown():
    # A timepoint string carrying no recognizable temporal expression (e.g.
    # "baseline") is unresolvable, so the series is labeled unknown rather than
    # ordered by input order.
    (trend,) = extract_measurement_trends(
        [
            {"entity": "m", "value": 10, "unit": "mm", "timepoint": "baseline"},
            {"entity": "m", "value": 20, "unit": "mm", "timepoint": "at diagnosis"},
        ]
    )
    assert trend["direction"] == "unknown"
    assert trend["ordered"] is False
    assert trend["comparable_count"] == 2
    assert all(point["normalized_timepoint"] is None for point in trend["points"])


# ---------------------------------------------------------------------------
# Advisory
# ---------------------------------------------------------------------------


def test_advisory_is_attached_to_every_trend():
    trends = extract_measurement_trends(
        [
            {
                "entity": "hemoglobin",
                "value": 9,
                "unit": "g/dL",
                "timepoint": "2026-01-01",
            },
            {
                "entity": "sodium",
                "value": 140,
                "unit": "mmol/L",
                "timepoint": "2026-01-01",
            },
        ]
    )
    assert trends
    for trend in trends:
        assert trend["advisory"] == TREND_ADVISORY
    assert "not a clinical judgment" in TREND_ADVISORY
    assert "descriptive summaries for clinician review" in TREND_ADVISORY


def test_empty_input_yields_no_trends():
    assert extract_measurement_trends([]) == []


# ---------------------------------------------------------------------------
# Stable JSON serialization (#3107)
# ---------------------------------------------------------------------------

SERIALIZED_FIXTURE = FIXTURE.with_name("measurement_trend_serialized.jsonl")


def _load_serialized_fixture() -> list[dict]:
    with SERIALIZED_FIXTURE.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_serialized_fixture_covers_each_trend_shape():
    rows = _load_serialized_fixture()

    assert all(row["metadata"]["synthetic"] is True for row in rows)
    assert {row["name"] for row in rows} == {
        "ordered",
        "mixed",
        "unknown",
        "incomparable",
    }


@pytest.mark.parametrize("row", _load_serialized_fixture(), ids=lambda row: row["name"])
def test_serialization_matches_golden_for_every_input_order(row):
    for permutation in itertools.permutations(row["points"]):
        trends = extract_measurement_trends(
            list(permutation), reference_date=row["reference_date"]
        )
        assert serialize_measurement_trends(trends) == row["expected"]


def test_serialization_sorts_groups_and_canonicalizes_entity_label():
    points = [
        {"entity": "weight", "value": 80, "unit": "kg", "timepoint": "2026-01-01"},
        {"entity": "Albumin", "value": 4.0, "unit": "g/dL", "timepoint": "2026-01-01"},
        {"entity": "Weight", "value": 78, "unit": "kg", "timepoint": "2026-02-01"},
        {"entity": "albumin", "value": 3.8, "unit": "g/dL", "timepoint": "2026-02-01"},
    ]

    outputs = {
        serialize_measurement_trends(extract_measurement_trends(list(order)))
        for order in itertools.permutations(points)
    }

    assert len(outputs) == 1
    serialized = json.loads(outputs.pop())
    # Grouping is case-insensitive; the label is the lowest spelling in the group.
    assert [trend["entity"] for trend in serialized] == ["Albumin", "Weight"]


def test_same_timepoint_tie_no_longer_depends_on_input_order():
    # 14 mm and 1.4 cm normalize to 0.014 and 0.013999999999999999: equal within
    # tolerance, so not a conflict, but not bitwise equal. Which one ended the
    # series used to follow input order and moved last_value and delta with it.
    base = {
        "entity": "Tumor size",
        "value": 12,
        "unit": "mm",
        "timepoint": "2026-01-05",
    }
    in_mm = {
        "entity": "Tumor size",
        "value": 14,
        "unit": "mm",
        "timepoint": "2026-02-05",
    }
    in_cm = {"entity": "Tumor size", "value": "1.4 cm", "timepoint": "2026-02-05"}

    forward = extract_measurement_trends([base, in_mm, in_cm])[0]
    backward = extract_measurement_trends([base, in_cm, in_mm])[0]

    assert forward["direction"] == backward["direction"] == "increasing"
    assert forward["last_value"] == backward["last_value"]
    assert forward["delta"] == backward["delta"]
    assert forward["points"] == backward["points"]


def test_serialization_is_compact_with_sorted_keys():
    trends = extract_measurement_trends(
        [
            {"entity": "Weight", "value": 80, "unit": "kg", "timepoint": "2026-01-01"},
            {"entity": "Weight", "value": 78, "unit": "kg", "timepoint": "2026-02-01"},
        ]
    )

    text = serialize_measurement_trends(trends)
    parsed = json.loads(text)

    assert text == json.dumps(parsed, separators=(",", ":"), sort_keys=True)
    assert set(parsed[0]) == set(trends[0])


def test_serialization_does_not_reorder_the_callers_trends():
    trends = extract_measurement_trends(
        [
            {"entity": "Weight", "value": 12, "unit": "furlongs"},
            {"entity": "Weight", "value": 80, "unit": "kg"},
            {"entity": "Weight", "value": 11, "unit": "fathoms"},
        ]
    )
    before = json.dumps(trends, sort_keys=True)

    serialize_measurement_trends(trends)

    assert json.dumps(trends, sort_keys=True) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("delta", float("nan")),
        ("first_value", float("inf")),
        ("last_value", float("-inf")),
    ],
)
def test_serialization_rejects_non_finite_derived_trend_values(field, value):
    trend = extract_measurement_trends(
        [
            {"entity": "Weight", "value": 80, "unit": "kg", "timepoint": "2026-01-01"},
            {"entity": "Weight", "value": 78, "unit": "kg", "timepoint": "2026-02-01"},
        ]
    )[0]
    trend[field] = value

    with pytest.raises(ValueError, match=f"{field} for 'Weight' must be finite"):
        serialize_measurement_trends([trend])


def test_serialization_rejects_non_finite_point_magnitude():
    trend = extract_measurement_trends(
        [
            {"entity": "Weight", "value": 80, "unit": "kg", "timepoint": "2026-01-01"},
            {"entity": "Weight", "value": 78, "unit": "kg", "timepoint": "2026-02-01"},
        ]
    )[0]
    trend["points"][1]["canonical_magnitude"] = float("nan")

    with pytest.raises(ValueError, match=r"points\[1\]\.canonical_magnitude"):
        serialize_measurement_trends([trend])


def test_serialization_of_no_trends_is_an_empty_array():
    assert serialize_measurement_trends([]) == "[]"
