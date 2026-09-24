"""Tests for strict evidence-bound laboratory value sequences."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.summary_lab_trends import (
    SUMMARY_LAB_TRENDS_ADVISORY,
    LabTrendObservation,
    render_lab_trend_summary,
)


def _evidence(character: str) -> str:
    return f"sha256:{character * 64}"


def test_renders_compatible_sequence_in_observation_time_order() -> None:
    result = render_lab_trend_summary(
        (
            LabTrendObservation(
                analyte="Synthetic Analyte",
                value=9.2,
                unit="mg/dL",
                observed_at="2026-03-01",
                evidence_id=_evidence("c"),
            ),
            LabTrendObservation(
                analyte="  synthetic   analyte ",
                value=8.1,
                unit="mg/dL",
                observed_at="2026-01-01",
                evidence_id=_evidence("a"),
            ),
            LabTrendObservation(
                analyte="SYNTHETIC ANALYTE",
                value=8.7,
                unit="mg/dL",
                observed_at="2026-02-01",
                evidence_id=_evidence("b"),
            ),
        )
    )

    assert result.status == "rendered"
    assert result.review_required is False
    assert result.sequence is not None
    assert result.sequence.analyte == "Synthetic Analyte"
    assert [point.observed_at for point in result.sequence.points] == [
        "2026-01-01",
        "2026-02-01",
        "2026-03-01",
    ]
    assert [point.value for point in result.sequence.points] == [0.081, 0.087, 0.092]
    assert result.sequence.unit == "g/L"
    assert result.advisory == SUMMARY_LAB_TRENDS_ADVISORY


def test_sequence_contains_no_clinical_interpretation() -> None:
    result = render_lab_trend_summary(
        (
            LabTrendObservation(
                "Synthetic B", 1, "mmol/L", "2026-01-01", _evidence("1")
            ),
            LabTrendObservation(
                "Synthetic B", 3, "mmol/L", "2026-02-01", _evidence("2")
            ),
        )
    )

    payload = result.to_dict()
    assert result.sequence is not None
    assert set(payload["sequence"]) == {"analyte", "unit", "points"}
    assert "direction" not in result.to_json()
    assert "increasing" not in result.to_json()
    assert "abnormal" not in result.to_json()


@pytest.mark.parametrize(
    ("observations", "expected_code"),
    [
        (
            (
                LabTrendObservation(None, 1, "mg/dL", "2026-01-01", _evidence("1")),
                LabTrendObservation(
                    "Synthetic C", 2, "mg/dL", "2026-02-01", _evidence("2")
                ),
            ),
            "missing_analyte",
        ),
        (
            (
                LabTrendObservation(
                    "Synthetic C", 1, None, "2026-01-01", _evidence("1")
                ),
                LabTrendObservation(
                    "Synthetic C", 2, "mg/dL", "2026-02-01", _evidence("2")
                ),
            ),
            "missing_unit",
        ),
        (
            (
                LabTrendObservation("Synthetic C", 1, "mg/dL", None, _evidence("1")),
                LabTrendObservation(
                    "Synthetic C", 2, "mg/dL", "2026-02-01", _evidence("2")
                ),
            ),
            "missing_observation_time",
        ),
        (
            (
                LabTrendObservation("Synthetic C", 1, "mg/dL", "2026-01-01", None),
                LabTrendObservation(
                    "Synthetic C", 2, "mg/dL", "2026-02-01", _evidence("2")
                ),
            ),
            "missing_evidence",
        ),
    ],
)
def test_incomplete_sequences_emit_codes_without_any_values(
    observations: tuple[LabTrendObservation, ...], expected_code: str
) -> None:
    result = render_lab_trend_summary(observations)

    assert result.status == "insufficient"
    assert result.sequence is None
    assert expected_code in result.insufficiency_codes
    payload = result.to_dict()
    assert "sequence" not in payload
    assert "value" not in result.to_json()


def test_incompatible_analytes_and_units_suppress_complete_sequence() -> None:
    result = render_lab_trend_summary(
        (
            LabTrendObservation(
                "Synthetic D", 90, "mg/dL", "2026-01-01", _evidence("1")
            ),
            LabTrendObservation(
                "Synthetic E", 5, "mmol/L", "2026-02-01", _evidence("2")
            ),
        )
    )

    assert result.sequence is None
    assert set(result.insufficiency_codes) == {
        "incompatible_analyte",
        "incompatible_unit",
    }


def test_mixed_date_and_datetime_precision_is_incompatible() -> None:
    result = render_lab_trend_summary(
        (
            LabTrendObservation(
                "Synthetic F", 1, "mg/dL", "2026-01-01", _evidence("1")
            ),
            LabTrendObservation(
                "Synthetic F",
                2,
                "mg/dL",
                "2026-02-01T12:00:00Z",
                _evidence("2"),
            ),
        )
    )

    assert result.sequence is None
    assert result.insufficiency_codes == ("incompatible_observation_time",)


def test_conflicting_values_at_one_time_are_not_rendered() -> None:
    result = render_lab_trend_summary(
        (
            LabTrendObservation(
                "Synthetic G", 1, "mg/dL", "2026-01-01", _evidence("1")
            ),
            LabTrendObservation(
                "Synthetic G", 2, "mg/dL", "2026-01-01", _evidence("2")
            ),
        )
    )

    assert result.sequence is None
    assert result.insufficiency_codes == (
        "conflicting_observation_time",
        "insufficient_observations",
    )


def test_single_observation_is_value_free_insufficiency() -> None:
    result = render_lab_trend_summary(
        (
            LabTrendObservation(
                "private-analyte",
                12345,
                "mg/dL",
                "2026-01-01",
                _evidence("4"),
            ),
        )
    )

    assert result.insufficiency_codes == ("insufficient_observations",)
    encoded = result.to_json()
    assert "private-analyte" not in encoded
    assert "12345" not in encoded
    assert _evidence("4") not in encoded


def test_duplicate_equal_points_merge_evidence() -> None:
    observations = (
        LabTrendObservation("Synthetic H", 1, "mg/dL", "2026-01-01", _evidence("5")),
        LabTrendObservation("Synthetic H", 1, "mg/dL", "2026-01-01", _evidence("6")),
        LabTrendObservation("Synthetic H", 2, "mg/dL", "2026-02-01", _evidence("7")),
    )

    result = render_lab_trend_summary(observations)

    assert result.sequence is not None
    assert len(result.sequence.points) == 2
    assert result.sequence.points[0].evidence_ids == (_evidence("5"), _evidence("6"))


def test_duplicate_time_without_a_second_time_is_insufficient() -> None:
    result = render_lab_trend_summary(
        (
            LabTrendObservation(
                "Synthetic H", 1, "mg/dL", "2026-01-01", _evidence("5")
            ),
            LabTrendObservation(
                "Synthetic H", 1, "mg/dL", "2026-01-01", _evidence("6")
            ),
        )
    )

    assert result.sequence is None
    assert result.insufficiency_codes == ("insufficient_observations",)


def test_renderer_is_deterministic_and_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    observations = (
        LabTrendObservation("Synthetic I", 1, "mg/dL", "2026-01-01", _evidence("8")),
        LabTrendObservation("Synthetic I", 2, "mg/dL", "2026-02-01", _evidence("9")),
    )

    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr("socket.socket", fail_network)
    first = render_lab_trend_summary(observations).to_json()
    second = render_lab_trend_summary(observations).to_json()

    assert first == second
    assert json.loads(first)["status"] == "rendered"
