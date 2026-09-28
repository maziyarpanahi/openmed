"""Tests for deterministic counts-only document-type calibration reports."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.sections.doctype import DOCUMENT_TYPES
from openmed.eval.doctype_calibration import (
    DOCTYPE_CALIBRATION_ARTIFACT,
    DocumentTypeCalibrationSample,
    DocumentTypeSupport,
    build_doctype_calibration_report,
    fingerprint_doctype_fixtures,
    fingerprint_doctype_model,
    render_doctype_calibration_json,
    render_doctype_calibration_markdown,
)

RAW_NOTE = "Patient Ada Lovelace, MRN-493021, called 555-0199."
RAW_FIXTURE_ID = "patient-Ada-Lovelace-493021"
RAW_MODEL_ID = "private/Ada-Lovelace-doctype-model"


def _samples() -> list[dict[str, object]]:
    return [
        {
            "expected_type": "progress_note",
            "predicted_type": "progress_note",
            "confidence": 0.9,
            "text": RAW_NOTE,
            "fixture_id": RAW_FIXTURE_ID,
        },
        {
            "expected_type": "progress_note",
            "predicted_type": "radiology_report",
            "confidence": 0.8,
            "text": RAW_NOTE,
            "fixture_id": RAW_FIXTURE_ID,
        },
        {
            "expected_type": "radiology_report",
            "predicted_type": "radiology_report",
            "confidence": 0.6,
            "text": RAW_NOTE,
            "fixture_id": RAW_FIXTURE_ID,
        },
        {
            "expected_type": "pathology_report",
            "predicted_type": "unknown",
            "confidence": 0.0,
            "text": RAW_NOTE,
            "fixture_id": RAW_FIXTURE_ID,
        },
    ]


def test_calibration_bins_and_per_type_support_match_hand_computed_counts() -> None:
    report = build_doctype_calibration_report(
        _samples(),
        model={"model_id": RAW_MODEL_ID, "revision": "synthetic-v1"},
        num_bins=2,
        abstention_thresholds=(0.5,),
    )

    assert report.sample_count == 4
    assert report.correct_count == 2
    assert report.unknown_prediction_count == 1
    assert report.accuracy == 0.5
    assert report.expected_calibration_error == pytest.approx(0.075)
    assert report.bins[0].to_dict() == {
        "absolute_gap": 0.0,
        "accuracy": 0.0,
        "correct_count": 0,
        "lower_bound": 0.0,
        "mean_confidence": 0.0,
        "sample_count": 1,
        "upper_bound": 0.5,
    }
    assert report.bins[1].sample_count == 3
    assert report.bins[1].correct_count == 2
    assert report.bins[1].mean_confidence == pytest.approx(2.3 / 3)
    assert report.bins[1].accuracy == pytest.approx(2 / 3)
    assert report.bins[1].absolute_gap == pytest.approx(0.1)

    per_type = {row.document_type: row for row in report.per_type_support}
    assert set(per_type) == set(DOCUMENT_TYPES)
    assert per_type["progress_note"].to_dict() == {
        "correct_count": 1,
        "document_type": "progress_note",
        "predicted_count": 1,
        "support": 2,
    }
    assert per_type["radiology_report"].to_dict() == {
        "correct_count": 1,
        "document_type": "radiology_report",
        "predicted_count": 2,
        "support": 1,
    }
    assert per_type["pathology_report"].support == 1
    assert per_type["pathology_report"].predicted_count == 0


def test_abstention_rates_are_deterministic_and_unknown_always_abstains() -> None:
    report = build_doctype_calibration_report(
        _samples(),
        model="synthetic-doctype-v1",
        num_bins=2,
        abstention_thresholds=(0.9, 0.5, 0.7, 0.5),
    )

    assert [row.threshold for row in report.abstention] == [0.5, 0.7, 0.9]
    assert [row.abstained_count for row in report.abstention] == [1, 2, 3]
    assert [row.abstention_rate for row in report.abstention] == [0.25, 0.5, 0.75]
    assert [row.retained_count for row in report.abstention] == [3, 2, 1]
    assert [row.retained_correct_count for row in report.abstention] == [2, 1, 1]
    assert [row.retained_accuracy for row in report.abstention] == [
        pytest.approx(2 / 3),
        0.5,
        1.0,
    ]


def test_json_and_markdown_are_stable_counts_only_reports() -> None:
    report = build_doctype_calibration_report(
        _samples(),
        model={"model_id": RAW_MODEL_ID, "revision": "synthetic-v1"},
        num_bins=2,
        abstention_thresholds=(0.5, 0.7),
    )

    first_json = render_doctype_calibration_json(report)
    second_json = render_doctype_calibration_json(report)
    first_markdown = render_doctype_calibration_markdown(report)
    second_markdown = render_doctype_calibration_markdown(report)

    assert first_json == second_json
    assert first_markdown == second_markdown
    assert json.loads(first_json)["artifact_type"] == DOCTYPE_CALIBRATION_ARTIFACT
    assert "## Calibration bins" in first_markdown
    assert "## Abstention" in first_markdown
    assert "## Per-type support" in first_markdown
    for forbidden in (
        RAW_NOTE,
        RAW_FIXTURE_ID,
        RAW_MODEL_ID,
        "fixture_id",
        '"text"',
        "expected_type",
        "predicted_type",
    ):
        assert forbidden not in first_json
        assert forbidden not in first_markdown
    assert report.model_fingerprint.startswith("sha256:")
    assert report.fixture_fingerprint.startswith("sha256:")


def test_model_and_fixture_fingerprints_are_stable() -> None:
    first_model = fingerprint_doctype_model(
        {"revision": "synthetic-v1", "model_id": RAW_MODEL_ID}
    )
    second_model = fingerprint_doctype_model(
        {"model_id": RAW_MODEL_ID, "revision": "synthetic-v1"}
    )
    first_fixture = fingerprint_doctype_fixtures(_samples())
    second_fixture = fingerprint_doctype_fixtures(reversed(_samples()))

    assert first_model == second_model
    assert first_fixture == second_fixture
    assert RAW_MODEL_ID not in first_model
    assert RAW_NOTE not in first_fixture
    assert fingerprint_doctype_model(first_model) == first_model


def test_complete_report_is_independent_of_sample_order() -> None:
    forward = build_doctype_calibration_report(
        _samples(),
        model="synthetic-doctype-v1",
        num_bins=3,
    )
    reverse = build_doctype_calibration_report(
        reversed(_samples()),
        model="synthetic-doctype-v1",
        num_bins=3,
    )

    assert forward.to_json() == reverse.to_json()
    assert forward.to_markdown() == reverse.to_markdown()


def test_report_writers_emit_only_the_rendered_artifacts(tmp_path) -> None:
    report = build_doctype_calibration_report(
        _samples(),
        model="synthetic-doctype-v1",
        num_bins=2,
        abstention_thresholds=(0.5,),
    )
    json_path = report.write_json(tmp_path / "calibration.json")
    markdown_path = report.write_markdown(tmp_path / "calibration.md")

    assert json_path.read_text(encoding="utf-8") == report.to_json() + "\n"
    assert markdown_path.read_text(encoding="utf-8") == report.to_markdown()
    assert RAW_NOTE not in json_path.read_text(encoding="utf-8")
    assert RAW_NOTE not in markdown_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_type", RAW_NOTE),
        ("predicted_type", RAW_NOTE),
        ("confidence", float("nan")),
    ],
)
def test_validation_errors_do_not_echo_rejected_values(
    field: str,
    value: object,
) -> None:
    sample = {
        "expected_type": "progress_note",
        "predicted_type": "progress_note",
        "confidence": 0.5,
    }
    sample[field] = value

    with pytest.raises(ValueError) as captured:
        build_doctype_calibration_report([sample], model="synthetic")

    assert RAW_NOTE not in str(captured.value)


def test_invalid_output_path_is_not_echoed() -> None:
    report = build_doctype_calibration_report(
        _samples(),
        model="synthetic-doctype-v1",
        num_bins=2,
        abstention_thresholds=(0.5,),
    )
    unsafe_path = f"{RAW_NOTE}\0.json"

    with pytest.raises(OSError) as captured:
        report.write_json(unsafe_path)

    assert RAW_NOTE not in str(captured.value)


def test_input_contract_is_bounded_and_uses_no_implicit_string_conversion() -> None:
    hostile = _HostileValue()

    with pytest.raises(TypeError, match="plain dictionaries"):
        DocumentTypeCalibrationSample.from_mapping(hostile)  # type: ignore[arg-type]
    with pytest.raises(
        RuntimeError, match="failed to read calibration samples"
    ) as error:
        build_doctype_calibration_report(hostile, model="synthetic")  # type: ignore[arg-type]
    assert RAW_NOTE not in str(error.value)
    with pytest.raises(ValueError) as error:
        DocumentTypeSupport(
            document_type=RAW_NOTE,
            support=1,
            predicted_count=1,
            correct_count=1,
        )
    assert RAW_NOTE not in str(error.value)
    with pytest.raises(ValueError, match="num_bins"):
        build_doctype_calibration_report(
            _samples(),
            model="synthetic",
            num_bins=0,
        )
    with pytest.raises(ValueError, match="at least one calibration sample"):
        build_doctype_calibration_report([], model="synthetic")
    with pytest.raises(ValueError, match="at least one abstention threshold"):
        build_doctype_calibration_report(
            _samples(),
            model="synthetic",
            abstention_thresholds=(),
        )


class _HostileValue:
    def __iter__(self):
        raise AssertionError(RAW_NOTE)

    def __str__(self) -> str:
        raise AssertionError(RAW_NOTE)
