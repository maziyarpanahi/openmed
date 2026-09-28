"""Offline acceptance tests for the clinical quality profiler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.interop.athena import load_athena_vocab
from openmed.structured.quality import (
    QualityGateError,
    enforce_completeness_floor,
    profile_jsonl,
    profile_results,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures" / "quality" / "profiler_batch.jsonl"
ATHENA = ROOT / "fixtures" / "quality" / "athena"


def _profile_fixture():
    return profile_jsonl(
        FIXTURE,
        athena_index=load_athena_vocab(ATHENA),
        required_fields=("condition", "drug", "measurement"),
    )


def test_grounding_coverage_matches_hand_computed_fixture_counts() -> None:
    report = _profile_fixture()

    assert report["grounding"] == {
        "total_spans": 6,
        "grounded_spans": 5,
        "ungrounded_spans": 1,
        "coverage": pytest.approx(5 / 6),
        "grounded_rate": pytest.approx(5 / 6),
        "by_domain": {
            "condition": {
                "total": 2,
                "grounded": 1,
                "ungrounded": 1,
                "coverage": pytest.approx(0.5),
                "grounded_rate": pytest.approx(0.5),
                "passed": False,
            },
            "drug": {
                "total": 2,
                "grounded": 2,
                "ungrounded": 0,
                "coverage": 1.0,
                "grounded_rate": 1.0,
                "passed": True,
            },
            "measurement": {
                "total": 2,
                "grounded": 2,
                "ungrounded": 0,
                "coverage": 1.0,
                "grounded_rate": 1.0,
                "passed": True,
            },
        },
    }
    assert report["completeness"]["per_note"] == [
        {
            "completeness_score": 1.0,
            "field_count": 3,
            "grounded_rate": 1.0,
            "grounded_span_count": 3,
            "missing_required_field_count": 0,
            "missing_required_fields": [],
            "missing_required_rate": 0.0,
            "note_index": 0,
            "null_density": 0.0,
            "null_field_count": 0,
            "passed": True,
            "required_field_count": 3,
            "span_count": 3,
            "ungrounded_rate": 0.0,
            "ungrounded_span_count": 0,
        },
        {
            "completeness_score": pytest.approx(2 / 3),
            "field_count": 3,
            "grounded_rate": pytest.approx(2 / 3),
            "grounded_span_count": 2,
            "missing_required_field_count": 0,
            "missing_required_fields": [],
            "missing_required_rate": 0.0,
            "note_index": 1,
            "null_density": 0.0,
            "null_field_count": 0,
            "passed": True,
            "required_field_count": 3,
            "span_count": 3,
            "ungrounded_rate": pytest.approx(1 / 3),
            "ungrounded_span_count": 1,
        },
    ]


def test_plausibility_flags_each_seeded_invalid_value_and_date() -> None:
    report = _profile_fixture()

    findings = report["plausibility"]["findings"]
    assert findings == [
        {
            "kind": "date",
            "path": "record[1].note_date",
            "reason": "out_of_range_or_invalid",
        },
        {
            "kind": "measurement",
            "path": "record[1].span[2]",
            "reason": "outside_reference_range",
        },
    ]
    assert report["plausibility"]["invalid_count"] == 2
    assert report["plausibility"]["checks"] == [
        {"name": "date_range", "passed": False, "invalid_count": 1},
        {"name": "laboratory_values", "passed": False, "invalid_count": 1},
        {"name": "vital_signs", "passed": True, "invalid_count": 0},
    ]


def test_field_level_measurements_and_dates_use_the_same_normalizers() -> None:
    report = profile_results(
        [
            {
                "fields": {
                    "measurement": {
                        "value": 1000,
                        "unit": "mg/dL",
                        "reference_range": "70-99 mg/dL",
                    }
                },
                "timing": {"measurement_date": "2200-01-01"},
            }
        ]
    )

    assert report["plausibility"]["invalid_count"] == 2
    assert {finding["kind"] for finding in report["plausibility"]["findings"]} == {
        "date",
        "measurement",
    }


def test_nested_span_dates_are_checked_without_emitting_the_date_value() -> None:
    report = profile_results(
        [
            {
                "entities": [
                    {
                        "label": "condition",
                        "start": 0,
                        "end": 1,
                        "date": "2200-01-01",
                    }
                ]
            }
        ]
    )

    assert report["plausibility"]["findings"] == [
        {
            "kind": "date",
            "path": "record[0].entities[0].date",
            "reason": "out_of_range_or_invalid",
        }
    ]
    assert "2200-01-01" not in report.to_json()


def test_vital_sign_normalizer_flags_all_and_only_seeded_invalid_values() -> None:
    report = profile_results(
        [
            {
                "note_text": "HR 72",
                "entities": [
                    {"label": "vital_sign", "text": "HR 72", "start": 0, "end": 5}
                ],
            },
            {
                "note_text": "HR 400",
                "entities": [
                    {
                        "label": "vital_sign",
                        "text": "HR 400",
                        "start": 0,
                        "end": 6,
                    }
                ],
            },
        ]
    )

    assert report["plausibility"]["findings"] == [
        {
            "kind": "vital_sign",
            "path": "record[1].span[0]",
            "reason": "outside_plausible_range",
        }
    ]


def test_report_is_deterministic_phi_free_and_offset_based() -> None:
    first = _profile_fixture()
    second = _profile_fixture()

    assert first.to_json() == second.to_json()
    serialized = first.to_json()
    for raw_marker in ("diabetes", "metformin", "aspirin", "glucose"):
        assert raw_marker not in serialized
    assert "record[1].note_date" in serialized
    assert "record[1].span[2]" in serialized
    assert "note_text" not in serialized


def test_per_field_completeness_exposes_null_and_missing_required_rates() -> None:
    report = profile_results(
        [
            {"fields": {"condition": {"concept_id": 1}}},
            {"fields": {"condition": None}},
        ],
        required_fields=("condition",),
    )

    condition = report["completeness"]["per_field"]["condition"]
    assert condition["note_total"] == 2
    assert condition["present"] == 1
    assert condition["null"] == 1
    assert condition["null_density"] == pytest.approx(0.5)
    assert condition["missing_required"] == 1
    assert condition["missing_required_rate"] == pytest.approx(0.5)
    assert report["completeness"]["per_note"][1]["passed"] is False


def test_per_field_null_density_counts_absent_fields_across_the_batch() -> None:
    report = profile_results(
        [
            {"fields": {"condition": {"concept_id": 1}}},
            {"fields": {"drug": {"concept_id": 2}}},
        ]
    )

    condition = report["completeness"]["per_field"]["condition"]
    assert condition["note_total"] == 2
    assert condition["present"] == 1
    assert condition["null"] == 1
    assert condition["null_density"] == pytest.approx(0.5)


def test_completeness_floor_is_a_pipeline_gate() -> None:
    records = [{"required_fields": ["condition"], "condition": None}]
    report = profile_results(
        records, required_fields=("condition",), completeness_floor=0.5
    )

    assert report.status == "fail"
    assert report.passed is False
    assert report["gate"] == {
        "completeness_floor": 0.5,
        "observed_completeness": 0.0,
        "passed": False,
    }
    with pytest.raises(QualityGateError) as error:
        enforce_completeness_floor(records, 0.5, required_fields=("condition",))
    assert error.value.report.status == "fail"
    assert "condition" not in str(error.value)


def test_span_conformance_reports_overlap_without_raw_text() -> None:
    report = profile_results(
        [
            {
                "note_text": "synthetic text",
                "entities": [
                    {"label": "condition", "text": "synthetic", "start": 0, "end": 9},
                    {"label": "drug", "text": "the", "start": 1, "end": 4},
                ],
            }
        ]
    )

    assert report["conformance"]["overlap_count"] == 1
    assert report["conformance"]["passed"] is False
    serialized = json.dumps(report.to_dict(), sort_keys=True)
    assert "synthetic text" not in serialized
    assert '"start": 0' in serialized
    assert '"end": 9' in serialized
