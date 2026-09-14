"""Tests for the PHI-safe clinical NLI error-slice report."""

from __future__ import annotations

import json

import pytest

from openmed.eval import (
    CLINICAL_NLI_PHENOMENA,
    NLIErrorSliceCase,
    NLIErrorSliceReport,
    build_nli_error_slice_report,
    render_nli_error_slice_report_json,
    render_nli_error_slice_report_markdown,
)


def _records() -> list[dict[str, object]]:
    return [
        {
            "fixture_id": "nli-negation-001",
            "phenomena": ["negation"],
            "gold_label": "entailment",
            "predicted_label": "entailment",
        },
        {
            "fixture_id": "nli-negation-002",
            "phenomena": ["negation", "temporality"],
            "gold_label": "contradiction",
            "predicted_label": "neutral",
        },
        {
            "fixture_id": "nli-context-003",
            "phenomena": ["temporality", "experiencer"],
            "gold_label": "neutral",
            "abstained": True,
            "predicted_label": None,
        },
        {
            "fixture_id": "nli-context-004",
            "phenomena": ["experiencer", "numbers"],
            "gold_label": "entailment",
            "predicted_label": "contradiction",
        },
        {
            "fixture_id": "nli-medication-005",
            "phenomena": ["numbers", "medication_status"],
            "gold_label": "contradiction",
            "predicted_label": "contradiction",
        },
        {
            "fixture_id": "nli-medication-006",
            "phenomena": ["medication_status"],
            "gold_label": "entailment",
            "predicted_label": "abstain",
        },
    ]


def test_report_has_each_predefined_slice_and_hand_counted_metrics() -> None:
    report = build_nli_error_slice_report(
        _records(),
        fixture_set_id="synthetic-nli-v1",
        model_id="synthetic-model-v1",
    )

    assert tuple(report.slices) == CLINICAL_NLI_PHENOMENA
    assert report.fixture_count == 6

    negation = report.slices["negation"]
    assert negation.fixture_ids == ("nli-negation-001", "nli-negation-002")
    assert negation.confusion_matrix["entailment"]["entailment"] == 1
    assert negation.confusion_matrix["contradiction"]["neutral"] == 1
    assert negation.correct_count == 1
    assert negation.error_count == 1
    assert negation.abstention_coverage == 1.0

    temporality = report.slices["temporality"]
    assert temporality.abstained_count == 1
    assert temporality.scored_count == 1
    assert temporality.abstention_coverage == 0.5
    assert temporality.confusion_matrix["neutral"]["abstain"] == 1

    assert report.summary == {
        "abstained_count": 3,
        "abstention_coverage": 0.7,
        "abstention_rate": 0.3,
        "correct_count": 3,
        "error_count": 4,
        "scored_count": 7,
        "slice_assignment_count": 10,
        "unique_fixture_count": 6,
    }


def test_input_order_does_not_change_json_or_markdown() -> None:
    first = build_nli_error_slice_report(_records())
    second = build_nli_error_slice_report(reversed(_records()))

    assert first.to_json() == second.to_json()
    assert first.to_markdown() == second.to_markdown()
    assert render_nli_error_slice_report_json(first) == first.to_json()
    assert render_nli_error_slice_report_markdown(first) == first.to_markdown()


def test_report_suppresses_raw_fields_but_retains_opaque_fixture_ids() -> None:
    rows = _records()
    rows[0] = {
        **rows[0],
        "premise": "raw-source-sentinel-premise",
        "hypothesis": "raw-source-sentinel-hypothesis",
        "examples": ["raw-source-sentinel-example"],
    }
    report = build_nli_error_slice_report(rows)
    serialized = report.to_json() + report.to_markdown()
    payload = json.loads(report.to_json())

    assert "raw-source-sentinel" not in serialized
    assert "nli-negation-001" in serialized
    assert "premise" not in payload
    assert "hypothesis" not in payload
    assert "examples" not in payload
    assert payload["provenance"]["raw_text_suppressed"] is True
    assert payload["provenance"]["examples_suppressed"] is True
    assert payload["provenance"]["network_required"] is False


def test_mapping_aliases_and_abstention_are_normalized() -> None:
    case = NLIErrorSliceCase.from_mapping(
        {
            "id": "nli-alias-001",
            "slice": "medication",
            "gold": "entail",
            "prediction": "abstention",
        }
    )

    assert case.to_dict() == {
        "abstained": True,
        "fixture_id": "nli-alias-001",
        "gold_label": "entailment",
        "phenomena": ["medication_status"],
        "predicted_label": "abstain",
    }
    assert (
        build_nli_error_slice_report(case).slices["medication_status"].abstained_count
        == 1
    )


def test_round_trip_rehydrates_only_safe_aggregate_evidence() -> None:
    report = build_nli_error_slice_report(_records(), model_id="synthetic-model-v1")
    restored = NLIErrorSliceReport.from_dict(report.to_dict())

    assert restored.to_dict() == report.to_dict()
    assert restored.provenance.model_digest.startswith("sha256:")


def test_invalid_input_errors_do_not_echo_sensitive_values() -> None:
    sensitive_value = "raw clinical note with an invented identifier"

    with pytest.raises(ValueError) as captured:
        build_nli_error_slice_report(
            [
                {
                    "fixture_id": sensitive_value,
                    "phenomena": ["negation"],
                    "gold_label": "entailment",
                    "predicted_label": "neutral",
                }
            ]
        )

    assert sensitive_value not in str(captured.value)


def test_duplicate_fixture_ids_and_unknown_slices_fail_closed() -> None:
    rows = _records()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="fixture identifiers"):
        build_nli_error_slice_report(rows)

    invalid = dict(_records()[0])
    invalid["phenomena"] = ["unsupported"]
    with pytest.raises(ValueError, match="predefined"):
        build_nli_error_slice_report([invalid])


def test_empty_report_is_stable_and_contains_zero_coverage_slices() -> None:
    report = build_nli_error_slice_report([])

    assert report.fixture_count == 0
    assert report.summary["abstention_coverage"] == 0.0
    assert all(
        slice_report.fixture_count == 0 for slice_report in report.slices.values()
    )
    assert report.to_json() == report.to_json()


def test_report_writes_json_and_markdown_without_network_access(tmp_path) -> None:
    report = build_nli_error_slice_report(_records())
    json_path = report.write_json(tmp_path / "nli-error-slices.json")
    markdown_path = report.write_markdown(tmp_path / "nli-error-slices.md")

    assert json_path.read_text(encoding="utf-8") == report.to_json() + "\n"
    assert markdown_path.read_text(encoding="utf-8") == report.to_markdown()
