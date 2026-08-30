"""Focused tests for privacy-safe form extraction and FHIR SDC output."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.interop.fhir.sdc import (
    SDC_QUESTIONNAIRE_RESPONSE_PROFILE,
    to_questionnaire,
    to_questionnaire_response,
    validate_questionnaire_response,
)
from openmed.structured.forms import (
    extract_form_fields,
    render_review_html,
    render_review_json,
)

FIXTURE = (
    Path(__file__).parents[2] / "fixtures" / "documents" / "synthetic_form_blocks.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_form_values_keep_provenance_and_flag_ambiguous_matches() -> None:
    source = _fixture()
    source["pages"][0]["blocks"].extend(
        [
            {
                "text": "Review status: pending",
                "page": 0,
                "bbox": [40, 150, 220, 160],
            },
            {
                "text": "Review status: complete",
                "page": 0,
                "bbox": [40, 170, 220, 180],
            },
        ]
    )
    result = extract_form_fields(source)

    field = next(field for field in result.fields if field.link_id == "field")
    assert field.redacted_value == "[IDENTIFIER]"
    assert field.value == "SYNTH-FIELD-01"
    assert field.provenance is not None
    assert field.provenance.page == 0
    assert result.review_required

    repeated = [field for field in result.fields if field.link_id == "review_status"]
    assert len(repeated) == 2
    assert all(field.review_required for field in repeated)
    assert all("ambiguous repeated field label" in field.warnings for field in repeated)


def test_review_artifacts_are_redacted_and_fhir_response_validates() -> None:
    result = extract_form_fields(
        _fixture(),
        schema=[
            {"linkId": "field", "text": "Field", "type": "string"},
            {"linkId": "value", "text": "Value", "type": "decimal"},
        ],
    )
    review_json = render_review_json(result)
    review_html = render_review_html(result)
    assert "SYNTH-FIELD-01" not in review_json
    assert "SYNTH-FIELD-01" not in review_html
    assert "[IDENTIFIER]" in review_json

    questionnaire = to_questionnaire(result)
    response = to_questionnaire_response(result, questionnaire=questionnaire)
    serialized = json.dumps(response, sort_keys=True)
    assert "SYNTH-FIELD-01" not in serialized
    assert SDC_QUESTIONNAIRE_RESPONSE_PROFILE in response["meta"]["profile"]
    assert validate_questionnaire_response(response, questionnaire) == []


def test_configured_detector_and_transformer_are_applied_before_serialization() -> None:
    result = extract_form_fields(
        [{"text": "Comment: Alice called 555-0100", "page": 2, "bbox": (1, 2, 3, 4)}],
        pii_detector=lambda value: [
            {
                "start": value.index("Alice"),
                "end": value.index("Alice") + 5,
                "label": "NAME",
            }
        ],
        transformer=lambda value: f"<masked:{len(value)}>",
    )

    field = result.fields[0]
    assert field.redacted_value == "<masked:5> called <masked:8>"
    response = to_questionnaire_response(result)
    assert "Alice" not in json.dumps(response)
    assert "555-0100" not in json.dumps(response)
