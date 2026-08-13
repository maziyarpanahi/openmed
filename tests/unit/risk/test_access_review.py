"""Tests for deterministic, value-free structured access reviews."""

from __future__ import annotations

import json

import pytest

from openmed.risk import (
    AccessReviewReport,
    AccessReviewValidationError,
    WorkflowRequirement,
    render_access_review,
    review_structured_access,
)


def _review() -> AccessReviewReport:
    return review_structured_access(
        {
            "triage": {
                "read": {"patient_id", "age", "missing_field"},
                "export": {"patient_id", "diagnosis"},
            }
        },
        {
            "properties": {
                "patient_id": {
                    "type": "string",
                    "example": "PATIENT-0007",
                },
                "age": {"type": "integer"},
                "diagnosis": {
                    "type": "string",
                    "example": "PRIVATE-DIAGNOSIS",
                },
                "unused": {"default": "RAW-SCHEMA-VALUE"},
            }
        },
        denied_fields={"diagnosis"},
    )


def test_review_reports_missing_excessive_and_denied_fields() -> None:
    report = _review()
    triage = report.workflow("triage")

    assert triage.read.missing_fields == ("missing_field",)
    assert triage.read.denied_fields == ()
    assert triage.read.excessive_fields == ("diagnosis", "unused")
    assert triage.read.complete is False
    assert triage.export.missing_fields == ()
    assert triage.export.denied_fields == ("diagnosis",)
    assert triage.export.excessive_fields == ("age", "unused")
    assert report.missing_fields == ("missing_field",)
    assert report.denied_fields == ("diagnosis",)
    assert report.complete is False


def test_report_is_deterministic_and_never_copies_schema_values() -> None:
    first = _review()
    second = review_structured_access(
        {
            "triage": {
                "export": ["diagnosis", "patient_id"],
                "read": ["missing_field", "patient_id", "age"],
            }
        },
        {
            "unused": {"default": "RAW-SCHEMA-VALUE"},
            "diagnosis": {"example": "PRIVATE-DIAGNOSIS"},
            "age": {"type": "integer"},
            "patient_id": {"example": "PATIENT-0007"},
        },
        denied_fields=["diagnosis"],
    )

    first_json = first.to_json()
    assert first_json == second.to_json()
    assert json.loads(first_json) == first.to_dict()
    for value in ("PATIENT-0007", "PRIVATE-DIAGNOSIS", "RAW-SCHEMA-VALUE"):
        assert value not in first_json
    assert "patient_id" in first_json


def test_markdown_renders_each_access_mode_without_values() -> None:
    report = _review()
    markdown = render_access_review(report)

    assert "### Workflow `triage`" in markdown
    assert "`missing_field`" in markdown
    assert "`diagnosis`" in markdown
    assert "Missing fields" in markdown
    assert "Excessive fields" in markdown
    assert "Denied fields" in markdown
    assert "PATIENT-0007" not in markdown
    assert "PRIVATE-DIAGNOSIS" not in markdown
    assert "RAW-SCHEMA-VALUE" not in markdown


def test_accepts_json_schema_and_explicit_workflow_requirements() -> None:
    report = review_structured_access(
        [
            WorkflowRequirement(
                "exporter",
                read_fields=["patient_id"],
                export_fields=["diagnosis"],
            )
        ],
        {"properties": {"patient_id": {}, "diagnosis": {}}},
        denied_fields={"export": ["diagnosis"]},
    )

    assert report.workflow("exporter").read.allowed_fields == ("patient_id",)
    assert report.workflow("exporter").read.complete is False
    assert report.workflow("exporter").export.denied_fields == ("diagnosis",)
    assert report.to_dict()["policy_denied_fields"] == ["diagnosis"]


def test_invalid_identifiers_are_rejected_without_echoing_input() -> None:
    secret = "SENSITIVE-RAW-VALUE"

    with pytest.raises(AccessReviewValidationError) as error:
        review_structured_access(
            {"triage": [secret + " with spaces"]},
            ["patient_id"],
        )

    assert secret not in str(error.value)


def test_public_report_payload_contains_no_mapping_metadata() -> None:
    report = review_structured_access(
        {"reader": {"read": {"patient_id"}}},
        {"patient_id": {"description": "RAW-PATIENT-DESCRIPTION"}},
    )

    payload = report.to_dict()
    assert payload["resource_fields"] == ["patient_id"]
    assert "RAW-PATIENT-DESCRIPTION" not in json.dumps(payload)
