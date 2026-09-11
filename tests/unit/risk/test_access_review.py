"""Tests for deterministic, value-free structured access reviews."""

from __future__ import annotations

import itertools
import json

import pytest

from openmed.risk import (
    AccessModeReview,
    AccessReviewReport,
    AccessReviewValidationError,
    WorkflowAccessReview,
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


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, MemoryError])
def test_iterator_errors_are_value_free_and_unchained(error_type: type[Exception]):
    secret = "RAW-SYNTHETIC-PATIENT"

    class ExplodingFields:
        def __iter__(self):
            raise error_type(secret)

    with pytest.raises(AccessReviewValidationError) as exc_info:
        review_structured_access(
            {"triage": ExplodingFields()},
            ["patient_id"],
        )

    assert str(exc_info.value) == "structured access review declarations are invalid"
    assert exc_info.value.__cause__ is None
    assert secret not in str(exc_info.value)


def test_schema_attribute_errors_are_value_free_and_unchained():
    secret = "RAW-SYNTHETIC-PATIENT"

    class ExplodingSchema:
        @property
        def fields(self):
            raise RuntimeError(secret)

    with pytest.raises(AccessReviewValidationError) as exc_info:
        review_structured_access({"triage": ["patient_id"]}, ExplodingSchema())

    assert str(exc_info.value) == "structured access review declarations are invalid"
    assert exc_info.value.__cause__ is None
    assert secret not in str(exc_info.value)


def test_field_and_workflow_iterables_are_bounded():
    with pytest.raises(AccessReviewValidationError):
        review_structured_access(
            {"triage": itertools.repeat("patient_id")},
            ["patient_id"],
        )

    requirement = WorkflowRequirement("triage", read_fields=["patient_id"])
    with pytest.raises(AccessReviewValidationError):
        review_structured_access(
            itertools.repeat(requirement),
            ["patient_id"],
        )


@pytest.mark.parametrize(
    ("workflows", "denied_fields"),
    [
        (
            {"read": ["patient_id"], "read_fields": ["diagnosis"]},
            (),
        ),
        (
            {"triage": {"read": ["patient_id"], "unexpected": ["diagnosis"]}},
            (),
        ),
        (
            {"triage": {"read": ["patient_id"]}},
            {"read": ["diagnosis"], "unexpected": ["patient_id"]},
        ),
    ],
)
def test_ambiguous_or_unknown_policy_fields_are_rejected(workflows, denied_fields):
    with pytest.raises(AccessReviewValidationError):
        review_structured_access(
            workflows,
            ["patient_id", "diagnosis"],
            denied_fields=denied_fields,
        )


def test_public_report_types_reject_contradictory_findings():
    with pytest.raises(AccessReviewValidationError):
        AccessModeReview(
            mode="read",
            requested_fields=("patient_id",),
            available_fields=("patient_id",),
            allowed_fields=("diagnosis",),
            missing_fields=(),
            excessive_fields=(),
            denied_fields=(),
        )

    read = AccessModeReview(
        mode="read",
        requested_fields=("patient_id",),
        available_fields=("patient_id",),
        allowed_fields=("patient_id",),
        missing_fields=(),
        excessive_fields=(),
        denied_fields=(),
    )
    export = AccessModeReview(
        mode="export",
        requested_fields=(),
        available_fields=("patient_id",),
        allowed_fields=(),
        missing_fields=(),
        excessive_fields=("patient_id",),
        denied_fields=(),
    )
    workflow = WorkflowAccessReview("triage", read=read, export=export)

    with pytest.raises(AccessReviewValidationError):
        AccessReviewReport(
            resource_fields=("diagnosis",),
            workflows=(workflow,),
        )
