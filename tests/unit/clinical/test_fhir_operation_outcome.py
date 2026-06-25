"""Tests for the FHIR R4 OperationOutcome builder (OM-360)."""

from dataclasses import dataclass, field

import pytest

from openmed.clinical.exporters.fhir import (
    OperationOutcomeIssue,
    from_validation_result,
    to_operation_outcome,
)


class TestToOperationOutcome:
    def test_error_and_warning_issues_map_to_outcome(self):
        issues = [
            {
                "severity": "error",
                "code": "required",
                "diagnostics": "Patient.name is required",
                "expression": "Patient.name",
            },
            {
                "severity": "warning",
                "code": "code-invalid",
                "diagnostics": "Unknown LOINC code",
                "expression": ["Observation.code.coding[0]"],
            },
        ]

        outcome = to_operation_outcome(issues)

        assert outcome["resourceType"] == "OperationOutcome"
        assert len(outcome["issue"]) == 2

        first, second = outcome["issue"]
        assert first == {
            "severity": "error",
            "code": "required",
            "diagnostics": "Patient.name is required",
            "expression": ["Patient.name"],
        }
        assert second == {
            "severity": "warning",
            "code": "code-invalid",
            "diagnostics": "Unknown LOINC code",
            "expression": ["Observation.code.coding[0]"],
        }

    def test_empty_issue_list_yields_all_ok_information_outcome(self):
        outcome = to_operation_outcome([])

        assert outcome["resourceType"] == "OperationOutcome"
        assert len(outcome["issue"]) == 1
        issue = outcome["issue"][0]
        assert issue["severity"] == "information"
        assert issue["code"] == "informational"
        assert issue["diagnostics"]

    def test_none_yields_all_ok_outcome(self):
        outcome = to_operation_outcome(None)
        assert outcome["issue"][0]["severity"] == "information"

    def test_single_mapping_is_treated_as_one_issue(self):
        outcome = to_operation_outcome(
            {"severity": "fatal", "code": "exception", "diagnostics": "boom"}
        )
        assert len(outcome["issue"]) == 1
        assert outcome["issue"][0]["severity"] == "fatal"

    def test_dataclass_issue_maps_to_outcome(self):
        outcome = to_operation_outcome(
            [
                OperationOutcomeIssue(
                    severity="error",
                    code="processing",
                    diagnostics="Could not process Bundle.entry[0]",
                    expression="Bundle.entry[0]",
                )
            ]
        )

        assert outcome["issue"][0] == {
            "severity": "error",
            "code": "processing",
            "diagnostics": "Could not process Bundle.entry[0]",
            "expression": ["Bundle.entry[0]"],
        }

    def test_severity_is_case_normalised(self):
        outcome = to_operation_outcome([{"severity": "Warning", "code": "invalid"}])
        assert outcome["issue"][0]["severity"] == "warning"

    def test_non_canonical_severity_alias_is_rejected(self):
        with pytest.raises(ValueError, match="invalid issue severity"):
            to_operation_outcome([{"severity": "info", "code": "informational"}])

    def test_diagnostics_and_expression_are_optional(self):
        outcome = to_operation_outcome([{"severity": "error", "code": "invalid"}])
        issue = outcome["issue"][0]
        assert issue == {"severity": "error", "code": "invalid"}
        assert "diagnostics" not in issue
        assert "expression" not in issue

    def test_expressions_plural_becomes_expression_list(self):
        outcome = to_operation_outcome(
            [
                {
                    "severity": "error",
                    "code": "invalid",
                    "diagnostics": "bad",
                    "expressions": ["Patient.gender", "Patient.name"],
                }
            ]
        )
        issue = outcome["issue"][0]
        assert issue["diagnostics"] == "bad"
        assert issue["expression"] == ["Patient.gender", "Patient.name"]

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="invalid issue severity"):
            to_operation_outcome([{"severity": "critical", "code": "invalid"}])

    def test_invalid_code_raises(self):
        with pytest.raises(ValueError, match="invalid issue code"):
            to_operation_outcome([{"severity": "error", "code": "not-a-code"}])

    def test_missing_severity_raises(self):
        with pytest.raises(ValueError, match="issue 'severity' is required"):
            to_operation_outcome([{"code": "invalid"}])

    def test_missing_code_raises(self):
        with pytest.raises(ValueError, match="issue 'code' is required"):
            to_operation_outcome([{"severity": "error"}])

    def test_unsupported_issue_item_raises(self):
        with pytest.raises(TypeError, match="severity/code fields"):
            to_operation_outcome([123])

    def test_string_issue_item_rejected(self):
        with pytest.raises(TypeError, match="severity/code fields"):
            to_operation_outcome(["something went wrong"])

    def test_string_argument_rejected(self):
        with pytest.raises(TypeError):
            to_operation_outcome("oops")

    def test_legacy_location_input_is_emitted_as_expression(self):
        outcome = to_operation_outcome(
            [
                {
                    "severity": "warning",
                    "code": "structure",
                    "diagnostics": "Deprecated path key",
                    "location": "Patient.name",
                }
            ]
        )

        issue = outcome["issue"][0]
        assert issue["expression"] == ["Patient.name"]
        assert "location" not in issue


@dataclass
class _Issue:
    severity: str
    code: str
    diagnostics: str | None = None
    expression: str | None = None


@dataclass
class _ConformanceResult:
    issues: list = field(default_factory=list)


@dataclass
class _BucketResult:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    information: list = field(default_factory=list)


class TestFromValidationResult:
    def test_adapts_issue_objects(self):
        result = _ConformanceResult(
            issues=[
                _Issue("error", "required", "Missing identifier", "Patient.identifier"),
                _Issue("warning", "code-invalid", "Unmapped code", "Condition.code"),
            ]
        )

        outcome = from_validation_result(result)

        assert [i["severity"] for i in outcome["issue"]] == ["error", "warning"]
        assert [i["code"] for i in outcome["issue"]] == ["required", "code-invalid"]
        assert outcome["issue"][0]["expression"] == ["Patient.identifier"]

    def test_adapts_bucketed_errors_and_warnings(self):
        result = _BucketResult(
            errors=["Patient.name is required"],
            warnings=[
                {"diagnostics": "deprecated field", "expression": "Patient.animal"}
            ],
        )

        outcome = from_validation_result(result)

        error_issue, warning_issue = outcome["issue"]
        assert error_issue["severity"] == "error"
        assert error_issue["code"] == "invalid"
        assert error_issue["diagnostics"] == "Patient.name is required"

        assert warning_issue["severity"] == "warning"
        assert warning_issue["code"] == "invalid"
        assert warning_issue["expression"] == ["Patient.animal"]

    def test_clean_result_yields_all_ok(self):
        outcome = from_validation_result(_ConformanceResult(issues=[]))
        assert len(outcome["issue"]) == 1
        assert outcome["issue"][0]["severity"] == "information"
        assert outcome["issue"][0]["code"] == "informational"

    def test_none_result_yields_all_ok(self):
        outcome = from_validation_result(None)
        assert outcome["issue"][0]["severity"] == "information"

    def test_mapping_result_with_issues_key(self):
        result = {
            "issues": [
                {"severity": "fatal", "code": "structure", "diagnostics": "bad json"}
            ]
        }
        outcome = from_validation_result(result)
        assert outcome["issue"][0] == {
            "severity": "fatal",
            "code": "structure",
            "diagnostics": "bad json",
        }

    def test_mapping_result_with_single_issue_mapping(self):
        result = {
            "issues": {
                "severity": "error",
                "code": "invalid",
                "diagnostics": "Malformed Patient resource",
            }
        }

        outcome = from_validation_result(result)

        assert outcome["issue"] == [
            {
                "severity": "error",
                "code": "invalid",
                "diagnostics": "Malformed Patient resource",
            }
        ]

    def test_bucket_string_is_not_split_into_characters(self):
        outcome = from_validation_result({"errors": "Patient.name is required"})

        assert outcome["issue"] == [
            {
                "severity": "error",
                "code": "invalid",
                "diagnostics": "Patient.name is required",
            }
        ]

    def test_bucket_entries_accept_message_and_path_aliases(self):
        result = {
            "warnings": [
                {
                    "message": "Unmapped code",
                    "path": "Observation.code.coding[0]",
                }
            ]
        }

        outcome = from_validation_result(result)

        assert outcome["issue"][0] == {
            "severity": "warning",
            "code": "invalid",
            "diagnostics": "Unmapped code",
            "expression": ["Observation.code.coding[0]"],
        }
