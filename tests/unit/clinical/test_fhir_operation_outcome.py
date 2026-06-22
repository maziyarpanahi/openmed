"""Tests for the FHIR R4 OperationOutcome builder (OM-360)."""

from dataclasses import dataclass, field

import pytest

from openmed.clinical.exporters.fhir import (
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

    def test_string_issue_becomes_diagnostics(self):
        outcome = to_operation_outcome(["something went wrong"])
        issue = outcome["issue"][0]
        assert issue["severity"] == "error"
        assert issue["code"] == "processing"
        assert issue["diagnostics"] == "something went wrong"

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

    def test_string_argument_rejected(self):
        with pytest.raises(TypeError):
            to_operation_outcome("oops")


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
