"""Offline tests for minimum-data clinical agent tool contracts."""

from __future__ import annotations

import json
from typing import Any

import pytest

from openmed.agent.tool_contract_lint import (
    LintReasonCode,
    LintSeverity,
    ToolContractLintReport,
    lint_tool_contract,
)

PURPOSE = "purpose:org.example/care-summary@1.0.0"
OTHER_PURPOSE = "purpose:org.example/research-export@1.0.0"


def _field(
    *,
    purpose: object = PURPOSE,
    minimum_data: object = "required",
    **schema: object,
) -> dict[str, object]:
    return {
        "type": "string",
        "x-openmed-purpose": purpose,
        "x-openmed-minimum-data": minimum_data,
        **schema,
    }


def _contract(
    properties: dict[object, object],
    *,
    required: list[object] | None = None,
    purpose: object = PURPOSE,
) -> dict[str, object]:
    return {
        "type": "object",
        "x-openmed-purpose": purpose,
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def _codes(report: ToolContractLintReport) -> list[LintReasonCode]:
    return [finding.reason_code for finding in report.findings]


def test_minimal_contract_passes_with_stable_value_free_json() -> None:
    schema = _contract(
        {
            "observations": {
                "type": "array",
                "x-openmed-purpose": PURPOSE,
                "x-openmed-minimum-data": "required",
                "items": {
                    "type": "object",
                    "properties": {"code": _field()},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
            "question": _field(),
        }
    )

    first = lint_tool_contract(schema)
    second = lint_tool_contract(schema)

    assert first.passed
    assert first.findings == ()
    assert first.to_json() == second.to_json()
    assert first.to_json() == (
        '{"findings":[],"passed":true,'
        '"schema_version":"openmed.agent.tool_contract_lint.v1"}'
    )


def test_overbroad_field_for_another_purpose_fails_review() -> None:
    schema = _contract(
        {
            "summary": _field(),
            "research_export": _field(purpose=OTHER_PURPOSE),
        }
    )

    report = lint_tool_contract(schema)

    assert not report.passed
    assert [finding.to_dict() for finding in report.findings] == [
        {
            "severity": "error",
            "reason_code": "overbroad_for_purpose",
            "schema_path": ("#/properties/research_export/x-openmed-purpose"),
        }
    ]


@pytest.mark.parametrize(
    ("schema", "expected_path"),
    [
        (
            {
                "type": "object",
                "properties": {"summary": _field()},
                "required": ["summary"],
            },
            "#/x-openmed-purpose",
        ),
        (
            _contract(
                {
                    "summary": {
                        "type": "string",
                        "x-openmed-minimum-data": "required",
                    }
                }
            ),
            "#/properties/summary/x-openmed-purpose",
        ),
    ],
)
def test_missing_purpose_is_reported_at_the_schema_path(
    schema: dict[str, object], expected_path: str
) -> None:
    report = lint_tool_contract(schema)

    assert any(
        finding.reason_code is LintReasonCode.MISSING_PURPOSE
        and finding.schema_path == expected_path
        for finding in report.findings
    )


@pytest.mark.parametrize(
    ("field", "required", "expected_code"),
    [
        (_field(minimum_data="optional"), ["context"], LintReasonCode.OPTIONAL_INPUT),
        (_field(), [], LintReasonCode.OPTIONAL_INPUT),
        (_field(minimum_data="derived"), ["context"], LintReasonCode.DERIVED_INPUT),
    ],
)
def test_optional_and_derived_inputs_fail_review(
    field: dict[str, object],
    required: list[object],
    expected_code: LintReasonCode,
) -> None:
    report = lint_tool_contract(_contract({"context": field}, required=required))

    assert not report.passed
    assert expected_code in _codes(report)
    assert all(finding.severity is LintSeverity.ERROR for finding in report.findings)


def test_missing_and_invalid_minimum_data_annotations_fail_closed() -> None:
    schema = _contract(
        {
            "missing": {
                "type": "string",
                "x-openmed-purpose": PURPOSE,
            },
            "invalid": _field(minimum_data="sometimes"),
        }
    )

    report = lint_tool_contract(schema)

    assert [finding.to_dict() for finding in report.findings] == [
        {
            "severity": "error",
            "reason_code": "invalid_minimum_data",
            "schema_path": "#/properties/invalid/x-openmed-minimum-data",
        },
        {
            "severity": "error",
            "reason_code": "missing_minimum_data",
            "schema_path": "#/properties/missing/x-openmed-minimum-data",
        },
    ]


def test_open_objects_and_schema_combinators_fail_closed() -> None:
    schema = _contract({"summary": _field()})
    del schema["additionalProperties"]
    schema["oneOf"] = [{"properties": {"extra": {"type": "string"}}}]

    report = lint_tool_contract(schema)

    assert [finding.to_dict() for finding in report.findings] == [
        {
            "severity": "error",
            "reason_code": "open_input_object",
            "schema_path": "#/additionalProperties",
        },
        {
            "severity": "error",
            "reason_code": "unsupported_schema_keyword",
            "schema_path": "#/oneOf",
        },
    ]


@pytest.mark.parametrize(
    "purpose",
    ["care summary", "purpose:org.example/care-summary", 42, None],
)
def test_invalid_or_unversioned_purposes_use_a_stable_reason(purpose: object) -> None:
    report = lint_tool_contract(_contract({"summary": _field()}, purpose=purpose))

    assert report.findings[0].to_dict() == {
        "severity": "error",
        "reason_code": "invalid_purpose",
        "schema_path": "#/x-openmed-purpose",
    }


def test_report_never_copies_examples_defaults_or_descriptions() -> None:
    sentinel = "Synthetic Person 8842 secret clinical value"
    schema = _contract(
        {
            "context": _field(
                minimum_data="optional",
                example=sentinel,
                default=sentinel,
                description=sentinel,
            )
        },
        required=[],
    )

    report = lint_tool_contract(schema)
    rendered = report.to_json()

    assert sentinel not in rendered
    assert sentinel not in repr(report)
    assert sentinel not in repr(report.findings[0])
    assert set(json.loads(rendered)) == {"findings", "passed", "schema_version"}


def test_paths_are_json_pointer_escaped_and_findings_are_deterministic() -> None:
    schema = _contract(
        {
            "b": _field(minimum_data="derived"),
            "a/b~c": _field(minimum_data="optional"),
        }
    )

    report = lint_tool_contract(schema)

    assert [finding.schema_path for finding in report.findings] == [
        "#/properties/a~1b~0c",
        "#/properties/b",
    ]


def test_cyclic_python_schema_fails_without_recursing_or_echoing_values() -> None:
    schema = _contract({})
    schema["properties"] = {"cycle": schema}
    schema["required"] = ["cycle"]

    report = lint_tool_contract(schema)

    assert [finding.to_dict() for finding in report.findings] == [
        {
            "severity": "error",
            "reason_code": "invalid_schema",
            "schema_path": "#/properties/cycle",
        }
    ]


@pytest.mark.parametrize(
    "schema",
    [
        None,
        [],
        {"x-openmed-purpose": PURPOSE, "properties": []},
        {
            "x-openmed-purpose": PURPOSE,
            "type": "object",
            "properties": {1: _field()},
        },
        {
            "x-openmed-purpose": PURPOSE,
            "type": "object",
            "properties": {"summary": _field()},
            "required": "summary",
        },
        _contract(
            {
                "summary": {
                    "x-openmed-purpose": PURPOSE,
                    "x-openmed-minimum-data": "required",
                }
            }
        ),
    ],
)
def test_malformed_schemas_return_value_free_invalid_schema_findings(
    schema: Any,
) -> None:
    report = lint_tool_contract(schema)

    assert LintReasonCode.INVALID_SCHEMA in _codes(report)
    assert all(
        set(finding.to_dict()) == {"severity", "reason_code", "schema_path"}
        for finding in report.findings
    )
