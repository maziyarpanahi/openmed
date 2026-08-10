"""Focused tests for value-free privacy report linting."""

from __future__ import annotations

import json

import pytest

from openmed.compliance.report_lint import (
    ReportFieldSpec,
    ReportLintError,
    ReportSchema,
    lint_report,
)

_HASH = "sha256:" + ("0" * 64)


def _schema() -> ReportSchema:
    return ReportSchema(
        {
            "report_hash": ReportFieldSpec("hash", required=True),
            "record_count": ReportFieldSpec(
                "count",
                required=True,
                maximum=10_000,
            ),
            "status": ReportFieldSpec(
                "code",
                allowed_values=("failed", "passed"),
                required=True,
            ),
            "summary": ReportFieldSpec(
                "object",
                fields={
                    "checked_count": ReportFieldSpec("count", required=True),
                    "source_hash": ReportFieldSpec("hash", required=True),
                },
            ),
        }
    )


def _valid_report() -> dict[str, object]:
    return {
        "summary": {"source_hash": _HASH, "checked_count": 3},
        "status": "passed",
        "record_count": 3,
        "report_hash": _HASH,
    }


def test_lint_accepts_allowlisted_hash_count_code_and_nested_fields() -> None:
    result = lint_report(_valid_report(), _schema())

    assert result.valid is True
    assert result.findings == ()
    assert result.checked_field_count == 4
    assert result.input_field_count == 4
    assert result.to_dict()["finding_count"] == 0


def test_lint_is_deterministic_for_mapping_order() -> None:
    first = _valid_report()
    second = {
        "report_hash": first["report_hash"],
        "record_count": first["record_count"],
        "status": first["status"],
        "summary": {
            "checked_count": first["summary"]["checked_count"],  # type: ignore[index]
            "source_hash": first["summary"]["source_hash"],  # type: ignore[index]
        },
    }

    assert (
        lint_report(first, _schema()).to_dict()
        == lint_report(
            second,
            _schema(),
        ).to_dict()
    )


def test_lint_classifies_forbidden_shapes_without_echoing_values() -> None:
    sensitive_value = "SYNTHETIC-PRIVATE-VALUE-1047"
    report = {
        "report_hash": sensitive_value,
        "record_count": -1,
        "status": "not-a-schema-code",
        "unapproved_field": sensitive_value,
    }

    result = lint_report(report, _schema())
    rendered = json.dumps(result.to_dict(), sort_keys=True)

    assert result.valid is False
    assert [finding.code for finding in result.findings] == [
        "count_out_of_bounds",
        "invalid_hash",
        "invalid_code",
        "unknown_key",
    ]
    assert sensitive_value not in str(result)
    assert sensitive_value not in rendered
    assert "unapproved_field" not in rendered
    assert all("value" not in finding.to_dict() for finding in result.findings)


def test_lint_rejects_free_text_and_unbounded_code_content() -> None:
    sensitive_value = "SYNTHETIC-NARRATIVE-VALUE"
    schema = ReportSchema(
        {
            "notes": ReportFieldSpec("text", max_length=128),
            "status": ReportFieldSpec(
                "code",
                max_length=4,
                allowed_values=("pass",),
            ),
        }
    )

    result = lint_report(
        {"notes": sensitive_value, "status": "passed"},
        schema,
    )

    assert [finding.code for finding in result.findings] == [
        "forbidden_text",
        "length_exceeded",
    ]
    assert sensitive_value not in str(result)


def test_lint_rejects_noncanonical_hashes_and_boolean_counts() -> None:
    result = lint_report(
        {
            "report_hash": "SYNTHETIC-RAW-HASH",
            "record_count": True,
            "status": "passed",
            "summary": {"source_hash": _HASH, "checked_count": 1},
        },
        _schema(),
    )

    assert [finding.code for finding in result.findings] == [
        "invalid_count",
        "invalid_hash",
    ]


def test_nested_arrays_are_bounded_and_linted_by_item_shape() -> None:
    schema = ReportSchema(
        {
            "digests": ReportFieldSpec(
                "array",
                max_items=2,
                item=ReportFieldSpec("hash"),
            )
        }
    )

    result = lint_report(
        {"digests": [_HASH, "SYNTHETIC-RAW-DIGEST"]},
        schema,
    )

    assert result.valid is False
    assert result.findings[0].code == "invalid_hash"
    assert result.findings[0].path == "$.digests[1]"

    too_many = lint_report({"digests": [_HASH, _HASH, _HASH]}, schema)
    assert [finding.code for finding in too_many.findings] == ["too_many_items"]


def test_strict_lint_raises_a_value_free_error() -> None:
    sensitive_value = "SYNTHETIC-EXCEPTION-VALUE"

    with pytest.raises(ReportLintError) as error:
        lint_report(
            {"report_hash": sensitive_value},
            _schema(),
            strict=True,
        )

    assert sensitive_value not in str(error.value)
    assert error.value.result.valid is False
