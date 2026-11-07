"""Focused tests for the counts-only tabular schema-drift gate."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from openmed.risk import (
    SchemaContract,
    SchemaDriftError,
    SchemaDriftReport,
    SchemaField,
    compare_schema_drift,
    enforce_schema_contract,
)


class _ExplodingMapping(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise RuntimeError("synthetic-sensitive-exception")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("synthetic-sensitive-exception")

    def __len__(self) -> int:
        return 1


def _contract() -> SchemaContract:
    return SchemaContract(
        version="v1",
        columns=(
            SchemaField(
                "subject_token",
                "string",
                role="direct_identifier",
                field_id="subject",
            ),
            SchemaField(
                "age_band",
                "integer",
                role="quasi_identifier",
                field_id="age",
            ),
            SchemaField(
                "measure",
                "float64",
                nullable=True,
                role="sensitive",
                field_id="measure",
            ),
            SchemaField(
                "source_system",
                "string",
                role="non_sensitive",
                field_id="source",
            ),
        ),
    )


def test_matching_schema_is_deterministic_and_counts_only() -> None:
    contract = _contract()

    first = compare_schema_drift(contract, list(contract.columns))
    second = compare_schema_drift(contract, list(reversed(contract.columns)))

    assert first == second
    assert first.passed is True
    assert first.has_drift is False
    assert first.counts == {
        "added": 0,
        "removed": 0,
        "renamed": 0,
        "type_changed": 0,
        "nullability_changed": 0,
        "role_changed": 0,
        "unsafe_role_drift": 0,
        "unsafe_schema_drift": 0,
        "version_mismatch": 0,
    }
    assert json.loads(first.to_json()) == first.to_dict()


def test_schema_changes_are_classified_without_exposing_column_names() -> None:
    contract = _contract()
    incoming = [
        SchemaField(
            "subject_reference",
            "string",
            role="direct_identifier",
            field_id="subject",
        ),
        SchemaField(
            "measure",
            "float64",
            nullable=True,
            role="sensitive",
            field_id="measure",
        ),
        SchemaField(
            "source_system",
            "integer",
            nullable=True,
            role="excluded",
            field_id="source",
        ),
        SchemaField(
            "new_signal",
            "string",
            role="non_sensitive",
            field_id="new-signal",
        ),
    ]

    report = compare_schema_drift(contract, incoming)

    assert report.counts == {
        "added": 1,
        "removed": 1,
        "renamed": 1,
        "type_changed": 1,
        "nullability_changed": 1,
        "role_changed": 1,
        "unsafe_role_drift": 1,
        "unsafe_schema_drift": 1,
        "version_mismatch": 0,
    }
    assert report.release_blocked is True
    evidence = report.to_json()
    assert "subject_reference" not in evidence
    assert "age_band" not in evidence
    assert "new_signal" not in evidence


def test_role_drift_blocks_when_it_enters_or_leaves_a_protected_role() -> None:
    contract = SchemaContract(
        "v1",
        (SchemaField("subject_token", "string", role="direct_identifier"),),
    )
    incoming = (SchemaField("subject_token", "string", role="non_sensitive"),)

    report = compare_schema_drift(contract, incoming)

    assert report.role_changed == 1
    assert report.unsafe_role_drift == 1
    assert report.unsafe_schema_drift == 0
    assert report.release_blocked is True


def test_safe_role_drift_is_reported_but_does_not_block() -> None:
    contract = SchemaContract(
        "v1",
        (SchemaField("source_system", "string", role="non_sensitive"),),
    )
    incoming = (SchemaField("source_system", "string", role="excluded"),)

    report = enforce_schema_contract(contract, incoming)

    assert report.role_changed == 1
    assert report.unsafe_role_drift == 0
    assert report.release_blocked is False


def test_protected_type_and_nullability_drift_blocks() -> None:
    contract = SchemaContract(
        "v1",
        (SchemaField("measure", "float64", role="sensitive"),),
    )
    incoming = (SchemaField("measure", "string", nullable=True, role="sensitive"),)

    report = compare_schema_drift(contract, incoming)

    assert report.type_changed == 1
    assert report.nullability_changed == 1
    assert report.unsafe_schema_drift == 1
    assert report.release_blocked is True


def test_version_mismatch_and_enforcement_error_are_counts_only() -> None:
    contract = SchemaContract.from_mapping(
        {
            "schema_version": "v1",
            "columns": {
                "patient_name": {
                    "type": "string",
                    "role": "direct_identifier",
                    "field_id": "private-field",
                }
            },
        }
    )
    incoming = {
        "schema_version": "v2",
        "columns": [
            {
                "name": "patient_name",
                "type": "string",
                "role": "direct_identifier",
                "field_id": "private-field",
            }
        ],
    }

    report = compare_schema_drift(contract, incoming)
    assert report.version_match is False
    assert report.counts["version_mismatch"] == 1
    assert report.release_blocked is True
    evidence = report.to_json()
    assert "patient_name" not in evidence
    assert "private-field" not in evidence

    with pytest.raises(SchemaDriftError) as exc_info:
        enforce_schema_contract(contract, incoming)

    assert exc_info.value.report == report
    assert "patient_name" not in str(exc_info.value)
    assert "private-field" not in str(exc_info.value)


def test_report_never_serializes_the_raw_contract_version() -> None:
    raw_version = "synthetic-sensitive-contract-version"
    contract = SchemaContract(
        raw_version,
        (SchemaField("subject_token", "string", role="direct_identifier"),),
    )
    incoming = {
        "version": "v2",
        "columns": [{"name": "subject_token", "type": "string", "role": "direct"}],
    }

    report = compare_schema_drift(contract, incoming)
    payload = report.to_dict()

    assert set(payload) == {
        "report_schema_version",
        "version_match",
        "release_blocked",
        "counts",
    }
    assert raw_version not in report.to_json()
    assert raw_version not in repr(report)
    with pytest.raises(SchemaDriftError) as exc_info:
        report.raise_if_blocked()
    assert raw_version not in str(exc_info.value)


def test_conflicting_stable_ids_are_not_matched_by_column_name() -> None:
    contract = SchemaContract(
        "v1",
        (
            SchemaField(
                "subject_token",
                "string",
                role="direct_identifier",
                field_id="old-id",
            ),
        ),
    )
    incoming = (
        SchemaField(
            "subject_token",
            "string",
            role="direct_identifier",
            field_id="new-id",
        ),
    )

    report = compare_schema_drift(contract, incoming)

    assert report.added == 1
    assert report.removed == 1
    assert report.unsafe_role_drift == 2
    assert report.release_blocked is True


def test_input_schema_is_closed_bounded_and_failure_text_is_sanitized() -> None:
    contract = _contract()
    invalid_inputs = (
        {
            "columns": [
                {
                    "name": "subject_token",
                    "type": "string",
                    "role": "direct_identifier",
                    "sample": "synthetic-sensitive-value",
                }
            ]
        },
        {
            "columns": [
                {
                    "name": "subject_token",
                    "column": "alternate",
                    "type": "string",
                }
            ]
        },
        _ExplodingMapping(),
    )

    for incoming in invalid_inputs:
        with pytest.raises(ValueError) as exc_info:
            compare_schema_drift(contract, incoming)
        error = str(exc_info.value)
        assert "synthetic-sensitive-value" not in error
        assert "synthetic-sensitive-exception" not in error

    with pytest.raises(ValueError, match="field limit"):
        SchemaContract("v1", tuple(contract.columns) * 1025)


def test_public_report_rejects_inconsistent_release_decisions() -> None:
    with pytest.raises(ValueError, match="release decision"):
        SchemaDriftReport(
            version_match=True,
            added=0,
            removed=0,
            renamed=0,
            type_changed=0,
            nullability_changed=0,
            role_changed=1,
            unsafe_role_drift=1,
            unsafe_schema_drift=0,
            release_blocked=False,
        )
