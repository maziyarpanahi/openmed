"""Focused tests for deterministic, value-free schema snapshot diffs."""

from __future__ import annotations

import json

import pytest

from openmed.structured.schema_snapshot import (
    SchemaSnapshot,
    compare_schema_snapshots,
    is_schema_compatible,
)


def test_snapshot_comparison_is_order_independent_and_reports_field_metadata() -> None:
    before = SchemaSnapshot(
        version="1.0.0",
        fields={
            "encounter.count": {"type": "int", "optional": True},
            "subject.identifier": {"type": "string", "optional": False},
        },
    )
    after = SchemaSnapshot(
        version="1.1.0",
        fields={
            "subject.identifier": {"type": "string", "optional": False},
            "encounter.note": {"type": "string", "optional": True},
            "encounter.count": {"type": "number", "optional": True},
        },
    )

    report = compare_schema_snapshots(before, after)

    assert [change.path for change in report.additions] == ["encounter.note"]
    assert report.removals == ()
    assert [change.path for change in report.changes] == ["encounter.count"]
    assert report.changes[0].reason == "type_widened"
    assert report.changes[0].before_type == "integer"
    assert report.changes[0].after_type == "number"
    assert report.compatible is True
    assert report.has_breaking_changes is False

    reversed_before = SchemaSnapshot(
        version="1.0.0",
        fields={
            "subject.identifier": {"type": "string", "optional": False},
            "encounter.count": {"type": "int", "optional": True},
        },
    )
    assert before.to_dict() == reversed_before.to_dict()


def test_required_addition_and_removal_fail_without_a_major_version_bump() -> None:
    before = SchemaSnapshot(
        version="2.2.0",
        fields={
            "subject.identifier": {"type": "string", "optional": False},
            "encounter.note": {"type": "string", "optional": True},
        },
    )
    after = SchemaSnapshot(
        version="2.3.0",
        fields={
            "encounter.count": {"type": "integer", "optional": False},
        },
    )

    report = compare_schema_snapshots(before, after)

    assert [change.path for change in report.additions] == ["encounter.count"]
    assert [change.path for change in report.removals] == [
        "encounter.note",
        "subject.identifier",
    ]
    assert {change.path for change in report.incompatible_changes} == {
        "encounter.count",
        "encounter.note",
        "subject.identifier",
    }
    assert report.compatible is False
    assert report.violations == ("breaking_change_requires_major_version_bump",)


def test_major_version_bump_records_breaking_changes_but_satisfies_rules() -> None:
    before = SchemaSnapshot(
        version="3.4.1",
        fields={"encounter.measurement": {"type": "string", "optional": False}},
    )
    after = SchemaSnapshot(
        version="4.0.0",
        fields={"encounter.measurement": {"type": "number", "optional": False}},
    )

    report = compare_schema_snapshots(before, after)

    assert report.compatible is True
    assert report.version_bump_satisfies_rules is True
    assert report.has_breaking_changes is True
    assert report.breaking_changes[0].reason == "type_changed"
    assert report.violations == ()


def test_optionality_and_nullable_type_changes_are_explicit() -> None:
    old = SchemaSnapshot(
        version="1.0.0",
        fields={
            "encounter.comment": {"type": "string", "optional": True},
            "encounter.status": {"type": "string", "optional": False},
        },
    )
    new = SchemaSnapshot(
        version="1.1.0",
        fields={
            "encounter.comment": {"type": "string", "optional": False},
            "encounter.status": {"type": ["null", "string"]},
        },
    )

    report = compare_schema_snapshots(old, new)

    assert [change.path for change in report.changes] == [
        "encounter.comment",
        "encounter.status",
    ]
    assert report.changes[0].reason == "field_became_required"
    assert report.changes[0].breaking is True
    assert report.changes[1].reason == "field_became_optional"
    assert report.changes[1].after_optional is True
    assert report.compatible is False


def test_reports_and_snapshots_never_serialize_example_values() -> None:
    sensitive_canary = "SYNTHETIC-PAYLOAD-CANARY"
    before = SchemaSnapshot.from_mapping(
        {
            "version": "1.0.0",
            "fields": [
                {
                    "path": "subject.identifier",
                    "type": "string",
                    "optional": False,
                    "example": sensitive_canary,
                    "description": "synthetic identifier example",
                }
            ],
        }
    )
    after = SchemaSnapshot(
        version="1.1.0",
        fields={
            "subject.identifier": {
                "type": "string",
                "optional": False,
                "default": sensitive_canary,
            },
            "encounter.note": {"type": "string", "optional": True},
        },
    )

    report = compare_schema_snapshots(before, after)
    serialized = json.dumps(
        {"snapshot": before.to_dict(), "report": report.to_dict()},
        sort_keys=True,
    )

    assert sensitive_canary not in serialized
    assert "example" not in serialized
    assert "default" not in serialized
    assert sensitive_canary not in before.to_json()
    assert json.loads(before.to_json()) == before.to_dict()


def test_invalid_metadata_errors_do_not_echo_sensitive_values() -> None:
    sensitive_canary = "SYNTHETIC-INVALID-OPTIONAL-CANARY"

    with pytest.raises(TypeError) as exc_info:
        SchemaSnapshot(
            version="1.0.0",
            fields={
                "subject.identifier": {
                    "type": "string",
                    "optional": sensitive_canary,
                }
            },
        )

    assert sensitive_canary not in str(exc_info.value)


def test_mapping_inputs_and_boolean_helper_use_the_same_rules() -> None:
    before = {
        "version": "1.0.0",
        "fields": {
            "encounter.count": {"type": "integer", "optional": True},
        },
    }
    after = {
        "version": "1.1.0",
        "fields": {
            "encounter.count": {"type": "number", "optional": True},
        },
    }

    report = compare_schema_snapshots(before, after, rules_version=1)

    assert report.compatible is True
    assert is_schema_compatible(before, after, rules_version=1) is True


@pytest.mark.parametrize(
    "field_metadata",
    [
        {"path": "different.path", "type": "string"},
        {"type": "string", "field_type": "integer"},
        {"type": "string", "optional": True, "required": True},
        {"type": "string", "optional": False, "nullable": True},
    ],
)
def test_conflicting_field_aliases_are_rejected_without_echoing_values(
    field_metadata: dict[str, object],
) -> None:
    sensitive_path = "synthetic.secret.path"

    with pytest.raises(ValueError) as exc_info:
        SchemaSnapshot(fields={sensitive_path: field_metadata})

    assert sensitive_path not in str(exc_info.value)
