"""Focused tests for the nested structured-redaction contract."""

from __future__ import annotations

import json

import pytest

from openmed.structured import (
    ACTION_HASH,
    ACTION_NULL,
    ACTION_REMOVE,
    ACTION_REPLACE,
    RedactionContract,
    RedactionContractError,
    RedactionRule,
    redact_resource,
)


def _synthetic_bundle() -> dict[str, object]:
    """Return an offline Bundle-shaped fixture with synthetic identifiers."""

    return {
        "resourceType": "Bundle",
        "entry": [
            {
                "fullUrl": "urn:synthetic:entry-001",
                "resource": {
                    "resourceType": "Patient",
                    "id": "synthetic-patient-001",
                    "name": [{"text": "synthetic-name-001"}],
                },
            },
            {
                "fullUrl": "urn:synthetic:entry-002",
                "resource": {
                    "resourceType": "Patient",
                    "id": "synthetic-patient-002",
                    "name": [{"text": "synthetic-name-002"}],
                },
            },
        ],
    }


def test_nested_wildcard_redaction_preserves_resource_shape_and_identifiers():
    source = _synthetic_bundle()
    contract = RedactionContract(
        rules=(
            RedactionRule(
                "entry[*].resource.name[*].text",
                action=ACTION_REPLACE,
                replacement="[SYNTHETIC_NAME]",
            ),
        )
    )

    result = redact_resource(source, contract)

    assert source["entry"][0]["resource"]["name"][0]["text"] == ("synthetic-name-001")
    assert [entry["fullUrl"] for entry in result.resource["entry"]] == [
        "urn:synthetic:entry-001",
        "urn:synthetic:entry-002",
    ]
    assert [entry["resource"]["id"] for entry in result.resource["entry"]] == [
        "synthetic-patient-001",
        "synthetic-patient-002",
    ]
    assert [
        entry["resource"]["name"][0]["text"] for entry in result.resource["entry"]
    ] == ["[SYNTHETIC_NAME]", "[SYNTHETIC_NAME]"]
    assert result.report.array_lengths_preserved is True
    assert result.report.matched_rule_count == 2
    assert result.report.resource_identifier_count == (
        result.report.resource_identifiers_preserved
    )
    assert "synthetic-name-001" not in json.dumps(result.to_audit_report())
    assert "synthetic-name-002" not in repr(result)


def test_nullability_and_remove_semantics_are_explicit_and_structure_safe():
    source = {
        "resourceType": "Example",
        "id": "synthetic-resource-001",
        "entry": [
            {"value": "synthetic-value-001"},
            None,
            {"value": None},
        ],
        "metadata": {
            "remove_me": "synthetic-secret-001",
            "optional": None,
        },
    }
    contract = RedactionContract(
        rules=(
            RedactionRule("entry[*].value", action=ACTION_REMOVE),
            RedactionRule("metadata.remove_me", action=ACTION_REMOVE),
            RedactionRule("metadata.optional", action=ACTION_REPLACE),
        )
    )

    result = redact_resource(source, contract)

    assert result.resource["entry"] == [{}, None, {"value": None}]
    assert len(result.resource["entry"]) == len(source["entry"])
    assert "remove_me" not in result.resource["metadata"]
    assert result.resource["metadata"]["optional"] is None
    assert result.resource["id"] == "synthetic-resource-001"
    assert result.report.removed_field_count == 2
    assert result.report.null_preserved_count == 2
    assert result.report.array_lengths_preserved is True
    assert "synthetic-secret-001" not in json.dumps(result.report.to_dict())


def test_hash_action_is_deterministic_and_does_not_report_the_source_value():
    source = {"token": "synthetic-token-001", "other": "synthetic-stable"}
    contract = RedactionContract(rules=(RedactionRule("token", action=ACTION_HASH),))

    first = redact_resource(source, contract)
    second = redact_resource(source, contract)

    assert first.resource == second.resource
    assert first.resource["token"].startswith("sha256:")
    assert first.report.to_dict() == second.report.to_dict()
    assert "synthetic-token-001" not in json.dumps(first.report.to_dict())


def test_ambiguous_wildcards_and_overlapping_rules_fail_closed():
    with pytest.raises(RedactionContractError):
        RedactionRule("entry.*.resource.id", action=ACTION_NULL)

    with pytest.raises(RedactionContractError):
        RedactionContract(
            rules=(
                RedactionRule("entry[*].resource.id", action=ACTION_NULL),
                RedactionRule("entry[0].resource.id", action=ACTION_NULL),
            )
        )


def test_container_selection_and_resource_type_changes_are_rejected():
    with pytest.raises(RedactionContractError) as container_error:
        redact_resource(
            _synthetic_bundle(),
            RedactionContract(
                rules=(RedactionRule("entry[*]", action=ACTION_REPLACE),)
            ),
        )
    assert "synthetic-patient-001" not in str(container_error.value)

    with pytest.raises(RedactionContractError):
        RedactionContract(
            rules=(RedactionRule("entry[*].resource.resourceType", action=ACTION_NULL),)
        )

    with pytest.raises(RedactionContractError):
        RedactionContract(
            rules=(RedactionRule("entry[*].resource.id", action=ACTION_HASH),)
        )


def test_compact_mapping_and_exact_index_paths_are_supported():
    source = _synthetic_bundle()
    contract = RedactionContract.from_mapping(
        {
            "entry[1].resource.name[0].text": {
                "action": ACTION_REPLACE,
                "replacement": "[SECOND_ONLY]",
            }
        }
    )

    result = redact_resource(source, contract)

    assert result.resource["entry"][0]["resource"]["name"][0]["text"] == (
        "synthetic-name-001"
    )
    assert result.resource["entry"][1]["resource"]["name"][0]["text"] == (
        "[SECOND_ONLY]"
    )
    assert result.report.applied_paths == ("entry[1].resource.name[0].text",)
