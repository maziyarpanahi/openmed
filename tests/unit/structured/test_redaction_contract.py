"""Focused tests for the nested structured-redaction contract."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import Any

import pytest

from openmed.structured import (
    ACTION_HASH,
    ACTION_KEEP,
    ACTION_NULL,
    ACTION_REMOVE,
    ACTION_REPLACE,
    RedactionContract,
    RedactionContractError,
    RedactionInputError,
    RedactionReport,
    RedactionResult,
    RedactionRule,
    redact_resource,
)
from openmed.structured.redaction_contract import (
    MAX_CONTAINER_ITEMS,
    MAX_PATH_SEGMENTS,
    MAX_REDACTION_MATCHES,
    MAX_REDACTION_RULES,
    MAX_REPLACEMENT_STRING_CHARS,
    MAX_RESOURCE_DEPTH,
    MAX_STRING_CHARS,
)


class _ExplodingMapping(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise RuntimeError("synthetic-secret-from-getitem")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("synthetic-secret-from-iterator")

    def __len__(self) -> int:
        return 1

    def items(self) -> Any:
        raise RuntimeError("synthetic-secret-from-items")


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


def test_ancestor_rules_and_preserved_paths_cannot_overlap():
    with pytest.raises(RedactionContractError):
        RedactionContract(
            rules=(
                RedactionRule("entry[*].resource", action=ACTION_REPLACE),
                RedactionRule(
                    "entry[0].resource.name[0].text",
                    action=ACTION_REPLACE,
                ),
            )
        )

    with pytest.raises(RedactionContractError):
        RedactionContract(
            rules=(RedactionRule("entry[*].resource.name", action=ACTION_REPLACE),),
            preserve_paths=("entry[0].resource",),
        )


@pytest.mark.parametrize(
    "action",
    [ACTION_HASH, ACTION_NULL, ACTION_REMOVE, ACTION_REPLACE],
)
def test_resource_root_cannot_be_transformed(action: str):
    with pytest.raises(RedactionContractError):
        RedactionContract(rules=(RedactionRule("$", action=action),))


def test_only_explicit_array_wildcards_are_accepted():
    with pytest.raises(RedactionContractError):
        RedactionRule("entry[].resource.name", action=ACTION_REPLACE)

    rule = RedactionRule("entry[*].resource.name", action=ACTION_KEEP)
    assert str(rule.path) == "entry[*].resource.name"


def test_mapping_policies_are_closed_and_do_not_echo_values():
    with pytest.raises(RedactionContractError) as unknown_option:
        RedactionContract.from_mapping(
            {
                "field": {
                    "action": ACTION_REPLACE,
                    "unsupported": "synthetic-secret-option",
                }
            }
        )
    assert "synthetic-secret-option" not in str(unknown_option.value)

    with pytest.raises(RedactionContractError):
        RedactionContract.from_mapping({"field": "synthetic-replacement"})

    with pytest.raises(RedactionContractError):
        RedactionContract.from_mapping(
            {"field": {"action": ACTION_HASH, "replacement": "not-allowed"}}
        )


def test_contract_and_path_limits_fail_closed():
    with pytest.raises(RedactionContractError):
        RedactionContract(
            rules=tuple(
                RedactionRule(f"field{index}")
                for index in range(MAX_REDACTION_RULES + 1)
            )
        )

    with pytest.raises(RedactionContractError):
        RedactionRule(tuple("field" for _ in range(MAX_PATH_SEGMENTS + 1)))


def test_resource_limits_are_enforced_before_transformation():
    with pytest.raises(RedactionInputError):
        redact_resource(
            [None] * (MAX_CONTAINER_ITEMS + 1),
            RedactionContract(),
        )

    nested: object = "synthetic-leaf"
    for _ in range(MAX_RESOURCE_DEPTH + 1):
        nested = {"child": nested}
    with pytest.raises(RedactionInputError):
        redact_resource(nested, RedactionContract())

    with pytest.raises(RedactionInputError):
        redact_resource(
            {"field": "x" * (MAX_STRING_CHARS + 1)},
            RedactionContract(),
        )

    with pytest.raises(RedactionInputError):
        redact_resource({"field": 1 << 64}, RedactionContract())


def test_match_limit_is_enforced_for_nested_wildcards():
    group_size = 101
    value_count = MAX_REDACTION_MATCHES // group_size + 1
    source = {
        "groups": [
            {"values": ["synthetic-value"] * value_count} for _ in range(group_size)
        ]
    }
    contract = RedactionContract.from_paths(
        ["groups[*].values[*]"],
        action=ACTION_REPLACE,
    )

    with pytest.raises(RedactionContractError):
        redact_resource(source, contract)


def test_hostile_mappings_fail_without_leaking_exception_values():
    with pytest.raises(RedactionContractError) as contract_error:
        RedactionContract(rules=_ExplodingMapping())
    assert "synthetic-secret" not in str(contract_error.value)

    with pytest.raises(RedactionInputError) as input_error:
        redact_resource(_ExplodingMapping(), RedactionContract())
    assert "synthetic-secret" not in str(input_error.value)


def test_input_is_deeply_copied_and_list_subclasses_are_rejected():
    source = {"nested": {"field": "synthetic-original"}}
    result = redact_resource(source, RedactionContract())
    result.resource["nested"]["field"] = "synthetic-mutated"
    assert source["nested"]["field"] == "synthetic-original"

    class _ListSubclass(list[Any]):
        pass

    with pytest.raises(RedactionInputError):
        redact_resource(_ListSubclass(["synthetic-value"]), RedactionContract())


def test_quoted_concrete_paths_round_trip_in_reports():
    result = redact_resource(
        {"synthetic]field": "synthetic-value"},
        RedactionContract.from_paths(
            [("synthetic]field",)],
            action=ACTION_REPLACE,
        ),
    )

    assert result.resource == {"synthetic]field": "[REDACTED]"}
    assert result.report.applied_paths == ('$["synthetic]field"]',)


def test_public_reports_reject_inconsistent_or_noncanonical_metadata():
    report = redact_resource(
        {"field": "synthetic-value"},
        RedactionContract.from_paths(["field"]),
    ).report

    with pytest.raises(RedactionContractError):
        replace(report, source_digest="not-a-digest")
    with pytest.raises(RedactionContractError):
        replace(report, matched_rule_count=2)
    with pytest.raises(RedactionContractError):
        replace(report, applied_paths=("$.field",))
    with pytest.raises(RedactionContractError):
        replace(report, array_lengths_preserved=False)

    with pytest.raises(RedactionContractError):
        RedactionResult(resource={}, report=object())  # type: ignore[arg-type]

    assert type(report) is RedactionReport


def test_invalid_scalar_and_strict_inputs_are_bounded_and_value_free():
    with pytest.raises(RedactionContractError):
        RedactionRule(
            "field",
            action=ACTION_REPLACE,
            replacement="x" * (MAX_REPLACEMENT_STRING_CHARS + 1),
        )

    with pytest.raises(RedactionContractError):
        redact_resource({"field": "synthetic-value"}, RedactionContract(), strict=1)

    with pytest.raises(RedactionInputError) as key_error:
        redact_resource(
            {"synthetic\nsecret": "synthetic-value"},
            RedactionContract(),
        )
    assert "synthetic" not in str(key_error.value)


@pytest.mark.parametrize("source,replacement", [(0, False), (True, 1), (1, 1.0)])
def test_replacement_counts_changes_between_distinct_json_scalar_types(
    source: object, replacement: object
):
    result = redact_resource(
        {"value": source},
        RedactionContract(rules=(RedactionRule("value", replacement=replacement),)),
    )

    assert type(result.resource["value"]) is type(replacement)
    assert result.report.changed_value_count == 1
    assert result.report.source_digest != result.report.output_digest


def test_array_removal_reports_nullification_without_removing_a_position():
    result = redact_resource(
        {"values": ["synthetic-value", None]},
        RedactionContract.from_paths(["values[*]"], action=ACTION_REMOVE),
    )

    assert result.resource == {"values": [None, None]}
    assert result.report.changed_value_count == 1
    assert result.report.nullified_value_count == 1
    assert result.report.null_preserved_count == 1
    assert result.report.removed_field_count == 0
