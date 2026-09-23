"""Tests for the synthetic FHIR R4 CapabilityStatement builders."""

from __future__ import annotations

import copy
import json
import socket

import pytest

from tests.fixtures.fhir_capabilities import (
    FHIR_R4_VERSION,
    SUPPORTED_RESOURCE_TYPE,
    UNSUPPORTED_RESOURCE_TYPE,
    build_conditional_capability_statement,
    build_create_capability_statement,
    build_read_only_capability_statement,
    build_transaction_capability_statement,
    build_unsupported_resource_capability_statement,
    build_update_capability_statement,
    with_invalid_capability_field,
    without_capability_field,
)


def _resource(statement: dict) -> dict:
    return statement["rest"][0]["resource"][0]


def _interaction_codes(items: list[dict]) -> list[str]:
    return [item["code"] for item in items]


@pytest.mark.parametrize(
    ("builder", "expected_interactions"),
    [
        (build_read_only_capability_statement, ["read", "search-type"]),
        (build_create_capability_statement, ["create"]),
        (build_update_capability_statement, ["update"]),
    ],
)
def test_resource_fixture_drives_one_interaction_outcome(
    builder,
    expected_interactions: list[str],
) -> None:
    resource = _resource(builder())

    assert resource["type"] == SUPPORTED_RESOURCE_TYPE
    assert _interaction_codes(resource["interaction"]) == expected_interactions
    assert resource["conditionalCreate"] is False
    assert resource["conditionalUpdate"] is False


def test_conditional_fixture_declares_both_conditional_write_modes() -> None:
    resource = _resource(build_conditional_capability_statement())

    assert _interaction_codes(resource["interaction"]) == ["create", "update"]
    assert resource["conditionalCreate"] is True
    assert resource["conditionalUpdate"] is True


def test_transaction_fixture_is_system_level_only() -> None:
    statement = build_transaction_capability_statement()
    rest = statement["rest"][0]

    assert "resource" not in rest
    assert _interaction_codes(rest["interaction"]) == ["transaction"]


def test_unsupported_fixture_omits_the_planned_resource_type() -> None:
    statement = build_unsupported_resource_capability_statement()
    declared_types = {
        resource["type"] for rest in statement["rest"] for resource in rest["resource"]
    }

    assert declared_types == {SUPPORTED_RESOURCE_TYPE}
    assert UNSUPPORTED_RESOURCE_TYPE not in declared_types


@pytest.mark.parametrize(
    "builder",
    [
        build_read_only_capability_statement,
        build_create_capability_statement,
        build_update_capability_statement,
        build_conditional_capability_statement,
        build_transaction_capability_statement,
        build_unsupported_resource_capability_statement,
    ],
)
def test_fixtures_are_minimal_r4_and_contain_no_server_or_clinical_data(
    builder,
) -> None:
    statement = builder()
    serialized = json.dumps(statement, sort_keys=True).lower()

    assert statement["resourceType"] == "CapabilityStatement"
    assert statement["fhirVersion"] == FHIR_R4_VERSION
    assert statement["format"] == ["json"]
    assert statement["rest"][0]["mode"] == "server"
    for forbidden in (
        "endpoint",
        "credential",
        "authorization",
        "organization",
        "publisher",
        "software",
        "implementation",
        "patient",
    ):
        assert forbidden not in serialized


def test_builders_are_deterministic_and_return_independent_values() -> None:
    first = build_create_capability_statement()
    second = build_create_capability_statement()

    assert first == second
    assert first is not second
    _resource(first)["interaction"][0]["code"] = "delete"
    assert _interaction_codes(_resource(second)["interaction"]) == ["create"]


def test_missing_field_helper_is_deterministic_and_does_not_mutate_input() -> None:
    statement = build_create_capability_statement()
    original = copy.deepcopy(statement)

    first = without_capability_field(statement, ("fhirVersion",))
    second = without_capability_field(statement, ("fhirVersion",))

    assert first == second
    assert "fhirVersion" not in first
    assert statement == original


def test_invalid_field_helper_supports_nested_paths_without_mutation() -> None:
    statement = build_create_capability_statement()
    original = copy.deepcopy(statement)

    invalid = with_invalid_capability_field(
        statement,
        ("rest", 0, "resource", 0, "interaction", 0, "code"),
        "not-an-interaction",
    )

    assert _interaction_codes(_resource(invalid)["interaction"]) == [
        "not-an-interaction"
    ]
    assert statement == original


@pytest.mark.parametrize("path", [(), ("missing",), "fhirVersion"])
def test_corruption_helpers_reject_missing_or_malformed_paths(path) -> None:
    statement = build_create_capability_statement()

    with pytest.raises(KeyError):
        without_capability_field(statement, path)
    with pytest.raises(KeyError):
        with_invalid_capability_field(statement, path, None)


def test_fixture_builders_run_without_network_access(monkeypatch) -> None:
    def fail_network(*_args, **_kwargs):
        raise AssertionError("CapabilityStatement fixtures must remain offline")

    monkeypatch.setattr(socket, "socket", fail_network)

    assert build_read_only_capability_statement()["resourceType"] == (
        "CapabilityStatement"
    )
