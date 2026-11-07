"""Offline tests for canonical agent governance identifiers."""

from __future__ import annotations

import traceback
from typing import Any

import pytest

import openmed.agent.identifiers as identifiers_module
from openmed.agent import (
    CapabilityId,
    GovernanceIdError,
    PolicyId,
    PurposeId,
    ToolId,
    WorkflowId,
)

IDENTIFIER_TYPES = (CapabilityId, PurposeId, PolicyId, WorkflowId, ToolId)
KIND_NAMES = ("capability", "purpose", "policy", "workflow", "tool")
NAMESPACE = "org.example"
LOCAL_NAME = "document-intake"
VERSIONED_VALUES = tuple(
    f"{kind}:{NAMESPACE}/{LOCAL_NAME}@1.2.3" for kind in KIND_NAMES
)
UNVERSIONED_VALUES = tuple(f"{kind}:{NAMESPACE}/{LOCAL_NAME}" for kind in KIND_NAMES)


def _parse_error(identifier_type: type[Any], value: object) -> GovernanceIdError:
    with pytest.raises(GovernanceIdError) as caught:
        identifier_type.parse(value)
    return caught.value


def test_each_typed_identifier_round_trips_without_normalization(subtests) -> None:
    for identifier_type, value in list(
        zip(IDENTIFIER_TYPES, VERSIONED_VALUES, strict=True)
    ) + list(zip(IDENTIFIER_TYPES, UNVERSIONED_VALUES, strict=True)):
        with subtests.test(identifier_type=identifier_type, value=value):
            identifier = identifier_type.parse(value)

            assert identifier.serialize() == value
            assert str(identifier) == value
            assert identifier_type.parse(identifier.serialize()) == identifier
            assert identifier.namespace == NAMESPACE
            assert identifier.local_name == LOCAL_NAME
            assert identifier.version == ("1.2.3" if "@" in value else None)


def test_direct_construction_uses_the_same_validation_as_parse(subtests) -> None:
    for identifier_type, value in list(
        zip(IDENTIFIER_TYPES, VERSIONED_VALUES, strict=True)
    ):
        with subtests.test(identifier_type=identifier_type, value=value):
            direct = identifier_type(value)
            parsed = identifier_type.parse(value)

            assert direct == parsed
            assert hash(direct) == hash(parsed)


def test_governance_ids_are_immutable_and_value_free_in_repr(subtests) -> None:
    for identifier_type, value in list(
        zip(IDENTIFIER_TYPES, VERSIONED_VALUES, strict=True)
    ):
        with subtests.test(identifier_type=identifier_type, value=value):
            identifier = identifier_type(value)

            with pytest.raises(AttributeError):
                identifier.value = "tool:org.example/replaced"  # type: ignore[misc]

            assert value not in repr(identifier)
            assert value not in repr([identifier])
            assert value not in repr((identifier,))
            assert value not in repr({"identifier": identifier})


def test_public_agent_exports_match_module_definitions() -> None:
    assert CapabilityId is identifiers_module.CapabilityId
    assert PurposeId is identifiers_module.PurposeId
    assert PolicyId is identifiers_module.PolicyId
    assert WorkflowId is identifiers_module.WorkflowId
    assert ToolId is identifiers_module.ToolId
    assert GovernanceIdError is identifiers_module.GovernanceIdError


def test_only_the_matching_typed_parser_accepts_each_kind(subtests) -> None:
    for row in range(5):
        with subtests.test(row=row):
            value = VERSIONED_VALUES[row]
            for column, identifier_type in enumerate(IDENTIFIER_TYPES):
                if column == row:
                    assert identifier_type.parse(value).serialize() == value
                else:
                    error = _parse_error(identifier_type, value)
                    assert error.code == "wrong_kind"
                    assert error.field_name == "kind"


def test_unknown_and_noncanonical_kinds_fail_closed() -> None:
    unknown = _parse_error(ToolId, "unknown:org.example/redact")
    assert unknown.code == "unknown_kind"

    for value in ("Tool:org.example/redact", "tOoL:org.example/redact"):
        error = _parse_error(ToolId, value)
        assert error.code == "invalid_identifier"

    for value in ("run_" + "0" * 32, "act_" + "0" * 32):
        with pytest.raises(GovernanceIdError):
            ToolId.parse(value)


def test_typed_objects_are_not_implicitly_accepted_as_strings() -> None:
    source = ToolId("tool:org.example/redact")

    error = _parse_error(ToolId, source)

    assert error.code == "invalid_identifier_type"
    assert source.serialize() not in str(error)


def test_direct_construction_and_parse_reject_the_same_inputs(subtests) -> None:
    for identifier_type, value in [
        (ToolId, "tool:org.example/Redact"),
        (ToolId, "tool:org.example/redact@01.2.3"),
        (ToolId, "policy:org.example/redact"),
        (CapabilityId, 42),
    ]:
        with subtests.test(identifier_type=identifier_type, value=value):
            parsed_error = _parse_error(identifier_type, value)

            with pytest.raises(GovernanceIdError) as direct:
                identifier_type(value)

            assert direct.value.code == parsed_error.code
            assert direct.value.field_name == parsed_error.field_name


def test_invalid_namespaces_are_rejected(subtests) -> None:
    for namespace in [
        "example",
        "org",
        "org..example",
        ".org.example",
        "org.example.",
        "org.-example",
        "org.example-",
        "org.example_name",
        "Org.example",
        "org.example name",
        "org.exämple",
    ]:
        with subtests.test(namespace=namespace):
            error = _parse_error(ToolId, f"tool:{namespace}/redact")

            assert error.code == "invalid_identifier"


def test_valid_namespace_labels_allow_digits_and_internal_hyphens(subtests) -> None:
    for namespace in [
        "a.b",
        "1.2",
        "123.example",
        "org.example-2",
        "a-b.9z",
        "a" * 63 + "." + "b" * 63,
    ]:
        with subtests.test(namespace=namespace):
            identifier = ToolId(f"tool:{namespace}/redact")

            assert identifier.namespace == namespace


def test_namespace_label_and_total_length_boundaries() -> None:
    label_63 = "a" * 63
    label_64 = "a" * 64
    namespace_253 = ".".join((label_63, label_63, label_63, "b" * 61))
    namespace_254 = ".".join((label_63, label_63, label_63, "b" * 62))

    assert len(namespace_253) == 253
    assert ToolId(f"tool:{namespace_253}/redact").namespace == namespace_253
    assert _parse_error(ToolId, f"tool:{namespace_254}/redact").code == (
        "namespace_too_long"
    )
    assert _parse_error(ToolId, f"tool:{label_64}.example/redact").code == (
        "invalid_identifier"
    )


def test_local_name_boundaries_and_trailing_hyphens_are_supported(subtests) -> None:
    for local_name in ["a", "a" * 64, "a-", "a--b", "a0", "a-0"]:
        with subtests.test(local_name=local_name):
            assert ToolId(f"tool:{NAMESPACE}/{local_name}").local_name == local_name


def test_invalid_local_names_are_rejected(subtests) -> None:
    for local_name in [
        "",
        "a" * 65,
        "0redact",
        "-redact",
        "Redact",
        "redact_name",
        "redact.name",
        "redact name",
    ]:
        with subtests.test(local_name=local_name):
            error = _parse_error(ToolId, f"tool:{NAMESPACE}/{local_name}")

            assert error.code == "invalid_identifier"


def test_versions_accept_ascii_nonnegative_decimal_components(subtests) -> None:
    for version in ["0.0.0", "1.2.3", "10.20.300", "9" * 100 + ".0.0"]:
        with subtests.test(version=version):
            value = f"tool:{NAMESPACE}/redact@{version}"
            identifier = ToolId(value)

            assert identifier.version == value.rsplit("@", 1)[1]
            assert identifier.serialize() == value


def test_noncanonical_versions_are_rejected(subtests) -> None:
    for version in [
        "01.2.3",
        "1.02.3",
        "1.2.03",
        "1.2",
        "1.2.3.4",
        "1.2.3-alpha",
        "1.2.3+build",
        "1.2.3@4.5.6",
        "+1.2.3",
        "-1.2.3",
        "v1.2.3",
        "",
        "１.２.３",
    ]:
        with subtests.test(version=version):
            error = _parse_error(ToolId, f"tool:{NAMESPACE}/redact@{version}")

            assert error.code == "invalid_identifier"


def test_identifier_length_boundaries_are_independent_of_field_limits() -> None:
    prefix = "tool:org.example/redact@"
    version_without_last_component = "1.2."
    last_component_length = 512 - len(prefix) - len(version_without_last_component)
    valid = prefix + version_without_last_component + "3" * last_component_length
    invalid = valid + "3"

    assert len(valid) == 512
    assert len(invalid) == 513
    assert ToolId(valid).serialize() == valid
    assert _parse_error(ToolId, invalid).code == "identifier_too_long"


def test_paths_urls_delimiters_controls_and_confusables_are_rejected(subtests) -> None:
    for value in [
        "tool:org.example/redact/extra",
        "tool::org.example/redact",
        "tool:org.example:redact",
        "tool:org.example\\redact",
        "tool:org.example/./redact",
        "tool:org.example/../redact",
        "/org.example/redact",
        "C:\\org.example\\redact",
        "https://org.example/redact",
        "file://org.example/redact",
        "tool:org.example/redact?query",
        "tool:org.example/redact#fragment",
        "tool:org.example/red%61ct",
        "tool:org.example%2Fredact",
        " tool:org.example/redact",
        "tool:org.example/redact ",
        "tool:org.example/red act",
        "tool:org.example/redact\x00",
        "tool:org.example/redact\r",
        "tool:org.example/redact\n",
        "tool:org.example/redact\t",
        "tool:org.example/redact\x7f",
        "tool:org．example/redact",
        "tool:org.examp1e/redаct",
        "tool:org.example/redact\u200b",
        "tool:org.example/redact\u202e",
        "tool:org.example/redact!",
    ]:
        with subtests.test(value=value):
            error = _parse_error(ToolId, value)

            assert error.code == "invalid_identifier"


class _CanaryObject:
    def __str__(self) -> str:
        raise AssertionError("__str__ must not be called")

    def __repr__(self) -> str:
        raise AssertionError("__repr__ must not be called")


def test_basic_non_string_inputs_are_rejected(subtests) -> None:
    for value in [None, True, 42, [], {}]:
        with subtests.test(value=value):
            error = _parse_error(ToolId, value)

            assert error.code == "invalid_identifier_type"


def test_bytes_inputs_are_not_decoded(subtests) -> None:
    for value in [b"tool:org.example/redact", bytearray(b"tool:org.example/redact")]:
        with subtests.test(value=value):
            error = _parse_error(ToolId, value)

            assert error.code == "invalid_identifier_type"


def test_custom_object_is_rejected_without_implicit_formatting() -> None:
    error = _parse_error(ToolId, _CanaryObject())

    assert error.code == "invalid_identifier_type"


class _StringSubclass(str):
    def __str__(self) -> str:
        raise AssertionError("__str__ must not be called")

    def __repr__(self) -> str:
        raise AssertionError("__repr__ must not be called")


def test_string_subclasses_are_rejected_by_the_strict_input_contract() -> None:
    error = _parse_error(ToolId, _StringSubclass("tool:org.example/redact"))

    assert error.code == "invalid_identifier_type"


def test_errors_are_value_free_in_messages_attributes_and_tracebacks() -> None:
    canary = "SYNTHETIC-CANARY-3042-not-a-real-record"
    value = f"tool:org.example/{canary}@1.2.3"

    with pytest.raises(GovernanceIdError) as caught:
        ToolId.parse(value)

    error = caught.value
    rendered = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    assert error.code == "invalid_identifier"
    assert error.field_name == "identifier"
    assert canary not in str(error)
    assert canary not in repr(error)
    assert canary not in repr(error.args)
    assert canary not in rendered
    assert all(canary not in repr(attribute) for attribute in vars(error).values())


def test_valid_identifier_repr_never_contains_the_identifier(subtests) -> None:
    for identifier_type in IDENTIFIER_TYPES:
        with subtests.test(identifier_type=identifier_type):
            value = f"{KIND_NAMES[IDENTIFIER_TYPES.index(identifier_type)]}:{NAMESPACE}/redact@1.2.3"

            identifier = identifier_type(value)

            assert value not in repr(identifier)
            assert "<redacted>" in repr(identifier)
