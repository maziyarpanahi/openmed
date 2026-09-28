"""Focused offline tests for FHIR CapabilityStatement write preflight."""

from __future__ import annotations

import socket
from dataclasses import fields

import pytest

from openmed.interop.fhir_capability_preflight import (
    MAX_CAPABILITY_RESOURCES,
    CapabilityStatementError,
    FHIRPreflightReason,
    FHIRPreflightStatus,
    FHIRWriteInteraction,
    FHIRWritePlan,
    parse_capability_statement,
    preflight_write_plan,
)
from tests.fixtures.fhir_capabilities import (
    SUPPORTED_RESOURCE_TYPE,
    UNSUPPORTED_RESOURCE_TYPE,
    build_capability_statement,
    build_conditional_capability_statement,
    build_create_capability_statement,
    build_read_only_capability_statement,
    build_transaction_capability_statement,
    build_update_capability_statement,
    with_invalid_capability_field,
    without_capability_field,
)


def _plan(
    interaction: FHIRWriteInteraction,
    *,
    conditional: bool = False,
    resource_type: str = SUPPORTED_RESOURCE_TYPE,
) -> FHIRWritePlan:
    return FHIRWritePlan(
        interaction=interaction,
        resource_type=resource_type,
        conditional=conditional,
    )


@pytest.mark.parametrize(
    ("statement", "plan"),
    [
        (
            build_create_capability_statement(),
            _plan(FHIRWriteInteraction.CREATE),
        ),
        (
            build_update_capability_statement(),
            _plan(FHIRWriteInteraction.UPDATE),
        ),
        (
            build_conditional_capability_statement(),
            _plan(FHIRWriteInteraction.CREATE, conditional=True),
        ),
        (
            build_conditional_capability_statement(),
            _plan(FHIRWriteInteraction.UPDATE, conditional=True),
        ),
        (
            build_transaction_capability_statement(),
            FHIRWritePlan(FHIRWriteInteraction.TRANSACTION),
        ),
    ],
)
def test_supported_write_plans_are_compatible(statement, plan) -> None:
    result = preflight_write_plan(statement, plan)

    assert result.status is FHIRPreflightStatus.COMPATIBLE
    assert result.reason_code is FHIRPreflightReason.SUPPORTED
    assert result.is_compatible is True
    assert result.to_dict() == {
        "status": "compatible",
        "reason_code": "supported",
    }


@pytest.mark.parametrize(
    ("statement", "plan", "reason"),
    [
        (
            build_read_only_capability_statement(),
            _plan(FHIRWriteInteraction.CREATE),
            FHIRPreflightReason.INTERACTION_NOT_SUPPORTED,
        ),
        (
            build_create_capability_statement(),
            _plan(FHIRWriteInteraction.UPDATE),
            FHIRPreflightReason.INTERACTION_NOT_SUPPORTED,
        ),
        (
            build_create_capability_statement(),
            _plan(FHIRWriteInteraction.CREATE, conditional=True),
            FHIRPreflightReason.CONDITIONAL_CREATE_NOT_SUPPORTED,
        ),
        (
            build_update_capability_statement(),
            _plan(FHIRWriteInteraction.UPDATE, conditional=True),
            FHIRPreflightReason.CONDITIONAL_UPDATE_NOT_SUPPORTED,
        ),
        (
            build_create_capability_statement(),
            _plan(
                FHIRWriteInteraction.CREATE,
                resource_type=UNSUPPORTED_RESOURCE_TYPE,
            ),
            FHIRPreflightReason.RESOURCE_NOT_SUPPORTED,
        ),
        (
            build_create_capability_statement(),
            FHIRWritePlan(FHIRWriteInteraction.TRANSACTION),
            FHIRPreflightReason.TRANSACTION_NOT_SUPPORTED,
        ),
    ],
)
def test_unsupported_write_plans_are_incompatible(statement, plan, reason) -> None:
    result = preflight_write_plan(statement, plan)

    assert result.status is FHIRPreflightStatus.INCOMPATIBLE
    assert result.reason_code is reason
    assert result.is_compatible is False


@pytest.mark.parametrize(
    ("field_name", "interaction", "reason"),
    [
        (
            "conditionalCreate",
            FHIRWriteInteraction.CREATE,
            FHIRPreflightReason.CONDITIONAL_CREATE_UNDECLARED,
        ),
        (
            "conditionalUpdate",
            FHIRWriteInteraction.UPDATE,
            FHIRPreflightReason.CONDITIONAL_UPDATE_UNDECLARED,
        ),
    ],
)
def test_undeclared_conditional_support_requires_review(
    field_name,
    interaction,
    reason,
) -> None:
    statement = without_capability_field(
        build_conditional_capability_statement(),
        ("rest", 0, "resource", 0, field_name),
    )

    result = preflight_write_plan(
        statement,
        _plan(interaction, conditional=True),
    )

    assert result.status is FHIRPreflightStatus.REVIEW
    assert result.reason_code is reason


def test_parser_normalizes_resource_and_system_metadata() -> None:
    statement = build_capability_statement(
        resource_interactions=("update", "create", "create"),
        conditional_create=True,
        system_interactions=("transaction",),
    )

    parsed = parse_capability_statement(statement)
    resource = parsed.for_resource(SUPPORTED_RESOURCE_TYPE)

    assert parsed.fhir_version == "4.0.1"
    assert parsed.system_interactions == frozenset({"transaction"})
    assert resource is not None
    assert resource.interactions == frozenset({"create", "update"})
    assert resource.conditional_create is True
    assert resource.conditional_update is False
    assert parsed.for_resource(UNSUPPORTED_RESOURCE_TYPE) is None


@pytest.mark.parametrize(
    "statement",
    [
        without_capability_field(
            build_create_capability_statement(),
            ("fhirVersion",),
        ),
        with_invalid_capability_field(
            build_create_capability_statement(),
            ("rest", 0, "resource", 0, "interaction", 0, "code"),
            "not-an-interaction",
        ),
        with_invalid_capability_field(
            build_create_capability_statement(),
            ("rest", 0, "resource", 0, "conditionalCreate"),
            "true",
        ),
        with_invalid_capability_field(
            build_create_capability_statement(),
            ("rest", 0, "mode"),
            [],
        ),
    ],
)
def test_malformed_statements_require_review(statement) -> None:
    result = preflight_write_plan(statement, _plan(FHIRWriteInteraction.CREATE))

    assert result.status is FHIRPreflightStatus.REVIEW
    assert result.reason_code is FHIRPreflightReason.CAPABILITY_STATEMENT_MALFORMED

    with pytest.raises(CapabilityStatementError):
        parse_capability_statement(statement)


def test_non_r4_statement_is_incompatible() -> None:
    statement = with_invalid_capability_field(
        build_create_capability_statement(),
        ("fhirVersion",),
        "5.0.0",
    )

    result = preflight_write_plan(statement, _plan(FHIRWriteInteraction.CREATE))

    assert result.status is FHIRPreflightStatus.INCOMPATIBLE
    assert result.reason_code is FHIRPreflightReason.FHIR_VERSION_NOT_SUPPORTED


def test_parser_rejects_capabilities_above_resource_bound() -> None:
    statement = build_create_capability_statement()
    resource = statement["rest"][0]["resource"][0]
    statement["rest"][0]["resource"] = [
        {**resource, "type": f"Resource{index}"}
        for index in range(MAX_CAPABILITY_RESOURCES + 1)
    ]

    result = preflight_write_plan(statement, _plan(FHIRWriteInteraction.CREATE))

    assert result.status is FHIRPreflightStatus.REVIEW
    assert result.reason_code is FHIRPreflightReason.CAPABILITY_STATEMENT_MALFORMED


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interaction": FHIRWriteInteraction.CREATE},
        {
            "interaction": FHIRWriteInteraction.TRANSACTION,
            "resource_type": "Bundle",
        },
        {
            "interaction": FHIRWriteInteraction.TRANSACTION,
            "conditional": True,
        },
    ],
)
def test_write_plan_rejects_ambiguous_or_incomplete_shapes(kwargs) -> None:
    with pytest.raises(ValueError):
        FHIRWritePlan(**kwargs)


def test_write_plan_has_no_payload_endpoint_identifier_or_credential_fields() -> None:
    assert {field.name for field in fields(FHIRWritePlan)} == {
        "interaction",
        "resource_type",
        "conditional",
    }


def test_preflight_runs_without_network_access(monkeypatch) -> None:
    def fail_network(*_args, **_kwargs):
        raise AssertionError("FHIR capability preflight must remain offline")

    monkeypatch.setattr(socket, "socket", fail_network)

    result = preflight_write_plan(
        build_create_capability_statement(),
        _plan(FHIRWriteInteraction.CREATE),
    )

    assert result.status is FHIRPreflightStatus.COMPATIBLE
