"""Offline SMART v2 scope normalization and least-privilege tests."""

from __future__ import annotations

import json

import pytest

from openmed.interop.smart_scope_audit import audit_smart_scopes, parse_smart_scope


@pytest.mark.parametrize(
    ("raw", "name", "context", "resource", "access"),
    [
        (
            "patient/Observation.sr",
            "patient/Observation.rs",
            "patient",
            "Observation",
            "rs",
        ),
        ("user/Condition.r", "user/Condition.r", "user", "Condition", "r"),
        ("system/*.cruds", "system/*.cruds", "system", "*", "cruds"),
        ("system/Patient.cud", "system/Patient.cud", "system", "Patient", "cud"),
        ("launch", "launch", "launch", None, ""),
        ("launch/patient", "launch/patient", "launch", None, ""),
        ("launch/encounter", "launch/encounter", "launch", None, ""),
    ],
)
def test_normalizes_smart_v2_scope(raw, name, context, resource, access) -> None:
    scope = parse_smart_scope(raw)
    assert (scope.name, scope.context, scope.resource_type, scope.access) == (
        name,
        context,
        resource,
        frozenset(access),
    )


@pytest.mark.parametrize(
    ("required", "requested", "missing", "excessive"),
    [
        (["patient/Observation.rs"], ["patient/Observation.rs"], [], []),
        (
            ["patient/Observation.rs"],
            ["patient/Observation.r"],
            ["patient/Observation.s"],
            [],
        ),
        (["user/Condition.r"], ["user/Condition.cr"], [], ["user/Condition.c"]),
        (["system/Patient.cu"], ["system/Patient.c"], ["system/Patient.u"], []),
        (
            ["patient/Observation.r"],
            ["user/Observation.r"],
            ["patient/Observation.r"],
            ["user/Observation.r"],
        ),
        (["patient/Observation.r"], ["patient/*.r"], [], ["patient/*.r"]),
        (["system/*.r"], ["system/Observation.r"], ["system/*.r"], []),
        (["system/*.r"], ["system/*.rs"], [], ["system/*.s"]),
        (["launch/patient"], ["launch"], ["launch/patient"], ["launch"]),
        (["launch/encounter"], ["launch/encounter"], [], []),
        (
            ["user/Observation.r"],
            ["user/Observation.r", "user/Condition.r"],
            [],
            ["user/Condition.r"],
        ),
    ],
)
def test_audit_cases(required, requested, missing, excessive) -> None:
    result = audit_smart_scopes(required_scopes=required, requested_scopes=requested)
    assert [item.scope for item in result.missing_scopes] == missing
    assert [item.scope for item in result.excessive_scopes] == excessive
    assert result.is_least_privilege is (not missing and not excessive)


def test_reason_codes_and_output_are_stable_and_content_free() -> None:
    result = audit_smart_scopes(
        required_scopes=["patient/Observation.rs", "launch/patient"],
        requested_scopes=["patient/*.r", "launch"],
    )
    assert result.to_dict() == {
        "missing_scopes": [
            {
                "reason_code": "missing_scope",
                "scope": "launch/patient",
                "resource_type": None,
            },
            {
                "reason_code": "missing_scope",
                "scope": "patient/Observation.s",
                "resource_type": "Observation",
            },
        ],
        "excessive_scopes": [
            {
                "reason_code": "excessive_scope",
                "scope": "launch",
                "resource_type": None,
            },
            {
                "reason_code": "overbroad_resource",
                "scope": "patient/*.r",
                "resource_type": "*",
            },
        ],
    }
    serialized = json.dumps(result.to_dict())
    assert "token" not in serialized and "http" not in serialized


@pytest.mark.parametrize(
    "value",
    [
        "patient/Observation.read",  # SMART v1 is not silently promoted.
        "patient/Observation.rr",
        "patient/Observation.",
        "patient/observation.r",
        "patient/Observation.*",
        "launch/custom-identifier",
        "offline_access",
        "https://example.invalid/token",
        "Bearer sensitive-value",
    ],
)
def test_unsupported_values_fail_without_echoing_input(value: str) -> None:
    with pytest.raises(ValueError) as error:
        parse_smart_scope(value)
    assert value not in str(error.value)


def test_input_order_and_duplicates_do_not_change_report() -> None:
    first = audit_smart_scopes(
        required_scopes=["user/Condition.r", "user/Observation.r"],
        requested_scopes=["user/Observation.rs", "user/Observation.rs"],
    )
    second = audit_smart_scopes(
        required_scopes=["user/Observation.r", "user/Condition.r"],
        requested_scopes=["user/Observation.sr"],
    )
    assert first.to_dict() == second.to_dict()
