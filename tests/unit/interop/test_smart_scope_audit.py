"""Tests for offline SMART scope comparison helpers."""

from __future__ import annotations

import pytest

from openmed.interop.smart_scope_audit import (
    audit_smart_scopes,
    normalize_smart_scope,
    parse_smart_scope,
)


def test_normalizes_context_and_operation_order() -> None:
    assert normalize_smart_scope(" Patient/SyntheticObservation.SR ") == (
        "patient/SyntheticObservation.rs"
    )

    scope = parse_smart_scope("user/SyntheticCarePlan.uc")
    assert scope.to_dict() == {
        "scope": "user/SyntheticCarePlan.cu",
        "context": "user",
        "resource_type": "SyntheticCarePlan",
        "operations": [
            {"code": "c", "name": "create"},
            {"code": "u", "name": "update"},
        ],
    }


def test_reports_missing_and_excessive_atomic_operations() -> None:
    result = audit_smart_scopes(
        workflow_id="synthetic-workflow",
        required_scopes=(
            "patient/SyntheticCondition.r",
            "patient/SyntheticObservation.r",
        ),
        declared_scopes=(
            "patient/SyntheticObservation.rs",
            "patient/SyntheticMedication.r",
        ),
    )

    assert result.status == "missing"
    assert [scope.value for scope in result.missing_scopes] == [
        "patient/SyntheticCondition.r"
    ]
    assert [scope.value for scope in result.excessive_scopes] == [
        "patient/SyntheticMedication.r",
        "patient/SyntheticObservation.s",
    ]


@pytest.mark.parametrize(
    ("v1", "v2"),
    [("read", "rs"), ("write", "cud"), ("*", "cruds")],
)
@pytest.mark.parametrize("context", ["patient", "user", "system"])
def test_v1_permissions_are_explicit_v2_equivalents(v1, v2, context) -> None:
    first, second = f"{context}/Observation.{v1}", f"{context}/Observation.{v2}"
    assert normalize_smart_scope(first) == second
    for required, declared in [(first, second), (second, first)]:
        result = audit_smart_scopes(
            workflow_id="synthetic",
            required_scopes=[required],
            declared_scopes=[declared],
        )
        assert result.status == "pass"


@pytest.mark.parametrize(
    ("required", "declared", "missing", "excessive"),
    [
        ("patient/Observation.r", "patient/*.r", [], ["patient/*.r"]),
        ("patient/*.r", "patient/Observation.r", ["patient/*.r"], []),
        ("patient/*.read", "patient/*.rs", [], []),
        (
            "patient/Observation.cu",
            "patient/Observation.write",
            [],
            ["patient/Observation.d"],
        ),
        (
            "patient/Observation.r",
            "patient/Observation.rs?category=lab",
            ["patient/Observation.r"],
            ["patient/Observation.s"],
        ),
        (
            "patient/Observation.rs?category=lab",
            "patient/Observation.rs",
            [],
            ["patient/Observation.rs"],
        ),
        (
            "patient/Observation.rs?category=lab",
            "patient/Observation.rs?category=lab",
            [],
            [],
        ),
        (
            "patient/Observation.r?category=lab",
            "patient/Observation.r?category=exam",
            ["patient/Observation.r"],
            ["patient/Observation.r"],
        ),
        (
            "patient/Observation.r?category=lab",
            "patient/*.r?category=lab",
            [],
            ["patient/*.r"],
        ),
        (
            "user/Observation.r",
            "patient/Observation.r",
            ["user/Observation.r"],
            ["patient/Observation.r"],
        ),
        (
            "patient/Observation.r",
            "openid fhirUser launch/patient offline_access",
            ["patient/Observation.r"],
            ["fhirUser", "launch/patient", "offline_access", "openid"],
        ),
    ],
)
def test_conservative_comparison_table(required, declared, missing, excessive) -> None:
    result = audit_smart_scopes(
        workflow_id="synthetic", required_scopes=required, declared_scopes=declared
    )
    # Query values never appear in evidence; compare the public base labels here.
    assert [
        scope.evidence_value.split("?")[0] for scope in result.missing_scopes
    ] == missing
    assert [
        scope.evidence_value.split("?")[0] for scope in result.excessive_scopes
    ] == excessive
    assert (result.status == "pass") is (not missing and not excessive)


def test_mixed_scope_string_and_iterable_are_equivalent() -> None:
    scopes = "openid fhirUser launch/patient offline_access patient/Observation.read"
    result = audit_smart_scopes(
        workflow_id="synthetic", required_scopes=scopes, declared_scopes=scopes.split()
    )
    assert result.status == "pass"
    assert not result.findings
    assert sum(scope.is_clinical for scope in result.declared_scopes) == 1
    assert parse_smart_scope("launch/patient").atoms() == frozenset()


@pytest.mark.parametrize(
    "scope", ["profile", "launch", "launch/encounter", "online_access"]
)
def test_other_recognized_non_clinical_scopes(scope) -> None:
    parsed = parse_smart_scope(scope)
    assert not parsed.is_clinical
    assert parsed.value == scope
    assert parsed.operations == ()


def test_union_of_operations_and_duplicates_is_deterministic() -> None:
    required = "patient/Observation.read"
    result = audit_smart_scopes(
        workflow_id="synthetic",
        required_scopes=required,
        declared_scopes=[
            "patient/Observation.s",
            "patient/Observation.r",
            "patient/Observation.r",
        ],
    )
    assert result.status == "pass"
    assert len(result.declared_scopes) == 2


def test_query_normalization_is_conservative_and_keeps_repeated_parameters() -> None:
    first = parse_smart_scope(
        "patient/Observation.rs?category=urn%3Asynthetic%7Clab&status=final"
    )
    second = parse_smart_scope(
        "patient/Observation.sr?status=final&category=urn:synthetic|lab"
    )
    assert first == second
    assert parse_smart_scope(
        "patient/Observation.r?category=lab&category=lab"
    ) != parse_smart_scope("patient/Observation.r?category=lab")
    result = audit_smart_scopes(
        workflow_id="synthetic",
        required_scopes=["patient/Observation.r?category=lab,exam"],
        declared_scopes=["patient/Observation.r?category=exam,lab"],
    )
    # Do not guess FHIR server-specific implication or OR-list equivalence.
    assert result.status == "missing"


def test_report_and_object_repr_hide_all_query_values_and_keys() -> None:
    import json

    secret = "SYNTHETIC_PRIVATE_VALUE"
    key = "synthetic_private_key"
    result = audit_smart_scopes(
        workflow_id="synthetic",
        required_scopes=[f"patient/Observation.r?{key}={secret}"],
        declared_scopes=[f"patient/Observation.rs?{key}={secret}"],
    )
    for evidence in [json.dumps(result.to_dict()), repr(result)]:
        assert secret not in evidence
        assert key not in evidence
    assert "constraint_digest=" in json.dumps(result.to_dict())
    assert (
        result.to_dict()
        == audit_smart_scopes(
            workflow_id="synthetic",
            required_scopes=[f"patient/Observation.r?{key}={secret}"],
            declared_scopes=[f"patient/Observation.rs?{key}={secret}"],
        ).to_dict()
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "unknown-synthetic-secret",
        "Bearer SYNTHETIC_SECRET",
        "https://example.invalid/private",
        "tenant/Observation.r",
        "patient/Observation.rr",
        "patient/Observation.rx",
        "patient/Observation.",
        "patient/Ob*servation.r",
        "patient/Observation.r?",
        "patient/Observation.r?category=",
        "patient/Observation.r?category",
        "patient/Observation.r?category=lab#fragment",
        "patient/Observation.r?category=%GG",
        "patient/Observation.r?category=%FF",
        "patient/Observation.r?category=%00",
        "patient/Observation.r?=secret",
        "patient/Observation.r?category=lab&",
        "launch/private-secret",
        None,
    ],
)
def test_invalid_scope_yields_value_free_finding_and_never_passes(value) -> None:
    import json

    from openmed.interop.smart_scope_grammar import SmartScopeFinding

    finding = parse_smart_scope(value)
    assert isinstance(finding, SmartScopeFinding)
    assert set(finding.to_dict()) == {"reason_code"}
    result = audit_smart_scopes(
        workflow_id="synthetic", required_scopes=[], declared_scopes=[value]
    )
    assert result.status == "invalid"
    assert not result.declared_scopes
    assert result.to_dict()["findings"] == [
        {"reason_code": finding.reason_code, "source": "declared", "index": 0}
    ]
    if value:
        assert value not in json.dumps(result.to_dict())
    assert isinstance(normalize_smart_scope(value), SmartScopeFinding)


def test_invalid_required_input_is_not_silently_dropped() -> None:
    result = audit_smart_scopes(
        workflow_id="synthetic", required_scopes=["unknown-secret"], declared_scopes=[]
    )
    assert result.status == "invalid"
    assert result.findings[0].source == "required"


def test_shared_comparison_matches_audit_and_returns_no_invalid_permissions() -> None:
    from openmed.interop.smart_scope_grammar import compare_smart_scopes

    scopes = dict(
        required_scopes="openid patient/Observation.read",
        declared_scopes="openid patient/Observation.rs?category=lab unknown-secret",
    )
    shared = compare_smart_scopes(**scopes)
    audited = audit_smart_scopes(workflow_id="synthetic", **scopes)
    assert shared.missing_scopes == audited.missing_scopes
    assert shared.excessive_scopes == audited.excessive_scopes
    assert shared.findings == audited.findings
