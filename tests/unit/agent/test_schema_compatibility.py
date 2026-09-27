"""Table-driven unit tests for agent artifact schema compatibility."""

from __future__ import annotations

import json
from typing import Any

import pytest

from openmed.agent import (
    COMPATIBILITY_SCHEMA_VERSION,
    ArtifactKind,
    CompatibilityOutcome,
    CompatibilityReason,
    CompatibilityResult,
    SchemaCompatibilityError,
    SchemaCompatibilityMatrix,
    SchemaRange,
    SemVer,
    check_schema_compatibility,
)


@pytest.mark.parametrize(
    (
        "version_str",
        "expected_major",
        "expected_minor",
        "expected_patch",
        "expected_prerelease",
    ),
    [
        ("0.0.1", 0, 0, 1, ()),
        ("1.0.0", 1, 0, 0, ()),
        ("1.2.3", 1, 2, 3, ()),
        ("2.10.345", 2, 10, 345, ()),
        ("1.0.0-alpha", 1, 0, 0, ("alpha",)),
        ("1.0.0-alpha.1", 1, 0, 0, ("alpha", 1)),
        ("1.0.0-0.3.7", 1, 0, 0, (0, 3, 7)),
        ("1.0.0-x.7.z.92", 1, 0, 0, ("x", 7, "z", 92)),
        ("1.0.0+20130313144700", 1, 0, 0, ()),
        ("1.0.0-beta+exp.sha.5114f85", 1, 0, 0, ("beta",)),
    ],
)
def test_semver_valid_parsing(
    version_str: str,
    expected_major: int,
    expected_minor: int,
    expected_patch: int,
    expected_prerelease: tuple[str | int, ...],
) -> None:
    parsed = SemVer.parse(version_str)
    assert parsed.major == expected_major
    assert parsed.minor == expected_minor
    assert parsed.patch == expected_patch
    assert parsed.prerelease == expected_prerelease
    assert SemVer.is_valid(version_str) is True


@pytest.mark.parametrize(
    "invalid_version",
    [
        "",
        "v1.0.0",
        "1.0",
        "1",
        "01.0.0",
        "1.01.0",
        "1.0.01",
        "1.0.0-01",
        "1.0.0.0",
        "1.0.0-",
        "1.0.0+",
        "1.0.0-alpha..1",
        "alpha.1.0.0",
        "1.0.0+invalid build",
        "1.0.0-alpha@beta",
        42,
        None,
        True,
        [],
    ],
)
def test_semver_invalid_parsing_raises_contract_error(invalid_version: Any) -> None:
    assert SemVer.is_valid(invalid_version) is False
    with pytest.raises(SchemaCompatibilityError) as caught:
        SemVer.parse(invalid_version)
    assert caught.value.code == "malformed_version"
    assert caught.value.field_name == "version"


@pytest.mark.parametrize(
    ("lesser", "greater"),
    [
        ("1.0.0", "2.0.0"),
        ("2.0.0", "2.1.0"),
        ("2.1.0", "2.1.1"),
        ("1.0.0-alpha", "1.0.0"),
        ("1.0.0-alpha", "1.0.0-alpha.1"),
        ("1.0.0-alpha.1", "1.0.0-alpha.beta"),
        ("1.0.0-alpha.beta", "1.0.0-beta"),
        ("1.0.0-beta", "1.0.0-beta.2"),
        ("1.0.0-beta.2", "1.0.0-beta.11"),
        ("1.0.0-beta.11", "1.0.0-rc.1"),
        ("1.0.0-rc.1", "1.0.0"),
        ("1.0.0-1", "1.0.0-alpha"),
    ],
)
def test_semver_strict_precedence_ordering(lesser: str, greater: str) -> None:
    a = SemVer.parse(lesser)
    b = SemVer.parse(greater)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert a != b
    assert not (a > b)
    assert not (b < a)


def test_semver_build_metadata_ignored_in_precedence() -> None:
    a = SemVer.parse("1.0.0+build.1")
    b = SemVer.parse("1.0.0+build.2")
    assert a == b
    assert not (a < b)
    assert not (b < a)


@pytest.mark.parametrize(
    ("kind", "incoming", "min_v", "max_v", "expected_outcome", "expected_reason"),
    [
        (
            ArtifactKind.EVIDENCE,
            "1.0.0",
            "1.0.0",
            "1.0.0",
            CompatibilityOutcome.COMPATIBLE,
            CompatibilityReason.EXACT_MATCH,
        ),
        (
            "evidence",
            "1.0.0",
            "1.0.0",
            "2.0.0",
            CompatibilityOutcome.COMPATIBLE,
            CompatibilityReason.EXACT_MATCH,
        ),
        (
            "evidence",
            "2.0.0",
            "1.0.0",
            "2.0.0",
            CompatibilityOutcome.COMPATIBLE,
            CompatibilityReason.EXACT_MATCH,
        ),
        (
            ArtifactKind.PREVIEW,
            "1.5.0",
            "1.0.0",
            "2.0.0",
            CompatibilityOutcome.COMPATIBLE,
            CompatibilityReason.WITHIN_RANGE,
        ),
        (
            "preview",
            "1.0.1",
            "1.0.0",
            "1.1.0",
            CompatibilityOutcome.COMPATIBLE,
            CompatibilityReason.WITHIN_RANGE,
        ),
        (
            ArtifactKind.FHIR,
            "0.9.0",
            "1.0.0",
            "2.0.0",
            CompatibilityOutcome.UPGRADE_REQUIRED,
            CompatibilityReason.UPGRADE_REQUIRED,
        ),
        (
            "fhir",
            "1.1.9",
            "1.2.0",
            "2.0.0",
            CompatibilityOutcome.UPGRADE_REQUIRED,
            CompatibilityReason.UPGRADE_REQUIRED,
        ),
        (
            ArtifactKind.OMOP,
            "2.0.1",
            "1.0.0",
            "2.0.0",
            CompatibilityOutcome.DOWNGRADE_REQUIRED,
            CompatibilityReason.DOWNGRADE_REQUIRED,
        ),
        (
            "omop",
            "3.0.0",
            "1.0.0",
            "2.5.0",
            CompatibilityOutcome.DOWNGRADE_REQUIRED,
            CompatibilityReason.DOWNGRADE_REQUIRED,
        ),
        (
            ArtifactKind.EVALUATION,
            "1.5.0-alpha.1",
            "1.0.0",
            "2.0.0",
            CompatibilityOutcome.UNSUPPORTED,
            CompatibilityReason.PRERELEASE_UNSUPPORTED,
        ),
        (
            "evidence",
            "not-a-version",
            "1.0.0",
            "2.0.0",
            CompatibilityOutcome.UNSUPPORTED,
            CompatibilityReason.MALFORMED_VERSION,
        ),
        (
            "evidence",
            "v1.0.0",
            "1.0.0",
            "2.0.0",
            CompatibilityOutcome.UNSUPPORTED,
            CompatibilityReason.MALFORMED_VERSION,
        ),
        (
            "evidence",
            "",
            "1.0.0",
            "2.0.0",
            CompatibilityOutcome.UNSUPPORTED,
            CompatibilityReason.MALFORMED_VERSION,
        ),
        (
            "unknown_kind",
            "1.5.0",
            "1.0.0",
            "2.0.0",
            CompatibilityOutcome.UNSUPPORTED,
            CompatibilityReason.UNKNOWN_ARTIFACT_KIND,
        ),
        (
            "clinical_notes",
            "1.0.0",
            "1.0.0",
            "2.0.0",
            CompatibilityOutcome.UNSUPPORTED,
            CompatibilityReason.UNKNOWN_ARTIFACT_KIND,
        ),
        (
            "evidence",
            "1.5.0",
            "2.0.0",
            "1.0.0",
            CompatibilityOutcome.UNSUPPORTED,
            CompatibilityReason.INVALID_RANGE,
        ),
        (
            "evidence",
            "1.5.0",
            "malformed",
            "2.0.0",
            CompatibilityOutcome.UNSUPPORTED,
            CompatibilityReason.MALFORMED_VERSION,
        ),
    ],
)
def test_table_driven_schema_compatibility(
    kind: Any,
    incoming: str,
    min_v: str,
    max_v: str,
    expected_outcome: CompatibilityOutcome,
    expected_reason: CompatibilityReason,
) -> None:
    result = check_schema_compatibility(
        artifact_kind=kind,
        incoming_version=incoming,
        min_version=min_v,
        max_version=max_v,
    )

    assert result.outcome == expected_outcome
    assert result.reason_code == expected_reason.value
    if expected_outcome == CompatibilityOutcome.COMPATIBLE:
        assert result.is_compatible is True
    else:
        assert result.is_compatible is False


def test_caller_declared_schema_range_object() -> None:
    supported_range = SchemaRange(min_version="1.0.0", max_version="2.2.0")
    result = check_schema_compatibility(
        artifact_kind=ArtifactKind.EVIDENCE,
        incoming_version="1.5.0",
        supported_range=supported_range,
    )
    assert result.is_compatible is True
    assert result.outcome == CompatibilityOutcome.COMPATIBLE
    assert result.reason_code == CompatibilityReason.WITHIN_RANGE.value
    assert result.min_version == "1.0.0"
    assert result.max_version == "2.2.0"


def test_schema_range_invalid_bounds_raise_contract_error() -> None:
    with pytest.raises(SchemaCompatibilityError) as caught:
        SchemaRange(min_version="2.0.0", max_version="1.0.0")
    assert caught.value.code == "invalid_range"
    assert caught.value.field_name == "min_version"


def test_prerelease_supported_when_explicitly_allowed() -> None:
    supported_range = SchemaRange(
        min_version="1.0.0-alpha.1",
        max_version="2.0.0",
        allow_prerelease=True,
    )
    result = check_schema_compatibility(
        artifact_kind=ArtifactKind.EVIDENCE,
        incoming_version="1.0.0-beta",
        supported_range=supported_range,
    )
    assert result.outcome == CompatibilityOutcome.COMPATIBLE
    assert result.reason_code == CompatibilityReason.WITHIN_RANGE.value


def test_no_compatibility_is_inferred_from_artifact_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_open(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(
            "Compatibility evaluation must never access the filesystem"
        )

    import builtins

    monkeypatch.setattr(builtins, "open", forbidden_open)

    result = check_schema_compatibility(
        artifact_kind=ArtifactKind.FHIR,
        incoming_version="1.0.0",
        min_version="1.0.0",
        max_version="1.5.0",
    )
    assert result.is_compatible is True


def test_unknown_mandatory_schema_kinds_fail_closed() -> None:
    result = check_schema_compatibility(
        artifact_kind="mandatory_phi_payload",
        incoming_version="1.0.0",
        min_version="1.0.0",
        max_version="1.0.0",
    )
    assert result.outcome == CompatibilityOutcome.UNSUPPORTED
    assert result.reason_code == "unknown_artifact_kind"
    assert result.is_compatible is False


def test_results_serialize_deterministically_using_artifact_kind_and_versions_only() -> (
    None
):
    result = CompatibilityResult(
        outcome=CompatibilityOutcome.COMPATIBLE,
        reason_code=CompatibilityReason.EXACT_MATCH.value,
        artifact_kind="evidence",
        incoming_version="1.0.0",
        min_version="1.0.0",
        max_version="2.0.0",
    )

    expected_dict = {
        "artifact_kind": "evidence",
        "incoming_version": "1.0.0",
        "outcome": "compatible",
        "reason_code": "exact_match",
        "min_version": "1.0.0",
        "max_version": "2.0.0",
    }

    assert result.to_dict() == expected_dict
    assert set(result.to_dict().keys()) == {
        "artifact_kind",
        "incoming_version",
        "outcome",
        "reason_code",
        "min_version",
        "max_version",
    }

    expected_json = (
        '{"artifact_kind":"evidence","incoming_version":"1.0.0",'
        '"max_version":"2.0.0","min_version":"1.0.0",'
        '"outcome":"compatible","reason_code":"exact_match"}'
    )
    assert result.to_json() == expected_json
    assert json.loads(result.to_json()) == result.to_dict()

    deserialized = CompatibilityResult.from_json(result.to_json())
    assert deserialized == result
    assert deserialized.to_json() == result.to_json()


def test_result_deserialization_rejects_unknown_fields() -> None:
    data = {
        "artifact_kind": "evidence",
        "incoming_version": "1.0.0",
        "outcome": "compatible",
        "reason_code": "exact_match",
        "extra_phi_payload": "patient_123",
    }
    with pytest.raises(SchemaCompatibilityError) as caught:
        CompatibilityResult.from_dict(data)
    assert caught.value.code == "unknown_field"


def test_result_deserialization_rejects_malformed_json() -> None:
    with pytest.raises(SchemaCompatibilityError) as caught:
        CompatibilityResult.from_json("{not-valid-json}")
    assert caught.value.code == "malformed_json"


def test_schema_compatibility_matrix_evaluates_multi_kind_policies() -> None:
    matrix = SchemaCompatibilityMatrix(
        {
            ArtifactKind.EVIDENCE: SchemaRange(
                min_version="1.0.0", max_version="2.0.0"
            ),
            ArtifactKind.PREVIEW: SchemaRange(min_version="1.0.0", max_version="1.5.0"),
        }
    )

    res_ev = matrix.check(ArtifactKind.EVIDENCE, "1.2.0")
    assert res_ev.is_compatible is True

    res_prev = matrix.check("preview", "0.9.0")
    assert res_prev.outcome == CompatibilityOutcome.UPGRADE_REQUIRED

    res_fhir = matrix.check(ArtifactKind.FHIR, "1.0.0")
    assert res_fhir.outcome == CompatibilityOutcome.UNSUPPORTED
    assert res_fhir.reason_code == "unknown_artifact_kind"

    res_unknown = matrix.check("unknown_random", "1.0.0")
    assert res_unknown.outcome == CompatibilityOutcome.UNSUPPORTED
    assert res_unknown.reason_code == "unknown_artifact_kind"


def test_schema_compatibility_exported_from_public_agent_api() -> None:
    import openmed.agent as agent

    assert agent.CompatibilityOutcome is CompatibilityOutcome
    assert agent.CompatibilityReason is CompatibilityReason
    assert agent.CompatibilityResult is CompatibilityResult
    assert agent.SchemaCompatibilityError is SchemaCompatibilityError
    assert agent.SchemaCompatibilityMatrix is SchemaCompatibilityMatrix
    assert agent.SchemaRange is SchemaRange
    assert agent.SemVer is SemVer
    assert agent.check_schema_compatibility is check_schema_compatibility
    assert agent.COMPATIBILITY_SCHEMA_VERSION == COMPATIBILITY_SCHEMA_VERSION
