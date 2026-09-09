"""Focused tests for the no-PHI exception taxonomy validator."""

from __future__ import annotations

import hashlib
import socket
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone

import pytest

from openmed.risk import (
    DEFAULT_EXCEPTION_TAXONOMY,
    ApprovalMetadata,
    EvidenceReference,
    ExceptionRecord,
    ExceptionTaxonomy,
    ExceptionTaxonomyError,
    TaxonomyFinding,
    TaxonomyRule,
    TaxonomyValidationResult,
    validate_audit,
    validate_audit_record,
    validate_exception_record,
    validate_telemetry_record,
)

_AS_OF = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)


def _digest(seed: str) -> str:
    return f"sha256:{hashlib.sha256(seed.encode('ascii')).hexdigest()}"


def _record(
    *,
    category: str = "local_suppression",
    reason_code: str = "false_positive_reviewed",
    scope: str = "telemetry",
    evidence: list[dict[str, str]] | None = None,
    expires_at: str = "2026-09-01T00:00:00Z",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "taxonomy_version": "1.0",
        "category": category,
        "reason_code": reason_code,
        "scope": scope,
        "evidence": evidence
        or [
            {"kind": "test", "digest": _digest("test-evidence")},
            {"kind": "review", "digest": _digest("review-evidence")},
        ],
        "expires_at": expires_at,
        "approval": {
            "status": "approved",
            "role": "privacy_reviewer",
            "approval_digest": _digest("approval-evidence"),
            "approved_at": "2026-08-09T00:00:00Z",
        },
    }


def test_valid_record_is_deterministic_and_safe() -> None:
    payload = _record()

    first = validate_telemetry_record(payload, as_of=_AS_OF)
    second = validate_telemetry_record(payload, as_of=_AS_OF)

    assert first == second
    assert first.valid is True
    assert first.record is not None
    assert first.record.to_dict() == payload
    assert first.record_digest is not None
    assert first.to_json() == second.to_json()
    assert "approval" in first.record.to_json()
    assert "owner" not in first.record.to_json()


def test_telemetry_and_audit_surfaces_are_explicit() -> None:
    telemetry = validate_telemetry_record(_record(), as_of=_AS_OF)
    audit = validate_audit_record(
        _record(scope="audit"),
        as_of=_AS_OF,
    )
    mismatched = validate_audit_record(_record(), as_of=_AS_OF)

    assert telemetry.valid is True
    assert telemetry.surface == "telemetry"
    assert audit.valid is True
    assert audit.surface == "audit"
    assert validate_audit(_record(scope="audit"), as_of=_AS_OF) == audit
    assert mismatched.valid is False
    assert "UNSUPPORTED_SCOPE" in mismatched.error_codes


@pytest.mark.parametrize(
    ("category", "reason_code", "evidence", "expires_at"),
    [
        (
            "local_suppression",
            "policy_exclusion",
            [
                {"kind": "test", "digest": _digest("suppression-test")},
                {"kind": "review", "digest": _digest("suppression-review")},
            ],
            "2026-09-01T00:00:00Z",
        ),
        (
            "local_allowance",
            "compatibility_boundary",
            [
                {"kind": "test", "digest": _digest("allowance-test")},
                {"kind": "review", "digest": _digest("allowance-review")},
            ],
            "2026-09-01T00:00:00Z",
        ),
        (
            "synthetic_fixture",
            "synthetic_only",
            [
                {"kind": "fixture", "digest": _digest("fixture")},
                {"kind": "test", "digest": _digest("fixture-test")},
            ],
            "2026-09-01T00:00:00Z",
        ),
        (
            "operational_fallback",
            "bounded_degradation",
            [
                {"kind": "incident", "digest": _digest("incident")},
                {"kind": "test", "digest": _digest("fallback-test")},
            ],
            "2026-08-16T00:00:00Z",
        ),
    ],
)
def test_each_taxonomy_category_requires_its_declared_evidence(
    category: str,
    reason_code: str,
    evidence: list[dict[str, str]],
    expires_at: str,
) -> None:
    result = validate_telemetry_record(
        _record(
            category=category,
            reason_code=reason_code,
            evidence=evidence,
            expires_at=expires_at,
        ),
        as_of=_AS_OF,
    )

    assert result.valid is True


def test_missing_required_evidence_is_rejected_without_echoing_payload() -> None:
    payload = _record(evidence=[{"kind": "test", "digest": _digest("test-only")}])

    result = validate_telemetry_record(payload, as_of=_AS_OF)

    assert result.valid is False
    assert result.error_codes == ("MISSING_REQUIRED_EVIDENCE",)
    assert "test-only" not in result.to_json()


def test_unknown_reason_and_arbitrary_payload_are_closed_world_rejections() -> None:
    payload = _record()
    payload["reason_code"] = "opaque-value-canary"
    payload["payload"] = "opaque-value-canary"

    result = validate_telemetry_record(payload, as_of=_AS_OF)

    assert result.valid is False
    assert "UNSUPPORTED_FIELD" in result.error_codes
    assert "UNSUPPORTED_REASON_CODE" in result.error_codes
    assert "opaque-value-canary" not in result.to_json()


def test_owner_and_free_form_approval_fields_are_not_accepted() -> None:
    payload = _record()
    approval = payload["approval"]
    assert isinstance(approval, dict)
    approval["owner"] = "opaque-value-canary"

    result = validate_telemetry_record(payload, as_of=_AS_OF)

    assert result.valid is False
    assert result.error_codes == ("INVALID_APPROVAL",)
    assert "opaque-value-canary" not in result.to_json()


def test_expiry_is_bounded_and_evaluated_only_at_explicit_as_of() -> None:
    expired = validate_telemetry_record(
        _record(expires_at="2026-08-09T12:00:00Z"),
        as_of=_AS_OF,
    )
    too_long = validate_telemetry_record(
        _record(expires_at="2026-11-08T00:00:00Z"),
        as_of=None,
    )

    assert expired.valid is False
    assert expired.error_codes == ("EXPIRED",)
    assert too_long.valid is False
    assert too_long.error_codes == ("EXPIRY_TOO_LONG",)


def test_versioned_taxonomy_round_trips_without_arbitrary_fields() -> None:
    serialized = DEFAULT_EXCEPTION_TAXONOMY.to_json()
    parsed = ExceptionTaxonomy.from_mapping(DEFAULT_EXCEPTION_TAXONOMY.to_dict())

    assert parsed == DEFAULT_EXCEPTION_TAXONOMY
    assert serialized == parsed.to_json()
    assert "payload" not in serialized
    assert "owner" not in serialized


def test_typed_record_round_trip_is_validated() -> None:
    record = ExceptionRecord.from_mapping(_record())

    result = validate_exception_record(record, as_of=_AS_OF)

    assert result.valid is True
    assert result.record == record
    assert result.record_digest == record.digest


def test_validation_does_not_open_a_network_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is not part of taxonomy validation")

    monkeypatch.setattr(socket, "socket", fail_socket)

    result = validate_telemetry_record(_record(), as_of=_AS_OF)

    assert result.valid is True


class _ExplodingMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise RuntimeError("opaque-value-canary")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("opaque-value-canary")

    def __len__(self) -> int:
        raise RuntimeError("opaque-value-canary")


def test_hostile_mapping_fails_closed_without_echoing_exception_text() -> None:
    result = validate_telemetry_record(_ExplodingMapping(), as_of=_AS_OF)

    assert result.valid is False
    assert result.error_codes == ("UNSUPPORTED_FIELD",)
    assert "opaque-value-canary" not in result.to_json()

    with pytest.raises(ExceptionTaxonomyError, match="invalid taxonomy") as caught:
        ExceptionTaxonomy.from_mapping(_ExplodingMapping())
    assert "opaque-value-canary" not in str(caught.value)


def test_oversized_evidence_stops_before_inspecting_entries() -> None:
    payload = _record()
    payload["evidence"] = [_ExplodingMapping()] * 9

    result = validate_telemetry_record(payload, as_of=_AS_OF)

    assert result.error_codes == ("TOO_MANY_EVIDENCE",)
    assert "opaque-value-canary" not in result.to_json()


def test_nested_hostile_mapping_is_a_safe_structural_finding() -> None:
    payload = _record()
    payload["evidence"] = [_ExplodingMapping()]

    result = validate_telemetry_record(payload, as_of=_AS_OF)

    assert result.valid is False
    assert result.error_codes == ("INVALID_EVIDENCE",)
    assert "opaque-value-canary" not in result.to_json()


def test_timestamps_require_explicit_utc() -> None:
    payload = _record(expires_at="2026-09-01T01:00:00+01:00")

    result = validate_telemetry_record(payload, as_of=_AS_OF)
    invalid_as_of = validate_telemetry_record(
        _record(),
        as_of="opaque-value-canary",
    )

    assert result.error_codes == ("INVALID_TIMESTAMP",)
    assert invalid_as_of.error_codes == ("INVALID_TIMESTAMP",)
    assert "opaque-value-canary" not in invalid_as_of.to_json()


def test_evidence_order_is_canonical_and_audit_records_round_trip() -> None:
    forward = ExceptionRecord.from_mapping(_record())
    payload = _record()
    evidence = payload["evidence"]
    assert isinstance(evidence, list)
    payload["evidence"] = list(reversed(evidence))
    reverse = ExceptionRecord.from_mapping(payload)
    audit = ExceptionRecord.from_mapping(_record(scope="audit"))

    assert forward == reverse
    assert forward.digest == reverse.digest
    assert audit.scope == "audit"
    assert validate_audit_record(audit, as_of=_AS_OF).valid is True


def test_public_dataclasses_reject_inconsistent_or_payload_bearing_states() -> None:
    digest = _digest("evidence")
    evidence = EvidenceReference(kind="test", digest=digest)
    approval = ApprovalMetadata(
        status="approved",
        role="privacy_reviewer",
        approval_digest=digest,
        approved_at=datetime(2026, 8, 9, tzinfo=timezone.utc),
    )
    record = ExceptionRecord(
        category="local_suppression",
        reason_code="false_positive_reviewed",
        scope="telemetry",
        evidence=(evidence, EvidenceReference(kind="review", digest=digest)),
        expires_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        approval=approval,
    )

    with pytest.raises(ExceptionTaxonomyError, match="unsupported taxonomy category"):
        TaxonomyRule(
            category="opaque-value-canary",
            reason_codes=("false_positive_reviewed",),
            required_evidence=("test",),
            max_expiry_days=1,
            allowed_scopes=("telemetry",),
        )
    with pytest.raises(ValueError, match="invalid result cannot contain record data"):
        TaxonomyValidationResult(
            surface="telemetry",
            valid=False,
            errors=(TaxonomyFinding("INVALID_TYPE", "$"),),
            record_digest=record.digest,
            record=record,
        )

    incomplete_record = ExceptionRecord(
        category="local_suppression",
        reason_code="false_positive_reviewed",
        scope="telemetry",
        evidence=(evidence,),
        expires_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        approval=approval,
    )
    with pytest.raises(ValueError, match="taxonomy-valid record"):
        TaxonomyValidationResult(
            surface="telemetry",
            valid=True,
            errors=(),
            record_digest=incomplete_record.digest,
            record=incomplete_record,
        )
