"""Tests for deterministic, aggregate-only audit retention planning."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from openmed.risk.audit_retention import (
    AuditRetentionPolicy,
    AuditRetentionReport,
    RetentionRule,
    scrub_audit_artifacts,
)

AS_OF = datetime(2026, 8, 9, 12, tzinfo=timezone.utc)


def _policy() -> AuditRetentionPolicy:
    return AuditRetentionPolicy(
        rules={
            "operational": RetentionRule.days(30),
            "legal_hold": RetentionRule(max_age=None, action="retain"),
        }
    )


def _artifacts() -> list[dict[str, object]]:
    return [
        {
            "artifact_id": "artifact-recent",
            "created_at": AS_OF - timedelta(days=3),
            "disposition": "operational",
            "counts": {"masked": 4, "reviewed": 2},
        },
        {
            "artifact_id": "artifact-expired",
            "created_at": AS_OF - timedelta(days=31),
            "disposition": "operational",
            "counts": {"masked": 8},
        },
        {
            "artifact_id": "artifact-held",
            "created_at": AS_OF - timedelta(days=365),
            "disposition": "legal_hold",
            "counts": {"reviewed": 1},
        },
    ]


def test_age_and_disposition_rules_are_deterministic_and_order_independent() -> None:
    artifacts = _artifacts()
    first = scrub_audit_artifacts(artifacts, _policy(), as_of=AS_OF)
    second = scrub_audit_artifacts(list(reversed(artifacts)), _policy(), now=AS_OF)

    assert first.to_dict() == second.to_dict()
    assert first.input_artifact_count == 3
    assert first.retained_artifact_count == 2
    assert first.deleted_artifact_count == 1
    assert first.deleted_artifacts[0].reason == "age_expired"
    assert {item.disposition for item in first.retained_artifacts} == {
        "operational",
        "legal_hold",
    }
    assert first.deleted_fingerprints
    assert first.remaining_fingerprint.startswith("sha256:")


def test_report_verifies_remaining_set_and_round_trips_without_raw_ids() -> None:
    artifacts = _artifacts()
    report = scrub_audit_artifacts(artifacts, _policy(), as_of=AS_OF)
    remaining = [artifacts[0], artifacts[2]]

    assert report.verify_remaining_artifacts(remaining)
    assert report.verify_remaining(remaining)
    assert not report.verify_remaining_artifacts(artifacts)

    encoded = report.to_json(indent=None)
    assert "artifact-recent" not in encoded
    assert "artifact-expired" not in encoded
    assert "artifact-held" not in encoded
    assert "counts" not in encoded
    restored = AuditRetentionReport.from_json(encoded)
    assert restored.to_dict() == report.to_dict()
    assert json.loads(encoded)["integrity_digest"] == report.integrity_digest


def test_report_rejects_tampering() -> None:
    report = scrub_audit_artifacts(_artifacts(), _policy(), as_of=AS_OF)
    payload = report.to_dict()
    payload["retained_artifact_count"] = 1

    with pytest.raises(ValueError, match="integrity digest"):
        AuditRetentionReport.from_dict(payload)


def test_known_raw_fields_are_rejected_without_echoing_their_value() -> None:
    sensitive = "synthetic-secret-value"
    artifact = {
        "artifact_id": "artifact-raw",
        "created_at": AS_OF,
        "disposition": "operational",
        "counts": {"masked": 1},
        "text": sensitive,
    }

    with pytest.raises(ValueError) as exc_info:
        scrub_audit_artifacts([artifact], _policy(), as_of=AS_OF)
    assert sensitive not in str(exc_info.value)


def test_invalid_disposition_fails_closed_without_echoing_identifier() -> None:
    sensitive = "synthetic-patient-token"
    artifact = {
        "artifact_id": sensitive,
        "created_at": AS_OF,
        "disposition": "unconfigured",
        "counts": {"masked": 1},
    }

    with pytest.raises(ValueError) as exc_info:
        scrub_audit_artifacts([artifact], _policy(), as_of=AS_OF)
    assert sensitive not in str(exc_info.value)


def test_exact_age_boundary_is_expired() -> None:
    artifact = {
        "artifact_id": "artifact-boundary",
        "created_at": AS_OF - timedelta(days=30),
        "disposition": "operational",
        "count": 1,
    }

    report = scrub_audit_artifacts([artifact], _policy(), as_of=AS_OF)

    assert report.deleted_artifact_count == 1
