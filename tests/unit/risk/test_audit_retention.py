"""Tests for deterministic, aggregate-only audit retention planning."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone

import pytest

import openmed.risk.audit_retention as audit_retention
from openmed.risk.audit_retention import (
    MAX_AUDIT_RETENTION_COUNT,
    AuditArtifact,
    AuditRetentionPolicy,
    AuditRetentionReport,
    DeletionFingerprint,
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

    with pytest.raises(ValueError):
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


def test_future_dated_artifact_fails_closed() -> None:
    artifact = {
        "artifact_id": "artifact-future",
        "created_at": AS_OF + timedelta(seconds=1),
        "disposition": "operational",
        "count": 1,
    }

    with pytest.raises(ValueError, match="must not follow"):
        scrub_audit_artifacts([artifact], _policy(), as_of=AS_OF)


@pytest.mark.parametrize(
    "extra",
    [
        {"note": "synthetic-sensitive-value"},
        {"id": "conflicting-alias"},
        {"event_count": 2},
    ],
)
def test_unknown_and_duplicate_artifact_fields_fail_closed(
    extra: dict[str, object],
) -> None:
    artifact = {**_artifacts()[0], **extra}

    with pytest.raises(ValueError) as exc_info:
        scrub_audit_artifacts([artifact], _policy(), as_of=AS_OF)
    assert "synthetic-sensitive-value" not in str(exc_info.value)


def test_duplicate_artifact_ids_are_rejected() -> None:
    artifacts = _artifacts()[:2]
    artifacts[1]["artifact_id"] = artifacts[0]["artifact_id"]

    with pytest.raises(ValueError, match="identifiers must be unique"):
        scrub_audit_artifacts(artifacts, _policy(), as_of=AS_OF)


def test_counts_and_policy_names_cannot_overwrite_after_normalization() -> None:
    artifact = _artifacts()[0]
    artifact["counts"] = {"Masked": 1, "masked": 2}

    with pytest.raises(ValueError, match="count names must be unique"):
        AuditArtifact.from_mapping(artifact)
    with pytest.raises(ValueError, match="dispositions must be unique"):
        AuditRetentionPolicy(
            rules={
                "Operational": RetentionRule.days(30),
                "operational": RetentionRule.days(10),
            }
        )


def test_counts_are_bounded_without_integer_coercion() -> None:
    artifact = _artifacts()[0]
    artifact["counts"] = {"masked": MAX_AUDIT_RETENTION_COUNT + 1}

    with pytest.raises(ValueError, match="bounded integers"):
        AuditArtifact.from_mapping(artifact)


def test_artifact_iterable_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit_retention, "MAX_AUDIT_RETENTION_ARTIFACTS", 2)

    with pytest.raises(ValueError, match="item limit"):
        scrub_audit_artifacts(
            itertools.repeat(_artifacts()[0]),
            _policy(),
            as_of=AS_OF,
        )


class _ExplodingMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise RuntimeError("synthetic-sensitive-value")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("synthetic-sensitive-value")

    def __len__(self) -> int:
        raise RuntimeError("synthetic-sensitive-value")


def test_hostile_mapping_errors_are_value_free() -> None:
    with pytest.raises(ValueError) as exc_info:
        scrub_audit_artifacts(
            [_ExplodingMapping()],
            _policy(),
            as_of=AS_OF,
        )

    assert "synthetic-sensitive-value" not in str(exc_info.value)


def test_report_json_is_strict_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = scrub_audit_artifacts(_artifacts(), _policy(), as_of=AS_OF)
    encoded = report.to_json(indent=None)
    duplicate = encoded.replace("{", '{"format":"duplicate",', 1)
    nonfinite = encoded.replace('"version":1', '"version":NaN')

    with pytest.raises(ValueError, match="invalid retention report JSON"):
        AuditRetentionReport.from_json(duplicate)
    with pytest.raises(ValueError, match="invalid retention report JSON"):
        AuditRetentionReport.from_json(nonfinite)
    with pytest.raises(ValueError, match="indentation"):
        report.to_json(indent=1_000_000)

    monkeypatch.setattr(audit_retention, "MAX_AUDIT_RETENTION_JSON_BYTES", 32)
    with pytest.raises(ValueError, match="size limit"):
        AuditRetentionReport.from_json(encoded)


def test_direct_evidence_rejects_unknown_reason() -> None:
    with pytest.raises(ValueError, match="reason is not supported"):
        DeletionFingerprint(
            artifact_fingerprint="sha256:" + "0" * 64,
            disposition="operational",
            age_seconds=1,
            reason="caller_supplied",
        )


def test_retain_rule_rejects_ignored_age_limit() -> None:
    with pytest.raises(ValueError, match="must not specify"):
        RetentionRule(max_age=timedelta(days=1), action="retain")
