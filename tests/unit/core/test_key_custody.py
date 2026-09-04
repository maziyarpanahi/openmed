"""Tests for privacy-safe key custody metadata validation."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from openmed.core.key_custody import (
    KeyCustodyMetadata,
    KeyCustodyValidationError,
    require_valid_key_custody_metadata,
    validate_key_custody_metadata,
)


def _metadata(**overrides):
    payload = {
        "key_id": "signing-primary-2026",
        "purpose": "signing",
        "algorithm": "HMAC-SHA256",
        "created_at": "2026-08-01T10:00:00Z",
        "state": "active",
    }
    payload.update(overrides)
    return payload


def test_valid_metadata_is_deterministic_and_reports_only_identifier_digests():
    records = [
        _metadata(),
        _metadata(key_id="surrogate-epoch-0001", purpose="surrogate"),
    ]

    first = validate_key_custody_metadata(records)
    second = validate_key_custody_metadata(records)

    assert first.valid
    assert first.to_dict() == second.to_dict()
    assert first.active_purposes == ("signing", "surrogate")
    assert first.state_counts == (
        ("active", 2),
        ("rotated", 0),
        ("retired", 0),
        ("destroyed", 0),
    )
    assert "signing-primary-2026" not in first.to_json()
    assert "surrogate-epoch-0001" not in first.to_json()
    assert json.loads(first.to_json()) == first.to_dict()


def test_dataclass_metadata_and_timezone_aware_datetime_are_supported():
    metadata = KeyCustodyMetadata(
        key_id="surrogate-epoch-0002",
        purpose="surrogate-vault",
        algorithm="hmac_sha256",
        created_at=datetime(2026, 8, 1, 10, tzinfo=timezone.utc),
    )

    result = validate_key_custody_metadata(metadata)

    assert result.valid
    assert result.records_checked == 1


def test_rotation_retirement_and_destruction_follow_ordered_transitions():
    result = validate_key_custody_metadata(
        _metadata(
            key_id="surrogate-epoch-0001",
            purpose="surrogate",
            state="destroyed",
            rotated_at="2026-08-02T10:00:00Z",
            retired_at="2026-08-03T10:00:00Z",
            destroyed_at="2026-08-04T10:00:00Z",
            transitions=(
                {"state": "active", "at": "2026-08-01T10:00:00Z"},
                {"state": "rotated", "at": "2026-08-02T10:00:00Z"},
                {"state": "retired", "at": "2026-08-03T10:00:00Z"},
                {"state": "destroyed", "at": "2026-08-04T10:00:00Z"},
            ),
        )
    )

    assert result.valid


def test_explicit_lifecycle_timestamps_must_match_transition_history():
    result = validate_key_custody_metadata(
        _metadata(
            key_id="surrogate-epoch-0001",
            purpose="surrogate",
            state="rotated",
            rotated_at="2026-08-03T10:00:00Z",
            transitions=(
                {"state": "active", "at": "2026-08-01T10:00:00Z"},
                {"state": "rotated", "at": "2026-08-02T10:00:00Z"},
            ),
        )
    )

    assert not result.valid
    assert any(
        violation.code == "transition_timestamp_mismatch"
        for violation in result.violations
    )


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        (
            {"state": "active", "retired_at": "2026-08-02T10:00:00Z"},
            "invalid_transition",
        ),
        (
            {"state": "destroyed", "destroyed_at": "2026-08-03T10:00:00Z"},
            "invalid_transition",
        ),
        (
            {
                "state": "retired",
                "retired_at": "2026-07-31T10:00:00Z",
            },
            "invalid_transition",
        ),
        (
            {
                "transitions": (
                    {"state": "active", "at": "2026-08-01T10:00:00Z"},
                    {"state": "destroyed", "at": "2026-08-02T10:00:00Z"},
                )
            },
            "invalid_transition",
        ),
    ],
)
def test_invalid_lifecycle_transitions_are_rejected(overrides, code):
    result = validate_key_custody_metadata(_metadata(**overrides))

    assert not result.valid
    assert code in {violation.code for violation in result.violations}


def test_only_one_active_key_may_hold_a_purpose():
    result = validate_key_custody_metadata(
        [
            _metadata(key_id="signing-primary-2026"),
            _metadata(key_id="signing-secondary-2026"),
        ]
    )

    assert not result.valid
    assert any(
        violation.code == "overlapping_purpose" for violation in result.violations
    )


def test_algorithm_must_fit_the_declared_purpose():
    result = validate_key_custody_metadata(
        _metadata(purpose="surrogate", algorithm="Ed25519")
    )

    assert not result.valid
    assert any(
        violation.code == "incompatible_algorithm" for violation in result.violations
    )


def test_retired_key_can_coexist_with_new_active_key_for_same_purpose():
    result = validate_key_custody_metadata(
        [
            _metadata(
                key_id="signing-retired-2025",
                state="retired",
                retired_at="2026-08-02T10:00:00Z",
            ),
            _metadata(key_id="signing-primary-2026"),
        ]
    )

    assert result.valid


def test_secret_like_fields_and_key_bytes_are_rejected_without_echoing_values():
    sensitive_value = b"synthetic-key-material-must-not-be-accepted"
    result = validate_key_custody_metadata(
        _metadata(private_key=sensitive_value, key_bytes=sensitive_value)
    )

    assert not result.valid
    assert {violation.code for violation in result.violations} >= {
        "sensitive_field_rejected",
        "key_material_rejected",
    }
    assert sensitive_value.decode() not in str(result.violations)
    assert sensitive_value.decode() not in result.to_json()

    with pytest.raises(KeyCustodyValidationError) as raised:
        require_valid_key_custody_metadata(
            _metadata(secret_value="synthetic-secret-value")
        )
    assert "synthetic-secret-value" not in str(raised.value)


def test_unknown_fields_are_rejected_and_raw_values_never_enter_report():
    raw_value = "synthetic-sensitive-metadata-value"
    result = validate_key_custody_metadata(_metadata(operator_note=raw_value))

    assert not result.valid
    assert any(violation.code == "unsupported_field" for violation in result.violations)
    assert raw_value not in result.to_json()


def test_overlapping_purposes_inside_one_record_are_rejected():
    payload = _metadata()
    payload.pop("purpose")
    payload["purposes"] = ("signing", "surrogate")

    result = validate_key_custody_metadata(payload)

    assert not result.valid
    assert any(
        violation.code == "overlapping_purpose" for violation in result.violations
    )
