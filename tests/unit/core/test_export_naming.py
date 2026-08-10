"""Focused tests for deterministic, privacy-safe export filenames."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from openmed.core.export_naming import (
    DEFAULT_FINGERPRINT_LENGTH,
    EXPORT_FILENAME_SCHEMA_VERSION,
    ExportArtifactMetadata,
    ExportNamingError,
    build_export_filename,
    export_naming_policy,
    fingerprint_for,
    short_fingerprint,
)

_FINGERPRINT = "sha256:" + "a" * 64


def test_filename_is_deterministic_and_identifies_typed_metadata() -> None:
    metadata = ExportArtifactMetadata(
        artifact_type="audit-report",
        format="JSON",
        schema_version="v1",
        fingerprint=_FINGERPRINT,
    )

    first = build_export_filename(metadata)
    second = build_export_filename(
        {
            "schema_version": "v1",
            "fingerprint": _FINGERPRINT,
            "format": "json",
            "artifact_type": "audit-report",
        }
    )

    assert first == second == "audit-report-json-schema-v1-aaaaaaaaaaaa.json"
    assert "/" not in first
    assert "\\" not in first


def test_fingerprint_sources_are_canonical_and_raw_free() -> None:
    first = fingerprint_for({"format": "json", "schema": 1})
    second = fingerprint_for({"schema": 1, "format": "json"})
    raw_identifier = "synthetic-patient-482901"

    assert first == second
    assert first.startswith("sha256:")
    assert len(short_fingerprint(raw_identifier)) == DEFAULT_FINGERPRINT_LENGTH
    assert raw_identifier not in short_fingerprint(raw_identifier)


def test_metadata_report_contains_only_safe_typed_values() -> None:
    raw_identifier = "synthetic-patient-482901"
    metadata = ExportArtifactMetadata(
        artifact_type="deidentified-record",
        format="json",
        schema_version=1,
        fingerprint=fingerprint_for(raw_identifier),
    )

    serialized = json.dumps(metadata.to_dict(), sort_keys=True)

    assert metadata.to_dict() == {
        "artifact_type": "deidentified-record",
        "format": "json",
        "schema_version": "1",
        "fingerprint": fingerprint_for(raw_identifier),
        "extension": "json",
    }
    assert raw_identifier not in serialized


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact_type", "audit/report"),
        ("format", r"json\\backup"),
        ("schema_version", "v1/next"),
        ("extension", "../json"),
        ("fingerprint", "sha256:" + "b" * 63 + "/"),
    ],
)
def test_path_syntax_is_rejected_without_echoing_values(field: str, value: str) -> None:
    fields = {
        "artifact_type": "audit-report",
        "format": "json",
        "schema_version": "v1",
        "fingerprint": _FINGERPRINT,
    }
    fields[field] = value

    with pytest.raises(ExportNamingError) as error:
        build_export_filename(**fields)

    assert value not in str(error.value)
    assert field in str(error.value)


@pytest.mark.parametrize(
    "value",
    [
        "synthetic-patient-482901",
        "synthetic-record-123456",
        "550e8400-e29b-41d4-a716-446655440000",
    ],
)
def test_raw_identifier_tokens_are_rejected_without_echoing_values(value: str) -> None:
    with pytest.raises(ExportNamingError) as error:
        build_export_filename(
            artifact_type=value,
            format="json",
            schema_version="v1",
            fingerprint=_FINGERPRINT,
        )

    assert value not in str(error.value)
    assert "raw identifier" in str(error.value)


def test_unstable_clock_components_are_absent_unless_explicitly_supplied() -> None:
    base = build_export_filename(
        artifact_type="audit-report",
        format="json",
        schema_version="v1",
        fingerprint=_FINGERPRINT,
    )
    explicit = build_export_filename(
        artifact_type="audit-report",
        format="json",
        schema_version="v1",
        fingerprint=_FINGERPRINT,
        explicit_timestamp="2024-01-02T03:04:05Z",
    )

    assert base == "audit-report-json-schema-v1-aaaaaaaaaaaa.json"
    assert explicit == "audit-report-json-schema-v1-aaaaaaaaaaaa-20240102t030405z.json"
    assert explicit != base


def test_explicit_datetime_is_normalized_deterministically() -> None:
    timestamp = datetime(2024, 1, 2, 3, 4, 5, 600, tzinfo=timezone.utc)

    assert build_export_filename(
        artifact_type="audit-report",
        format="json",
        schema_version="v1",
        fingerprint=_FINGERPRINT,
        timestamp=timestamp,
    ).endswith("-20240102t030405000600z.json")


def test_unsupported_metadata_fields_are_rejected_without_serializing_values() -> None:
    with pytest.raises(ExportNamingError, match="unsupported fields") as error:
        build_export_filename(
            {
                "artifact_type": "audit-report",
                "format": "json",
                "schema_version": "v1",
                "fingerprint": _FINGERPRINT,
                "subject_id": "synthetic-subject-482901",
            }
        )

    assert "synthetic-subject-482901" not in str(error.value)


def test_policy_report_is_stable_and_value_free() -> None:
    report = export_naming_policy()

    assert report == {
        "schema_version": EXPORT_FILENAME_SCHEMA_VERSION,
        "hash_algorithm": "sha256",
        "default_fingerprint_length": 12,
        "fingerprint_length": {"minimum": 6, "maximum": 64},
        "timestamp_policy": "omitted_unless_explicitly_supplied",
        "raw_identifier_policy": "reject",
        "path_policy": "relative_single_filename",
    }
