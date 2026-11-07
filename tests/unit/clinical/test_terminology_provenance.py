"""Focused synthetic tests for terminology snapshot provenance."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from openmed.clinical.terminology.provenance import (
    EXPIRY_STATUS_EXPIRED,
    EXPIRY_STATUS_EXPIRING,
    EXPIRY_STATUS_FRESH,
    ExpiryPolicy,
    SnapshotExpiredError,
    SnapshotManifest,
    build_freshness_report,
    build_provenance_report,
    build_snapshot_manifest,
    checksum_bytes,
    checksum_file,
    load_snapshot_manifest,
    render_freshness_report,
    require_fresh_snapshot,
    save_snapshot_manifest,
)

_IMPORTED_AT = "2026-01-01T00:00:00Z"
_AS_OF = "2026-01-20T00:00:00Z"


def _manifest(
    *,
    source_name: str = "synthetic terminology",
    source_version: str = "2026.01",
    max_age_days: int | None = 30,
    reject_expired: bool = True,
) -> SnapshotManifest:
    return build_snapshot_manifest(
        source_name,
        source_version,
        b"synthetic terminology snapshot bytes",
        imported_at=_IMPORTED_AT,
        expiry_policy=ExpiryPolicy(
            max_age_days=max_age_days,
            reject_expired=reject_expired,
        ),
    )


def test_manifest_records_metadata_without_snapshot_values() -> None:
    manifest = build_snapshot_manifest(
        "synthetic source",
        "2026.01",
        b"synthetic term that must not be retained",
        imported_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        expiry_policy={"max_age_days": 30, "on_expiry": "reject"},
    )

    payload = manifest.to_dict()
    serialized = manifest.to_json()

    assert payload["source_name"] == "synthetic source"
    assert payload["source_version"] == "2026.01"
    assert payload["checksum"].startswith("sha256:")
    assert payload["imported_at"] == _IMPORTED_AT
    assert payload["expiry_policy"] == {
        "max_age_days": 30,
        "reject_expired": True,
    }
    assert "synthetic term that must not be retained" not in serialized


def test_manifest_serialization_round_trip_and_file_io(tmp_path: Path) -> None:
    manifest = _manifest()
    path = save_snapshot_manifest(manifest, tmp_path / "manifest.json")

    assert load_snapshot_manifest(path) == manifest
    assert json.loads(path.read_text(encoding="utf-8")) == manifest.to_dict()
    assert manifest.version == manifest.source_version
    assert manifest.import_time == manifest.imported_at
    assert manifest.source_checksum == manifest.checksum


def test_reports_are_deterministic_and_sort_manifests() -> None:
    first = _manifest(source_name="z source")
    second = _manifest(source_name="a source", max_age_days=25)

    report_one = build_provenance_report([first, second])
    report_two = build_provenance_report([second, first])
    freshness_one = build_freshness_report(
        [first, second], _AS_OF, expiring_within_days=10
    )
    freshness_two = build_freshness_report(
        [second, first], _AS_OF, expiring_within_days=10
    )

    assert report_one.to_json() == report_two.to_json()
    assert freshness_one.to_json() == freshness_two.to_json()
    assert [record.source_name for record in freshness_one.snapshots] == [
        "a source",
        "z source",
    ]
    assert freshness_one.ok is True
    assert freshness_one.snapshots[0].status == EXPIRY_STATUS_EXPIRING
    assert freshness_one.snapshots[1].status == EXPIRY_STATUS_FRESH


def test_expired_snapshot_is_rejected_only_when_policy_requires_it() -> None:
    strict = _manifest(max_age_days=10, reject_expired=True)
    permissive = _manifest(max_age_days=10, reject_expired=False)

    strict_report = build_freshness_report([strict], _AS_OF)
    assert strict_report.snapshots[0].status == EXPIRY_STATUS_EXPIRED
    assert strict_report.rejection_required is True
    with pytest.raises(SnapshotExpiredError, match="expired"):
        require_fresh_snapshot(strict, _AS_OF)

    assert require_fresh_snapshot(permissive, _AS_OF) == permissive


def test_checksum_and_rendering_are_offline_and_value_free(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.bin"
    snapshot_path.write_bytes(b"synthetic bytes only")
    manifest = build_snapshot_manifest(
        "local synthetic source",
        "v1",
        snapshot_path,
        imported_at=_IMPORTED_AT,
        expiry_policy=None,
    )

    assert manifest.checksum == checksum_bytes(b"synthetic bytes only")
    rendered = render_freshness_report([manifest], _AS_OF)
    assert "# Terminology Snapshot Freshness Report" in rendered
    assert "no_expiry_policy" in rendered
    assert "synthetic bytes only" not in rendered


def test_checksum_file_streams_snapshots_larger_than_one_chunk(tmp_path: Path) -> None:
    snapshot = b"a" * (1024 * 1024) + b"bounded-tail"
    snapshot_path = tmp_path / "large-snapshot.bin"
    snapshot_path.write_bytes(snapshot)

    assert checksum_file(snapshot_path) == checksum_bytes(snapshot)


def test_invalid_manifest_does_not_echo_sensitive_input() -> None:
    with pytest.raises(ValueError) as error:
        SnapshotManifest(
            source_name="synthetic source",
            source_version="v1",
            checksum="not-a-checksum",
            imported_at=_IMPORTED_AT,
        )

    assert "not-a-checksum" not in str(error.value)
