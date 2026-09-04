"""Tests for privacy-safe multimodal asset manifests."""

from __future__ import annotations

import json

import pytest

from openmed.multimodal.asset_manifest import (
    MAX_MANIFEST_BYTE_SIZE,
    MAX_MANIFEST_COUNT,
    MAX_MANIFEST_DURATION_SECONDS,
    AssetManifest,
    AssetManifestError,
)

VALID_DIGEST = "a" * 64


@pytest.mark.parametrize(
    "payload",
    [
        {
            "asset_id": "img-001",
            "media_type": "image/png",
            "sha256": VALID_DIGEST,
            "byte_size": 1024,
            "width": 640,
            "height": 480,
        },
        {
            "asset_id": "pdf-001",
            "media_type": "application/pdf",
            "sha256": VALID_DIGEST,
            "byte_size": 2048,
            "pages": 4,
        },
        {
            "asset_id": "dicom-001",
            "media_type": "application/dicom",
            "sha256": VALID_DIGEST,
            "byte_size": 4096,
            "frames": 12,
            "width": 512,
            "height": 512,
        },
        {
            "asset_id": "audio-001",
            "media_type": "audio/wav",
            "sha256": VALID_DIGEST,
            "byte_size": 8192,
            "duration_seconds": 3.5,
        },
    ],
)
def test_valid_asset_examples_round_trip_with_stable_json(payload):
    manifest = AssetManifest.from_dict(payload)

    assert AssetManifest.from_json(manifest.to_json()) == manifest
    assert json.loads(manifest.to_json()) == manifest.to_dict()
    assert list(json.loads(manifest.to_json())) == sorted(manifest.to_dict())
    assert manifest.to_json() == manifest.to_json()


def test_to_dict_uses_stable_order_and_omits_unset_fields():
    manifest = AssetManifest.from_dict(
        {
            "media_type": "application/pdf",
            "byte_size": 20,
            "sha256": VALID_DIGEST,
            "asset_id": "asset-123",
            "pages": 2,
        }
    )

    assert list(manifest.to_dict()) == [
        "version",
        "asset_id",
        "media_type",
        "sha256",
        "byte_size",
        "pages",
    ]
    assert "width" not in manifest.to_dict()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sha256", "A" * 64),
        ("sha256", "abc"),
        ("sha256", "g" * 64),
        ("byte_size", 0),
        ("byte_size", -1),
        ("byte_size", 2.2),
        ("byte_size", True),
        ("byte_size", MAX_MANIFEST_BYTE_SIZE + 1),
        ("pages", -1),
        ("pages", MAX_MANIFEST_COUNT + 1),
        ("width", 0),
        ("height", -2),
        ("frames", 1.5),
        ("duration_seconds", 0.0),
        ("duration_seconds", -1.0),
        ("duration_seconds", float("inf")),
        ("duration_seconds", float("nan")),
        ("duration_seconds", MAX_MANIFEST_DURATION_SECONDS + 1),
        ("version", 2),
        ("version", 1.0),
        ("version", True),
        ("media_type", "text/plain"),
        ("media_type", "Image/PNG"),
    ],
)
def test_invalid_field_values_fail_closed(field, value):
    payload = {
        "version": 1,
        "asset_id": "asset-001",
        "media_type": "image/png",
        "sha256": VALID_DIGEST,
        "byte_size": 1024,
    }
    payload[field] = value

    with pytest.raises(AssetManifestError):
        AssetManifest.from_dict(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"asset_id": "asset-001", "media_type": "image/png", "byte_size": 1},
        {
            "asset_id": "asset-001",
            "media_type": "image/png",
            "sha256": VALID_DIGEST,
            "byte_size": 1,
            "description": "Patient note",
        },
        "not-a-dict",
    ],
)
def test_missing_unknown_and_non_mapping_payloads_fail_closed(payload):
    with pytest.raises(AssetManifestError):
        AssetManifest.from_dict(payload)  # type: ignore[arg-type]


def test_unknown_free_text_field_fails_without_echo():
    payload = {
        "asset_id": "asset-001",
        "media_type": "image/png",
        "sha256": VALID_DIGEST,
        "byte_size": 1,
        "description": "synthetic sentinel chart text",
    }

    with pytest.raises(AssetManifestError) as exc_info:
        AssetManifest.from_dict(payload)

    assert payload["description"] not in str(exc_info.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("asset_id", "/tmp/patient.png"),
        ("asset_id", "s3://bucket/patient.png"),
        ("asset_id", "C:\\patient\\scan.dcm"),
        ("media_type", "https://example.test/image/png"),
        ("sha256", "/tmp/" + VALID_DIGEST),
    ],
)
def test_paths_urls_and_location_like_values_fail_without_echo(field, value):
    payload = {
        "version": 1,
        "asset_id": "asset-001",
        "media_type": "image/png",
        "sha256": VALID_DIGEST,
        "byte_size": 1024,
    }
    payload[field] = value

    with pytest.raises(AssetManifestError) as exc_info:
        AssetManifest.from_dict(payload)

    assert value not in str(exc_info.value)


@pytest.mark.parametrize("payload", ["{", b"\xff", 42])
def test_malformed_json_fails_closed(payload):
    with pytest.raises(AssetManifestError):
        AssetManifest.from_json(payload)  # type: ignore[arg-type]


def test_duplicate_json_fields_fail_closed_without_echoing_values() -> None:
    payload = (
        '{"asset_id":"asset-001","asset_id":"synthetic-secret",'
        f'"media_type":"image/png","sha256":"{VALID_DIGEST}","byte_size":1}}'
    )

    with pytest.raises(AssetManifestError) as exc_info:
        AssetManifest.from_json(payload)

    assert "synthetic-secret" not in str(exc_info.value)


def test_scalar_subclasses_do_not_bypass_strict_types() -> None:
    class IntSubclass(int):
        pass

    with pytest.raises(AssetManifestError):
        AssetManifest(
            asset_id="asset-001",
            media_type="image/png",
            sha256=VALID_DIGEST,
            byte_size=IntSubclass(1),
        )


def test_manifest_contract_is_available_from_public_multimodal_api():
    import openmed.multimodal as multimodal

    assert multimodal.AssetManifest is AssetManifest
    assert multimodal.AssetManifestError is AssetManifestError
    assert multimodal.MANIFEST_VERSION == 1
