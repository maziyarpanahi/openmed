"""Tests for bounded multimodal media-type detection."""

from __future__ import annotations

import pytest

from openmed.multimodal.media_type import (
    MAX_MEDIA_TYPE_PREFIX_BYTES,
    MediaTypeStatus,
    detect_media_type,
    validate_media_type,
)


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        (b"%PDF-1.7\n", "application/pdf"),
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff\xe0", "image/jpeg"),
        (b"II*\x00", "image/tiff"),
        (b"MM\x00*", "image/tiff"),
        (b"\x00" * 128 + b"DICM", "application/dicom"),
        (b"RIFF\x10\x00\x00\x00WAVE", "audio/wav"),
    ],
)
def test_detect_media_type_from_synthetic_prefix(prefix, expected):
    assert detect_media_type(prefix) == expected


@pytest.mark.parametrize(
    "prefix",
    [
        b"",
        b"%PD",
        b"\x89PNG",
        b"\xff\xd8",
        b"RIFF\x10\x00\x00\x00",
        b"\x00" * 128 + b"DIC",
        b"not-a-supported-format",
    ],
)
def test_truncated_ambiguous_and_unsupported_prefixes_are_unknown(prefix):
    assert detect_media_type(prefix) is None


def test_detection_accepts_bytes_like_inputs_and_uses_a_bounded_prefix():
    payload = memoryview(b"%PDF-1.7" + b"x" * (MAX_MEDIA_TYPE_PREFIX_BYTES + 500))

    assert detect_media_type(payload) == "application/pdf"
    assert MAX_MEDIA_TYPE_PREFIX_BYTES == 132


def test_detection_never_recognizes_a_signature_beyond_the_prefix_bound() -> None:
    payload = b"x" * MAX_MEDIA_TYPE_PREFIX_BYTES + b"%PDF-1.7"

    assert detect_media_type(payload) is None


@pytest.mark.parametrize(
    ("prefix", "declared", "expected"),
    [
        (b"%PDF-1.7", "application/pdf", MediaTypeStatus.MATCH),
        (b"%PDF-1.7", "image/png", MediaTypeStatus.MISMATCH),
        (b"%PDF-1.7", "application/octet-stream", MediaTypeStatus.MISMATCH),
        (b"unknown", "application/pdf", MediaTypeStatus.UNKNOWN),
    ],
)
def test_validate_media_type_returns_stable_categories(prefix, declared, expected):
    assert validate_media_type(prefix, declared) is expected


@pytest.mark.parametrize("value", ["%PDF-1.7", 42, None])
def test_detector_rejects_non_bytes_like_values(value):
    with pytest.raises(TypeError, match="prefix must be bytes-like"):
        detect_media_type(value)  # type: ignore[arg-type]


def test_validator_rejects_invalid_declared_media_type_without_echoing_value():
    declared = "https://patient.example/scan.dcm"

    with pytest.raises(ValueError) as exc_info:
        validate_media_type(b"%PDF-1.7", declared)

    assert declared not in str(exc_info.value)


def test_media_type_contract_is_available_from_public_multimodal_api():
    import openmed.multimodal as multimodal

    assert multimodal.detect_media_type is detect_media_type
    assert multimodal.validate_media_type is validate_media_type
    assert multimodal.MediaTypeStatus is MediaTypeStatus
