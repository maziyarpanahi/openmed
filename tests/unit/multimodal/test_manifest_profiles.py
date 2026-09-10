import math

import pytest

from openmed.multimodal.asset_manifest import AssetManifest
from openmed.multimodal.manifest_profiles import (
    AUDIO_V1,
    DICOM_V1,
    IMAGE_V1,
    PDF_V1,
    ManifestProfileError,
    ValidationFinding,
    validate_manifest_metadata,
)


@pytest.mark.parametrize(
    "manifest, expected_findings",
    [
        # IMAGE Profile
        pytest.param(
            {"width": 800, "height": 600},
            [],
            id="image_valid",
        ),
        pytest.param(
            {"height": 600},
            [ValidationFinding("width", "missing_required")],
            id="image_missing_width",
        ),
        pytest.param(
            {"width": 800},
            [ValidationFinding("height", "missing_required")],
            id="image_missing_height",
        ),
        pytest.param(
            {"width": 0, "height": 600},
            [ValidationFinding("width", "invalid_zero")],
            id="image_zero_width",
        ),
        pytest.param(
            {"width": 800, "height": 0},
            [ValidationFinding("height", "invalid_zero")],
            id="image_zero_height",
        ),
        pytest.param(
            {"width": True, "height": 600},
            [ValidationFinding("width", "invalid_boolean")],
            id="image_boolean_width",
        ),
        pytest.param(
            {"width": 800, "height": False},
            [ValidationFinding("height", "invalid_boolean")],
            id="image_boolean_height",
        ),
        pytest.param(
            {"width": math.inf, "height": 600},
            [ValidationFinding("width", "non_finite_numeric")],
            id="image_inf_width",
        ),
        pytest.param(
            {"width": 800, "height": math.nan},
            [ValidationFinding("height", "non_finite_numeric")],
            id="image_nan_height",
        ),
        pytest.param(
            {"width": 800, "height": 600, "pages": 10},
            [ValidationFinding("pages", "inapplicable_present")],
            id="image_inapplicable_pages",
        ),
    ],
)
def test_image_profile(manifest, expected_findings):
    assert validate_manifest_metadata(IMAGE_V1, manifest) == expected_findings


@pytest.mark.parametrize(
    "manifest, expected_findings",
    [
        # PDF Profile
        pytest.param(
            {"pages": 5},
            [],
            id="pdf_valid",
        ),
        pytest.param(
            {},
            [ValidationFinding("pages", "missing_required")],
            id="pdf_missing_pages",
        ),
        pytest.param(
            {"pages": 0},
            [ValidationFinding("pages", "invalid_zero")],
            id="pdf_zero_pages",
        ),
        pytest.param(
            {"pages": True},
            [ValidationFinding("pages", "invalid_boolean")],
            id="pdf_boolean_pages",
        ),
        pytest.param(
            {"pages": math.nan},
            [ValidationFinding("pages", "non_finite_numeric")],
            id="pdf_nan_pages",
        ),
        pytest.param(
            {"pages": 5, "duration_seconds": 120.5},
            [ValidationFinding("duration_seconds", "inapplicable_present")],
            id="pdf_inapplicable_duration",
        ),
    ],
)
def test_pdf_profile(manifest, expected_findings):
    assert validate_manifest_metadata(PDF_V1, manifest) == expected_findings


@pytest.mark.parametrize(
    "manifest, expected_findings",
    [
        # DICOM Profile
        pytest.param(
            {"frames": 1, "width": 512, "height": 512},
            [],
            id="dicom_valid",
        ),
        pytest.param(
            {"width": 512, "height": 512},
            [ValidationFinding("frames", "missing_required")],
            id="dicom_missing_frames",
        ),
        pytest.param(
            {"frames": 1, "height": 512},
            [ValidationFinding("width", "missing_required")],
            id="dicom_missing_width",
        ),
        pytest.param(
            {"frames": 1, "width": 512},
            [ValidationFinding("height", "missing_required")],
            id="dicom_missing_height",
        ),
        pytest.param(
            {"frames": 0, "width": 512, "height": 512},
            [ValidationFinding("frames", "invalid_zero")],
            id="dicom_zero_frames",
        ),
        pytest.param(
            {"frames": 1, "width": 0, "height": 512},
            [ValidationFinding("width", "invalid_zero")],
            id="dicom_zero_dimensions",
        ),
        pytest.param(
            {"frames": False, "width": 512, "height": 512},
            [ValidationFinding("frames", "invalid_boolean")],
            id="dicom_boolean_frames",
        ),
        pytest.param(
            {"frames": 1, "width": math.inf, "height": 512},
            [ValidationFinding("width", "non_finite_numeric")],
            id="dicom_inf_dimensions",
        ),
        pytest.param(
            {"frames": 1, "width": 512, "height": 512, "pages": 10},
            [ValidationFinding("pages", "inapplicable_present")],
            id="dicom_inapplicable_pages",
        ),
    ],
)
def test_dicom_profile(manifest, expected_findings):
    assert validate_manifest_metadata(DICOM_V1, manifest) == expected_findings


@pytest.mark.parametrize(
    "manifest, expected_findings",
    [
        # AUDIO Profile
        pytest.param(
            {"duration_seconds": 120.5},
            [],
            id="audio_valid",
        ),
        pytest.param(
            {},
            [ValidationFinding("duration_seconds", "missing_required")],
            id="audio_missing_duration",
        ),
        pytest.param(
            {"duration_seconds": 0},
            [ValidationFinding("duration_seconds", "invalid_zero")],
            id="audio_zero_duration",
        ),
        pytest.param(
            {"duration_seconds": True},
            [ValidationFinding("duration_seconds", "invalid_boolean")],
            id="audio_boolean_duration",
        ),
        pytest.param(
            {"duration_seconds": math.nan},
            [ValidationFinding("duration_seconds", "non_finite_numeric")],
            id="audio_nan_duration",
        ),
        pytest.param(
            {"duration_seconds": -math.inf},
            [ValidationFinding("duration_seconds", "non_finite_numeric")],
            id="audio_neg_inf_duration",
        ),
        pytest.param(
            {"duration_seconds": 120.5, "width": 100},
            [ValidationFinding("width", "inapplicable_present")],
            id="audio_inapplicable_width",
        ),
    ],
)
def test_audio_profile(manifest, expected_findings):
    assert validate_manifest_metadata(AUDIO_V1, manifest) == expected_findings


def test_determinism_and_multiple_findings():
    """Test that findings are returned deterministically and multiple issues are reported."""
    manifest = {
        "width": 0,  # invalid_zero
        "height": True,  # invalid_boolean
        # "pages" is missing, which is inapplicable for audio, but we test AUDIO_V1
        "frames": 5,  # inapplicable for audio
    }
    # For audio, required is duration_seconds. Inapplicable are width, height, pages, frames.
    # Sorted order of fields for audio: duration_seconds, frames, pages, height, width
    # But python sort order of strings: duration_seconds, frames, height, pages, width

    findings = validate_manifest_metadata(AUDIO_V1, manifest)

    expected = [
        ValidationFinding("duration_seconds", "missing_required"),
        ValidationFinding("frames", "inapplicable_present"),
        ValidationFinding("height", "inapplicable_present"),
        ValidationFinding("width", "inapplicable_present"),
    ]
    assert findings == expected


def test_no_file_access_with_source_path():
    """Ensure that the validator ignores source paths and strictly operates on metadata."""
    manifest = {
        "width": 800,
        "height": 600,
        "source_path": "/this/path/does/not/exist.png",
    }
    findings = validate_manifest_metadata(IMAGE_V1, manifest)

    # Validation should succeed without trying to open source_path.
    # source_path is not in the declared fields, so it's ignored deterministically.
    assert findings == []


@pytest.mark.parametrize(
    ("profile", "manifest", "expected"),
    (
        (IMAGE_V1, {"width": -1, "height": 2}, "out_of_range"),
        (PDF_V1, {"pages": 1.5}, "invalid_type"),
        (
            DICOM_V1,
            {"frames": 1, "width": 2**31, "height": 512},
            "out_of_range",
        ),
        (AUDIO_V1, {"duration_seconds": -1.0}, "out_of_range"),
    ),
)
def test_profile_values_enforce_types_and_ranges(profile, manifest, expected):
    findings = validate_manifest_metadata(profile, manifest)

    assert len(findings) == 1
    assert findings[0].reason_code == expected


def test_findings_cannot_retain_arbitrary_fields_or_reasons() -> None:
    with pytest.raises(ManifestProfileError):
        ValidationFinding("source_path", "invalid_type")
    with pytest.raises(ManifestProfileError):
        ValidationFinding("width", "synthetic private value")


def test_profile_contract_is_available_from_public_multimodal_api() -> None:
    import openmed.multimodal as multimodal

    assert multimodal.IMAGE_V1 is IMAGE_V1
    assert multimodal.validate_manifest_metadata is validate_manifest_metadata
    assert multimodal.ValidationFinding is ValidationFinding


def test_profile_validates_canonical_asset_manifest_without_decoding() -> None:
    manifest = AssetManifest(
        asset_id="synthetic-image-1",
        media_type="image/png",
        sha256="a" * 64,
        byte_size=128,
        width=800,
        height=600,
    )

    assert validate_manifest_metadata(IMAGE_V1, manifest) == []
