import math

import pytest

from openmed.multimodal.manifest_profiles import (
    AUDIO_V1,
    DICOM_V1,
    IMAGE_V1,
    PDF_V1,
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
            {"width": 800, "height": 600, "page_count": 10},
            [ValidationFinding("page_count", "inapplicable_present")],
            id="image_inapplicable_page_count",
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
            {"page_count": 5},
            [],
            id="pdf_valid",
        ),
        pytest.param(
            {},
            [ValidationFinding("page_count", "missing_required")],
            id="pdf_missing_page_count",
        ),
        pytest.param(
            {"page_count": 0},
            [ValidationFinding("page_count", "invalid_zero")],
            id="pdf_zero_page_count",
        ),
        pytest.param(
            {"page_count": True},
            [ValidationFinding("page_count", "invalid_boolean")],
            id="pdf_boolean_page_count",
        ),
        pytest.param(
            {"page_count": math.nan},
            [ValidationFinding("page_count", "non_finite_numeric")],
            id="pdf_nan_page_count",
        ),
        pytest.param(
            {"page_count": 5, "duration": 120.5},
            [ValidationFinding("duration", "inapplicable_present")],
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
            {"frame_count": 1, "width": 512, "height": 512},
            [],
            id="dicom_valid",
        ),
        pytest.param(
            {"width": 512, "height": 512},
            [ValidationFinding("frame_count", "missing_required")],
            id="dicom_missing_frame_count",
        ),
        pytest.param(
            {"frame_count": 1, "height": 512},
            [ValidationFinding("width", "missing_required")],
            id="dicom_missing_width",
        ),
        pytest.param(
            {"frame_count": 1, "width": 512},
            [ValidationFinding("height", "missing_required")],
            id="dicom_missing_height",
        ),
        pytest.param(
            {"frame_count": 0, "width": 512, "height": 512},
            [ValidationFinding("frame_count", "invalid_zero")],
            id="dicom_zero_frame_count",
        ),
        pytest.param(
            {"frame_count": 1, "width": 0, "height": 512},
            [ValidationFinding("width", "invalid_zero")],
            id="dicom_zero_dimensions",
        ),
        pytest.param(
            {"frame_count": False, "width": 512, "height": 512},
            [ValidationFinding("frame_count", "invalid_boolean")],
            id="dicom_boolean_frame_count",
        ),
        pytest.param(
            {"frame_count": 1, "width": math.inf, "height": 512},
            [ValidationFinding("width", "non_finite_numeric")],
            id="dicom_inf_dimensions",
        ),
        pytest.param(
            {"frame_count": 1, "width": 512, "height": 512, "page_count": 10},
            [ValidationFinding("page_count", "inapplicable_present")],
            id="dicom_inapplicable_page_count",
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
            {"duration": 120.5},
            [],
            id="audio_valid",
        ),
        pytest.param(
            {},
            [ValidationFinding("duration", "missing_required")],
            id="audio_missing_duration",
        ),
        pytest.param(
            {"duration": 0},
            [ValidationFinding("duration", "invalid_zero")],
            id="audio_zero_duration",
        ),
        pytest.param(
            {"duration": True},
            [ValidationFinding("duration", "invalid_boolean")],
            id="audio_boolean_duration",
        ),
        pytest.param(
            {"duration": math.nan},
            [ValidationFinding("duration", "non_finite_numeric")],
            id="audio_nan_duration",
        ),
        pytest.param(
            {"duration": -math.inf},
            [ValidationFinding("duration", "non_finite_numeric")],
            id="audio_neg_inf_duration",
        ),
        pytest.param(
            {"duration": 120.5, "width": 100},
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
        # "page_count" is missing, which is inapplicable for audio, but we test AUDIO_V1
        "frame_count": 5,  # inapplicable for audio
    }
    # For audio, required is duration. Inapplicable are width, height, page_count, frame_count.
    # Sorted order of fields for audio: duration, frame_count, page_count, height, width
    # But python sort order of strings: duration, frame_count, height, page_count, width

    findings = validate_manifest_metadata(AUDIO_V1, manifest)

    expected = [
        ValidationFinding("duration", "missing_required"),
        ValidationFinding("frame_count", "inapplicable_present"),
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
