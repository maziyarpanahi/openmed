import dataclasses
import json
import math

import pytest

from openmed.multimodal.asset_limits import (
    DESKTOP_V1,
    LIMIT_FIELDS,
    LIMIT_REASON_CODES,
    MAX_PIXEL_PRODUCT,
    MOBILE_V1,
    AssetLimitError,
    LimitFinding,
    LimitProfile,
    evaluate_asset_limits,
)
from openmed.multimodal.asset_manifest import (
    MAX_MANIFEST_BYTE_SIZE,
    MAX_MANIFEST_COUNT,
    AssetManifest,
)

SHA = "a" * 64


def manifest(**fields):
    base = {
        "asset_id": "asset-1",
        "media_type": "image/png",
        "sha256": SHA,
        "byte_size": 1,
    }
    base.update(fields)
    return base


# --- defaults -----------------------------------------------------------------


def test_default_profiles_match_the_agreed_policy():
    assert MOBILE_V1.to_dict() == {
        "name": "mobile",
        "version": "1.0",
        "max_byte_size": 64 * 1024**2,
        "max_pages": 10,
        "max_pixels": 10_000_000,
        "max_total_pixels": 25_000_000,
        "max_frames": 128,
        "max_duration_seconds": 300,
    }
    assert DESKTOP_V1.to_dict() == {
        "name": "desktop",
        "version": "1.0",
        "max_byte_size": 256 * 1024**2,
        "max_pages": 100,
        "max_pixels": 40_000_000,
        "max_total_pixels": 100_000_000,
        "max_frames": 1_024,
        "max_duration_seconds": 1_800,
    }


def test_desktop_page_and_pixel_ceilings_match_the_pdf_renderer_defaults():
    from openmed.multimodal import render_pdf

    assert DESKTOP_V1.max_pages == render_pdf._DEFAULT_MAX_PAGES
    assert DESKTOP_V1.max_pixels == render_pdf._DEFAULT_MAX_PAGE_PIXELS
    assert DESKTOP_V1.max_total_pixels == render_pdf._DEFAULT_MAX_TOTAL_PIXELS


def test_profiles_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        MOBILE_V1.max_pages = 5  # type: ignore[misc]


def test_with_limits_returns_a_revalidated_copy():
    tighter = MOBILE_V1.with_limits(max_pages=5, max_duration_seconds=60.5)
    assert tighter.max_pages == 5
    assert tighter.max_duration_seconds == 60.5
    assert MOBILE_V1.max_pages == 10
    with pytest.raises(AssetLimitError):
        MOBILE_V1.with_limits(max_pages=0)
    with pytest.raises(AssetLimitError):
        MOBILE_V1.with_limits(max_widgets=1)


# --- profile validation --------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"max_byte_size": 0}, id="zero_bytes"),
        pytest.param({"max_pages": -1}, id="negative_pages"),
        pytest.param({"max_frames": True}, id="boolean_frames"),
        pytest.param({"max_pixels": 1.5}, id="float_pixels"),
        pytest.param({"max_pages": MAX_MANIFEST_COUNT + 1}, id="pages_over_bound"),
        pytest.param(
            {"max_byte_size": MAX_MANIFEST_BYTE_SIZE + 1}, id="bytes_over_bound"
        ),
        pytest.param({"max_duration_seconds": math.inf}, id="infinite_duration"),
        pytest.param({"max_duration_seconds": 0}, id="zero_duration"),
        pytest.param({"max_pixels": 30_000_000}, id="pixels_above_total"),
        pytest.param({"version": "2.0"}, id="unsupported_version"),
        pytest.param({"name": "Mobile"}, id="uppercase_name"),
        pytest.param({"name": ""}, id="empty_name"),
        pytest.param({"name": "a/b"}, id="path_like_name"),
    ],
)
def test_invalid_profiles_are_rejected(overrides):
    values = MOBILE_V1.to_dict()
    values.update(overrides)
    with pytest.raises(AssetLimitError):
        LimitProfile(**values)


# --- boundaries, per representable rule ----------------------------------------


def _finding(field_name, limit, observed):
    return LimitFinding(field_name, "limit_exceeded", limit, observed)


@pytest.mark.parametrize("modality", ["image", "pdf", "dicom", "audio"])
def test_bytes_boundary_on_every_modality(modality):
    extra = {
        "image": {"width": 10, "height": 10},
        "pdf": {"pages": 1},
        "dicom": {"width": 10, "height": 10, "frames": 1},
        "audio": {"duration_seconds": 1.0},
    }[modality]
    limit = MOBILE_V1.max_byte_size
    below = evaluate_asset_limits(
        MOBILE_V1, manifest(byte_size=limit - 1, **extra), modality
    )
    equal = evaluate_asset_limits(
        MOBILE_V1, manifest(byte_size=limit, **extra), modality
    )
    above = evaluate_asset_limits(
        MOBILE_V1, manifest(byte_size=limit + 1, **extra), modality
    )
    assert not [f for f in below if f.field_name == "byte_size"]
    assert not [f for f in equal if f.field_name == "byte_size"]
    assert [f for f in above if f.field_name == "byte_size"] == [
        _finding("byte_size", limit, limit + 1)
    ]


def test_pdf_pages_boundary():
    limit = MOBILE_V1.max_pages
    for pages, expect in (
        (limit - 1, []),
        (limit, []),
        (limit + 1, [_finding("pages", limit, limit + 1)]),
    ):
        findings = evaluate_asset_limits(MOBILE_V1, manifest(pages=pages), "pdf")
        assert [f for f in findings if f.field_name == "pages"] == expect


def test_pdf_pixel_rules_are_always_unevaluable_and_reported():
    findings = evaluate_asset_limits(MOBILE_V1, manifest(pages=1), "pdf")
    assert findings == [
        LimitFinding("pixels", "insufficient_metadata", MOBILE_V1.max_pixels, None),
        LimitFinding(
            "total_pixels", "insufficient_metadata", MOBILE_V1.max_total_pixels, None
        ),
    ]


def test_pdf_pixels_are_never_inferred_from_page_count_or_smuggled_geometry():
    # Even if a caller passes geometry a PDF manifest must not carry, the PDF
    # pixel rules stay unevaluable rather than computing from it.
    findings = evaluate_asset_limits(
        MOBILE_V1, manifest(pages=1, width=100_000, height=100_000), "pdf"
    )
    assert [f.reason_code for f in findings] == [
        "insufficient_metadata",
        "insufficient_metadata",
    ]
    assert all(f.observed is None for f in findings)


@pytest.mark.parametrize("modality", ["image", "dicom"])
def test_pixels_per_unit_boundary(modality):
    limit = MOBILE_V1.max_pixels  # 10_000_000 = 2_500 * 4_000
    frames = {"frames": 1} if modality == "dicom" else {}
    for width, expect in (
        (2_499, []),
        (2_500, []),
        (2_501, [_finding("pixels", limit, 2_501 * 4_000)]),
    ):
        findings = evaluate_asset_limits(
            MOBILE_V1, manifest(width=width, height=4_000, **frames), modality
        )
        assert [f for f in findings if f.field_name == "pixels"] == expect


def test_image_total_pixels_is_width_times_height():
    tight = MOBILE_V1.with_limits(max_pixels=100, max_total_pixels=100)
    assert evaluate_asset_limits(tight, manifest(width=10, height=10), "image") == []
    findings = evaluate_asset_limits(tight, manifest(width=10, height=11), "image")
    assert findings == [
        _finding("pixels", 100, 110),
        _finding("total_pixels", 100, 110),
    ]


def test_dicom_total_pixels_is_width_times_height_times_frames():
    limit = MOBILE_V1.max_total_pixels  # 25_000_000 = 1_000 * 1_000 * 25
    for frames, expect in (
        (24, []),
        (25, []),
        (26, [_finding("total_pixels", limit, 26_000_000)]),
    ):
        findings = evaluate_asset_limits(
            MOBILE_V1, manifest(width=1_000, height=1_000, frames=frames), "dicom"
        )
        assert [f for f in findings if f.field_name == "total_pixels"] == expect


def test_dicom_frames_boundary():
    limit = MOBILE_V1.max_frames
    tiny = MOBILE_V1.with_limits(max_total_pixels=MOBILE_V1.max_total_pixels)
    for frames, expect in (
        (limit - 1, []),
        (limit, []),
        (limit + 1, [_finding("frames", limit, limit + 1)]),
    ):
        findings = evaluate_asset_limits(
            tiny, manifest(width=2, height=2, frames=frames), "dicom"
        )
        assert [f for f in findings if f.field_name == "frames"] == expect


def test_audio_duration_boundary_accepts_integers_and_floats():
    limit = MOBILE_V1.max_duration_seconds
    assert (
        evaluate_asset_limits(MOBILE_V1, manifest(duration_seconds=limit), "audio")
        == []
    )
    assert (
        evaluate_asset_limits(MOBILE_V1, manifest(duration_seconds=299.9), "audio")
        == []
    )
    assert evaluate_asset_limits(
        MOBILE_V1, manifest(duration_seconds=300.5), "audio"
    ) == [_finding("duration_seconds", limit, 300.5)]


# --- applicability and missing metadata -----------------------------------------


def test_inapplicable_rules_produce_no_findings():
    # Audio has no pages rule; a pages value is simply not evaluated.
    assert (
        evaluate_asset_limits(
            MOBILE_V1, manifest(duration_seconds=1, pages=10_000), "audio"
        )
        == []
    )
    # An image has no frames or duration rule.
    assert (
        evaluate_asset_limits(
            MOBILE_V1,
            manifest(width=1, height=1, frames=10_000, duration_seconds=10_000),
            "image",
        )
        == []
    )


@pytest.mark.parametrize(
    "modality, fields, missing",
    [
        pytest.param("image", {}, ["pixels", "total_pixels"], id="image_no_geometry"),
        pytest.param(
            "image", {"width": 10}, ["pixels", "total_pixels"], id="image_no_height"
        ),
        pytest.param(
            "dicom",
            {"width": 10, "height": 10},
            ["total_pixels", "frames"],
            id="dicom_no_frames",
        ),
        pytest.param(
            "dicom", {"frames": 2}, ["pixels", "total_pixels"], id="dicom_no_geometry"
        ),
        pytest.param("audio", {}, ["duration_seconds"], id="audio_no_duration"),
        pytest.param("pdf", {}, ["pages", "pixels", "total_pixels"], id="pdf_no_pages"),
    ],
)
def test_missing_inputs_are_reported_never_assumed_safe(modality, fields, missing):
    findings = evaluate_asset_limits(MOBILE_V1, manifest(**fields), modality)
    assert [f.field_name for f in findings] == missing
    assert all(
        f.reason_code == "insufficient_metadata" and f.observed is None
        for f in findings
    )


def test_missing_byte_size_in_a_mapping_is_reported():
    findings = evaluate_asset_limits(MOBILE_V1, {"width": 1, "height": 1}, "image")
    assert findings == [
        LimitFinding(
            "byte_size", "insufficient_metadata", MOBILE_V1.max_byte_size, None
        )
    ]


# --- ordering, output, inputs --------------------------------------------------------


def test_multiple_findings_are_returned_in_the_documented_order():
    findings = evaluate_asset_limits(
        MOBILE_V1,
        manifest(
            byte_size=MOBILE_V1.max_byte_size + 1,
            width=4_000,
            height=4_000,
            frames=MOBILE_V1.max_frames + 1,
        ),
        "dicom",
    )
    assert [f.field_name for f in findings] == [
        "byte_size",
        "pixels",
        "total_pixels",
        "frames",
    ]
    assert [LIMIT_FIELDS.index(f.field_name) for f in findings] == sorted(
        LIMIT_FIELDS.index(f.field_name) for f in findings
    )
    assert findings[1].observed == 16_000_000
    assert findings[2].observed == 16_000_000 * (MOBILE_V1.max_frames + 1)


def test_findings_serialise_deterministically_and_carry_only_numbers():
    findings = evaluate_asset_limits(MOBILE_V1, manifest(pages=11), "pdf")
    payload = json.dumps([f.to_dict() for f in findings], sort_keys=False)
    assert payload == (
        '[{"field_name": "pages", "reason_code": "limit_exceeded", "limit": 10, "observed": 11}, '
        '{"field_name": "pixels", "reason_code": "insufficient_metadata", "limit": 10000000, "observed": null}, '
        '{"field_name": "total_pixels", "reason_code": "insufficient_metadata", "limit": 25000000, "observed": null}]'
    )
    assert "asset-1" not in payload and SHA not in payload


def test_accepts_a_canonical_asset_manifest_instance():
    instance = AssetManifest.from_dict(
        manifest(media_type="audio/wav", duration_seconds=301)
    )
    assert evaluate_asset_limits(MOBILE_V1, instance, "audio") == [
        _finding("duration_seconds", 300, 301)
    ]


def test_findings_cannot_be_constructed_outside_the_allowlists():
    with pytest.raises(AssetLimitError):
        LimitFinding("path", "limit_exceeded", 1, 1)
    with pytest.raises(AssetLimitError):
        LimitFinding("pages", "too_big", 1, 1)
    with pytest.raises(AssetLimitError):
        LimitFinding("pages", "insufficient_metadata", 1, 5)
    with pytest.raises(AssetLimitError):
        LimitFinding("pages", "limit_exceeded", 1, True)
    assert set(LIMIT_REASON_CODES) == {"limit_exceeded", "insufficient_metadata"}


# --- checked arithmetic and content-free errors -----------------------------------


def test_overflow_bound_is_the_cube_of_the_manifest_count_bound():
    assert MAX_PIXEL_PRODUCT == MAX_MANIFEST_COUNT**3


def test_maximal_inputs_multiply_exactly_without_wraparound():
    big = MAX_MANIFEST_COUNT
    findings = evaluate_asset_limits(
        DESKTOP_V1, manifest(width=big, height=big, frames=big), "dicom"
    )
    total = next(f for f in findings if f.field_name == "total_pixels")
    assert total.observed == big**3 == MAX_PIXEL_PRODUCT
    assert total.reason_code == "limit_exceeded"


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"width": True, "height": 10}, id="boolean_width"),
        pytest.param({"width": 10.0, "height": 10}, id="float_width"),
        pytest.param({"width": 0, "height": 10}, id="zero_width"),
        pytest.param({"width": -1, "height": 10}, id="negative_width"),
        pytest.param(
            {"width": MAX_MANIFEST_COUNT + 1, "height": 10}, id="width_over_bound"
        ),
        pytest.param(
            {"width": 10, "height": 10, "byte_size": 2**70}, id="bytes_over_bound"
        ),
        pytest.param({"duration_seconds": math.nan}, id="nan_duration"),
        pytest.param({"duration_seconds": -1.0}, id="negative_duration"),
    ],
)
def test_inputs_outside_the_manifest_contract_are_rejected(fields):
    with pytest.raises(AssetLimitError):
        evaluate_asset_limits(MOBILE_V1, manifest(**fields), "image")


def test_errors_are_content_free():
    secret_id = "patient-record-77"
    with pytest.raises(AssetLimitError) as excinfo:
        evaluate_asset_limits(
            MOBILE_V1, manifest(asset_id=secret_id, width=True, height=10), "image"
        )
    message = str(excinfo.value)
    assert secret_id not in message and SHA not in message and "True" not in message
    with pytest.raises(AssetLimitError) as excinfo:
        evaluate_asset_limits(MOBILE_V1, manifest(width=1, height=1), "hologram")
    assert "hologram" not in str(excinfo.value)
    with pytest.raises(TypeError):
        evaluate_asset_limits(MOBILE_V1, 42, "image")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        evaluate_asset_limits({"max_pages": 1}, manifest(), "image")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [10**1000, -(10**1000)])
def test_unbounded_numeric_inputs_have_stable_errors(value):
    with pytest.raises(AssetLimitError):
        MOBILE_V1.with_limits(max_duration_seconds=value)
    with pytest.raises(AssetLimitError):
        evaluate_asset_limits(MOBILE_V1, manifest(duration_seconds=value), "audio")
    with pytest.raises(AssetLimitError):
        LimitFinding("pixels", "limit_exceeded", value, None)
    with pytest.raises(AssetLimitError):
        LimitFinding("pixels", "limit_exceeded", 1, value)
