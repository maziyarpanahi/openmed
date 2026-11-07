"""Synthetic waveform metadata, ceilings, and decoded-memory controls."""

import json
from dataclasses import replace

import pytest

from openmed.multimodal import (
    DESKTOP_V1,
    DESKTOP_V1_1,
    DICOM_NON_PIXEL_V1,
    MOBILE_V1_1,
    WAVEFORM_V1,
    AssetManifest,
    AssetManifestError,
    ValidationFinding,
    evaluate_asset_limits,
    validate_manifest_metadata,
)
from openmed.multimodal.asset_limits import AssetLimitError
from openmed.multimodal.asset_manifest import MAX_MANIFEST_BYTE_SIZE, MAX_MANIFEST_COUNT
from openmed.multimodal.batch_memory import (
    BatchOutcome,
    MemoryEstimationPolicy,
    plan_batch_memory,
)


def waveform(**overrides):
    fields = dict(
        asset_id="synthetic-1",
        media_type="application/edf",
        sha256="a" * 64,
        byte_size=256,
        channel_count=12,
        sample_rate_hz=500,
        samples_per_channel=5000,
        duration_seconds=10,
    )
    fields.update(overrides)
    return AssetManifest(**fields)


@pytest.mark.parametrize(
    "media_type,subprofile",
    [
        ("application/edf", None),
        ("application/wfdb", None),
        ("application/dicom", "waveform"),
    ],
)
def test_waveform_containers_roundtrip_and_validate(media_type, subprofile):
    asset = waveform(media_type=media_type, dicom_subprofile=subprofile)
    assert AssetManifest.from_json(asset.to_json()) == asset
    assert validate_manifest_metadata(WAVEFORM_V1, asset) == []
    assert evaluate_asset_limits(MOBILE_V1_1, asset, "waveform") == []


@pytest.mark.parametrize(
    "field",
    [
        "channel_count",
        "sample_rate_hz",
        "samples_per_channel",
        "duration_seconds",
    ],
)
def test_missing_waveform_fields_have_stable_codes(field):
    fields = waveform().to_dict()
    del fields[field]
    assert validate_manifest_metadata(WAVEFORM_V1, fields) == [
        ValidationFinding(field, "missing_required")
    ]
    findings = evaluate_asset_limits(MOBILE_V1_1, fields, "waveform")
    assert [(f.field_name, f.reason_code) for f in findings] == [
        (field, "insufficient_metadata"),
        ("total_samples", "insufficient_metadata"),
    ]


@pytest.mark.parametrize(
    "field",
    ["channel_count", "sample_rate_hz", "samples_per_channel", "duration_seconds"],
)
@pytest.mark.parametrize(
    "value,code",
    [
        (True, "invalid_boolean"),
        (0, "invalid_zero"),
        (-1, "out_of_range"),
        (float("inf"), "non_finite_numeric"),
        (float("nan"), "non_finite_numeric"),
        ("synthetic private payload", "invalid_type"),
        (10**1000, "out_of_range"),
    ],
)
def test_invalid_numbers_are_content_free(field, value, code):
    fields = waveform().to_dict()
    fields[field] = value
    assert validate_manifest_metadata(WAVEFORM_V1, fields) == [
        ValidationFinding(field, code)
    ]
    with pytest.raises(AssetManifestError) as caught:
        AssetManifest.from_dict(fields)
    assert "synthetic private payload" not in str(caught.value)


@pytest.mark.parametrize(
    "field,value", [("channel_count", 1.5), ("samples_per_channel", 1.5)]
)
def test_waveform_counts_must_be_integers(field, value):
    fields = waveform().to_dict()
    fields[field] = value
    assert validate_manifest_metadata(WAVEFORM_V1, fields) == [
        ValidationFinding(field, "invalid_type")
    ]


def test_fractional_rate_and_endpoint_tolerance():
    assert (
        validate_manifest_metadata(
            WAVEFORM_V1, waveform(sample_rate_hz=2.5, samples_per_channel=25)
        )
        == []
    )
    assert (
        validate_manifest_metadata(WAVEFORM_V1, waveform(samples_per_channel=4999))
        == []
    )
    asset = waveform(samples_per_channel=4998)
    assert validate_manifest_metadata(WAVEFORM_V1, asset) == [
        ValidationFinding("samples_per_channel", "sample_count_mismatch")
    ]
    plan = plan_batch_memory(
        [asset],
        MemoryEstimationPolicy(waveform_bytes_per_sample=4),
        budget_bytes=300000,
        overhead_bytes=0,
    )
    assert plan.assets[0].reason_code == "invalid_metadata"
    assert plan.assets[0].field_name == "samples_per_channel"


@pytest.mark.parametrize("field", ["pages", "width", "height", "frames"])
def test_waveform_rejects_pixel_and_page_metadata(field):
    assert validate_manifest_metadata(WAVEFORM_V1, waveform(**{field: 1})) == [
        ValidationFinding(field, "inapplicable_present")
    ]


def test_non_pixel_dicom_has_only_byte_limits_and_no_guessed_memory():
    asset = AssetManifest(
        asset_id="synthetic-sr",
        media_type="application/dicom",
        sha256="a" * 64,
        byte_size=132,
        dicom_subprofile="non_pixel",
    )
    assert validate_manifest_metadata(DICOM_NON_PIXEL_V1, asset) == []
    assert evaluate_asset_limits(DESKTOP_V1, asset, "dicom_non_pixel") == []
    plan = plan_batch_memory(
        [asset], MemoryEstimationPolicy(), budget_bytes=1024, overhead_bytes=0
    )
    assert plan.outcome is BatchOutcome.REJECT
    assert plan.assets[0].field_name == "decoded_bytes"


@pytest.mark.parametrize(
    "media_type,subprofile",
    [
        ("image/png", "waveform"),
        ("application/dicom+json", "waveform"),
        ("application/dicom", "synthetic private payload"),
        ("application/dicom", True),
    ],
)
def test_subprofiles_are_controlled_and_dicom_only(media_type, subprofile):
    with pytest.raises(AssetManifestError, match="dicom_subprofile is unsupported"):
        waveform(media_type=media_type, dicom_subprofile=subprofile)


@pytest.mark.parametrize(
    "field,ceiling",
    [
        ("channel_count", 12),
        ("sample_rate_hz", 500),
        ("samples_per_channel", 5000),
        ("duration_seconds", 10),
        ("total_samples", 60000),
    ],
)
def test_inclusive_waveform_ceilings(field, ceiling):
    profile = MOBILE_V1_1.with_limits(**{"max_" + field: ceiling})
    assert evaluate_asset_limits(profile, waveform(), "waveform") == []
    profile = profile.with_limits(**{"max_" + field: ceiling - 1})
    assert [
        (f.field_name, f.reason_code, f.observed)
        for f in evaluate_asset_limits(profile, waveform(), "waveform")
    ] == [(field, "limit_exceeded", ceiling)]


def test_total_samples_charge_rate_duration_even_when_count_understates():
    asset = waveform(samples_per_channel=1)
    findings = evaluate_asset_limits(
        MOBILE_V1_1.with_limits(max_total_samples=59999), asset, "waveform"
    )
    assert findings[-1].observed == 60000


def test_channel_rate_duration_overflow_saturates_even_at_maximum_ceiling():
    asset = waveform(
        channel_count=MAX_MANIFEST_COUNT,
        sample_rate_hz=MAX_MANIFEST_COUNT,
        samples_per_channel=MAX_MANIFEST_COUNT,
        duration_seconds=MAX_MANIFEST_COUNT,
    )
    profile = DESKTOP_V1_1.with_limits(max_total_samples=MAX_MANIFEST_BYTE_SIZE)
    findings = evaluate_asset_limits(profile, asset, "waveform")
    assert findings[-1].field_name == "total_samples"
    assert findings[-1].reason_code == "arithmetic_saturated"
    assert findings[-1].observed == MAX_MANIFEST_BYTE_SIZE
    plan = plan_batch_memory(
        [asset],
        MemoryEstimationPolicy(waveform_bytes_per_sample=1),
        budget_bytes=MAX_MANIFEST_BYTE_SIZE,
        overhead_bytes=0,
    )
    assert plan.outcome is BatchOutcome.REJECT
    assert plan.assets[0].reason_code == "estimate_saturated"
    assert plan.assets[0].estimated_bytes is None


def test_channel_samples_bytes_product_overflow_rejects():
    asset = waveform(
        channel_count=MAX_MANIFEST_COUNT,
        samples_per_channel=MAX_MANIFEST_COUNT,
        sample_rate_hz=1,
        duration_seconds=MAX_MANIFEST_COUNT,
    )
    plan = plan_batch_memory(
        [asset],
        MemoryEstimationPolicy(waveform_bytes_per_sample=4),
        budget_bytes=MAX_MANIFEST_BYTE_SIZE,
        overhead_bytes=0,
    )
    assert plan.assets[0].reason_code == "estimate_saturated"


def test_memory_split_and_exact_budget_including_overhead():
    asset = waveform()
    policy = MemoryEstimationPolicy(waveform_bytes_per_sample=4)
    plan = plan_batch_memory(
        [asset, asset], policy, budget_bytes=240016, overhead_bytes=16
    )
    assert plan.outcome is BatchOutcome.SPLIT
    assert [b.estimated_bytes for b in plan.batches] == [240016, 240016]
    assert [b.positions for b in plan.batches] == [(0,), (1,)]
    assert "synthetic-1" not in plan.to_json()
    assert "a" * 64 not in plan.to_json()


def test_no_missing_policy_factor_is_guessed():
    plan = plan_batch_memory(
        [waveform()],
        MemoryEstimationPolicy(),
        budget_bytes=MAX_MANIFEST_BYTE_SIZE,
        overhead_bytes=0,
    )
    assert plan.assets[0].field_name == "waveform_bytes_per_sample"
    assert plan.assets[0].reason_code == "insufficient_metadata"


def test_waveform_profile_limits_are_versioned():
    assert MOBILE_V1_1.to_dict()["max_channel_count"] == 32
    assert DESKTOP_V1_1.version == "1.1"
    with pytest.raises(AssetLimitError):
        DESKTOP_V1.with_limits(max_channel_count=32)
    with pytest.raises(AssetLimitError):
        replace(DESKTOP_V1_1, max_sample_rate_hz=10**1000)
    with pytest.raises(AssetLimitError):
        evaluate_asset_limits(DESKTOP_V1, waveform(), "waveform")


@pytest.mark.parametrize(
    "fields",
    [
        dict(media_type="image/png", width=2, height=3),
        dict(media_type="application/pdf", pages=2),
        dict(media_type="application/dicom", width=2, height=3, frames=1),
        dict(media_type="audio/wav", duration_seconds=2.5),
    ],
)
def test_legacy_manifest_json_has_exact_original_fields(fields):
    original = dict(
        version=1, asset_id="synthetic-legacy", sha256="b" * 64, byte_size=132, **fields
    )
    assert AssetManifest.from_dict(original).to_json() == json.dumps(
        original, sort_keys=True, separators=(",", ":")
    )
    assert "max_channel_count" not in DESKTOP_V1.to_dict()
    assert "waveform_bytes_per_sample" not in MemoryEstimationPolicy().to_dict()
