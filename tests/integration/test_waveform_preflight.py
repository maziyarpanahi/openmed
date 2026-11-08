"""Offline composition of waveform manifests, limits, digest and memory plans."""

import hashlib
import io

import pytest

from openmed.multimodal import (
    DESKTOP_V1,
    MOBILE_V1_1,
    WAVEFORM_V1,
    AssetManifest,
    PreflightError,
    PreflightStatus,
    preflight_asset,
)
from openmed.multimodal.batch_memory import MemoryEstimationPolicy, plan_batch_memory

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "media_type,subprofile,source",
    [
        ("application/edf", None, b"0       " + b" " * 248),
        ("application/dicom", "waveform", b"\0" * 128 + b"DICM"),
    ],
)
def test_waveform_preflight_and_memory_compose(media_type, subprofile, source):
    manifest = AssetManifest(
        asset_id="synthetic-ecg",
        media_type=media_type,
        sha256=hashlib.sha256(source).hexdigest(),
        byte_size=len(source),
        dicom_subprofile=subprofile,
        channel_count=12,
        sample_rate_hz=500,
        samples_per_channel=5000,
        duration_seconds=10,
    )
    stream = io.BytesIO(source)
    report = preflight_asset(manifest, stream)
    assert report.status is PreflightStatus.ACCEPT
    assert report.metadata_profile == WAVEFORM_V1
    assert report.limit_profile.version == "1.1"
    assert stream.tell() == 0
    assert not stream.closed
    plan = plan_batch_memory(
        [manifest],
        MemoryEstimationPolicy(waveform_bytes_per_sample=4),
        budget_bytes=240000,
        overhead_bytes=0,
    )
    assert plan.assets[0].estimated_bytes == 240000


def test_wfdb_metadata_is_admissible_but_header_text_does_not_prove_media_type():
    source = b"synthetic 12 500 5000\n"
    manifest = AssetManifest(
        asset_id="synthetic-wfdb",
        media_type="application/wfdb",
        sha256=hashlib.sha256(source).hexdigest(),
        byte_size=len(source),
        channel_count=12,
        sample_rate_hz=500,
        samples_per_channel=5000,
        duration_seconds=10,
    )
    report = preflight_asset(manifest, source)
    assert report.modality == "waveform"
    assert report.status is PreflightStatus.ABSTAIN
    assert [(f.check, f.reason_code) for f in report.findings] == [
        ("media_type", "unknown")
    ]
    assert report.digest.byte_count == len(source)


def test_non_pixel_dicom_preflight_has_no_pixel_findings():
    source = b"\0" * 128 + b"DICM"
    manifest = AssetManifest(
        asset_id="synthetic-sr",
        media_type="application/dicom",
        dicom_subprofile="non_pixel",
        sha256=hashlib.sha256(source).hexdigest(),
        byte_size=len(source),
    )
    report = preflight_asset(manifest, source)
    assert report.status is PreflightStatus.ACCEPT
    assert report.modality == "dicom_non_pixel"
    assert report.limit_profile is DESKTOP_V1


def test_waveform_preflight_collects_stable_metadata_and_limit_findings():
    source = b"\0" * 128 + b"DICM"
    manifest = AssetManifest(
        asset_id="synthetic-ecg",
        media_type="application/dicom",
        dicom_subprofile="waveform",
        sha256=hashlib.sha256(source).hexdigest(),
        byte_size=len(source),
        channel_count=33,
        sample_rate_hz=500,
        samples_per_channel=1,
        duration_seconds=10,
    )
    report = preflight_asset(manifest, source, limit_profile=MOBILE_V1_1)
    assert report.status is PreflightStatus.ABSTAIN
    assert [(f.check, f.field_name, f.reason_code) for f in report.findings] == [
        ("metadata", "samples_per_channel", "sample_count_mismatch"),
        ("limits", "channel_count", "limit_exceeded"),
    ]
    assert report.findings[1].observed == 33
    with pytest.raises(PreflightError, match="waveform_limit_profile_required"):
        preflight_asset(manifest, source, limit_profile=DESKTOP_V1)


def test_malformed_private_metadata_abstains_without_reading_or_echoing():
    class NoRead:
        def read(self, size):
            raise AssertionError("malformed manifests must not read sources")

    report = preflight_asset(
        {
            "patient_name": "Synthetic private name",
            "source_path": "/private/synthetic.edf",
        },
        NoRead(),
    )
    assert report.status is PreflightStatus.ABSTAIN
    assert report.manifest is None
    assert report.findings[0].reason_code == "malformed_manifest"
    assert "Synthetic private name" not in report.to_json()
    assert "/private/synthetic.edf" not in report.to_json()


def test_waveform_digest_and_type_negative_controls():
    source = b"0       " + b" " * 248
    manifest = AssetManifest(
        asset_id="synthetic-1",
        media_type="application/edf",
        sha256="a" * 64,
        byte_size=len(source),
        channel_count=1,
        sample_rate_hz=1,
        samples_per_channel=1,
        duration_seconds=1,
    )
    report = preflight_asset(manifest, source)
    assert [(f.check, f.reason_code) for f in report.findings] == [
        ("digest", "sha256_mismatch")
    ]
    mismatched = b"\0" * 128 + b"DICM"
    report = preflight_asset(manifest, mismatched)
    assert report.findings[0].reason_code == "mismatch"
    assert report.status is PreflightStatus.ABSTAIN
