"""Real shared preflight parity and independent ordered batch-policy controls."""

import hashlib
from dataclasses import replace

import pytest

from openmed.multimodal.asset_manifest import MAX_MANIFEST_BYTE_SIZE, AssetManifest
from openmed.multimodal.batch_preflight import (
    BatchPreflightAsset,
    BatchPreflightLimits,
    preflight_asset_batch,
)
from openmed.multimodal.preflight import preflight_asset

PNG = b"\x89PNG\r\n\x1a\n"
PDF = b"%PDF-1.7\n"
DICOM = b"\x00" * 128 + b"DICM"
WAV = b"RIFF\x00\x00\x00\x00WAVE"
LIMITS = BatchPreflightLimits(1000000, 100, 1000000, 100, 100, 10)


def asset(code, source=PNG, media="image/png", **fields):
    if media.startswith("image/"):
        fields = {"width": 8, "height": 8} | fields
    return BatchPreflightAsset(
        AssetManifest(
            code, media, hashlib.sha256(source).hexdigest(), len(source), **fields
        ),
        source,
    )


def findings(report):
    return [(f.position, f.reason, f.resource) for f in report.findings]


def test_mixed_packet_preserves_actual_single_reports_and_input_order():
    assets = [
        asset("z-image"),
        asset("a-pdf", PDF, "application/pdf", pages=1),
        asset("d-dicom", DICOM, "application/dicom", width=4, height=4, frames=1),
        asset("b-audio", WAV, "audio/wav", duration_seconds=1),
    ]
    report = preflight_asset_batch(assets, limits=LIMITS)
    assert [row.position for row in report.assets] == list(range(4))
    for row, item in zip(report.assets, assets):
        assert (
            row.preflight.to_dict()
            == preflight_asset(item.manifest, item.source).to_dict()
        )
    assert report.to_json() == preflight_asset_batch(assets, limits=LIMITS).to_json()
    # Current shared PDF preflight lacks raster geometry; never invent acceptance.
    assert report.assets[1].status == "abstain"
    assert report.totals["pixels"] is None
    assert report.outcome == "hold"


def test_clean_image_packet_can_accept():
    report = preflight_asset_batch([asset("image")], limits=LIMITS)
    assert report.outcome == "accept"
    assert report.totals["pixels"] == 64


@pytest.mark.parametrize(
    "resource,item,maximum",
    [
        ("bytes", asset("image"), 1),
        ("pages", asset("pdf", PDF, "application/pdf", pages=2), 1),
        ("pixels", asset("image"), 63),
        (
            "frames",
            asset("dicom", DICOM, "application/dicom", width=1, height=1, frames=2),
            1,
        ),
        ("duration", asset("wav", WAV, "audio/wav", duration_seconds=2), 1),
        ("rejections", asset("bad", b"\x00\x01"), 0),
    ],
)
def test_each_aggregate_ceiling_is_independent(resource, item, maximum):
    report = preflight_asset_batch(
        [item], limits=replace(LIMITS, **{resource: maximum})
    )
    assert (0, "aggregate_limit_exceeded", resource) in findings(report)
    assert report.outcome == "hold"


def test_duplicate_id_and_content_are_separate_ordered_findings():
    report = preflight_asset_batch([asset("same"), asset("same")], limits=LIMITS)
    assert (1, "duplicate_asset_id", None) in findings(report)
    assert (1, "duplicate_content", None) in findings(report)
    assert report.outcome == "hold"


@pytest.mark.parametrize(
    "policy,last_status,outcome",
    [
        ("continue", "accept", "hold"),
        ("hold", "not_evaluated", "hold"),
        ("fail_fast", "not_evaluated", "fail_fast"),
    ],
)
def test_policy_modes_keep_one_result_per_asset(policy, last_status, outcome):
    report = preflight_asset_batch(
        [asset("bad", b"\x00\x01"), asset("good")], limits=LIMITS, policy=policy
    )
    assert len(report.assets) == 2
    assert report.assets[0].status == "abstain"
    assert report.assets[1].status == last_status
    assert report.outcome == outcome


def test_global_byte_admission_prevents_source_reads_even_in_continue_mode():
    class MustNotRead:
        def read(self, size):
            raise AssertionError("source must not be read over aggregate byte ceiling")

    item = BatchPreflightAsset(asset("bounded").manifest, MustNotRead())
    report = preflight_asset_batch(
        [item], limits=replace(LIMITS, bytes=1), policy="continue"
    )
    assert report.assets[0].status == "not_evaluated"
    assert (0, "aggregate_limit_exceeded", "bytes") in findings(report)


def test_checked_aggregate_overflow_does_not_wrap_or_read_later_sources():
    first = asset("large")
    first = BatchPreflightAsset(
        replace(first.manifest, byte_size=MAX_MANIFEST_BYTE_SIZE), first.source
    )
    report = preflight_asset_batch(
        [first, asset("second")], limits=replace(LIMITS, bytes=MAX_MANIFEST_BYTE_SIZE)
    )
    assert report.totals["bytes"] is None
    assert (1, "aggregate_overflow", "bytes") in findings(report)
    assert report.assets[1].status == "not_evaluated"


def test_unknown_header_and_source_error_are_distinct_without_echo():
    class BrokenSource:
        def read(self, size):
            raise OSError("synthetic/private/path credentials transcript")

    item = BatchPreflightAsset(asset("broken").manifest, BrokenSource())
    report = preflight_asset_batch([asset("unknown", b"\x00\x01"), item], limits=LIMITS)
    assert report.assets[0].status == "abstain"
    assert any(f.reason_code == "unknown" for f in report.assets[0].preflight.findings)
    assert report.assets[1].status == "error"
    assert (1, "source_error", None) in findings(report)
    serialized = report.to_json()
    assert "synthetic/private" not in serialized
    assert "credentials transcript" not in serialized
    assert "BrokenSource" not in serialized


def test_missing_geometry_never_contributes_assumed_zero():
    manifest = AssetManifest(
        "missing", "image/png", hashlib.sha256(PNG).hexdigest(), len(PNG)
    )
    report = preflight_asset_batch([BatchPreflightAsset(manifest, PNG)], limits=LIMITS)
    assert report.totals["pixels"] is None
    assert (0, "aggregate_insufficient_metadata", "pixels") in findings(report)
    assert report.outcome == "hold"


def test_no_heavy_decoders_or_models_are_loaded(monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "PIL",
            "pydicom",
            "torch",
            "transformers",
            "cv2",
        }
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert preflight_asset_batch([asset("image")], limits=LIMITS).outcome == "accept"


@pytest.mark.parametrize(
    "changes",
    [
        {"bytes": True},
        {"duration": float("inf")},
        {"pages": -1},
        {"pixels": -1},
        {"frames": -1},
        {"rejections": -1},
    ],
)
def test_malformed_aggregate_policies_reject(changes):
    with pytest.raises(ValueError):
        replace(LIMITS, **changes)


def test_unknown_modes_and_unbounded_inputs_reject_before_source_reads():
    with pytest.raises(ValueError):
        preflight_asset_batch(
            [asset("image")], limits=LIMITS, policy="synthetic/private"
        )
    calls = []

    def infinite():
        while True:
            calls.append(1)
            yield asset("image")

    with pytest.raises(ValueError):
        preflight_asset_batch(infinite(), limits=LIMITS)
    assert len(calls) == 10001


def test_source_is_excluded_from_input_repr():
    item = asset("opaque", b"synthetic-private-source")
    assert "synthetic-private-source" not in repr(item)
