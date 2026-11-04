"""Measured degradation, bounded decoding and inference suppression controls."""

import builtins
import json
import logging
import socket
import struct
import threading
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from io import BytesIO

import pytest

from openmed.multimodal.vision.providers.base import (
    ImageInput,
    ImageRegion,
    LocalProviderMetadata,
    Prompt,
    RuntimeBackend,
    RuntimeMetadata,
    VLMRequest,
    VLMResult,
)
from openmed.multimodal.vision.safety import (
    ImageQualityError,
    QualityAction,
    QualityPolicy,
    QualityReason,
    assess_image_quality,
    run_quality_gated_vlm,
)

Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")
ImageFilter = pytest.importorskip("PIL.ImageFilter")
ImageFile = pytest.importorskip("PIL.ImageFile")


@pytest.fixture
def policy():
    # Synthetic fixture thresholds, not clinically qualified minimums.
    return QualityPolicy(
        min_short_side=32,
        min_long_side=64,
        min_luminance=0.1,
        max_luminance=0.9,
        min_contrast=0.05,
        min_sharpness=0.00001,
        max_dark_fraction=0.5,
        max_bright_fraction=0.5,
    )


@pytest.fixture
def pattern():
    image = Image.new("L", (64, 64))
    image.putdata(
        [48 if (x // 8 + y // 8) % 2 else 192 for y in range(64) for x in range(64)]
    )
    return image


def encoded(image, fmt="PNG", **kwargs):
    out = BytesIO()
    image.save(out, format=fmt, **kwargs)
    return ImageInput(
        data=out.getvalue(),
        media_type=f"image/{fmt.lower()}",
        width=image.width,
        height=image.height,
    )


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_supported_images_are_measured_deterministically_without_network(
    policy, pattern, fmt, monkeypatch
):
    def deny(*args, **kwargs):
        pytest.fail("unexpected network access")

    monkeypatch.setattr(socket, "socket", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    source = encoded(pattern, fmt)
    first = assess_image_quality(source, policy)
    second = assess_image_quality(source, policy)
    assert first.accepted and first.action is QualityAction.CONTINUE
    assert first.to_json() == second.to_json()
    assert first.image_digest == source.digest
    assert first.metrics.width == first.metrics.height == 64
    assert 0 < first.metrics.contrast <= 0.5 and first.metrics.sharpness > 0


def test_real_blur_lowers_measured_sharpness_and_blocks_the_stricter_task(
    policy, pattern
):
    blurred = pattern.filter(ImageFilter.GaussianBlur(4))
    original = assess_image_quality(encoded(pattern), policy)
    degraded = assess_image_quality(encoded(blurred), policy)
    assert degraded.metrics.sharpness < original.metrics.sharpness
    threshold = (degraded.metrics.sharpness + original.metrics.sharpness) / 2
    stricter = replace(policy, min_sharpness=threshold)
    assert assess_image_quality(encoded(pattern), stricter).accepted
    report = assess_image_quality(encoded(blurred), stricter)
    assert QualityReason.SHARPNESS in report.reasons
    assert report.action is QualityAction.REACQUIRE


@pytest.mark.parametrize(
    "value,reason",
    [
        (0, QualityReason.DARK),
        (255, QualityReason.BRIGHT),
        (128, QualityReason.CONTRAST),
    ],
)
def test_exposure_and_flat_image_controls(policy, value, reason):
    report = assess_image_quality(encoded(Image.new("L", (64, 64), value)), policy)
    assert reason in report.reasons and not report.accepted
    assert QualityReason.SHARPNESS in report.reasons


def test_resolution_uses_source_geometry_not_the_analysis_thumbnail(policy, pattern):
    small = pattern.resize((16, 16))
    assert (
        QualityReason.RESOLUTION in assess_image_quality(encoded(small), policy).reasons
    )
    large = pattern.resize((1024, 1024), Image.Resampling.NEAREST)
    report = assess_image_quality(
        encoded(large), replace(policy, min_short_side=800, min_long_side=1000)
    )
    assert report.accepted
    assert report.metrics.width == 1024 and report.metrics.sample_width == 512


def test_blank_margin_requirement_detects_foreground_at_the_crop_edge(policy):
    policy = replace(
        policy,
        min_luminance=0,
        max_luminance=1,
        max_bright_fraction=1,
        max_dark_fraction=1,
        max_boundary_fraction=0.1,
        background_luminance=1,
    )
    interior = Image.new("L", (64, 64), 255)
    ImageDraw.Draw(interior).rectangle((16, 16, 48, 48), fill=0)
    assert assess_image_quality(encoded(interior), policy).accepted
    clipped = interior.copy()
    ImageDraw.Draw(clipped).rectangle((0, 16, 16, 48), fill=0)
    report = assess_image_quality(encoded(clipped), policy)
    assert QualityReason.BOUNDARY in report.reasons
    assert report.metrics.boundary_fraction > policy.max_boundary_fraction
    disabled = replace(policy, max_boundary_fraction=None, background_luminance=None)
    assert assess_image_quality(encoded(clipped), disabled).accepted


def test_corruption_and_declared_mismatches_fail_closed(policy, pattern):
    source = encoded(pattern)
    broken = replace(source, data=source.data[:60])
    assert not assess_image_quality(broken, policy).accepted
    assert assess_image_quality(replace(source, width=65), policy).reasons == (
        QualityReason.GEOMETRY_MISMATCH,
    )
    assert assess_image_quality(
        replace(source, media_type="image/jpeg"), policy
    ).reasons == (QualityReason.FORMAT_MISMATCH,)


def test_huge_header_is_rejected_before_decoder_allocation(policy, monkeypatch):
    ihdr = struct.pack(">IIBBBBB", 10000, 10000, 8, 2, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(ihdr))
        + b"IHDR"
        + ihdr
        + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    )
    source = ImageInput(data=data, media_type="image/png", width=10000, height=10000)
    monkeypatch.setattr(
        Image, "open", lambda *a, **k: pytest.fail("decoder was reached")
    )
    report = assess_image_quality(source, policy)
    assert report.reasons == (QualityReason.RESOURCE_LIMIT,)


def test_truncated_image_tolerance_is_never_enabled_or_silently_trusted(
    policy, pattern, monkeypatch
):
    source = encoded(pattern)
    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)
    assert assess_image_quality(source, policy).reasons == (
        QualityReason.DECODER_UNAVAILABLE,
    )
    assert ImageFile.LOAD_TRUNCATED_IMAGES is True


def test_missing_optional_decoder_abstains(policy, pattern, monkeypatch):
    source = encoded(pattern)
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("synthetic-private-import-value")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    report = assess_image_quality(source, policy)
    assert report.reasons == (QualityReason.DECODER_UNAVAILABLE,)
    assert "synthetic-private-import-value" not in report.to_json()


def test_transparency_animation_and_unhandled_modes_require_review(policy, pattern):
    rgba = pattern.convert("RGBA")
    assert assess_image_quality(encoded(rgba), policy).accepted
    rgba.putalpha(128)
    assert assess_image_quality(encoded(rgba), policy).reasons == (
        QualityReason.UNSUPPORTED_IMAGE,
    )
    animated = encoded(
        pattern,
        save_all=True,
        append_images=[pattern.transpose(Image.Transpose.FLIP_LEFT_RIGHT)],
    )
    assert assess_image_quality(animated, policy).reasons == (
        QualityReason.UNSUPPORTED_IMAGE,
    )
    palette = pattern.convert("P")
    assert assess_image_quality(encoded(palette), policy).reasons == (
        QualityReason.UNSUPPORTED_IMAGE,
    )


def test_too_narrow_analysis_cannot_produce_a_passing_focus_score(policy):
    source = encoded(Image.new("L", (1000, 3), 128))
    assert assess_image_quality(source, policy).reasons == (
        QualityReason.SAMPLE_TOO_SMALL,
    )


def test_metadata_debug_logs_and_decoder_exceptions_never_escape(
    policy, pattern, caplog, monkeypatch
):
    private = "SYNTHETIC-PRIVATE-ICC-NAME"
    payload = zlib.compress(b"synthetic-profile")
    chunk = b"iCCP" + private.encode() + b"\0\0" + payload
    source = encoded(pattern)
    # Valid ICC-name chunk exercises Pillow's actual DEBUG metadata path.
    data = (
        source.data[:33]
        + struct.pack(">I", len(chunk) - 4)
        + chunk
        + struct.pack(">I", zlib.crc32(chunk))
        + source.data[33:]
    )
    source = replace(source, data=data)
    with caplog.at_level(logging.DEBUG, logger="PIL.PngImagePlugin"):
        report = assess_image_quality(source, policy)
        assert private not in caplog.text + report.to_json() + repr(report)
        logging.getLogger("PIL.PngImagePlugin").debug("outside-gate-record")
    assert "outside-gate-record" in caplog.text

    def fail(*args, **kwargs):
        logging.getLogger("PIL.Image").error(private)
        raise ValueError(private)

    monkeypatch.setattr(Image, "open", fail)
    report = assess_image_quality(source, policy)
    assert report.reasons == (QualityReason.DECODE_FAILED,)
    assert private not in caplog.text + report.to_json() + repr(report)


def test_concurrent_assessments_serialize_shared_warning_contexts(
    policy, pattern, monkeypatch
):
    source = encoded(pattern.resize((512, 512), Image.Resampling.NEAREST))
    original_context = warnings.catch_warnings
    original_filters = list(warnings.filters)
    lock = threading.Lock()
    active = 0
    peak = 0

    @contextmanager
    def tracked_context(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            with original_context(*args, **kwargs):
                yield
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(warnings, "catch_warnings", tracked_context)
    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(
            pool.map(lambda _: assess_image_quality(source, policy), range(8))
        )
    assert peak == 1 and active == 0
    assert warnings.filters == original_filters
    assert len({report.to_json() for report in reports}) == 1
    assert all(report.accepted for report in reports)


@pytest.mark.parametrize(
    "change",
    [
        {"min_short_side": True},
        {"min_short_side": 0},
        {"min_long_side": 4},
        {"min_luminance": float("nan")},
        {"min_luminance": 1},
        {"min_contrast": 0.6},
        {"min_sharpness": -1},
        {"max_dark_fraction": 2},
        {"max_boundary_fraction": 0.1},
    ],
)
def test_malformed_task_policies_raise_value_free_errors(policy, change):
    with pytest.raises(ImageQualityError, match="^invalid image quality contract$"):
        replace(policy, **change)


def test_quality_failure_never_accesses_provider_metadata_or_inference(policy):
    class UnreachableProvider:
        @property
        def metadata(self):
            pytest.fail("provider admission must follow quality assessment")

        def infer(self, request):
            pytest.fail("failed input reached inference")

    request = VLMRequest(
        image=encoded(Image.new("L", (64, 64), 0)), prompt=Prompt(text="synthetic task")
    )
    report, result = run_quality_gated_vlm(UnreachableProvider(), request, policy)
    assert not report.accepted and result is None


def test_whole_image_statistics_never_silently_qualify_a_selected_crop(policy, pattern):
    source = encoded(pattern)
    region = ImageRegion(image_digest=source.digest, x0=0, y0=0, x1=16, y1=16)
    request = VLMRequest(
        image=source, prompt=Prompt(text="synthetic task"), regions=(region,)
    )
    report, result = run_quality_gated_vlm(None, request, policy)
    assert report.reasons == (QualityReason.UNASSESSED_REGION,)
    assert report.action is QualityAction.REVIEW and result is None


def test_passing_quality_retains_existing_provider_and_human_review_gates(
    policy, pattern
):
    class Provider:
        metadata = LocalProviderMetadata(
            backend=RuntimeBackend.SYNTHETIC,
            model_digest="a" * 64,
            preprocessing_digest="b" * 64,
        )
        calls = 0

        def infer(self, request):
            self.calls += 1
            return VLMResult(
                request_digest=request.digest,
                runtime=RuntimeMetadata(
                    provider=self.metadata, duration_ms=0, output_tokens=1
                ),
                text="synthetic draft",
            )

    provider = Provider()
    request = VLMRequest(image=encoded(pattern), prompt=Prompt(text="synthetic task"))
    report, result = run_quality_gated_vlm(provider, request, policy)
    assert report.accepted and provider.calls == 1
    assert result.reviewed_text(reviewer_confirmed=True) == "synthetic draft"
    assert json.loads(result.to_json())["review_required"] is True
    assert "synthetic draft" not in report.to_json() + result.to_json()
