"""Synthetic code geometry, leakage, residual and negative-control regressions."""

from __future__ import annotations

import hashlib
import json
import traceback
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from openmed.multimodal import (
    CodeDetectionError,
    CodeRegion,
    FakeCodeRegionDetector,
    RasterRedactionPage,
    ZXingCodeRegionDetector,
    detect_code_regions,
    redact_document,
    redact_image,
    render_redacted_raster_pages,
)
from openmed.multimodal.ocr import FakeOcrEngine

SENTINEL = "SYNTHETIC-MRN-3816-مريض"
REGIONS = (
    CodeRegion("linear", (10, 10, 110, 40)),
    CodeRegion("qr", (10, 60, 58, 108)),
    CodeRegion("datamatrix", (80, 60, 128, 108)),
)


def synthetic_codes():
    """Embed sentinel bits in code-like rasters; no symbology qualification."""
    image = Image.new("RGB", (160, 140), "white")
    draw = ImageDraw.Draw(image)
    bits = "".join(f"{byte:08b}" for byte in SENTINEL.encode())
    for region in REGIONS:
        x0, y0, x1, y1 = region.bbox
        for index, bit in enumerate(bits):
            if region.symbology == "linear":
                x = x0 + index % (x1 - x0)
                if bit == "1":
                    draw.line((x, y0, x, y1 - 1), fill="black")
            else:
                x = x0 + (index % 16) * 3
                y = y0 + ((index // 16) % 16) * 3
                if bit == "1":
                    draw.rectangle((x, y, x + 2, y + 2), fill="black")
    return image


class PixelAwareCodes:
    """Locate configured symbols until their complete region is blacked out."""

    def detect(self, image):
        image = image if isinstance(image, Image.Image) else Image.fromarray(image)
        return [
            {"symbology": region.symbology, "bbox": region.bbox, "payload": SENTINEL}
            for region in REGIONS
            if image.convert("RGB").crop(region.bbox).getextrema() != ((0, 0),) * 3
        ]


class UnavailableCodes:
    def detect(self, image):
        raise CodeDetectionError(SENTINEL)


def assert_covered(image):
    for region in REGIONS:
        assert image.convert("RGB").crop(region.bbox).getextrema() == ((0, 0),) * 3


def test_image_codes_redacted_without_ocr_identifiers_and_payload_reports(
    tmp_path, caplog
):
    source = tmp_path / "synthetic.png"
    synthetic_codes().save(source)
    result = redact_image(
        source,
        ocr_engine=FakeOcrEngine(()),
        code_detector=PixelAwareCodes(),
        code_detection_policy="fail_closed",
    )
    with Image.open(BytesIO(result.redacted_bytes)) as decoded:
        assert_covered(decoded)
    assert len(result.redaction_boxes) == 3
    assert result.residual_report.code_coverage.clean
    assert (
        result.code_coverage.regions[0].payload_sha256
        == hashlib.sha256(SENTINEL.encode()).hexdigest()
    )
    metadata = result.to_document().metadata
    assert metadata["code_coverage"]["count"] == 3
    report = json.dumps(
        {k: v for k, v in metadata.items() if k != "redacted_image_bytes"}
    )
    assert SENTINEL not in report + caplog.text + repr(result.code_coverage)
    assert "payload_sha256" not in report


@pytest.mark.parametrize("with_output", [False, True])
def test_document_dispatch_honors_code_provider_and_policy(
    tmp_path, monkeypatch, with_output
):
    import openmed.multimodal.base as base

    monkeypatch.setattr(base, "ensure_multimodal_available", lambda: None)
    source = tmp_path / "synthetic.png"
    synthetic_codes().save(source)
    result = redact_document(
        source,
        policy={
            **({"output_path": tmp_path / "out.png"} if with_output else {}),
            "code_detection_policy": "fail_closed",
        },
        models={"ocr_engine": FakeOcrEngine(()), "code_detector": PixelAwareCodes()},
    )
    assert result.metadata["code_coverage"]["count"] == 3


@pytest.mark.parametrize("extension", ["png", "jpg", "tiff"])
def test_no_codes_preserve_existing_encoded_outputs(tmp_path, extension, monkeypatch):
    from openmed.multimodal import code_regions

    source = tmp_path / f"control.{extension}"
    Image.new("RGB", (32, 32), (80, 140, 180)).save(source)
    monkeypatch.setattr(
        code_regions.ZXingCodeRegionDetector, "detect", UnavailableCodes().detect
    )
    old = redact_image(source, ocr_engine=FakeOcrEngine(()), verify=False)
    new = redact_image(
        source, ocr_engine=FakeOcrEngine(()), code_detector=FakeCodeRegionDetector()
    )
    assert old.redacted_bytes == new.redacted_bytes
    assert old.code_coverage.coverage_incomplete
    assert new.code_coverage.clean


def test_multiframe_tiff_has_frame_indexed_code_boxes(tmp_path):
    source = tmp_path / "multi.tiff"
    first = synthetic_codes()
    first.save(source, save_all=True, append_images=[synthetic_codes()])
    result = redact_image(
        source, ocr_engine=FakeOcrEngine(()), code_detector=PixelAwareCodes()
    )
    assert [r.page for r in result.code_coverage.regions] == [0, 0, 0, 1, 1, 1]
    assert result.residual_report.code_coverage.clean


@pytest.mark.parametrize("kind", ["missing", "generator"])
def test_unavailability_is_explicit_and_provider_errors_are_sanitized(
    kind, tmp_path, caplog
):
    class Provider:
        def detect(self, image):
            if kind == "generator":
                yield REGIONS[0]
            raise RuntimeError(SENTINEL)

    image = synthetic_codes()
    report = detect_code_regions(image, detector=Provider())
    assert report.coverage_incomplete and not report.clean
    with pytest.raises(CodeDetectionError) as caught:
        detect_code_regions(image, detector=Provider(), policy="fail_closed")
    assert (
        SENTINEL not in "".join(traceback.format_exception(caught.value)) + caplog.text
    )
    source = tmp_path / "source.png"
    target = tmp_path / "output.png"
    image.save(source)
    with pytest.raises(CodeDetectionError, match="unavailable"):
        redact_image(
            source,
            output_path=target,
            ocr_engine=FakeOcrEngine(()),
            code_detector=UnavailableCodes(),
            code_detection_policy="fail_closed",
        )
    assert not target.exists()


def test_residual_codes_reject_image_before_publication(tmp_path):
    source = tmp_path / "source.png"
    synthetic_codes().save(source)
    target = tmp_path / "output.png"
    target.write_bytes(b"existing-output")
    with pytest.raises(CodeDetectionError, match="residual"):
        redact_image(
            source,
            output_path=target,
            ocr_engine=FakeOcrEngine(()),
            code_detector=FakeCodeRegionDetector(REGIONS),
        )
    assert target.read_bytes() == b"existing-output"


@pytest.mark.parametrize(
    "bbox", [(0, 0, 200, 30), (10, 0, 1, 30), (0, 0, float("nan"), 30), (True, 0, 1, 1)]
)
def test_invalid_geometry_fails_closed_even_under_incomplete_policy(bbox):
    with pytest.raises(CodeDetectionError, match="geometry"):
        detect_code_regions(
            synthetic_codes(), detector=FakeCodeRegionDetector([CodeRegion("qr", bbox)])
        )


def test_unknown_symbology_cannot_echo_payload_and_region_limit_is_enforced():
    result = detect_code_regions(
        synthetic_codes(),
        detector=FakeCodeRegionDetector([CodeRegion(SENTINEL, (0, 0, 4, 4))]),
    )
    assert result.regions[0].symbology == "other"
    assert SENTINEL not in json.dumps(result.to_dict())
    with pytest.raises(CodeDetectionError, match="limit"):
        detect_code_regions(
            synthetic_codes(), detector=FakeCodeRegionDetector(REGIONS), max_regions=2
        )


@pytest.mark.parametrize(
    "format_name,kind",
    [("Code128", "linear"), ("QRCode", "qr"), ("DataMatrix", "datamatrix")],
)
def test_optional_backend_never_accesses_decoded_payload(
    monkeypatch, format_name, kind
):
    from openmed.multimodal import code_regions

    class BackendCode:
        format = SimpleNamespace(name=format_name)
        position = SimpleNamespace(
            **{
                name: SimpleNamespace(x=x, y=y)
                for name, x, y in [
                    ("top_left", 10, 10),
                    ("top_right", 29, 10),
                    ("bottom_left", 10, 29),
                    ("bottom_right", 29, 29),
                ]
            }
        )

        @property
        def text(self):
            pytest.fail("adapter accessed decoded payload")

    def read(image, *, return_errors):
        assert return_errors
        return [BackendCode()]

    monkeypatch.setattr(
        code_regions.importlib,
        "import_module",
        lambda _: SimpleNamespace(read_barcodes=read),
    )
    report = detect_code_regions(
        synthetic_codes(), detector=ZXingCodeRegionDetector(), policy="fail_closed"
    )
    assert report.regions[0] == CodeRegion(kind, (10, 10, 30, 30))


@pytest.mark.parametrize("output_format", ["png", "pdf"])
def test_raster_exports_merge_codes_and_verify_pixels(output_format):
    if output_format == "pdf":
        pikepdf = pytest.importorskip("pikepdf")
    image = synthetic_codes()
    result = render_redacted_raster_pages(
        [RasterRedactionPage(image, [(140, 120, 150, 130)], (160.0, 140.0))],
        output_format=output_format,
        code_detector=PixelAwareCodes(),
        code_detection_policy="fail_closed",
    )
    assert result.pages[0]["code_coverage"]["count"] == 3
    assert len(result.pages[0]["regions_pixels"]) == 4
    assert result.pages[0]["residual_code_coverage"]["count"] == 0
    if output_format == "png":
        with Image.open(BytesIO(result.content)) as decoded:
            assert_covered(decoded)
    else:
        with pikepdf.Pdf.open(BytesIO(result.content)) as pdf:
            stream = pdf.pages[0].Resources.XObject.Im0
            assert_covered(Image.frombytes("RGB", (160, 140), stream.read_bytes()))
    assert SENTINEL not in json.dumps(result.to_dict())


def test_raster_residual_code_rejected():
    with pytest.raises(CodeDetectionError, match="residual"):
        render_redacted_raster_pages(
            [RasterRedactionPage(synthetic_codes())],
            output_format="png",
            code_detector=FakeCodeRegionDetector(REGIONS),
        )


def test_direct_optional_adapter_errors_do_not_echo_provider_detail(monkeypatch):
    from openmed.multimodal import code_regions

    def broken_import(name):
        raise RuntimeError(SENTINEL)

    monkeypatch.setattr(code_regions.importlib, "import_module", broken_import)
    with pytest.raises(CodeDetectionError) as caught:
        ZXingCodeRegionDetector().detect(synthetic_codes())
    assert SENTINEL not in "".join(traceback.format_exception(caught.value))


def test_broken_mapping_accessor_errors_do_not_echo_provider_detail():
    class BadMapping(dict):
        def get(self, *args):
            raise CodeDetectionError(SENTINEL)

    class Provider:
        def detect(self, image):
            return [BadMapping()]

    with pytest.raises(CodeDetectionError) as caught:
        detect_code_regions(synthetic_codes(), detector=Provider())
    assert SENTINEL not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("policy", ["fail_closed", "coverage_incomplete"])
def test_residual_detector_outage_follows_policy_before_writing(tmp_path, policy):
    class Provider:
        calls = 0

        def detect(self, image):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError(SENTINEL)
            return REGIONS

    source = tmp_path / "codes.png"
    synthetic_codes().save(source)
    output = tmp_path / "safe.png"
    kwargs = dict(
        output_path=output,
        ocr_engine=FakeOcrEngine(()),
        code_detector=Provider(),
        code_detection_policy=policy,
    )
    if policy == "fail_closed":
        with pytest.raises(CodeDetectionError, match="unavailable"):
            redact_image(source, **kwargs)
        assert not output.exists()
    else:
        result = redact_image(source, **kwargs)
        assert result.residual_report.code_coverage.coverage_incomplete
        assert not result.code_coverage.coverage_incomplete
        with Image.open(output) as decoded:
            assert_covered(decoded)


def test_raster_exhausted_region_budget_allows_later_empty_pages():
    pytest.importorskip("pikepdf")
    image = Image.new("RGB", (20, 20), "white")
    result = render_redacted_raster_pages(
        [
            RasterRedactionPage(image, [(0, 0, 4, 4)], (20.0, 20.0)),
            RasterRedactionPage(image, [], (20.0, 20.0)),
        ],
        output_format="pdf",
        max_regions=1,
        code_detector=FakeCodeRegionDetector(),
    )
    assert len(result.pages) == 2
