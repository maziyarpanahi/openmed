"""Optional local barcode backend integration using generated sentinel symbols."""

from io import BytesIO

import pytest

from openmed.multimodal import (
    RasterRedactionPage,
    ZXingCodeRegionDetector,
    detect_code_regions,
    redact_image,
    render_redacted_pdf,
    render_redacted_raster_pages,
)
from openmed.multimodal.ocr import FakeOcrEngine

pytestmark = pytest.mark.integration


@pytest.fixture
def codes():
    zxing = pytest.importorskip("zxingcpp")
    Image = pytest.importorskip("PIL.Image")
    canvas = Image.new("RGB", (1024, 512), "white")
    for name, position in [
        ("Code128", (20, 20)),
        ("QRCode", (20, 200)),
        ("DataMatrix", (280, 200)),
    ]:
        pixels = zxing.write_barcode(
            getattr(zxing.BarcodeFormat, name),
            "SYNTHETIC-MRN-3816",
            width=240,
            height=120 if name == "Code128" else 180,
        )
        symbol = Image.fromarray(pixels).convert("RGB")
        canvas.paste(symbol, position)
    return canvas


def test_real_local_detector_redacts_all_three_symbologies(codes, tmp_path):
    Image = pytest.importorskip("PIL.Image")
    detector = ZXingCodeRegionDetector()
    before = detect_code_regions(codes, detector=detector, policy="fail_closed")
    assert {r.symbology for r in before.regions} == {"linear", "qr", "datamatrix"}
    source = tmp_path / "codes.png"
    codes.save(source)
    result = redact_image(
        source,
        ocr_engine=FakeOcrEngine(()),
        code_detector=detector,
        code_detection_policy="fail_closed",
    )
    assert result.residual_report.code_coverage.clean
    with Image.open(BytesIO(result.redacted_bytes)) as decoded:
        for region in before.regions:
            assert decoded.crop(region.bbox).getextrema() == ((0, 0),) * 3
        assert detect_code_regions(
            decoded, detector=detector, policy="fail_closed"
        ).clean


def test_both_pdf_rasterizers_redact_generated_codes(codes, tmp_path):
    pytest.importorskip("pikepdf")
    pdfplumber = pytest.importorskip("pdfplumber")
    detector = ZXingCodeRegionDetector()
    exported = render_redacted_raster_pages(
        [RasterRedactionPage(codes, size_points=(512.0, 256.0))],
        output_format="pdf",
        code_detector=detector,
        code_detection_policy="fail_closed",
    )
    assert exported.pages[0]["code_coverage"]["count"] >= 3
    source = tmp_path / "source.pdf"
    codes.save(source, format="PDF", resolution=144)
    output = tmp_path / "redacted.pdf"
    result = render_redacted_pdf(
        source,
        output,
        [],
        render_dpi=144,
        code_detector=detector,
        code_detection_policy="fail_closed",
    )
    assert result.passed
    assert {r.symbology for r in result.code_coverage.regions} == {
        "linear",
        "qr",
        "datamatrix",
    }
    assert result.residual_code_coverage.clean
    with pdfplumber.open(output) as pdf:
        assert detect_code_regions(
            pdf.pages[0].to_image(resolution=144).original,
            detector=detector,
            policy="fail_closed",
        ).clean


def test_pdf_residual_code_detection_rejects_publication(codes, tmp_path):
    from openmed.multimodal import CodeDetectionError, FakeCodeRegionDetector

    pytest.importorskip("pikepdf")
    source = tmp_path / "codes.pdf"
    codes.save(source, format="PDF", resolution=144)
    located = detect_code_regions(
        codes, detector=ZXingCodeRegionDetector(), policy="fail_closed"
    )
    output = tmp_path / "output.pdf"
    with pytest.raises(CodeDetectionError, match="residual"):
        render_redacted_pdf(
            source, output, [], code_detector=FakeCodeRegionDetector(located.regions)
        )
    assert not output.exists()


def test_image_only_pdf_codes_keep_text_evidence_truthful(codes, tmp_path):
    pytest.importorskip("pikepdf")
    source = tmp_path / "codes.pdf"
    codes.save(source, format="PDF", resolution=144)
    result = render_redacted_pdf(
        source, tmp_path / "output.pdf", [], code_detection_policy="fail_closed"
    )
    assert all(not region.source_text_found for region in result.text_removal.regions)
    assert all(
        not region.source_text_required for region in result.text_removal.regions
    )
    assert result.text_removal.passed and result.residual_code_coverage.clean
