"""In-memory export preserves pixels and cannot retain hidden representations."""

import hashlib
import json
from io import BytesIO

import pytest
from PIL import Image, ImageDraw, PngImagePlugin

from openmed.multimodal import (
    RasterExportError,
    RasterRedactionPage,
    render_redacted_raster_pages,
)
from openmed.multimodal import render_raster as renderer


@pytest.fixture
def image():
    with Image.new("RGB", (300, 160), "white") as source:
        draw = ImageDraw.Draw(source)
        draw.text((10, 15), "Anna Beispiel", fill="black")
        draw.text((10, 60), "Keine Dyspnoe. LVEF 55 %.", fill="black")
        source.info["private_marker"] = "Anna Beispiel"
        yield source


def test_png_is_lossless_outside_burned_pixels_and_strips_metadata(image):
    original = image.tobytes()
    result = render_redacted_raster_pages(
        [RasterRedactionPage(image, [(8, 10, 90, 30)])], output_format="png"
    )
    assert (
        image.tobytes() == original and image.info["private_marker"] == "Anna Beispiel"
    )
    with Image.open(BytesIO(result.content)) as decoded:
        assert decoded.info == {} and decoded.mode == "RGB"
        assert decoded.crop((5, 7, 93, 33)).getextrema() == ((0, 0),) * 3
        assert (
            decoded.crop((0, 40, 300, 160)).tobytes()
            == image.crop((0, 40, 300, 160)).tobytes()
        )
    report = result.to_dict()
    assert (
        report["text_layer"] == "absent" and report["source_metadata"] == "not_copied"
    )
    assert "Anna" not in json.dumps(report) and "Anna" not in repr(result)
    assert (
        "content" not in report
        and report["sha256"] == hashlib.sha256(result.content).hexdigest()
    )


def test_transparency_is_flattened_on_white_and_hidden_values_are_removed():
    with Image.new("RGBA", (20, 20), (20, 50, 80, 0)) as source:
        result = render_redacted_raster_pages(
            [RasterRedactionPage(source)], output_format="png"
        )
        with Image.open(BytesIO(result.content)) as decoded:
            assert decoded.getextrema() == ((255, 255),) * 3
        assert source.getpixel((0, 0)) == (20, 50, 80, 0)


def test_pdf_is_multipage_without_any_source_text_or_metadata(image):
    pikepdf = pytest.importorskip("pikepdf")
    pages = [
        RasterRedactionPage(image, [(8, 10, 90, 30)], (72.0, 38.4)),
        RasterRedactionPage(image, [], (144.0, 76.8)),
    ]
    result = render_redacted_raster_pages(pages, output_format="pdf")
    repeat = render_redacted_raster_pages(pages, output_format="pdf")
    assert result.content == repeat.content
    assert (
        result.media_type == "application/pdf" and result.to_dict()["page_count"] == 2
    )
    with pikepdf.Pdf.open(BytesIO(result.content)) as pdf:
        assert len(pdf.pages) == 2 and not pdf.attachments
        assert "/Info" not in pdf.trailer and "/Metadata" not in pdf.Root
        assert all(b"BT" not in p.Contents.read_bytes() for p in pdf.pages)
        assert [float(v) for v in pdf.pages[0].MediaBox] == [0, 0, 72, 38.4]
        assert pdf.pages[1].Resources.XObject.Im0.read_bytes() == image.tobytes()


@pytest.mark.parametrize("kind", ["unburned", "outside"])
def test_pixel_verifier_rejects_incorrect_burn_before_encoding(
    image, monkeypatch, kind
):
    def broken(image, regions):
        if kind == "outside":
            for bbox in regions:
                image.paste((0, 0, 0), bbox)
            image.putpixel((250, 150), (1, 2, 3))

    monkeypatch.setattr(renderer, "_burn", broken)
    with pytest.raises(
        RasterExportError, match="verification_failed|outside_pixels_changed"
    ):
        render_redacted_raster_pages(
            [RasterRedactionPage(image, [(8, 10, 90, 30)])], output_format="png"
        )


@pytest.mark.parametrize(
    "kind", ["metadata", "text", "pixels", "geometry", "attachment"]
)
def test_pdf_verifier_rejects_auxiliary_content_or_encoding_damage(
    image, monkeypatch, kind
):
    pikepdf = pytest.importorskip("pikepdf")
    append = renderer._append_page

    def corrupt(pdf, library, raster, **options):
        append(pdf, library, raster, **options)
        page = pdf.pages[-1]
        if kind == "metadata":
            pdf.docinfo["/Author"] = "Anna Beispiel"
        elif kind == "text":
            page.Contents = pdf.make_stream(
                page.Contents.read_bytes()
                + b"BT /OpenMedSafeText 10 Tf (Anna Beispiel) Tj ET"
            )
        elif kind == "pixels":
            page.Resources.XObject.Im0.write(b"\xff" * (300 * 160 * 3))
        elif kind == "geometry":
            page.MediaBox = pikepdf.Array([0, 0, 75, 38.4])
        else:
            pdf.attachments["private.txt"] = b"Anna Beispiel"

    monkeypatch.setattr(renderer, "_append_page", corrupt)
    with pytest.raises(
        RasterExportError, match="encoding_verification_failed"
    ) as error:
        render_redacted_raster_pages(
            [RasterRedactionPage(image, [(8, 10, 90, 30)], (72, 38.4))],
            output_format="pdf",
        )
    assert "Anna" not in str(error.value)


def test_png_verifier_rejects_added_text_metadata(image, monkeypatch):
    save = Image.Image.save

    def corrupt(self, fp, **options):
        info = PngImagePlugin.PngInfo()
        info.add_text("Author", "Anna Beispiel")
        return save(self, fp, pnginfo=info, **options)

    monkeypatch.setattr(Image.Image, "save", corrupt)
    with pytest.raises(RasterExportError, match="encoding_verification_failed"):
        render_redacted_raster_pages([RasterRedactionPage(image)], output_format="png")


@pytest.mark.parametrize(
    "bbox",
    [
        (0, 0, 0, 1),
        (-1, 0, 10, 10),
        (0, 0, 301, 10),
        (0.1, 0, 10, 10),
        (True, 0, 10, 10),
        (0, 0, 10),
    ],
)
def test_invalid_pixel_geometry_fails_closed(image, bbox):
    with pytest.raises(RasterExportError, match="invalid_geometry"):
        render_redacted_raster_pages(
            [RasterRedactionPage(image, [bbox])], output_format="png"
        )


@pytest.mark.parametrize(
    "options,code",
    [
        ({"max_page_pixels": 47999}, "page_pixel_limit"),
        ({"max_total_pixels": 47999}, "total_pixel_limit"),
        ({"max_regions": 1}, "region_limit"),
        ({"max_output_bytes": 10}, "output_limit"),
        ({"padding_pixels": -1}, "invalid_options"),
        ({"max_pages": True}, "invalid_limits"),
    ],
)
def test_resource_limits(image, options, code):
    with pytest.raises(RasterExportError, match=code):
        render_redacted_raster_pages(
            [RasterRedactionPage(image, [(0, 0, 1, 1), (0, 0, 1, 1)])],
            output_format="png",
            **options,
        )


@pytest.mark.parametrize("points", [None, (0, 10), (float("nan"), 10), (15000, 10)])
def test_pdf_requires_valid_physical_dimensions(image, points):
    pytest.importorskip("pikepdf")
    with pytest.raises(RasterExportError, match="invalid_page_size"):
        render_redacted_raster_pages(
            [RasterRedactionPage(image, [], points)], output_format="pdf"
        )


def test_empty_and_multiple_png_pages_are_rejected(image):
    with pytest.raises(RasterExportError, match="no_pages"):
        render_redacted_raster_pages([], output_format="png")
    with pytest.raises(RasterExportError, match="page_limit"):
        render_redacted_raster_pages(
            [RasterRedactionPage(image)] * 2, output_format="png"
        )


def test_cancellation_between_pages_releases_pdf_without_returning_partial_bytes(image):
    pytest.importorskip("pikepdf")
    cancelled = False

    def pages():
        nonlocal cancelled
        yield RasterRedactionPage(image, [], (72, 38.4))
        cancelled = True
        yield RasterRedactionPage(image, [], (72, 38.4))

    with pytest.raises(RasterExportError, match="cancelled"):
        render_redacted_raster_pages(
            pages(), output_format="pdf", cancel_check=lambda: cancelled
        )


def test_native_errors_do_not_echo_source(image, monkeypatch):
    def failed(*args):
        raise RuntimeError("Anna Beispiel")

    monkeypatch.setattr(renderer, "_burn", failed)
    with pytest.raises(RasterExportError, match="^raster_export_failed$") as error:
        render_redacted_raster_pages([RasterRedactionPage(image)], output_format="png")
    assert error.value.__suppress_context__
