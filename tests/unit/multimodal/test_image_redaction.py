"""Tests for plain raster-image PHI pixel redaction."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pytest

import openmed.multimodal.base as base
from openmed.multimodal import redact_document, redact_image, verify_image_metadata
from openmed.multimodal.ocr import OcrResult, OcrWord

Image = pytest.importorskip("PIL.Image")
ImageChops = pytest.importorskip("PIL.ImageChops")
ImageDraw = pytest.importorskip("PIL.ImageDraw")
ImageSequence = pytest.importorskip("PIL.ImageSequence")
PngImagePlugin = pytest.importorskip("PIL.PngImagePlugin")


WORDS = (
    OcrWord("Patient", (10.0, 16.0, 72.0, 34.0), 0.99, page=0),
    OcrWord("John", (76.0, 16.0, 118.0, 34.0), 0.98, page=0),
    OcrWord("Doe", (122.0, 16.0, 154.0, 34.0), 0.97, page=0),
)


class SequencedOcrEngine:
    """Return one deterministic OCR result per recognize call."""

    name = "sequence"

    def __init__(self, pages: Iterable[Iterable[OcrWord]]) -> None:
        self._pages = [tuple(page) for page in pages]
        self.languages: list[list[str] | None] = []

    def recognize(self, image, *, languages=None) -> OcrResult:
        self.languages.append(list(languages) if languages is not None else None)
        words = self._pages.pop(0) if self._pages else ()
        return OcrResult(words=tuple(words), metadata={"engine": self.name})


def _write_burned_in_png(path: Path) -> None:
    image = Image.new("RGB", (220, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 18), "Patient John Doe", fill="black")
    image.save(path)


def _john_doe_detector(text: str, *, lang: str | None = None):
    assert lang in {None, "en", "fr"}
    if "John" not in text:
        return {"entities": []}
    start = text.index("John")
    end = text.index("Doe") + len("Doe")
    return {
        "entities": [
            {
                "start": start,
                "end": end,
                "label": "PERSON",
                "confidence": 0.95,
            }
        ]
    }


def test_png_burned_in_phi_is_pixel_redacted_and_reocr_clean(tmp_path):
    source = tmp_path / "scan.png"
    _write_burned_in_png(source)
    engine = SequencedOcrEngine([WORDS, ()])

    result = redact_image(
        source,
        models={"detector": _john_doe_detector, "ocr_engine": engine},
        lang="en",
    )

    assert result.residual_report is not None
    assert result.residual_report.clean
    assert result.residual_report.residual_count == 0
    assert result.changed_pixel_count > 0
    assert result.modified_pixels
    assert len(result.redaction_boxes) == 1
    assert result.redaction_boxes[0].bbox == (76.0, 16.0, 154.0, 34.0)
    assert result.metadata_report.clean

    original = Image.open(source).convert("RGB")
    redacted = Image.open(BytesIO(result.redacted_bytes)).convert("RGB")
    assert ImageChops.difference(original, redacted).getbbox() is not None


def test_three_frame_tiff_redacts_each_frame(tmp_path):
    source = tmp_path / "scan.tiff"
    frames = []
    for index in range(3):
        frame = Image.new("RGB", (100, 44), "white")
        draw = ImageDraw.Draw(frame)
        draw.text((10, 14), f"John {index}", fill="black")
        frames.append(frame)
    frames[0].save(source, save_all=True, append_images=frames[1:])

    word = OcrWord("John", (10.0, 12.0, 54.0, 30.0), 0.99, page=0)
    engine = SequencedOcrEngine([(word,), (word,), (word,), (), (), ()])

    def detector(text: str, *, lang: str | None = None):
        return {
            "entities": [
                {"start": match.start(), "end": match.end(), "label": "PERSON"}
                for match in re.finditer("John", text)
            ]
        }

    result = redact_image(
        source,
        models={"detector": detector, "ocr_engine": engine},
    )

    assert result.frame_count == 3
    assert [box.page for box in result.redaction_boxes] == [0, 1, 2]
    assert len(result.changed_pixels_by_frame) == 3
    assert all(count > 0 for count in result.changed_pixels_by_frame)
    assert result.residual_report is not None
    assert result.residual_report.clean

    with Image.open(BytesIO(result.redacted_bytes)) as output:
        assert sum(1 for _ in ImageSequence.Iterator(output)) == 3


def test_exif_and_xmp_metadata_are_stripped(tmp_path):
    jpeg = tmp_path / "exif.jpg"
    exif = Image.Exif()
    exif[315] = "John Doe"
    Image.new("RGB", (40, 30), "white").save(jpeg, exif=exif)
    assert Image.open(jpeg).info.get("exif")

    jpeg_result = redact_image(
        jpeg,
        models={"ocr_engine": SequencedOcrEngine([()])},
        verify=False,
    )
    assert verify_image_metadata(jpeg_result.redacted_bytes).clean
    assert "exif" not in Image.open(BytesIO(jpeg_result.redacted_bytes)).info

    png = tmp_path / "xmp.png"
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("XML:com.adobe.xmp", "<xmp>Patient John Doe</xmp>")
    Image.new("RGB", (40, 30), "white").save(png, pnginfo=png_info)
    assert "XML:com.adobe.xmp" in Image.open(png).info

    png_result = redact_image(
        png,
        models={"ocr_engine": SequencedOcrEngine([()])},
        verify=False,
    )
    assert verify_image_metadata(png_result.redacted_bytes).clean
    assert (
        "XML:com.adobe.xmp" not in Image.open(BytesIO(png_result.redacted_bytes)).info
    )


def test_redact_document_dispatches_png_to_pixel_redaction_handler(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(base, "_missing_multimodal_dependencies", lambda: [])
    source = tmp_path / "scan.png"
    _write_burned_in_png(source)
    engine = SequencedOcrEngine([WORDS, ()])

    doc = redact_document(
        source,
        models={"detector": _john_doe_detector, "ocr_engine": engine},
        lang="fr",
    )

    assert doc.text == "Patient John Doe"
    assert doc.metadata["format"] == "image"
    assert doc.metadata["image_format"] == "PNG"
    assert doc.metadata["pixel_redaction"]["modified_pixels"] is True
    assert doc.metadata["residual_report"]["clean"] is True
    assert doc.metadata["metadata_report"]["clean"] is True
    assert isinstance(doc.metadata["redacted_image_bytes"], bytes)
    assert engine.languages == [["fr"], ["fr"]]
