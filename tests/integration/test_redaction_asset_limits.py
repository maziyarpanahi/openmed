"""Synthetic offline media admission through the actual redaction handlers."""

from __future__ import annotations

from io import BytesIO

import pytest

from openmed.multimodal import DESKTOP_V1, OcrResult, redact_image
from openmed.multimodal.documents_pdf import _pdf_handler, _render_redacted_pdf
from tests.fixtures.multimodal import redaction_assets as assets

pytestmark = pytest.mark.integration


class EmptyOcr:
    name = "synthetic-empty"

    def recognize(self, image, *, languages=None):
        return OcrResult(words=())


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "TIFF", "MPO", "APNG"])
def test_within_limit_image_pixels_and_results_are_unchanged(tmp_path, image_format):
    Image = pytest.importorskip("PIL.Image")
    source = tmp_path / ("image." + image_format.lower())
    frames = [Image.new("RGB", (8, 8), "white")]
    if image_format in {"TIFF", "MPO", "APNG"}:
        frames.append(Image.new("RGB", (8, 8), "black"))
    if len(frames) > 1:
        frames[0].save(
            source,
            format="PNG" if image_format == "APNG" else image_format,
            save_all=True,
            append_images=frames[1:],
        )
    else:
        frames[0].save(source, format=image_format)
    output = tmp_path / "default.tiff" if image_format in {"MPO", "APNG"} else None
    default = redact_image(
        source, output_path=output, ocr_engine=EmptyOcr(), verify=False
    )
    explicit = redact_image(
        source,
        policy={"asset_limit_profile": DESKTOP_V1},
        output_path=tmp_path / "explicit.tiff" if output is not None else None,
        ocr_engine=EmptyOcr(),
        verify=False,
    )
    assert default.redacted_bytes == explicit.redacted_bytes
    assert default.changed_pixel_count == 0
    assert default.frame_count == len(frames)
    assert default.extracted_document.text == explicit.extracted_document.text == ""
    with Image.open(BytesIO(default.redacted_bytes)) as output:
        for index, frame in enumerate(frames):
            output.seek(index)
            assert output.convert("RGB").tobytes() == frame.tobytes()


def test_within_limit_pdf_text_and_raster_output_are_unchanged():
    pytest.importorskip("pdfplumber")
    source = assets.pdf(2)
    default = _pdf_handler(BytesIO(source))
    explicit = _pdf_handler(BytesIO(source), policy={"asset_limit_profile": DESKTOP_V1})
    assert default.text == explicit.text == ""
    assert default.spans == explicit.spans == ()
    assert default.metadata["page_count"] == explicit.metadata["page_count"] == 2
    rendered = _render_redacted_pdf(BytesIO(source), ())
    alternate = _render_redacted_pdf(
        BytesIO(source), (), policy={"asset_limit_profile": DESKTOP_V1}
    )
    assert rendered == alternate


def test_within_limit_dicom_pixels_are_unchanged(monkeypatch, tmp_path):
    pydicom = pytest.importorskip("pydicom")
    np = pytest.importorskip("numpy")
    from openmed.multimodal import dicom as dicom_mod

    meta = pydicom.dataset.FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.10.543.1"
    dataset = pydicom.dataset.FileDataset(
        None, {}, file_meta=meta, preamble=b"\0" * 128
    )
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Rows = dataset.Columns = 8
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.NumberOfFrames = 2
    pixels = np.zeros((2, 8, 8), dtype=np.uint8)
    dataset.PixelData = pixels.tobytes()
    source = tmp_path / "synthetic.dcm"
    dataset.save_as(source, enforce_file_format=True)
    # An empty synthetic OCR result requires no detector/model invocation.
    monkeypatch.setattr(dicom_mod, "_detect_pixel_entities", lambda *args, **kwargs: ())
    for index, policy in enumerate((None, {"asset_limit_profile": DESKTOP_V1})):
        destination = tmp_path / f"output-{index}.dcm"
        result = dicom_mod.redact_dicom_pixels(
            source, policy=policy, output_path=destination, ocr_engine=EmptyOcr()
        )
        assert result.frames_processed == 2 and result.redaction_count == 0
        assert result.residual_report.passed
        assert np.array_equal(pydicom.dcmread(destination).pixel_array, pixels)
