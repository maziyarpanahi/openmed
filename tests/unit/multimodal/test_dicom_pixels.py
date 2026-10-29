"""Tests for DICOM burned-in pixel-text OCR redaction."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

import openmed.multimodal.dicom as dicom_mod
from openmed.multimodal import (
    OcrResult,
    OcrWord,
    redact_dicom_pixels,
    redact_document,
)
from openmed.processing.outputs import PredictionResult

pydicom = pytest.importorskip("pydicom")
np = pytest.importorskip("numpy")
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def test_redact_dicom_pixels_redacts_burned_in_phi_and_residual_is_clean(
    monkeypatch,
    tmp_path: Path,
):
    _generic_model_misses(monkeypatch)
    source = _write_pixel_dicom(tmp_path / "phi.dcm", _single_frame_pixels())
    output = tmp_path / "redacted.dcm"

    result = redact_dicom_pixels(
        source,
        output_path=output,
        ocr_engine=_BurnedInOcrEngine(_name_words()),
        model_name="stub",
    )

    redacted = pydicom.dcmread(output).pixel_array
    assert redacted[7:21, 7:59].max() == 0
    assert result.frames_processed == 1
    assert result.redaction_count == 2
    assert result.residual_report.passed
    assert result.residual_report.residual_entity_count == 0

    serialized = json.dumps(result.to_audit_report(), sort_keys=True)
    assert "Jane" not in serialized
    assert "Doe" not in serialized
    assert "MRN-12345" not in serialized


def test_header_seeded_recognizer_recovers_name_generic_model_misses(
    monkeypatch,
    tmp_path: Path,
):
    calls: list[str] = []

    def fake_model(text: str, **_kwargs):
        calls.append(text)
        return PredictionResult(
            text=text,
            entities=[],
            model_name="stub",
            timestamp=datetime.now().isoformat(),
        )

    monkeypatch.setattr(dicom_mod, "_extract_dicom_pixel_phi", fake_model)
    source = _write_pixel_dicom(tmp_path / "phi.dcm", _single_frame_pixels())

    result = redact_dicom_pixels(
        source,
        output_path=tmp_path / "redacted.dcm",
        ocr_engine=_BurnedInOcrEngine(_name_words()),
        model_name="stub",
    )

    assert calls == ["Jane Doe"]
    assert result.redaction_count == 2
    assert {finding.sources for finding in result.findings} == {
        ("custom:deny",),
    }


def test_redact_dicom_pixels_redacts_every_frame(monkeypatch, tmp_path: Path):
    _generic_model_misses(monkeypatch)
    pixels = np.zeros((2, 32, 96), dtype=np.uint8)
    pixels[:, 8:20, 8:58] = 255
    source = _write_pixel_dicom(tmp_path / "multi.dcm", pixels)
    output = tmp_path / "multi-redacted.dcm"

    result = redact_dicom_pixels(
        source,
        output_path=output,
        ocr_engine=_BurnedInOcrEngine(_name_words()),
        model_name="stub",
    )

    redacted = pydicom.dcmread(output).pixel_array
    assert redacted[0, 7:21, 7:59].max() == 0
    assert redacted[1, 7:21, 7:59].max() == 0
    assert result.frames_processed == 2
    assert result.redaction_count == 4
    assert result.residual_report.passed


def test_redact_document_runs_header_and_pixel_pass(monkeypatch, tmp_path: Path):
    _generic_model_misses(monkeypatch)
    source = _write_pixel_dicom(tmp_path / "phi.dcm", _single_frame_pixels())
    output = tmp_path / "document-redacted.dcm"

    document = redact_document(
        source,
        policy={
            "output_path": output,
            "date_shift_days": 5,
            "ocr_engine": _BurnedInOcrEngine(_name_words()),
            "model_name": "stub",
        },
    )

    redacted = pydicom.dcmread(output)
    assert str(redacted.PatientName) == ""
    assert redacted.PatientID == ""
    assert redacted.pixel_array[7:21, 7:59].max() == 0
    assert document.metadata["format"] == "dicom"
    assert document.metadata["dicom_header_deid"]["action_count"] > 0
    assert document.metadata["dicom_pixel_redaction"]["residual_report"]["passed"]


@pytest.mark.parametrize("in_place", [False, True])
def test_residual_failure_does_not_write_unsafe_pixel_artifact(
    monkeypatch,
    tmp_path: Path,
    in_place: bool,
):
    _generic_model_misses(monkeypatch)
    source = _write_pixel_dicom(tmp_path / "phi.dcm", _single_frame_pixels())
    original = source.read_bytes()
    output = None if in_place else tmp_path / "unsafe-redacted.dcm"

    with pytest.raises(ValueError, match="residual OCR PHI verification failed"):
        redact_dicom_pixels(
            source,
            output_path=output,
            ocr_engine=_PersistentOcrEngine(_name_words()),
            model_name="stub",
            fail_on_residual=True,
        )

    assert source.read_bytes() == original
    if output is not None:
        assert not output.exists()


class _BurnedInOcrEngine:
    name = "burned-in-test"

    def __init__(self, words: tuple[OcrWord, ...]) -> None:
        self._words = words

    def recognize(self, image, *, languages=None):
        del languages
        array = np.asarray(image)
        words = []
        for word in self._words:
            x0, y0, x1, y1 = (int(value) for value in word.bbox)
            if array[y0:y1, x0:x1].max(initial=0) > 0:
                words.append(word)
        return OcrResult(words=tuple(words), metadata={"engine": self.name})


class _PersistentOcrEngine(_BurnedInOcrEngine):
    def recognize(self, image, *, languages=None):
        del image, languages
        return OcrResult(words=self._words, metadata={"engine": self.name})


def _generic_model_misses(monkeypatch) -> None:
    def fake_model(text: str, **_kwargs):
        return PredictionResult(
            text=text,
            entities=[],
            model_name="stub",
            timestamp=datetime.now().isoformat(),
        )

    monkeypatch.setattr(dicom_mod, "_extract_dicom_pixel_phi", fake_model)


def _name_words() -> tuple[OcrWord, ...]:
    return (
        OcrWord("Jane", (8.0, 8.0, 30.0, 20.0), 0.99),
        OcrWord("Doe", (31.0, 8.0, 58.0, 20.0), 0.99),
    )


def _single_frame_pixels():
    pixels = np.zeros((32, 96), dtype=np.uint8)
    pixels[8:20, 8:58] = 255
    return pixels


def _write_pixel_dicom(path: Path, pixels) -> Path:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = str(file_meta.MediaStorageSOPInstanceUID)
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.PatientName = "DOE^Jane"
    dataset.PatientID = "MRN-12345"
    dataset.PatientBirthDate = "19800102"
    dataset.StudyDate = "20200101"
    dataset.SeriesDate = "20200102"
    dataset.ContentDate = "20200103"
    dataset.Rows = int(pixels.shape[-2])
    dataset.Columns = int(pixels.shape[-1])
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    if pixels.ndim == 3:
        dataset.NumberOfFrames = int(pixels.shape[0])
    dataset.PixelData = np.ascontiguousarray(pixels).tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return path


@pytest.mark.parametrize("verified", [False, True])
def test_pixel_markers_only_attest_verified_cleaning(monkeypatch, tmp_path, verified):
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence

    from openmed.multimodal import DicomPixelStatus

    _generic_model_misses(monkeypatch)
    source = _write_pixel_dicom(tmp_path / "phi.dcm", _single_frame_pixels())
    dataset = pydicom.dcmread(source)
    dataset.PatientIdentityRemoved = "YES"
    dataset.DeidentificationMethodCodeSequence = Sequence([Dataset()])
    dataset.save_as(source, enforce_file_format=True)
    result = redact_dicom_pixels(
        source,
        ocr_engine=_BurnedInOcrEngine(_name_words()),
        model_name="stub",
        verify_residual=verified,
    )
    cleaned = pydicom.dcmread(source)
    assert result.pixel_status is (
        DicomPixelStatus.CLEANED if verified else DicomPixelStatus.NOT_CLEANED
    )
    assert cleaned.BurnedInAnnotation == ("NO" if verified else "YES")
    assert cleaned.PatientIdentityRemoved == "NO"
    assert "DeidentificationMethodCodeSequence" not in cleaned


@pytest.mark.parametrize("mode", ["remove", "burn"])
def test_overlay_is_removed_or_burned_before_ocr(monkeypatch, tmp_path, mode):
    _generic_model_misses(monkeypatch)
    pixels = np.zeros((32, 96), dtype=np.uint8)
    source = _write_pixel_dicom(tmp_path / "overlay.dcm", pixels)
    dataset = pydicom.dcmread(source)
    mask = np.zeros_like(pixels)
    mask[8:20, 8:58] = 1
    dataset.add_new((0x6000, 0x0010), "US", 32)
    dataset.add_new((0x6000, 0x0011), "US", 96)
    dataset.add_new((0x6000, 0x0040), "CS", "G")
    dataset.add_new((0x6000, 0x0050), "SS", [1, 1])
    dataset.add_new((0x6000, 0x0100), "US", 1)
    dataset.add_new((0x6000, 0x0102), "US", 0)
    dataset.add_new(
        (0x6000, 0x3000), "OW", np.packbits(mask.ravel(), bitorder="little").tobytes()
    )
    dataset.save_as(source, enforce_file_format=True)
    result = redact_dicom_pixels(
        source,
        policy={"overlay_mode": mode},
        ocr_engine=_BurnedInOcrEngine(_name_words()),
        model_name="stub",
    )
    cleaned = pydicom.dcmread(source)
    assert not any(0x6000 <= tag.group <= 0x60FF for tag in cleaned.keys())
    assert cleaned.pixel_array.max() == 0
    assert result.redaction_count == (2 if mode == "burn" else 0)
    assert result.carrier_actions


def test_embedded_overlay_unused_bits_are_removed_from_stored_pixel_bytes(tmp_path):
    from openmed.multimodal import deidentify_dicom_headers

    pixels = np.full((32, 96), 5, dtype=np.uint16)
    pixels[8:20, 8:58] |= 1 << 15
    source = _write_pixel_dicom(tmp_path / "embedded.dcm", pixels)
    dataset = pydicom.dcmread(source)
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.add_new((0x6000, 0x0010), "US", 32)
    dataset.add_new((0x6000, 0x0011), "US", 96)
    dataset.add_new((0x6000, 0x0100), "US", 16)
    dataset.add_new((0x6000, 0x0102), "US", 15)
    dataset.save_as(source, enforce_file_format=True)
    deidentify_dicom_headers(source)
    cleaned = pydicom.dcmread(source)
    assert np.frombuffer(cleaned.PixelData, dtype="<u2").max() == 5
    assert cleaned.pixel_array.min() == 5
    assert (0x6000, 0x0102) not in cleaned


def test_combined_dispatch_redacts_embedded_document_once(monkeypatch, tmp_path):
    from openmed.multimodal import ExtractedDocument, base, register_handler

    _generic_model_misses(monkeypatch)
    source = _write_pixel_dicom(tmp_path / "encap.dcm", _single_frame_pixels())
    dataset = pydicom.dcmread(source)
    dataset.EncapsulatedDocument = b"%PDF-SYNTHETIC_PAYLOAD"
    dataset.MIMETypeOfEncapsulatedDocument = "application/pdf"
    dataset.save_as(source, enforce_file_format=True)
    calls = []

    def handler(stream, **kwargs):
        calls.append(stream.read())
        return ExtractedDocument(
            text="", metadata={"redacted_document_bytes": b"%PDF-CLEAN!"}
        )

    monkeypatch.setitem(base._HANDLERS, ".pdf", [])
    register_handler(".pdf", handler, requires_multimodal=False)
    result = redact_document(
        source,
        policy={
            "ocr_engine": _BurnedInOcrEngine(_name_words()),
            "model_name": "stub",
            "redact_encapsulated_documents": True,
            "document_models": lambda _: [],
        },
    )
    assert len(calls) == 1
    cleaned = pydicom.dcmread(source)
    assert cleaned.EncapsulatedDocument.rstrip(b"\0") == b"%PDF-CLEAN!"
    assert cleaned.PatientIdentityRemoved == "YES"
    assert cleaned.BurnedInAnnotation == "NO"
    assert result.metadata["dicom_pixel_redaction"]["pixel_status"] == "pixels_cleaned"


def _add_unclean_nested_pixels(source):
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence

    dataset = pydicom.dcmread(source)
    child = Dataset()
    child.add_new((0x7FE0, 0x0010), "OB", b"SYNTHETIC_CHILD!")
    child.BurnedInAnnotation = "YES"
    child.PatientIdentityRemoved = "YES"
    dataset.ReferencedImageSequence = Sequence([child])
    dataset.save_as(source, enforce_file_format=True)


def test_pixel_outcome_covers_nested_pixels_without_inheriting_clean_claims(
    monkeypatch, tmp_path
):
    from openmed.multimodal import DicomPixelStatus

    _generic_model_misses(monkeypatch)
    source = _write_pixel_dicom(tmp_path / "nested.dcm", _single_frame_pixels())
    _add_unclean_nested_pixels(source)
    result = redact_dicom_pixels(
        source, ocr_engine=_BurnedInOcrEngine(_name_words()), model_name="stub"
    )
    cleaned = pydicom.dcmread(source)
    assert result.pixel_status is DicomPixelStatus.NOT_CLEANED
    assert cleaned.PatientIdentityRemoved == "NO"
    assert cleaned.ReferencedImageSequence[0].PatientIdentityRemoved == "NO"


@pytest.mark.parametrize("in_place", [False, True])
def test_combined_refusal_does_not_publish_partial_pixels(
    monkeypatch, tmp_path, in_place
):
    from openmed.multimodal import DicomDeidentificationError

    _generic_model_misses(monkeypatch)
    source = _write_pixel_dicom(tmp_path / "nested.dcm", _single_frame_pixels())
    _add_unclean_nested_pixels(source)
    original = source.read_bytes()
    output = source if in_place else tmp_path / "existing.dcm"
    if not in_place:
        output.write_bytes(b"original destination")
    with pytest.raises(DicomDeidentificationError, match="pixels_not_cleaned"):
        redact_document(
            source,
            policy={
                "output_path": output,
                "fail_on_unclean_pixels": True,
                "ocr_engine": _BurnedInOcrEngine(_name_words()),
                "model_name": "stub",
            },
        )
    assert source.read_bytes() == original
    if not in_place:
        assert output.read_bytes() == b"original destination"


@pytest.mark.parametrize("pixel_api", [False, True])
def test_malformed_overlay_refusal_is_value_free(tmp_path, pixel_api):
    from openmed.multimodal import DicomDeidentificationError, deidentify_dicom_headers

    source = _write_pixel_dicom(tmp_path / "malformed.dcm", _single_frame_pixels())
    dataset = pydicom.dcmread(source)
    dataset.add_new((0x6000, 0x0102), "LO", "SYNTHETIC_PRIVATE_SENTINEL")
    dataset.add_new((0x6000, 0x3000), "OW", b"overlay data")
    dataset.save_as(source, enforce_file_format=True)
    original = source.read_bytes()
    with pytest.raises(DicomDeidentificationError) as exc:
        (redact_dicom_pixels if pixel_api else deidentify_dicom_headers)(source)
    assert str(exc.value) == "embedded_overlay_not_cleanable"
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "keyword", ["FloatPixelData", "DoubleFloatPixelData", "PixelDataProviderURL"]
)
def test_unsupported_pixel_storage_cannot_be_empty_success(tmp_path, keyword):
    from openmed.multimodal import DicomDeidentificationError

    source = _write_pixel_dicom(tmp_path / "alternate.dcm", _single_frame_pixels())
    dataset = pydicom.dcmread(source)
    del dataset.PixelData
    setattr(
        dataset,
        keyword,
        "https://example.invalid/synthetic-private"
        if keyword == "PixelDataProviderURL"
        else b"synthetic pixels",
    )
    dataset.save_as(source, enforce_file_format=True)
    original = source.read_bytes()
    with pytest.raises(DicomDeidentificationError) as exc:
        redact_dicom_pixels(source)
    assert exc.value.reason_code == (
        "external_pixel_data_unsupported"
        if keyword == "PixelDataProviderURL"
        else "pixel_data_unsupported"
    )
    assert source.read_bytes() == original
