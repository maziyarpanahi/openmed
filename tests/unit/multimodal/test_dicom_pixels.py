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
