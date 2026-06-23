"""Tests for OCR engine adapters and engine selection.

These run without Tesseract/PaddleOCR binaries: the deterministic fake engine
verifies the shared OcrResult contract, and the real adapters are checked for
structural conformance and for clear errors when their dependency is absent.
"""

from __future__ import annotations

import importlib.util

import pytest

import openmed.multimodal.ocr as ocr_mod
from openmed.multimodal.exceptions import MissingDependencyError
from openmed.multimodal.ocr import (
    FakeOcrEngine,
    OcrEngine,
    OcrResult,
    OcrWord,
    PaddleOcrEngine,
    TesseractEngine,
    ocr,
    resolve_engine,
)


def _assert_ocr_result_shape(result):
    assert isinstance(result, OcrResult)
    for word in result.words:
        assert isinstance(word.text, str)
        assert len(word.bbox) == 4
        assert all(isinstance(v, (int, float)) for v in word.bbox)
        assert isinstance(word.confidence, float)
        assert isinstance(word.page, int)


def test_fake_engine_satisfies_ocr_contract():
    result = FakeOcrEngine([OcrWord("x", (0.0, 0.0, 1.0, 1.0), 0.5, 0)]).recognize(
        "img"
    )
    _assert_ocr_result_shape(result)


def test_real_adapters_conform_to_engine_protocol():
    # Adapters must be constructible and conform without importing their deps.
    assert isinstance(TesseractEngine(), OcrEngine)
    assert isinstance(PaddleOcrEngine(), OcrEngine)
    assert TesseractEngine().name == "tesseract"
    assert PaddleOcrEngine().name == "paddleocr"


def test_resolve_engine_by_name():
    assert isinstance(resolve_engine("tesseract"), TesseractEngine)
    assert isinstance(resolve_engine("paddleocr"), PaddleOcrEngine)


def test_resolve_engine_passes_through_instance():
    fake = FakeOcrEngine([])
    assert resolve_engine(fake) is fake


def test_unknown_engine_name_raises_value_error():
    with pytest.raises(ValueError):
        resolve_engine("does-not-exist")


@pytest.mark.skipif(
    importlib.util.find_spec("pytesseract") is not None,
    reason="pytesseract is installed",
)
def test_missing_tesseract_yields_actionable_error():
    with pytest.raises(MissingDependencyError) as excinfo:
        ocr("image.png", engine="tesseract")
    message = str(excinfo.value).lower()
    assert "tesseract" in message


def test_auto_select_without_any_engine_is_actionable(monkeypatch):
    monkeypatch.setattr(ocr_mod, "_engine_available", lambda name: False)
    with pytest.raises(MissingDependencyError) as excinfo:
        ocr("image.png")
    message = str(excinfo.value).lower()
    assert "tesseract" in message or "paddle" in message
    assert "multimodal" in message
