"""Tests for the EasyOCR adapter without importing the real backend."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import openmed.multimodal.ocr as ocr_mod
from openmed.multimodal.exceptions import MissingDependencyError
from openmed.multimodal.ocr import EasyOcrEngine, OcrResult, ocr, resolve_engine


def _assert_result_shape(result: OcrResult) -> None:
    assert isinstance(result, OcrResult)
    for word in result.words:
        assert isinstance(word.text, str)
        assert len(word.bbox) == 4
        assert all(isinstance(value, float) for value in word.bbox)
        assert isinstance(word.confidence, float)
        assert isinstance(word.page, int)


def test_easyocr_engine_maps_polygon_predictions_to_ocr_contract(monkeypatch):
    seen = {}
    polygon = [(1, 2), (41, 2), (41, 12), (1, 12)]

    class FakeReader:
        def __init__(self, languages):
            seen["languages"] = languages

        def readtext(self, image):
            seen["image"] = image
            return [(polygon, "MRN", 0.88)]

    monkeypatch.setattr(
        ocr_mod,
        "_import_backend",
        lambda module, instruction: SimpleNamespace(Reader=FakeReader),
    )

    result = ocr(Path("scan.png"), engine="easyocr", languages=["fr"])

    _assert_result_shape(result)
    assert seen == {"languages": ["fr"], "image": "scan.png"}
    assert result.metadata["engine"] == "easyocr"
    assert result.words[0].text == "MRN"
    assert result.words[0].bbox == (1.0, 2.0, 41.0, 12.0)
    assert result.words[0].confidence == 0.88
    assert result.words[0].page == 0

    doc = result.to_document()
    span = doc.location_at(doc.text.index("MRN"))
    assert span is not None
    assert span.bbox == (1.0, 2.0, 41.0, 12.0)
    assert doc.text_for(span) == "MRN"


def test_easyocr_adapter_splits_detected_text_into_word_results(monkeypatch):
    polygon = [(10, 20), (50, 20), (50, 40), (10, 40)]

    class FakeReader:
        def __init__(self, languages):
            pass

        def readtext(self, image):
            return [(polygon, "Jane Doe", 0.91)]

    monkeypatch.setattr(
        ocr_mod,
        "_import_backend",
        lambda module, instruction: SimpleNamespace(Reader=FakeReader),
    )

    result = EasyOcrEngine().recognize(object())

    assert [word.text for word in result.words] == ["Jane", "Doe"]
    assert result.words[0].bbox == pytest.approx((10.0, 20.0, 32.85714285714286, 40.0))
    assert result.words[1].bbox == pytest.approx((32.85714285714286, 20.0, 50.0, 40.0))


def test_resolve_engine_accepts_easyocr_name():
    assert isinstance(resolve_engine("easyocr"), EasyOcrEngine)


def test_auto_select_chooses_easyocr_when_it_is_the_installed_engine(monkeypatch):
    monkeypatch.setattr(ocr_mod, "_engine_available", lambda name: name == "easyocr")

    assert "easyocr" in ocr_mod._AUTO_ORDER
    assert isinstance(resolve_engine(), EasyOcrEngine)


def test_missing_easyocr_yields_actionable_error(monkeypatch):
    def fake_import_module(name):
        if name == "easyocr":
            raise ImportError("missing")
        return importlib.import_module(name)

    monkeypatch.setattr(ocr_mod.importlib, "import_module", fake_import_module)

    with pytest.raises(MissingDependencyError) as excinfo:
        EasyOcrEngine().recognize(object())

    message = str(excinfo.value).lower()
    assert "easyocr" in message
    assert "openmed[multimodal]" in message


def test_importing_openmed_core_does_not_import_easyocr(monkeypatch):
    monkeypatch.delitem(sys.modules, "easyocr", raising=False)

    import openmed.core  # noqa: F401

    assert "easyocr" not in sys.modules
