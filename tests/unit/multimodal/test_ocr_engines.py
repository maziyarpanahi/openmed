"""Tests for OCR engine adapters and engine selection.

These run without Tesseract/PaddleOCR binaries: the deterministic fake engine
verifies the shared OcrResult contract, and the real adapters are checked for
structural conformance and for clear errors when their dependency is absent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


def test_tesseract_adapter_maps_words_to_shared_result(monkeypatch):
    image = object()

    class FakePytesseract:
        Output = SimpleNamespace(DICT="dict")
        pytesseract = SimpleNamespace(TesseractNotFoundError=RuntimeError)

        @staticmethod
        def image_to_data(loaded_image, *, lang, output_type):
            assert loaded_image is image
            assert lang == "eng"  # English default
            assert output_type == "dict"
            return {
                "text": ["", "Patient", "Doe"],
                "left": [0, 10, 80],
                "top": [0, 20, 20],
                "width": [0, 60, 30],
                "height": [0, 15, 15],
                "conf": ["-1", "96.0", "87.5"],
                "page_num": [1, 1, 2],
            }

    monkeypatch.setattr(
        ocr_mod,
        "_import_backend",
        lambda module, instruction: FakePytesseract,
    )

    result = TesseractEngine().recognize(image)

    _assert_ocr_result_shape(result)
    assert [word.text for word in result.words] == ["Patient", "Doe"]
    assert result.words[0].bbox == (10.0, 20.0, 70.0, 35.0)
    assert result.words[0].confidence == 0.96
    assert result.words[0].page == 0
    assert result.words[1].page == 1


def test_tesseract_binary_error_is_actionable(monkeypatch):
    image = object()

    class FakeTesseractNotFoundError(RuntimeError):
        pass

    class FakePytesseract:
        Output = SimpleNamespace(DICT="dict")
        pytesseract = SimpleNamespace(TesseractNotFoundError=FakeTesseractNotFoundError)

        @staticmethod
        def image_to_data(image, *, lang, output_type):
            raise FakeTesseractNotFoundError("missing binary")

    monkeypatch.setattr(
        ocr_mod,
        "_import_backend",
        lambda module, instruction: FakePytesseract,
    )

    with pytest.raises(MissingDependencyError) as excinfo:
        TesseractEngine().recognize(image)

    message = str(excinfo.value).lower()
    assert "tesseract" in message
    assert "system" in message


def test_paddle_adapter_maps_flat_predictions_to_shared_result(monkeypatch):
    seen = {}
    box = [(1, 2), (41, 2), (41, 12), (1, 12)]

    class FakePaddleOCR:
        def __init__(self, *, show_log, lang):
            seen["show_log"] = show_log
            seen["lang"] = lang

        def ocr(self, image):
            seen["image"] = image
            return [[box, ("MRN", 0.88)]]

    fake_module = SimpleNamespace(PaddleOCR=FakePaddleOCR)
    monkeypatch.setattr(
        ocr_mod,
        "_import_backend",
        lambda module, instruction: fake_module,
    )

    result = PaddleOcrEngine().recognize(Path("scan.png"))

    _assert_ocr_result_shape(result)
    assert seen == {"show_log": False, "lang": "en", "image": "scan.png"}
    assert result.words[0].text == "MRN"
    assert result.words[0].bbox == (1, 2, 41, 12)
    assert result.words[0].confidence == 0.88
    assert result.words[0].page == 0


def test_paddle_adapter_maps_paged_predictions_to_shared_result(monkeypatch):
    first_box = [(1, 2), (21, 2), (21, 12), (1, 12)]
    second_box = [(3, 4), (33, 4), (33, 14), (3, 14)]

    class FakePaddleOCR:
        def __init__(self, *, show_log, lang):
            pass

        def ocr(self, image):
            return [
                [[first_box, ("Jane", 0.91)]],
                [[second_box, ("Doe", 0.89)]],
            ]

    fake_module = SimpleNamespace(PaddleOCR=FakePaddleOCR)
    monkeypatch.setattr(
        ocr_mod,
        "_import_backend",
        lambda module, instruction: fake_module,
    )

    result = PaddleOcrEngine().recognize("scan.png")

    _assert_ocr_result_shape(result)
    assert [word.text for word in result.words] == ["Jane", "Doe"]
    assert [word.page for word in result.words] == [0, 1]


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
