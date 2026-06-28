"""Tests for the docTR OCR adapter."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import openmed.multimodal.ocr as ocr_module
from openmed.multimodal.exceptions import MissingDependencyError
from openmed.multimodal.ocr import (
    DocTrEngine,
    OcrEngine,
    OcrResult,
    OcrWord,
    available_ocr_engines,
    ocr,
    run_doctr_ocr,
)


class MockWord:
    def __init__(self, value, confidence, xmin, ymin, xmax, ymax):
        self.value = value
        self.confidence = confidence
        self.geometry = ((xmin, ymin), (xmax, ymax))


class MockLine:
    def __init__(self, words):
        self.words = words


class MockBlock:
    def __init__(self, lines):
        self.lines = lines


class MockPage:
    def __init__(self, dimensions, blocks):
        self.dimensions = dimensions
        self.blocks = blocks


class MockDocument:
    def __init__(self, pages):
        self.pages = pages


def _mock_document():
    return MockDocument(
        [
            MockPage(
                (1000, 500),
                [
                    MockBlock(
                        [MockLine([MockWord("Clinical", 0.991234, 0.1, 0.2, 0.3, 0.4)])]
                    )
                ],
            )
        ]
    )


def _predictor_factory(calls):
    def factory(*, pretrained):
        calls["pretrained"] = pretrained

        def predictor(document):
            calls["document"] = document
            return _mock_document()

        return predictor

    return factory


def _document_loader(calls):
    def loader(image):
        calls["image"] = image
        return "loaded-document"

    return loader


def test_run_doctr_ocr_maps_relative_boxes_to_absolute_pixels():
    calls = {}

    result = run_doctr_ocr(
        Path("sample_invoice.jpg"),
        predictor_factory=_predictor_factory(calls),
        document_loader=_document_loader(calls),
    )

    assert calls == {
        "pretrained": True,
        "image": Path("sample_invoice.jpg"),
        "document": "loaded-document",
    }
    assert result == OcrResult(
        words=(
            OcrWord(
                text="Clinical",
                bbox=(50.0, 200.0, 150.0, 400.0),
                confidence=0.991234,
                page=0,
            ),
        ),
        metadata={"engine": "doctr"},
    )


def test_doctr_result_projects_absolute_boxes_to_document_spans():
    result = run_doctr_ocr(
        Path("sample_invoice.jpg"),
        predictor_factory=_predictor_factory({}),
        document_loader=_document_loader({}),
    )

    doc = result.to_document()

    span = doc.location_at(doc.text.index("Clinical"))
    assert span is not None
    assert span.bbox == (50.0, 200.0, 150.0, 400.0)
    assert span.page == 0
    assert span.metadata["confidence"] == 0.991234


def test_doctr_engine_satisfies_ocr_contract():
    engine = DocTrEngine(
        predictor_factory=_predictor_factory({}),
        document_loader=lambda image: image,
    )

    result = engine.recognize("scan-1.png")

    assert isinstance(engine, OcrEngine)
    assert isinstance(result, OcrResult)
    assert result.words == (
        OcrWord(
            text="Clinical",
            bbox=(50.0, 200.0, 150.0, 400.0),
            confidence=0.991234,
            page=0,
        ),
    )


def test_ocr_doctr_uses_doctr_engine(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        ocr_module, "_load_doctr_predictor", lambda: _predictor_factory(calls)
    )
    monkeypatch.setattr(ocr_module, "_load_doctr_document", _document_loader(calls))

    result = ocr(["scan-1.png"], engine="doctr")

    assert calls["image"] == ["scan-1.png"]
    assert result.words[0].text == "Clinical"


def test_ocr_auto_selects_installed_doctr(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        ocr_module, "_engine_available", lambda engine: engine == "doctr"
    )
    monkeypatch.setattr(
        ocr_module, "_load_doctr_predictor", lambda: _predictor_factory(calls)
    )
    monkeypatch.setattr(ocr_module, "_load_doctr_document", _document_loader(calls))

    ocr("scan-1.png")

    assert available_ocr_engines() == ("doctr",)
    assert calls["image"] == "scan-1.png"


def test_missing_doctr_raises_actionable_import_error(monkeypatch):
    real_import_module = importlib.import_module

    def missing_doctr(module):
        if module == "doctr.models":
            raise ImportError("missing doctr")
        return real_import_module(module)

    monkeypatch.setattr(ocr_module.importlib, "import_module", missing_doctr)

    with pytest.raises(MissingDependencyError, match="python-doctr") as excinfo:
        ocr("scan-1.png", engine="doctr")
    assert "openmed[multimodal]" in str(excinfo.value)


def test_importing_ocr_module_does_not_import_doctr():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import openmed.multimodal.ocr; print('doctr' in sys.modules)",
        ],
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_unknown_ocr_engine_raises_value_error():
    with pytest.raises(ValueError, match="Unknown OCR engine"):
        ocr("scan-1.png", engine="made-up")
