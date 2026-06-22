"""Tests for the docTR OCR adapter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import openmed.multimodal.ocr as ocr_module
from openmed.multimodal.ocr import OcrResult, available_ocr_engines, ocr, run_doctr_ocr


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

        def predictor(images):
            calls["images"] = images
            return _mock_document()

        return predictor

    return factory


def test_run_doctr_ocr_maps_relative_boxes_to_absolute_pixels():
    calls = {}

    results = run_doctr_ocr(
        Path("sample_invoice.jpg"),
        predictor_factory=_predictor_factory(calls),
    )

    assert calls == {"pretrained": True, "images": ["sample_invoice.jpg"]}
    assert results == [
        OcrResult(
            text="Clinical",
            bbox=(50, 200, 150, 400),
            confidence=0.991234,
            page=0,
        )
    ]


def test_ocr_doctr_uses_doctr_engine(monkeypatch):
    calls = {}
    monkeypatch.setattr(ocr_module, "_engine_available", lambda engine: True)
    monkeypatch.setattr(
        ocr_module, "_load_doctr_predictor", lambda: _predictor_factory(calls)
    )

    results = ocr(["scan-1.png"], engine="doctr")

    assert calls["images"] == ["scan-1.png"]
    assert results[0].text == "Clinical"


def test_ocr_auto_selects_installed_doctr(monkeypatch):
    calls = {}
    monkeypatch.setattr(ocr_module, "_engine_available", lambda engine: True)
    monkeypatch.setattr(
        ocr_module, "_load_doctr_predictor", lambda: _predictor_factory(calls)
    )

    ocr("scan-1.png")

    assert available_ocr_engines() == ("doctr",)
    assert calls["images"] == ["scan-1.png"]


def test_missing_doctr_raises_actionable_import_error(monkeypatch):
    monkeypatch.setattr(ocr_module, "_engine_available", lambda engine: False)

    with pytest.raises(ImportError, match="openmed\\[multimodal\\]"):
        ocr("scan-1.png", engine="doctr")


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
    with pytest.raises(ValueError, match="Unsupported OCR engine"):
        ocr("scan-1.png", engine="made-up")
