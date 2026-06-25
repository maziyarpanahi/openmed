"""Tests for the OCR result contract and the OCR -> document bridge."""

from __future__ import annotations

import pytest

import openmed.multimodal.base as base
import openmed.multimodal.ocr as ocr_mod
from openmed.multimodal import ExtractedDocument, redact_document
from openmed.multimodal.ocr import FakeOcrEngine, OcrResult, OcrWord, ocr

WORDS = [
    OcrWord("Patient", (10.0, 10.0, 80.0, 30.0), 0.99, page=0),
    OcrWord("John", (85.0, 10.0, 130.0, 30.0), 0.98, page=0),
    OcrWord("Doe", (135.0, 10.0, 175.0, 30.0), 0.97, page=0),
]


def test_ocr_returns_result_with_word_bboxes_and_confidences():
    result = ocr("ignored.png", engine=FakeOcrEngine(WORDS))
    assert isinstance(result, OcrResult)
    assert [w.text for w in result.words] == ["Patient", "John", "Doe"]
    assert all(len(w.bbox) == 4 for w in result.words)
    assert all(0.0 <= w.confidence <= 1.0 for w in result.words)


def test_result_text_joins_words():
    result = FakeOcrEngine(WORDS).recognize("x")
    assert result.text == "Patient John Doe"


def test_to_document_projects_word_bboxes():
    doc = FakeOcrEngine(WORDS).recognize("x").to_document()
    assert isinstance(doc, ExtractedDocument)
    assert doc.text == "Patient John Doe"

    span = doc.location_at(doc.text.index("John"))
    assert span is not None
    assert span.bbox == (85.0, 10.0, 130.0, 30.0)
    assert doc.text_for(span) == "John"


def test_to_document_preserves_confidence_metadata():
    doc = FakeOcrEngine(WORDS).recognize("x").to_document()
    span = doc.location_at(doc.text.index("Doe"))
    assert span.metadata.get("confidence") == 0.97


def test_redact_document_bridges_image_through_ocr(monkeypatch):
    # Pretend the multimodal extra is installed and OCR resolves to the fake.
    monkeypatch.setattr(base, "_missing_multimodal_dependencies", lambda: [])
    monkeypatch.setattr(
        ocr_mod, "resolve_engine", lambda engine=None: FakeOcrEngine(WORDS)
    )

    doc = redact_document("scan.png")
    assert isinstance(doc, ExtractedDocument)
    # A detected PHI word projects back to its source pixel bbox.
    span = doc.location_at(doc.text.index("John"))
    assert span.bbox == (85.0, 10.0, 130.0, 30.0)
    assert span.page == 0
