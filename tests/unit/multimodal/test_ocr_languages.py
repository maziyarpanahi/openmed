"""Tests for OCR language-pack configuration (issue #315).

Covers the OpenMed PII language -> OCR engine identifier mapping for all 12
wired languages, threading a ``languages`` selection through ``ocr()`` to the
adapter, and the English default.
"""

from __future__ import annotations

import pytest

from openmed.multimodal.ocr import (
    SUPPORTED_OCR_LANGUAGES,
    FakeOcrEngine,
    OcrWord,
    ocr,
    paddle_language,
    tesseract_language,
)

# The 12 wired OpenMed PII languages.
PII_LANGUAGES = ("en", "fr", "de", "it", "es", "nl", "hi", "te", "pt", "ar", "ja", "tr")

WORDS = [OcrWord("x", (0.0, 0.0, 1.0, 1.0), 0.9, 0)]


def test_ocr_forwards_languages_to_adapter():
    engine = FakeOcrEngine(WORDS)
    ocr("scan.png", engine=engine, languages=["fr"])
    assert engine.last_languages == ["fr"]


def test_ocr_accepts_a_single_language_string():
    engine = FakeOcrEngine(WORDS)
    ocr("scan.png", engine=engine, languages="de")
    assert engine.last_languages == ["de"]


def test_default_language_is_english_when_unspecified():
    engine = FakeOcrEngine(WORDS)
    ocr("scan.png", engine=engine)
    assert engine.last_languages == ["en"]
    assert tesseract_language() == "eng"
    assert paddle_language() == "en"


def test_language_map_covers_all_twelve_pii_languages():
    assert set(PII_LANGUAGES) <= set(SUPPORTED_OCR_LANGUAGES)
    assert len(SUPPORTED_OCR_LANGUAGES) == 12
    for lang in PII_LANGUAGES:
        assert tesseract_language([lang])  # non-empty identifier
        assert paddle_language([lang])


def test_tesseract_identifier_mapping():
    assert tesseract_language(["fr"]) == "fra"
    assert tesseract_language(["de"]) == "deu"
    assert tesseract_language(["ja"]) == "jpn"
    # Multiple languages join with '+' per Tesseract's convention.
    assert tesseract_language(["en", "fr"]) == "eng+fra"


def test_paddle_identifier_mapping():
    assert paddle_language(["en"]) == "en"
    assert paddle_language(["fr"]) == "fr"
    assert paddle_language(["de"]) == "german"
    assert paddle_language(["ja"]) == "japan"


def test_unsupported_language_raises_value_error():
    with pytest.raises(ValueError):
        tesseract_language(["zz"])
    with pytest.raises(ValueError):
        paddle_language(["zz"])
