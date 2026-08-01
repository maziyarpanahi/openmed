"""Tests for Indic native-digit folding with offset preservation."""

from __future__ import annotations

import pytest

import openmed
from openmed.core.pii_entity_merger import find_semantic_units
from openmed.core.pii_i18n import get_patterns_for_language, validate_aadhaar
from openmed.core.script_detect import normalize_for_pii_detection
from openmed.processing.outputs import PredictionResult
from openmed.processing.text import (
    INDIC_DIGIT_SCRIPTS,
    DigitFolding,
    detect_with_digit_folding,
    fold_indic_digits,
)

# Base code point of the 0..9 decimal block for each script.
_BASES = {
    "devanagari": 0x0966,
    "bengali": 0x09E6,
    "gurmukhi": 0x0A66,
    "gujarati": 0x0AE6,
    "odia": 0x0B66,
    "tamil": 0x0BE6,
    "telugu": 0x0C66,
    "kannada": 0x0CE6,
    "malayalam": 0x0D66,
}


def _native(base: int, digits: str) -> str:
    return "".join(chr(base + int(d)) for d in digits)


# --------------------------------------------------------------------------
# All nine digit sets fold to ASCII
# --------------------------------------------------------------------------


@pytest.mark.parametrize("script", list(_BASES))
def test_all_nine_scripts_fold_to_ascii(script):
    base = _BASES[script]
    native = "".join(chr(base + d) for d in range(10))

    result = fold_indic_digits(native)

    assert isinstance(result, DigitFolding)
    assert result.text == "0123456789"


def test_exported_script_names_cover_all_nine():
    assert set(INDIC_DIGIT_SCRIPTS) == set(_BASES)


# --------------------------------------------------------------------------
# Idempotence / no-op on ASCII / offset preservation
# --------------------------------------------------------------------------


def test_ascii_is_a_noop():
    assert fold_indic_digits("2024-01-15").text == "2024-01-15"


def test_folding_is_idempotent():
    once = fold_indic_digits(_native(0x0966, "12345")).text
    twice = fold_indic_digits(once).text
    assert once == twice == "12345"


def test_offsets_are_length_preserving_and_map_back():
    original = f"जन्म तिथि {_native(0x0966, '15')}-०१-२०२४"
    result = fold_indic_digits(original)

    assert len(result.text) == len(original)
    # Folding is 1:1, so a span on the folded text indexes the original.
    digit_start = result.text.index("15")
    start, end = result.to_original_span(digit_start, digit_start + 2)
    assert original[start:end] == _native(0x0966, "15")


def test_tamil_traditional_signs_are_not_folded():
    # Tamil decimal digits fold, but the traditional number signs (U+0BF0-0BF2)
    # are non-positional and must be left untouched.
    assert fold_indic_digits(_native(0x0BE6, "10")).text == "10"
    assert fold_indic_digits("௰௱௲").text == "௰௱௲"


# --------------------------------------------------------------------------
# Checksum / pattern validation after folding, original surface preserved
# --------------------------------------------------------------------------


def test_devanagari_aadhaar_passes_verhoeff_after_folding():
    native_aadhaar = _native(0x0966, "234123412346")
    assert not validate_aadhaar(native_aadhaar)  # native digits fail as-is
    assert validate_aadhaar(fold_indic_digits(native_aadhaar).text)


def test_mixed_native_ascii_phone_detected_as_single_span():
    original = f"{_native(0x0966, '987')}-654-3210"
    patterns = get_patterns_for_language("hi")

    def matcher(folded: str):
        return [(u[0], u[1], u[2]) for u in find_semantic_units(folded, patterns)]

    results = detect_with_digit_folding(original, matcher)

    phones = [r for r in results if r[2] == "phone_number"]
    assert len(phones) == 1
    start, end, _ = phones[0]
    # The single span covers the whole mixed-digit phone, native digits intact.
    assert original[start:end] == original
    assert "९" in original[start:end]


def test_detect_maps_native_digit_date_back_to_source():
    original = _native(0x0C66, "2024") + "-01-15"  # Telugu year
    patterns = get_patterns_for_language("en")

    def matcher(folded: str):
        return [(u[0], u[1], u[2]) for u in find_semantic_units(folded, patterns)]

    results = detect_with_digit_folding(original, matcher)
    dates = [r for r in results if r[2] == "date"]
    assert dates
    start, end, _ = dates[0]
    assert original[start:end] == original


def test_detection_normalization_folds_digits_and_preserves_source_offsets():
    original = f"आधार {_native(0x0966, '234123412346')}"
    result = normalize_for_pii_detection(original)

    assert result.text == "आधार 234123412346"
    assert result.folded_native_digits == 12
    assert result.to_metadata()["folded_native_digits"] == 12
    start = result.text.index("234123412346")
    assert result.remap_span(start, len(result.text)) == (start, len(original))


def test_extract_pii_detects_native_digit_aadhaar_and_returns_original(monkeypatch):
    native_aadhaar = _native(0x0966, "234123412346")
    original = f"आधार संख्या: {native_aadhaar}"

    def fake_analyze_text(text, **_kwargs):
        return PredictionResult(
            text=text,
            entities=[],
            model_name="fixture-pii-model",
            timestamp="2026-01-01T00:00:00",
        )

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze_text)

    result = openmed.extract_pii(
        original,
        model_name="fixture-pii-model",
        lang="hi",
    )

    national_ids = [
        entity for entity in result.entities if entity.label == "national_id"
    ]
    assert len(national_ids) == 1
    assert national_ids[0].text == native_aadhaar
    assert original[national_ids[0].start : national_ids[0].end] == native_aadhaar
