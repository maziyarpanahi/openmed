"""Tests for the opt-in Unicode and native-digit detection normalizer."""

from __future__ import annotations

import pytest

from openmed.core.pii_i18n import validate_egyptian_national_id
from openmed.core.text_normalize import (
    normalize_digits,
    normalize_for_detection,
    normalize_unicode,
    normalize_unicode_with_offsets,
)


@pytest.mark.parametrize(
    ("native", "ascii"),
    [
        ("٠١٢٣٤٥٦٧٨٩", "0123456789"),  # Arabic-Indic
        ("۰۱۲۳۴۵۶۷۸۹", "0123456789"),  # Persian/Eastern Arabic-Indic
        ("०१२३४५६७८९", "0123456789"),  # Devanagari
        ("౦౧౨౩౪౫౬౭౮౯", "0123456789"),  # Telugu
        ("০১২৩৪৫৬৭৮৯", "0123456789"),  # Bengali
        ("๐๑๒๓๔๕๖๗๘๙", "0123456789"),  # Thai
        ("０１２３４５６７８９", "0123456789"),  # Fullwidth
    ],
)
def test_normalize_digits_folds_requested_native_digit_sets(native, ascii):
    assert normalize_digits(native) == ascii


def test_normalize_digits_leaves_letters_and_other_text_untouched():
    text = "Patient أحمد Telugu తెలుగు ABC xyz"

    assert normalize_digits(text) == text


def test_normalize_unicode_folds_fullwidth_latin_and_ligatures():
    assert normalize_unicode("ＯｐｅｎＭｅｄ ﬁ") == "OpenMed fi"


def test_unicode_offset_map_expansion_round_trips_exact_ligature_span():
    original = "prefix ﬁle suffix"

    normalized, offset_map = normalize_for_detection(original)

    assert normalized == "prefix file suffix"
    normalized_start = normalized.index("file")
    normalized_end = normalized_start + len("file")
    original_start, original_end = offset_map.to_original_span(
        normalized_start,
        normalized_end,
    )

    assert original[original_start:original_end] == "ﬁle"
    ligature_start = original.index("ﬁ")
    assert offset_map[normalized_start : normalized_start + 2] == (
        ligature_start,
        ligature_start,
    )


def test_unicode_offset_map_handles_composition_without_offset_drift():
    original = "before e\u0301 after"

    normalized, offset_map = normalize_unicode_with_offsets(original)

    assert normalized == "before é after"
    normalized_start = normalized.index("é")
    source_start = original.index("e\u0301")
    assert offset_map.to_original_span(normalized_start, normalized_start + 1) == (
        source_start,
        source_start + 2,
    )


def test_detection_normalization_folds_digits_and_preserves_their_offsets():
    original = "رقم: ۱۲۳۴۵۶"

    normalized, offset_map = normalize_for_detection(original)

    assert normalized == "رقم: 123456"
    start = normalized.index("123456")
    end = start + 6
    assert offset_map.to_original_span(start, end) == (
        original.index("۱"),
        len(original),
    )


def test_normalized_arabic_national_id_passes_existing_validator():
    ascii_id = "29801011234567"
    native_id = "٢٩٨٠١٠١١٢٣٤٥٦٧"

    assert validate_egyptian_national_id(ascii_id)
    assert validate_egyptian_national_id(normalize_digits(native_id))


def test_invalid_text_types_fail_clearly():
    with pytest.raises(TypeError, match="text must be a string"):
        normalize_digits(None)
    with pytest.raises(TypeError, match="text must be a string"):
        normalize_unicode_with_offsets(None)
