"""Tests for deterministic Indic transliteration."""

from __future__ import annotations

import unicodedata

import pytest

from openmed.processing.transliteration import (
    INDIC_SCRIPTS,
    LOSSY_CASES,
    from_latin,
    romanized_to_iso15919,
    to_latin,
    transliterate,
    transliteration_key,
)

_RAMA_BY_SCRIPT = {
    "Devanagari": "राम",
    "Bengali": "রাম",
    "Gurmukhi": "ਰਾਮ",
    "Gujarati": "રામ",
    "Odia": "ରାମ",
    "Tamil": "ராம",
    "Telugu": "రామ",
    "Kannada": "ರಾಮ",
    "Malayalam": "രാമ",
}


@pytest.mark.parametrize(
    "name",
    ("राम", "अनिता", "किरण", "लक्ष्मी", "अर्जुन", "सीता", "अक्का"),
)
def test_devanagari_lossless_subset_round_trips_through_iso15919(name):
    pivot = to_latin(name, "Devanagari").text

    assert from_latin(pivot, "Devanagari") == unicodedata.normalize("NFC", name)


def test_all_nine_scripts_share_the_iso15919_pivot():
    assert tuple(_RAMA_BY_SCRIPT) == INDIC_SCRIPTS

    for script, source in _RAMA_BY_SCRIPT.items():
        assert to_latin(source, script).text == "rāma"
        assert from_latin("rāma", script) == source
        assert transliterate("राम", script, source_script="Devanagari") == source


def test_cross_script_and_romanized_names_share_one_key():
    expected = transliteration_key("राम", "Devanagari")

    assert transliteration_key("ராம", "Tamil") == expected
    assert transliteration_key("RĀMA", "ISO 15919") == expected
    assert transliteration_key("rAma", "ITRANS") == expected
    assert transliteration_key("rAma", "Harvard-Kyoto") == expected


@pytest.mark.parametrize(
    ("itrans_source", "harvard_kyoto_source", "expected"),
    (
        ("rAma", "rAma", "rāma"),
        ("lakShmI", "lakSmI", "lakṣmī"),
        ("kR^iShNa", "kRSNa", "kr̥ṣṇa"),
        ("chandra", "candra", "candra"),
    ),
)
def test_itrans_and_harvard_kyoto_normalize_to_the_same_pivot(
    itrans_source,
    harvard_kyoto_source,
    expected,
):
    itrans = romanized_to_iso15919(itrans_source, scheme="itrans")
    harvard_kyoto = romanized_to_iso15919(harvard_kyoto_source, scheme="harvard-kyoto")

    assert itrans == harvard_kyoto == expected


def test_fixed_romanization_inputs_decode_to_the_same_devanagari_text():
    assert from_latin("lakShmI", "Devanagari", scheme="itrans") == "लक्ष्मी"
    assert from_latin("lakSmI", "Devanagari", scheme="harvard-kyoto") == "लक्ष्मी"


def test_offset_map_and_iso_output_are_idempotent_for_mixed_script_text():
    source = "राम ராம rāma"
    first = to_latin(source)
    second = to_latin(first.text)

    assert first.text == second.text == "rāma rāma rāma"
    assert first.remap_span(0, 4) == (0, 3)
    assert first.remap_span(5, 9) == (4, 7)
    assert first.remap_span(10, 14) == (8, 12)
    assert len(first.offset_starts) == len(first.text)
    assert len(first.offset_ends) == len(first.text)


def test_decomposed_iso_input_is_nfc_with_source_offsets_preserved():
    source = "ra\u0304ma"

    result = to_latin(source)

    assert result.text == "rāma"
    assert unicodedata.is_normalized("NFC", result.text)
    assert result.remap_span(1, 2) == (1, 3)


def test_explicit_schwa_and_anusvara_policies_document_lossiness():
    assert to_latin("राम", schwa_policy="preserve").text == "rāma"
    assert to_latin("राम", schwa_policy="source").text == "rām"
    assert to_latin("राम।", schwa_policy="source").text == "rām।"
    assert to_latin("அருண", schwa_policy="source").text == "aruṇa"
    assert to_latin("अंक", anusvara_policy="marker").text == "aṁka"
    assert to_latin("अंक", anusvara_policy="homorganic").text == "aṅka"
    assert LOSSY_CASES
    assert any("schwa" in case for case in LOSSY_CASES)
    assert any("anusvara" in case for case in LOSSY_CASES)


def test_invalid_scheme_and_script_fail_closed():
    with pytest.raises(ValueError, match="scheme"):
        romanized_to_iso15919("rAma", scheme="unknown")
    with pytest.raises(ValueError, match="target script"):
        from_latin("rāma", "Sinhala")
    with pytest.raises(ValueError, match="target script"):
        from_latin("rāma", "Urdu")
