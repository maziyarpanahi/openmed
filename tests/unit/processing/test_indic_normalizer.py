"""Acceptance tests for the offset-safe Indic Unicode normalizer."""

from __future__ import annotations

import unicodedata

import pytest

from openmed.processing.text import INDIC_SCRIPTS, IndicNormalizer

SCRIPT_BLOCKS = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Odia": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
}


def test_multiscript_normalization_is_idempotent():
    normalizer = IndicNormalizer()
    text = (
        "Latin ऩील अङ्क र्\u200d बंगলা ਡਾਕਟਰઃ ગુજરાતી | ଵୈଦ୍ୟ: தமிழ் తెలుగు ಕನ್ನಡ ന്\u200d രോഗി"
    )

    once = normalizer.normalize(text)

    assert normalizer.normalize(once) == once
    assert set(normalizer.normalize_with_offsets(text).scripts) == set(INDIC_SCRIPTS)


def test_encoding_variants_collapse_to_byte_identical_text():
    normalizer = IndicNormalizer()

    assert normalizer.normalize("ऩ") == normalizer.normalize("न\u093c")
    assert normalizer.normalize("अंक") == normalizer.normalize("अङ्क")
    assert normalizer.normalize("क्\u200dष") == normalizer.normalize("क्ष")
    assert normalizer.normalize("क्\u200cष") == normalizer.normalize("क्ष")


def test_normalization_offset_map_round_trips_multiscript_phi_spans():
    normalizer = IndicNormalizer()
    raw_tokens = [
        "न\u093cील",
        "রোগী",
        "ੱਕੰ",
        "ડૉક્ટર",
        "ଵୈଦ୍ୟ",
        "நோயாளி",
        "రోగి",
        "ರೋಗಿ",
        "ന്\u200d",
    ]
    raw = "Latin " + " ".join(raw_tokens) + " end"
    normalized = normalizer.normalize_with_offsets(raw)
    normalized_cursor = 0
    raw_cursor = len("Latin ")

    for token in raw_tokens:
        canonical = normalizer.normalize(token)
        normalized_start = normalized.text.index(canonical, normalized_cursor)
        normalized_end = normalized_start + len(canonical)
        assert normalized.remap_span(normalized_start, normalized_end) == (
            raw_cursor,
            raw_cursor + len(token),
        )
        normalized_cursor = normalized_end
        raw_cursor += len(token) + 1


@pytest.mark.parametrize(
    ("raw", "expected", "script"),
    [
        ("ന്\u200d", "ൻ", "Malayalam"),
        ("र्\u200d", "र्", "Devanagari"),
        ("ੱਕੰ", "ਕ੍ਕਂ", "Gurmukhi"),
        ("ଵ", "ବ", "Odia"),
    ],
)
def test_required_script_specific_cases(raw: str, expected: str, script: str):
    assert IndicNormalizer().normalize(raw, script=script) == expected


@pytest.mark.parametrize(
    ("raw", "expected", "script"),
    [
        ("কো", "কো", "Bengali"),
        ("கொ", "கொ", "Tamil"),
        ("காெ", "கொ", "Tamil"),
        ("కై", "కై", "Telugu"),
        ("ಕೇ", "ಕೇ", "Kannada"),
        ("കൊ", "കൊ", "Malayalam"),
        ("अग्नि|", "अग्नि।", "Devanagari"),
        ("নাম:", "নামঃ", "Bengali"),
        ("र्\u200d:", "र्ः", "Devanagari"),
    ],
)
def test_vowels_and_punctuation_use_script_specific_canonical_forms(
    raw: str,
    expected: str,
    script: str,
):
    assert IndicNormalizer().normalize(raw, script=script) == expected


def test_tunable_flags_are_explicit_and_safe_by_default():
    default = IndicNormalizer()

    assert "़" in default.normalize("क़")
    assert IndicNormalizer(remove_nuktas=True).normalize("क़") == "क"
    assert IndicNormalizer(joiner_policy="preserve").normalize("क्\u200dष") == (
        "क्\u200dष"
    )
    assert IndicNormalizer(normalize_chandra=False).normalize("अँ") == "अँ"
    assert IndicNormalizer(nasals_mode="preserve").normalize("अङ्क") == "अङ्क"
    assert IndicNormalizer(normalize_vowel_ending=True).normalize("नाम") == "नाम्"


def test_default_never_drops_script_bearing_codepoints_across_all_blocks():
    normalizer = IndicNormalizer()

    for script, (start, end) in SCRIPT_BLOCKS.items():
        for codepoint in range(start, end + 1):
            char = chr(codepoint)
            if unicodedata.category(char)[0] not in {"L", "M"}:
                continue
            output = normalizer.normalize(char, script=script)
            assert output, f"U+{codepoint:04X} was dropped for {script}"
            assert any(
                block_start <= ord(output_char) <= block_end
                for output_char in output
                for block_start, block_end in SCRIPT_BLOCKS.values()
            ), f"U+{codepoint:04X} lost its script-bearing representation"


def test_invalid_policy_values_fail_closed():
    with pytest.raises(ValueError, match="nasals_mode"):
        IndicNormalizer(nasals_mode="guess")
    with pytest.raises(ValueError, match="joiner_policy"):
        IndicNormalizer(joiner_policy="guess")
