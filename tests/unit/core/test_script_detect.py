from openmed.core.pii_i18n import NATIONAL_ID_ONLY_LANGUAGES, SUPPORTED_LANGUAGES
from openmed.core.script_detect import (
    SCRIPT_LANGUAGE_HINTS,
    SUPPORTED_SCRIPTS,
    UNKNOWN_SCRIPT,
    candidate_languages_for_script,
    detect_script,
    segment_by_script,
)


def _assert_offsets_cover_text(
    segments: list[tuple[int, int, str]],
    text: str,
) -> None:
    cursor = 0
    for start, end, script in segments:
        assert script
        assert start == cursor
        assert start < end
        assert text[start:end]
        cursor = end
    assert cursor == len(text)


def test_detect_script_classifies_single_script_samples():
    samples = {
        "Patient John Smith": "Latin",
        "المريض أحمد علي": "Arabic",
        "患者 佐藤花子": "Han",
        "かな カタカナ": "Hiragana/Katakana",
        "환자 김민수": "Hangul",
        "Пациент Иван": "Cyrillic",
        "मरीज़ अनिता शर्मा": "Devanagari",
        "రోగి సీత రెడ్డి": "Telugu",
        "Ασθενής Νίκος": "Greek",
        "מטופל דוד כהן": "Hebrew",
        "ผู้ป่วย สมชาย": "Thai",
    }

    for text, script in samples.items():
        assert detect_script(text) == script


def test_detect_script_ignores_neutral_characters():
    assert detect_script("  MRN-12345  ") == "Latin"
    assert detect_script("12345 / --") == UNKNOWN_SCRIPT


def test_segment_by_script_mixed_latin_arabic_offsets_cover_text():
    text = "Patient Ahmad راجع العيادة 5mg"
    segments = list(segment_by_script(text))

    _assert_offsets_cover_text(segments, text)
    assert [script for _, _, script in segments] == ["Latin", "Arabic", "Latin"]
    assert "".join(text[start:end] for start, end, _ in segments) == text


def test_segment_by_script_mixed_latin_han_offsets_cover_text():
    text = "MRN 42 患者 佐藤 visited"
    segments = list(segment_by_script(text))

    _assert_offsets_cover_text(segments, text)
    assert [script for _, _, script in segments] == ["Latin", "Han", "Latin"]
    assert "".join(text[start:end] for start, end, _ in segments) == text


def test_script_language_hints_cover_detectable_scripts():
    expected_scripts = set(SUPPORTED_SCRIPTS) | {UNKNOWN_SCRIPT}

    assert expected_scripts <= set(SCRIPT_LANGUAGE_HINTS)
    for script in expected_scripts:
        hints = candidate_languages_for_script(script)
        assert hints
        assert set(hints) <= SUPPORTED_LANGUAGES | NATIONAL_ID_ONLY_LANGUAGES
