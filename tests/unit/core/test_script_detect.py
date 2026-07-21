from openmed.core.pii_i18n import (
    DEFAULT_PII_MODELS,
    NATIONAL_ID_ONLY_LANGUAGES,
    SUPPORTED_LANGUAGES,
    USER_SUPPLIED_MODEL_LANGUAGES,
)
from openmed.core.script_detect import (
    SCRIPT_LANGUAGE_HINTS,
    SUPPORTED_SCRIPTS,
    UNKNOWN_SCRIPT,
    candidate_languages_for_script,
    detect_script,
    normalize_for_pii_detection,
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
        "রোগী অনিতা": "Bengali",
        "ਮਰੀਜ਼ ਅਨੀਤਾ": "Gurmukhi",
        "દર્દી અનીતા": "Gujarati",
        "ରୋଗୀ ଅନିତା": "Odia",
        "நோயாளி அனிதா": "Tamil",
        "రోగి సీత రెడ్డి": "Telugu",
        "ರೋಗಿ ಅನಿತಾ": "Kannada",
        "രോഗി അനിത": "Malayalam",
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
    routing_languages = (
        SUPPORTED_LANGUAGES | NATIONAL_ID_ONLY_LANGUAGES | USER_SUPPLIED_MODEL_LANGUAGES
    )

    assert expected_scripts <= set(SCRIPT_LANGUAGE_HINTS)
    for script in expected_scripts:
        hints = candidate_languages_for_script(script)
        assert hints
        assert set(hints) <= routing_languages


def test_indic_and_arabic_script_language_hints_are_exact():
    expected_hints = {
        "Devanagari": ("hi", "mr", "ne"),
        "Bengali": ("bn", "as"),
        "Gurmukhi": ("pa",),
        "Gujarati": ("gu",),
        "Odia": ("or",),
        "Tamil": ("ta",),
        "Telugu": ("te",),
        "Kannada": ("kn",),
        "Malayalam": ("ml",),
        "Arabic": ("ar", "ur"),
    }

    for script, languages in expected_hints.items():
        assert candidate_languages_for_script(script) == languages


def test_routing_only_languages_do_not_claim_bundled_models():
    expected_languages = {
        "as",
        "bn",
        "gu",
        "kn",
        "ml",
        "mr",
        "ne",
        "or",
        "pa",
        "ta",
        "ur",
    }

    assert USER_SUPPLIED_MODEL_LANGUAGES == expected_languages
    assert USER_SUPPLIED_MODEL_LANGUAGES.isdisjoint(DEFAULT_PII_MODELS)
    assert candidate_languages_for_script("Latin") == (
        "en",
        "fr",
        "de",
        "it",
        "es",
        "nl",
        "pt",
        "tr",
    )


def test_normalize_for_pii_detection_folds_obfuscation_with_offset_map():
    text = "Patient J\u200bo\u0301hn D\u03bfe"
    normalized = normalize_for_pii_detection(text)

    assert normalized.text == "Patient John Doe"
    assert normalized.changed
    assert normalized.mixed_script
    assert normalized.removed_zero_width == 1
    assert normalized.stripped_combining_marks == 1
    assert normalized.folded_confusables == 1
    assert normalized.remap_span(8, 16) == (8, len(text))
    assert "Patient" not in normalized.to_metadata()


def test_normalize_for_pii_detection_routes_indic_runs_and_preserves_marks():
    text = "Patient न\u093cील ന്\u200d"
    normalized = normalize_for_pii_detection(text)

    assert normalized.text == "Patient ऩील ൻ"
    assert normalized.indic_scripts == ("Devanagari", "Malayalam")
    assert normalized.indic_changes > 0
    assert normalized.removed_zero_width == 1
    assert "ी" in normalized.text
    name_start = normalized.text.index("ऩील")
    assert normalized.remap_span(name_start, name_start + len("ऩील")) == (8, 12)
