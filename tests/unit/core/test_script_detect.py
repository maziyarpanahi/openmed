from hypothesis import given
from hypothesis import strategies as st

from openmed.core.language_pack_catalog import (
    DEFAULT_PII_MODELS as BUILTIN_DEFAULT_PII_MODELS,
)
from openmed.core.pii_i18n import (
    DEFAULT_PII_MODELS,
    NATIONAL_ID_ONLY_LANGUAGES,
    OPTIONAL_PII_MODEL,
    SUPPORTED_LANGUAGES,
    USER_SUPPLIED_MODEL_LANGUAGES,
)
from openmed.core.script_detect import (
    CONFUSABLE_DATA_LICENSE,
    CONFUSABLE_DATA_URL,
    CONFUSABLE_DATA_VERSION,
    SCRIPT_LANGUAGE_HINTS,
    SUPPORTED_SCRIPTS,
    UNKNOWN_SCRIPT,
    candidate_languages_for_script,
    confusable_skeleton,
    detect_mixed_script,
    detect_script,
    is_han_dominant,
    mixed_script_spans,
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
        "ታካሚ ሰላም ተስፋዬ": "Ethiopic",
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


def test_detect_script_covers_all_ethiopic_unicode_blocks():
    samples = ("ሀ", "ᎀ", "ⶀ", "ꬁ", "𞟠")

    assert all(detect_script(char) == "Ethiopic" for char in samples)


def test_segment_by_script_mixed_amharic_latin_has_exact_offsets():
    text = "ታካሚ Selam፡ ቀጠሮ"

    assert list(segment_by_script(text)) == [
        (0, 4, "Ethiopic"),
        (4, 11, "Latin"),
        (11, 14, "Ethiopic"),
    ]


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
    assert USER_SUPPLIED_MODEL_LANGUAGES.isdisjoint(BUILTIN_DEFAULT_PII_MODELS)
    for language in expected_languages - {"ne", "ur"}:
        assert DEFAULT_PII_MODELS[language] == OPTIONAL_PII_MODEL
    assert candidate_languages_for_script("Latin") == (
        "en",
        "fr",
        "de",
        "it",
        "es",
        "nl",
        "pt",
        "tr",
        "sw",
        "zu",
        "xh",
    )


def test_han_script_routes_to_chinese_candidate_language():
    text = "患者王芳因心房颤动入院"

    script = detect_script(text)

    assert script == "Han"
    assert candidate_languages_for_script(script)[0] == "zh"


def test_han_dominance_detection_supports_language_routing():
    assert is_han_dominant("患者王芳因心房颤动入院")
    assert is_han_dominant("患者A")
    assert not is_han_dominant("患者AB")
    assert not is_han_dominant("Patient John Smith")


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


def test_normalize_for_pii_detection_strips_standalone_ethiopic_mark():
    normalized = normalize_for_pii_detection("\u135f")

    assert normalized.text == ""
    assert normalized.stripped_combining_marks == 1


@given(
    before=st.lists(st.sampled_from(tuple("ሀለሐመሠረሰቀበተነአከወዘየደገጠጸፈፐ")), min_size=1),
    after=st.lists(st.sampled_from(tuple("ሀለሐመሠረሰቀበተነአከወዘየደገጠጸፈፐ")), max_size=8),
    prefix=st.sampled_from(("", "ስም፡ ", "Patient ")),
    suffix=st.sampled_from(("", "።", " visited")),
)
def test_ethiopic_combining_mark_remaps_without_offset_drift(
    before: list[str],
    after: list[str],
    prefix: str,
    suffix: str,
):
    marked_value = f"{''.join(before)}\u135f{''.join(after)}"
    text = f"{prefix}{marked_value}{suffix}"
    normalized = normalize_for_pii_detection(text)
    value_start = len(prefix)
    value_end = value_start + len(marked_value)
    grapheme_start = value_start + len(before) - 1
    grapheme_end = grapheme_start + 2

    assert normalized.text == text
    assert normalized.stripped_combining_marks == 0
    assert normalized.remap_span(value_start, value_end) == (value_start, value_end)
    assert normalized.remap_span(grapheme_start, grapheme_end) == (
        grapheme_start,
        grapheme_end,
    )
    assert text[grapheme_start:grapheme_end].endswith("\u135f")


def test_confusable_skeleton_covers_cross_script_width_and_invisible_attacks():
    attacked = "J\u043ehn D\u03bfe D\u3007E \uff2d\uff32\uff2e A\u200b1001"

    assert confusable_skeleton(attacked) == "John Doe DOE MRN A1001"
    assert CONFUSABLE_DATA_VERSION == "17.0.0"
    assert CONFUSABLE_DATA_LICENSE == "Unicode-3.0"
    assert CONFUSABLE_DATA_URL.endswith("/17.0.0/security/confusables.txt")


def test_mixed_script_detector_flags_only_identifier_local_script_mixing():
    text = "Patient J\u043ehn met \u4f50\u85e4 after discharge"

    findings = mixed_script_spans(text)

    assert detect_mixed_script(text)
    assert len(findings) == 1
    assert findings[0].scripts == ("Cyrillic", "Latin")
    assert text[findings[0].start : findings[0].end] == "J\u043ehn"
    assert findings[0].confusable_count == 1
    assert not detect_mixed_script("Patient John met \u4f50\u85e4 after discharge")


def test_han_confusable_normalization_preserves_original_offsets():
    text = "Patient D\u3007E arrived"
    normalized = normalize_for_pii_detection(text)

    assert normalized.text == "Patient DOE arrived"
    assert normalized.mixed_script
    assert normalized.remap_span(8, 11) == (8, 11)
    assert text[slice(*normalized.remap_span(8, 11))] == "D\u3007E"
