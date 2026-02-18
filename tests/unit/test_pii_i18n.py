"""Tests for multilingual PII detection support (pii_i18n module)."""

import pytest
import re

from openmed.core.pii_i18n import (
    SUPPORTED_LANGUAGES,
    LANGUAGE_NAMES,
    LANGUAGE_MODEL_PREFIX,
    DEFAULT_PII_MODELS,
    LANGUAGE_MONTH_NAMES,
    LANGUAGE_PII_PATTERNS,
    LANGUAGE_FAKE_DATA,
    validate_french_nir,
    validate_german_steuer_id,
    validate_italian_codice_fiscale,
    validate_spanish_dni,
    validate_spanish_nie,
    get_patterns_for_language,
)
from openmed.core.pii_entity_merger import PIIPattern, PII_PATTERNS


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Test module-level constants."""

    def test_supported_languages(self):
        assert SUPPORTED_LANGUAGES == {"en", "fr", "de", "it", "es"}

    def test_language_names_keys(self):
        assert set(LANGUAGE_NAMES.keys()) == SUPPORTED_LANGUAGES

    def test_language_model_prefix(self):
        assert LANGUAGE_MODEL_PREFIX["en"] == ""
        assert LANGUAGE_MODEL_PREFIX["fr"] == "French-"
        assert LANGUAGE_MODEL_PREFIX["de"] == "German-"
        assert LANGUAGE_MODEL_PREFIX["it"] == "Italian-"
        assert LANGUAGE_MODEL_PREFIX["es"] == "Spanish-"

    def test_default_pii_models_all_languages(self):
        assert set(DEFAULT_PII_MODELS.keys()) == SUPPORTED_LANGUAGES

    def test_default_pii_models_naming(self):
        assert "French" in DEFAULT_PII_MODELS["fr"]
        assert "German" in DEFAULT_PII_MODELS["de"]
        assert "Italian" in DEFAULT_PII_MODELS["it"]
        assert "Spanish" in DEFAULT_PII_MODELS["es"]
        # English has no language prefix
        assert "French" not in DEFAULT_PII_MODELS["en"]
        assert "German" not in DEFAULT_PII_MODELS["en"]

    def test_month_names_all_languages(self):
        for lang in SUPPORTED_LANGUAGES:
            assert lang in LANGUAGE_MONTH_NAMES
            assert len(LANGUAGE_MONTH_NAMES[lang]) == 12


# ---------------------------------------------------------------------------
# French NIR Validator Tests
# ---------------------------------------------------------------------------


class TestValidateFrenchNIR:
    """Tests for validate_french_nir()."""

    def test_valid_nir(self):
        # number = 1000000000000, key = 97 - (1000000000000 % 97) = 47
        valid_nir = "1000000000000" + "47"
        assert validate_french_nir(valid_nir) is True

    def test_valid_nir_with_spaces(self):
        assert validate_french_nir("1 00 00 00 000 000 47") is True

    def test_invalid_nir_wrong_length(self):
        assert validate_french_nir("12345") is False

    def test_invalid_nir_bad_first_digit(self):
        assert validate_french_nir("300000000000047") is False

    def test_invalid_nir_wrong_checksum(self):
        assert validate_french_nir("100000000000048") is False

    def test_valid_nir_female(self):
        # number = 2000000000000, key = 97 - (2000000000000 % 97) = 94
        assert validate_french_nir("200000000000094") is True


# ---------------------------------------------------------------------------
# German Steuer-ID Validator Tests
# ---------------------------------------------------------------------------


class TestValidateGermanSteuerId:
    """Tests for validate_german_steuer_id()."""

    def test_valid_steuer_id(self):
        assert validate_german_steuer_id("12345678912") is True

    def test_valid_steuer_id_with_spaces(self):
        assert validate_german_steuer_id("1234 5678 912") is True

    def test_invalid_steuer_id_first_digit_zero(self):
        assert validate_german_steuer_id("01234567891") is False

    def test_invalid_steuer_id_wrong_length(self):
        assert validate_german_steuer_id("123456789") is False

    def test_invalid_steuer_id_too_many_repeats(self):
        assert validate_german_steuer_id("11223344556") is False

    def test_invalid_steuer_id_no_repeats(self):
        assert validate_german_steuer_id("12345678900") is False


# ---------------------------------------------------------------------------
# Italian Codice Fiscale Validator Tests
# ---------------------------------------------------------------------------


class TestValidateItalianCodiceFiscale:
    """Tests for validate_italian_codice_fiscale()."""

    def test_valid_codice_fiscale(self):
        assert validate_italian_codice_fiscale("RSSMRA85M01H501Z") is True

    def test_valid_codice_fiscale_lowercase(self):
        assert validate_italian_codice_fiscale("rssmra85m01h501z") is True

    def test_valid_codice_fiscale_with_spaces(self):
        assert validate_italian_codice_fiscale("RSS MRA 85M01 H501Z") is True

    def test_invalid_codice_fiscale_wrong_length(self):
        assert validate_italian_codice_fiscale("RSSMRA85M01H50") is False

    def test_invalid_codice_fiscale_wrong_format(self):
        assert validate_italian_codice_fiscale("1234567890123456") is False

    def test_invalid_codice_fiscale_wrong_pattern(self):
        assert validate_italian_codice_fiscale("12SMRA85M01H501Z") is False


# ---------------------------------------------------------------------------
# Spanish DNI Validator Tests
# ---------------------------------------------------------------------------


class TestValidateSpanishDNI:
    """Tests for validate_spanish_dni()."""

    def test_valid_dni(self):
        # 12345678 % 23 = 14 -> letter 'Z'
        assert validate_spanish_dni("12345678Z") is True

    def test_valid_dni_with_spaces(self):
        assert validate_spanish_dni("1234 5678 Z") is True

    def test_invalid_dni_wrong_length(self):
        assert validate_spanish_dni("1234567Z") is False

    def test_invalid_dni_wrong_letter(self):
        assert validate_spanish_dni("12345678A") is False

    def test_invalid_dni_no_letter(self):
        assert validate_spanish_dni("123456789") is False

    def test_valid_dni_another(self):
        # 00000000 % 23 = 0 -> letter 'T'
        assert validate_spanish_dni("00000000T") is True


# ---------------------------------------------------------------------------
# Spanish NIE Validator Tests
# ---------------------------------------------------------------------------


class TestValidateSpanishNIE:
    """Tests for validate_spanish_nie()."""

    def test_valid_nie_x(self):
        # X prefix -> 0, number = 01234567, 1234567 % 23 = 1234567 mod 23
        # 1234567 / 23 = 53676.8..., 53676 * 23 = 1234548, 1234567 - 1234548 = 19
        # letter at index 19 = 'L'
        assert validate_spanish_nie("X1234567L") is True

    def test_valid_nie_y(self):
        # Y prefix -> 1, number = 11234567, 11234567 % 23
        # 11234567 / 23 = 488459.4..., 488459 * 23 = 11234557, 11234567 - 11234557 = 10
        # letter at index 10 = 'X'
        assert validate_spanish_nie("Y1234567X") is True

    def test_valid_nie_z(self):
        # Z prefix -> 2, number = 21234567, 21234567 % 23
        # 21234567 / 23 = 923242.0..., 923042 * 23 = 21229966
        # Actually: 21234567 // 23 = 923242, 923242 * 23 = 21234566
        # 21234567 - 21234566 = 1 -> letter at index 1 = 'R'
        assert validate_spanish_nie("Z1234567R") is True

    def test_invalid_nie_wrong_prefix(self):
        assert validate_spanish_nie("A1234567L") is False

    def test_invalid_nie_wrong_length(self):
        assert validate_spanish_nie("X123456L") is False

    def test_invalid_nie_wrong_letter(self):
        assert validate_spanish_nie("X1234567A") is False


# ---------------------------------------------------------------------------
# Language-specific PII Patterns Tests
# ---------------------------------------------------------------------------


class TestLanguagePIIPatterns:
    """Tests for language-specific PII patterns."""

    def test_french_patterns_exist(self):
        assert "fr" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["fr"]) > 0

    def test_german_patterns_exist(self):
        assert "de" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["de"]) > 0

    def test_italian_patterns_exist(self):
        assert "it" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["it"]) > 0

    def test_spanish_patterns_exist(self):
        assert "es" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["es"]) > 0

    def test_all_patterns_are_pii_pattern(self):
        for lang, patterns in LANGUAGE_PII_PATTERNS.items():
            for p in patterns:
                assert isinstance(p, PIIPattern), f"Pattern in {lang} is not PIIPattern"

    # French date patterns
    def test_french_date_slash(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["fr"] if p.entity_type == "date"]
        texts = ["15/01/1970", "1/1/2020"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"French date pattern should match '{text}'"

    def test_french_date_month_name(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["fr"] if p.entity_type == "date"]
        text = "15 janvier 2020"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"French date pattern should match '{text}'"

    # German date patterns
    def test_german_date_dot(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["de"] if p.entity_type == "date"]
        texts = ["15.01.1970", "1.1.2020"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"German date pattern should match '{text}'"

    def test_german_date_month_name(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["de"] if p.entity_type == "date"]
        text = "15 Januar 2020"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"German date pattern should match '{text}'"

    # Italian date patterns
    def test_italian_date_slash(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["it"] if p.entity_type == "date"]
        texts = ["15/01/1970", "1/1/2020"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Italian date pattern should match '{text}'"

    def test_italian_date_month_name(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["it"] if p.entity_type == "date"]
        text = "15 gennaio 2020"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Italian date pattern should match '{text}'"

    # Spanish date patterns
    def test_spanish_date_slash(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["es"] if p.entity_type == "date"]
        texts = ["15/01/1970", "1/1/2020"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Spanish date pattern should match '{text}'"

    def test_spanish_date_month_name(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["es"] if p.entity_type == "date"]
        text = "15 de enero de 2020"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Spanish date pattern should match '{text}'"

    # French phone patterns
    def test_french_phone(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["fr"] if p.entity_type == "phone_number"]
        texts = ["+33 6 12 34 56 78", "06 12 34 56 78"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"French phone pattern should match '{text}'"

    # German phone patterns
    def test_german_phone(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["de"] if p.entity_type == "phone_number"]
        texts = ["+49 30 1234567", "030 1234567"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"German phone pattern should match '{text}'"

    # Italian phone patterns
    def test_italian_phone(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["it"] if p.entity_type == "phone_number"]
        texts = ["+39 333 123 4567", "333 123 4567"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Italian phone pattern should match '{text}'"

    # Spanish phone patterns
    def test_spanish_phone(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["es"] if p.entity_type == "phone_number"]
        texts = ["+34 612 345 678", "612 345 678"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Spanish phone pattern should match '{text}'"

    # National ID patterns
    def test_french_nir_pattern(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["fr"] if p.entity_type == "national_id"]
        assert len(patterns) >= 1
        text = "1 85 05 78 006 084 36"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "French NIR pattern should match"

    def test_german_steuer_id_pattern(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["de"] if p.entity_type == "national_id"]
        assert len(patterns) >= 1

    def test_italian_codice_fiscale_pattern(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["it"] if p.entity_type == "national_id"]
        assert len(patterns) >= 1
        text = "RSSMRA85M01H501Z"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Italian Codice Fiscale pattern should match"

    def test_spanish_dni_pattern(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["es"] if p.entity_type == "national_id"]
        assert len(patterns) >= 1
        text = "12345678Z"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Spanish DNI pattern should match"

    def test_spanish_nie_pattern(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["es"] if p.entity_type == "national_id"]
        text = "X1234567L"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Spanish NIE pattern should match"


# ---------------------------------------------------------------------------
# get_patterns_for_language Tests
# ---------------------------------------------------------------------------


class TestGetPatternsForLanguage:
    """Tests for get_patterns_for_language()."""

    def test_english_returns_base_patterns(self):
        patterns = get_patterns_for_language("en")
        assert len(patterns) == len(PII_PATTERNS)

    def test_french_includes_base_and_language(self):
        fr_patterns = get_patterns_for_language("fr")
        base_count = len(PII_PATTERNS)
        lang_count = len(LANGUAGE_PII_PATTERNS["fr"])
        assert len(fr_patterns) == base_count + lang_count

    def test_german_includes_base_and_language(self):
        de_patterns = get_patterns_for_language("de")
        base_count = len(PII_PATTERNS)
        lang_count = len(LANGUAGE_PII_PATTERNS["de"])
        assert len(de_patterns) == base_count + lang_count

    def test_italian_includes_base_and_language(self):
        it_patterns = get_patterns_for_language("it")
        base_count = len(PII_PATTERNS)
        lang_count = len(LANGUAGE_PII_PATTERNS["it"])
        assert len(it_patterns) == base_count + lang_count

    def test_spanish_includes_base_and_language(self):
        es_patterns = get_patterns_for_language("es")
        base_count = len(PII_PATTERNS)
        lang_count = len(LANGUAGE_PII_PATTERNS["es"])
        assert len(es_patterns) == base_count + lang_count

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            get_patterns_for_language("ja")

    def test_all_returned_patterns_are_pii_pattern(self):
        for lang in SUPPORTED_LANGUAGES:
            patterns = get_patterns_for_language(lang)
            for p in patterns:
                assert isinstance(p, PIIPattern)


# ---------------------------------------------------------------------------
# Language Fake Data Tests
# ---------------------------------------------------------------------------


class TestLanguageFakeData:
    """Tests for LANGUAGE_FAKE_DATA."""

    def test_all_languages_have_fake_data(self):
        for lang in SUPPORTED_LANGUAGES:
            assert lang in LANGUAGE_FAKE_DATA

    def test_required_keys_present(self):
        required_keys = {"NAME", "EMAIL", "PHONE", "DATE", "LOCATION"}
        for lang in SUPPORTED_LANGUAGES:
            data = LANGUAGE_FAKE_DATA[lang]
            for key in required_keys:
                assert key in data, f"Missing '{key}' in LANGUAGE_FAKE_DATA['{lang}']"

    def test_french_names_are_french(self):
        names = LANGUAGE_FAKE_DATA["fr"]["NAME"]
        assert any("Dupont" in n or "Martin" in n for n in names)

    def test_german_names_are_german(self):
        names = LANGUAGE_FAKE_DATA["de"]["NAME"]
        assert any("M\u00fcller" in n or "Schmidt" in n for n in names)

    def test_italian_names_are_italian(self):
        names = LANGUAGE_FAKE_DATA["it"]["NAME"]
        assert any("Rossi" in n or "Bianchi" in n for n in names)

    def test_spanish_names_are_spanish(self):
        names = LANGUAGE_FAKE_DATA["es"]["NAME"]
        assert any("L\u00f3pez" in n or "Garc\u00eda" in n for n in names)

    def test_french_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["fr"]["PHONE"]
        assert any("+33" in p or p.startswith("0") for p in phones)

    def test_german_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["de"]["PHONE"]
        assert any("+49" in p for p in phones)

    def test_italian_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["it"]["PHONE"]
        assert any("+39" in p for p in phones)

    def test_spanish_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["es"]["PHONE"]
        assert any("+34" in p for p in phones)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
