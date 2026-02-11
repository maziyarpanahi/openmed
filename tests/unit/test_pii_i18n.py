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
    get_patterns_for_language,
)
from openmed.core.pii_entity_merger import PIIPattern, PII_PATTERNS


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Test module-level constants."""

    def test_supported_languages(self):
        assert SUPPORTED_LANGUAGES == {"en", "fr", "de", "it"}

    def test_language_names_keys(self):
        assert set(LANGUAGE_NAMES.keys()) == SUPPORTED_LANGUAGES

    def test_language_model_prefix(self):
        assert LANGUAGE_MODEL_PREFIX["en"] == ""
        assert LANGUAGE_MODEL_PREFIX["fr"] == "French-"
        assert LANGUAGE_MODEL_PREFIX["de"] == "German-"
        assert LANGUAGE_MODEL_PREFIX["it"] == "Italian-"

    def test_default_pii_models_all_languages(self):
        assert set(DEFAULT_PII_MODELS.keys()) == SUPPORTED_LANGUAGES

    def test_default_pii_models_naming(self):
        assert "French" in DEFAULT_PII_MODELS["fr"]
        assert "German" in DEFAULT_PII_MODELS["de"]
        assert "Italian" in DEFAULT_PII_MODELS["it"]
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
        # Valid NIR: 1 85 05 78 006 084 36
        # number = 1850578006084, key = 97 - (1850578006084 % 97)
        # 1850578006084 % 97 = 1850578006084 - 19*97*... let's compute:
        # Actually let's construct a valid one.
        # For digits: 1 85 05 78 006 084, first 13 = 1850578006084
        # 1850578006084 mod 97 = ?
        # 1850578006084 / 97 ≈ 19077092846.2268
        # 19077092846 * 97 = 1850477846062
        # 1850578006084 - 1850477846062 = 100160022
        # That's too large. Let me just test with a known computed value.
        # Build: digits = "185057800608436"
        # first_13 = 1850578006084
        # 1850578006084 % 97 steps:
        # Actually, let me just pick simpler numbers.
        # number = 1000000000000, key = 97 - (1000000000000 % 97)
        # 1000000000000 % 97: 1000000000000/97 ≈ 10309278350.515
        # 10309278350 * 97 = 999999999950
        # 1000000000000 - 999999999950 = 50
        # key = 97 - 50 = 47
        valid_nir = "1000000000000" + "47"
        assert validate_french_nir(valid_nir) is True

    def test_valid_nir_with_spaces(self):
        # Same as above with spaces
        assert validate_french_nir("1 00 00 00 000 000 47") is True

    def test_invalid_nir_wrong_length(self):
        assert validate_french_nir("12345") is False

    def test_invalid_nir_bad_first_digit(self):
        # First digit must be 1 or 2
        assert validate_french_nir("300000000000047") is False

    def test_invalid_nir_wrong_checksum(self):
        # Valid structure but wrong check digits
        assert validate_french_nir("100000000000048") is False

    def test_valid_nir_female(self):
        # Female NIR starts with 2
        # number = 2000000000000, key = 97 - (2000000000000 % 97)
        # 2000000000000 % 97: 2000000000000/97 ≈ 20618556701.03
        # 20618556701 * 97 = 1999999999997
        # 2000000000000 - 1999999999997 = 3
        # key = 97 - 3 = 94
        assert validate_french_nir("200000000000094") is True


# ---------------------------------------------------------------------------
# German Steuer-ID Validator Tests
# ---------------------------------------------------------------------------


class TestValidateGermanSteuerId:
    """Tests for validate_german_steuer_id()."""

    def test_valid_steuer_id(self):
        # 11 digits, first digit != 0, exactly one digit appears 2+ times
        # e.g., 12345678912 (digit 1 appears twice -> first_ten = 1234567891)
        assert validate_german_steuer_id("12345678912") is True

    def test_valid_steuer_id_with_spaces(self):
        assert validate_german_steuer_id("1234 5678 912") is True

    def test_invalid_steuer_id_first_digit_zero(self):
        assert validate_german_steuer_id("01234567891") is False

    def test_invalid_steuer_id_wrong_length(self):
        assert validate_german_steuer_id("123456789") is False

    def test_invalid_steuer_id_too_many_repeats(self):
        # Two different digits each appear multiple times -> multi_count > 1
        assert validate_german_steuer_id("11223344556") is False

    def test_invalid_steuer_id_no_repeats(self):
        # First 10 digits all unique -> 0 multi_count
        assert validate_german_steuer_id("12345678900") is False


# ---------------------------------------------------------------------------
# Italian Codice Fiscale Validator Tests
# ---------------------------------------------------------------------------


class TestValidateItalianCodiceFiscale:
    """Tests for validate_italian_codice_fiscale()."""

    def test_valid_codice_fiscale(self):
        # Format: LLLLLLDDLDDLDDDL
        assert validate_italian_codice_fiscale("RSSMRA85M01H501Z") is True

    def test_valid_codice_fiscale_lowercase(self):
        assert validate_italian_codice_fiscale("rssmra85m01h501z") is True

    def test_valid_codice_fiscale_with_spaces(self):
        assert validate_italian_codice_fiscale("RSS MRA 85M01 H501Z") is True

    def test_invalid_codice_fiscale_wrong_length(self):
        assert validate_italian_codice_fiscale("RSSMRA85M01H50") is False

    def test_invalid_codice_fiscale_wrong_format(self):
        # All digits
        assert validate_italian_codice_fiscale("1234567890123456") is False

    def test_invalid_codice_fiscale_wrong_pattern(self):
        # Wrong letter/digit positions
        assert validate_italian_codice_fiscale("12SMRA85M01H501Z") is False


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

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            get_patterns_for_language("es")

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

    def test_french_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["fr"]["PHONE"]
        assert any("+33" in p or p.startswith("0") for p in phones)

    def test_german_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["de"]["PHONE"]
        assert any("+49" in p for p in phones)

    def test_italian_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["it"]["PHONE"]
        assert any("+39" in p for p in phones)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
