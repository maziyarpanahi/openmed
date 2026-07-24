"""Tests for multilingual PII detection support (pii_i18n module)."""

import json
import random
import re
import warnings
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from faker import Faker

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.locales import LANG_TO_LOCALE
from openmed.core.anonymizer.providers import (
    HungarianTAJProvider,
    generate_hungarian_taj,
)
from openmed.core.anonymizer.providers.clinical_ids import (
    KENYA_MFL_SYNTHETIC_MAX,
    KENYA_MFL_SYNTHETIC_MIN,
    NIGERIA_HFR_SYNTHETIC_SERIAL_MAX,
    NIGERIA_HFR_SYNTHETIC_SERIAL_MIN,
    generate_african_phone,
    generate_bulgarian_egn,
    generate_egyptian_national_id,
    generate_estonian_isikukood,
    generate_ethiopia_fayda,
    generate_jmbg,
    generate_moroccan_cin,
    generate_mpesa_transaction_code,
    generate_philhealth_pin,
    generate_portuguese_nif,
    generate_russian_oms,
    generate_russian_snils,
    generate_rwanda_id,
    generate_tanzania_nida,
    generate_uganda_nin,
    generate_vietnamese_cccd,
    generate_vietnamese_cmnd,
    register_clinical_providers,
)
from openmed.core.pii_entity_merger import PII_PATTERNS, PIIPattern, find_semantic_units
from openmed.core.pii_i18n import (
    AADHAAR_PII_PATTERNS,
    AFRICAN_MOBILE_PII_PATTERNS,
    AFRICAN_MOBILE_PLANS,
    DEFAULT_PII_MODELS,
    INDIA_HEALTH_ID_PII_PATTERNS,
    INDIC_NER_LANGUAGES,
    LANGUAGE_FAKE_DATA,
    LANGUAGE_MODEL_PREFIX,
    LANGUAGE_MONTH_NAMES,
    LANGUAGE_NAMES,
    LANGUAGE_PII_PATTERNS,
    LOCALE_PII_PATTERNS,
    MRZ_PII_PATTERNS,
    NATIONAL_ID_ONLY_LANGUAGES,
    SUPPORTED_LANGUAGES,
    USCC_PII_PATTERNS,
    AfricanMobilePlan,
    build_african_mobile_pattern,
    get_patterns_for_language,
    normalize_arabic_indic_digits,
    validate_bic,
    validate_bulgarian_egn,
    validate_croatian_oib,
    validate_czech_rodne_cislo,
    validate_czechoslovak_rodne_cislo,
    validate_danish_cpr,
    validate_dutch_bsn,
    validate_egyptian_national_id,
    validate_estonian_isikukood,
    validate_ethiopia_fayda,
    validate_finnish_hetu,
    validate_french_nir,
    validate_german_steuer_id,
    validate_ghana_card_pin,
    validate_greek_amka,
    validate_hungarian_taj,
    validate_iban,
    validate_indonesian_nik,
    validate_israeli_teudat_zehut,
    validate_italian_codice_fiscale,
    validate_jmbg,
    validate_kenya_maisha_namba,
    validate_kenya_mfl_code,
    validate_kenya_national_id,
    validate_korean_rrn,
    validate_latvian_personas_kods,
    validate_malaysian_mykad,
    validate_mobile_money_paybill,
    validate_mobile_money_till,
    validate_momo_reference,
    validate_moroccan_cin,
    validate_mpesa_transaction_code,
    validate_nigeria_bvn,
    validate_nigeria_hfr_code,
    validate_nigeria_nin,
    validate_norwegian_fodselsnummer,
    validate_pakistani_cnic,
    validate_philhealth_pin,
    validate_philsys_psn,
    validate_portuguese_cnpj,
    validate_portuguese_cpf,
    validate_portuguese_nif,
    validate_russian_oms,
    validate_russian_snils,
    validate_rwanda_id,
    validate_south_african_id,
    validate_spanish_dni,
    validate_spanish_nie,
    validate_swedish_personnummer,
    validate_tanzania_nida,
    validate_thai_national_id,
    validate_turkish_tckn,
    validate_uganda_nin,
    validate_ukrainian_rnokpp,
    validate_vietnamese_cccd,
    validate_vietnamese_cmnd,
)

# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Test module-level constants."""

    def test_supported_languages(self):
        assert SUPPORTED_LANGUAGES == {
            "am",
            "bn",
            "en",
            "fr",
            "de",
            "it",
            "es",
            "nl",
            "hi",
            "te",
            "ta",
            "pt",
            "ar",
            "he",
            "ja",
            "tr",
            "id",
            "th",
            "ko",
            "ro",
            "ru",
            "sv",
            "da",
            "no",
            "sw",
            "zu",
            "xh",
            "zh",
            "uk",
            "cs",
            "el",
        }

    def test_national_id_only_languages(self):
        assert NATIONAL_ID_ONLY_LANGUAGES == {
            "af",
            "ha",
            "ig",
            "yo",
            "pl",
            "lv",
            "sk",
            "ms",
            "tl",
            "hu",
            "et",
            "sr",
            "hr",
            "bg",
            "fi",
            "vi",
            "rw",
            "ur",
        }

    def test_language_names_keys(self):
        assert set(LANGUAGE_NAMES.keys()) == SUPPORTED_LANGUAGES | INDIC_NER_LANGUAGES

    def test_language_model_prefix(self):
        assert LANGUAGE_MODEL_PREFIX["am"] == "Amharic-"
        assert LANGUAGE_MODEL_PREFIX["en"] == ""
        assert LANGUAGE_MODEL_PREFIX["fr"] == "French-"
        assert LANGUAGE_MODEL_PREFIX["de"] == "German-"
        assert LANGUAGE_MODEL_PREFIX["it"] == "Italian-"
        assert LANGUAGE_MODEL_PREFIX["es"] == "Spanish-"
        assert LANGUAGE_MODEL_PREFIX["nl"] == "Dutch-"
        assert LANGUAGE_MODEL_PREFIX["hi"] == "Hindi-"
        assert LANGUAGE_MODEL_PREFIX["bn"] == "Bengali-"
        assert LANGUAGE_MODEL_PREFIX["ta"] == "Tamil-"
        assert LANGUAGE_MODEL_PREFIX["te"] == "Telugu-"
        assert LANGUAGE_MODEL_PREFIX["pt"] == "Portuguese-"
        assert LANGUAGE_MODEL_PREFIX["ar"] == "Arabic-"
        assert LANGUAGE_MODEL_PREFIX["he"] == "Hebrew-"
        assert LANGUAGE_MODEL_PREFIX["ja"] == "Japanese-"
        assert LANGUAGE_MODEL_PREFIX["tr"] == "Turkish-"
        assert LANGUAGE_MODEL_PREFIX["id"] == "Indonesian-"
        assert LANGUAGE_MODEL_PREFIX["th"] == "Thai-"
        assert LANGUAGE_MODEL_PREFIX["ko"] == "Korean-"
        assert LANGUAGE_MODEL_PREFIX["ro"] == "Romanian-"
        assert LANGUAGE_MODEL_PREFIX["sv"] == "Swedish-"
        assert LANGUAGE_MODEL_PREFIX["da"] == "Danish-"
        assert LANGUAGE_MODEL_PREFIX["no"] == "Norwegian-"
        assert LANGUAGE_MODEL_PREFIX["sw"] == "Swahili-"
        assert LANGUAGE_MODEL_PREFIX["zu"] == "isiZulu-"
        assert LANGUAGE_MODEL_PREFIX["xh"] == "isiXhosa-"
        assert LANGUAGE_MODEL_PREFIX["zh"] == "Chinese-"
        assert LANGUAGE_MODEL_PREFIX["uk"] == "Ukrainian-"
        assert LANGUAGE_MODEL_PREFIX["cs"] == "Czech-"
        assert LANGUAGE_MODEL_PREFIX["el"] == "Greek-"

    def test_default_pii_models_all_languages(self):
        assert set(DEFAULT_PII_MODELS.keys()) == SUPPORTED_LANGUAGES | (
            INDIC_NER_LANGUAGES - {"bn", "hi", "ta", "te"}
        )

    def test_default_pii_models_naming(self):
        assert DEFAULT_PII_MODELS["am"] == "OpenMed/privacy-filter-multilingual"
        assert "French" in DEFAULT_PII_MODELS["fr"]
        assert "German" in DEFAULT_PII_MODELS["de"]
        assert "Italian" in DEFAULT_PII_MODELS["it"]
        assert "Spanish" in DEFAULT_PII_MODELS["es"]
        assert "Dutch" in DEFAULT_PII_MODELS["nl"]
        assert "Hindi" in DEFAULT_PII_MODELS["hi"]
        assert "Bengali" in DEFAULT_PII_MODELS["bn"]
        assert "Tamil" in DEFAULT_PII_MODELS["ta"]
        assert "Telugu" in DEFAULT_PII_MODELS["te"]
        assert "Portuguese" in DEFAULT_PII_MODELS["pt"]
        assert "Arabic" in DEFAULT_PII_MODELS["ar"]
        assert DEFAULT_PII_MODELS["he"] == "OpenMed/privacy-filter-multilingual"
        assert "Japanese" in DEFAULT_PII_MODELS["ja"]
        assert "Turkish" in DEFAULT_PII_MODELS["tr"]
        assert DEFAULT_PII_MODELS["id"] == "OpenMed/privacy-filter-multilingual"
        assert DEFAULT_PII_MODELS["th"] == "OpenMed/privacy-filter-multilingual"
        assert (
            DEFAULT_PII_MODELS["ko"]
            == "OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1"
        )
        assert DEFAULT_PII_MODELS["ro"] == "OpenMed/privacy-filter-multilingual"
        assert DEFAULT_PII_MODELS["sv"] == "OpenMed/privacy-filter-multilingual"
        assert DEFAULT_PII_MODELS["da"] == "OpenMed/privacy-filter-multilingual"
        assert DEFAULT_PII_MODELS["no"] == "OpenMed/privacy-filter-multilingual"
        assert DEFAULT_PII_MODELS["sw"] == "OpenMed/privacy-filter-multilingual"
        assert DEFAULT_PII_MODELS["zu"] == "OpenMed/privacy-filter-multilingual"
        assert DEFAULT_PII_MODELS["xh"] == "OpenMed/privacy-filter-multilingual"
        assert "Chinese" in DEFAULT_PII_MODELS["zh"]
        assert DEFAULT_PII_MODELS["uk"] == "OpenMed/privacy-filter-multilingual"
        assert DEFAULT_PII_MODELS["cs"] == "OpenMed/privacy-filter-multilingual"
        assert DEFAULT_PII_MODELS["el"] == "OpenMed/privacy-filter-multilingual"
        # English has no language prefix
        assert "French" not in DEFAULT_PII_MODELS["en"]
        assert "German" not in DEFAULT_PII_MODELS["en"]

    def test_month_names_all_languages(self):
        for lang in SUPPORTED_LANGUAGES:
            assert lang in LANGUAGE_MONTH_NAMES
            assert len(LANGUAGE_MONTH_NAMES[lang]) == 12


class TestEastAfricanNationalIds:
    """Validator, detector, surrogate, and fixture coverage for OM-857."""

    def test_tanzania_nida_validator_accepts_supported_renderings(self):
        compact = "19850712123456789012"
        hyphenated = "19850712-12345-67890-12"

        assert validate_tanzania_nida(compact)
        assert validate_tanzania_nida(hyphenated)
        assert not validate_tanzania_nida("19850230123456789012")
        assert not validate_tanzania_nida("18991231123456789012")
        assert not validate_tanzania_nida("19850712-1234-567890-12")
        assert not validate_tanzania_nida(None)

    def test_uganda_nin_validator_checks_class_gender_and_length(self):
        assert validate_uganda_nin("CM123456789ABC")
        assert validate_uganda_nin("cf123456789abc")
        assert validate_uganda_nin("RFABCDEFGHIJKL")
        assert not validate_uganda_nin("XM123456789ABC")
        assert not validate_uganda_nin("CX123456789ABC")
        assert not validate_uganda_nin("CM123456789AB")
        assert not validate_uganda_nin("CM123456789AB!")
        assert not validate_uganda_nin(None)

    def test_rwanda_id_validator_checks_birth_year_and_gender(self):
        assert validate_rwanda_id("1198571234567890")
        assert not validate_rwanda_id("1189971234567890")
        assert not validate_rwanda_id("1198561234567890")
        assert not validate_rwanda_id(f"1{date.today().year + 1:04d}71234567890")
        assert not validate_rwanda_id("119857123456789")
        assert not validate_rwanda_id(None)

    def test_ethiopia_fayda_validator_checks_verhoeff_and_leading_digit(self):
        assert validate_ethiopia_fayda("234123412346")
        assert not validate_ethiopia_fayda("234123412347")
        assert not validate_ethiopia_fayda("134123412346")
        assert not validate_ethiopia_fayda("23412341234")
        assert not validate_ethiopia_fayda("2341-2341-2346")
        assert not validate_ethiopia_fayda("FAN234123412346")
        assert not validate_ethiopia_fayda(None)

    def test_cross_format_non_collision(self):
        samples = {
            validate_tanzania_nida: "19850712123456789012",
            validate_uganda_nin: "CF123456789ABC",
            validate_rwanda_id: "1198571234567890",
            validate_ethiopia_fayda: "234123412346",
        }
        for expected_validator, value in samples.items():
            assert expected_validator(value)
            for other_validator in samples:
                if other_validator is not expected_validator:
                    assert not other_validator(value)

    def test_generated_fayda_values_and_single_digit_mutations(self):
        rng = random.Random(857)
        for sample_index in range(1_000):
            value = generate_ethiopia_fayda(rng=rng)
            assert validate_ethiopia_fayda(value)

            mutation_index = sample_index % len(value)
            replacement = str((int(value[mutation_index]) + 1) % 10)
            mutated = value[:mutation_index] + replacement + value[mutation_index + 1 :]
            assert not validate_ethiopia_fayda(mutated)

    def test_generated_structural_ids_all_pass_their_validators(self):
        rng = random.Random(857)
        generators_and_validators = (
            (generate_tanzania_nida, validate_tanzania_nida),
            (generate_uganda_nin, validate_uganda_nin),
            (generate_rwanda_id, validate_rwanda_id),
        )
        for generator, validator in generators_and_validators:
            assert all(validator(generator(rng=rng)) for _ in range(1_000))

    def test_surrogates_preserve_embedded_fields(self):
        nida_source = "19850712123456789012"
        nida = generate_tanzania_nida(nida_source, rng=random.Random(1))
        assert int(nida[:4]) // 10 == 198

        for prefix in ("CM", "CF"):
            nin = generate_uganda_nin(
                f"{prefix}123456789ABC",
                rng=random.Random(2),
            )
            assert nin.startswith(prefix)

        rwanda_source = "1198571234567890"
        rwanda = generate_rwanda_id(rwanda_source, rng=random.Random(3))
        assert rwanda[0] == rwanda_source[0]
        assert int(rwanda[1:5]) // 10 == int(rwanda_source[1:5]) // 10
        assert rwanda[5] == rwanda_source[5]
        assert validate_rwanda_id(rwanda)

    def test_compact_and_hyphenated_nida_map_to_same_surrogate(self):
        anonymizer = Anonymizer(lang="sw", consistent=True, seed=857)
        compact = "19850712123456789012"
        hyphenated = "19850712-12345-67890-12"

        compact_surrogate = anonymizer.surrogate(compact, "national_id")
        hyphenated_surrogate = anonymizer.surrogate(hyphenated, "national_id")

        assert compact_surrogate == hyphenated_surrogate
        assert validate_tanzania_nida(compact_surrogate)

    @pytest.mark.parametrize(
        ("lang", "source", "validator"),
        (
            ("en_TZ", "19850712123456789012", validate_tanzania_nida),
            ("en_UG", "CF123456789ABC", validate_uganda_nin),
            ("rw", "1198571234567890", validate_rwanda_id),
            ("am", "234123412346", validate_ethiopia_fayda),
            ("en_ET", "234123412346", validate_ethiopia_fayda),
        ),
    )
    def test_anonymizer_dispatches_east_african_locale_surrogates(
        self,
        lang,
        source,
        validator,
    ):
        surrogate = Anonymizer(lang=lang, consistent=True, seed=857).surrogate(
            source,
            "national_id",
        )
        assert surrogate != source
        assert validator(surrogate)

    @pytest.mark.parametrize(
        ("alias", "value"),
        (
            ("sw", "19850712123456789012"),
            ("en_tz", "19850712-12345-67890-12"),
            ("en_ug", "CF123456789ABC"),
            ("rw", "1198571234567890"),
            ("am", "234123412346"),
            ("en_et", "234123412346"),
        ),
    )
    def test_locale_alias_patterns_validate_expected_ids(self, alias, value):
        patterns = LOCALE_PII_PATTERNS[alias]
        assert any(
            match
            and pattern.validator is not None
            and pattern.validator(match.group(0))
            for pattern in patterns
            if (match := re.search(pattern.pattern, value, pattern.flags))
        )

    @pytest.mark.parametrize(
        ("lang", "locale", "contextual", "bare"),
        (
            (
                "sw",
                None,
                "Namba ya NIDA: 19850712123456789012",
                "19850712123456789012",
            ),
            (
                "en",
                "en_UG",
                "NIRA NIN: CF123456789ABC",
                "CF123456789ABC",
            ),
            (
                "rw",
                None,
                "Indangamuntu: 1198571234567890",
                "1198571234567890",
            ),
            (
                "am",
                None,
                "Fayda FAN: 234123412346",
                "234123412346",
            ),
        ),
    )
    def test_context_gated_formats_require_keyword_context(
        self,
        lang,
        locale,
        contextual,
        bare,
    ):
        from openmed.core.safety_sweep import safety_sweep

        assert any(
            entity.label == "national_id"
            for entity in safety_sweep(contextual, [], lang=lang, locale=locale)
        )
        assert not any(
            entity.label == "national_id"
            for entity in safety_sweep(bare, [], lang=lang, locale=locale)
        )

    def test_east_africa_fixture_round_trip_has_zero_identifier_leakage(self):
        from openmed.core.pii import (
            _apply_safety_sweep_to_result,
            _build_deidentification_result,
        )
        from openmed.processing.outputs import PredictionResult

        fixture_path = Path("tests/fixtures/pii/east_africa_synthetic_notes.jsonl")
        rows = [
            json.loads(line)
            for line in fixture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert {row["id_type"] for row in rows} == {
            "fayda_fan",
            "nida_nin",
            "nin",
            "rwanda_id",
        }
        for row in rows:
            empty_result = PredictionResult(
                text=row["text"],
                entities=[],
                model_name="offline-safety-sweep",
                timestamp="2026-07-18T00:00:00Z",
                metadata={},
            )
            swept_result, added_count = _apply_safety_sweep_to_result(
                row["text"],
                empty_result,
                lang=row["language"],
                locale=row.get("locale"),
            )
            result = _build_deidentification_result(
                row["text"],
                swept_result,
                effective_method="replace",
                keep_year=False,
                date_shift_days=None,
                keep_mapping=False,
                lang=row["language"],
                consistent=True,
                seed=857,
                locale=row.get("locale"),
                use_safety_sweep=True,
            )

            assert added_count == len(row["identifiers"])
            assert all(
                identifier not in result.deidentified_text
                for identifier in row["identifiers"]
            )


# ---------------------------------------------------------------------------
# Financial Identifier Validator Tests
# ---------------------------------------------------------------------------


class TestFinancialIdentifierValidators:
    """Tests for IBAN and SWIFT/BIC financial identifiers."""

    @pytest.mark.parametrize(
        "iban",
        [
            "GB82 WEST 1234 5698 7654 32",
            "DE89 3704 0044 0532 0130 00",
            "ES91 2100 0418 4502 0005 1332",
            "FR14 2004 1010 0505 0001 3M02 606",
            "NL91 ABNA 0417 1643 00",
        ],
    )
    def test_validate_iban_accepts_known_valid_synthetic_values(self, iban):
        assert validate_iban(iban) is True

    @pytest.mark.parametrize(
        "iban",
        [
            "GB83 WEST 1234 5698 7654 32",
            "DE89 3704 0044 0532 0130",
            "ZZ12 1234 5678 9012",
            "GB82 WEST 1234 5698 7654 3!",
        ],
    )
    def test_validate_iban_rejects_bad_checksum_length_or_shape(self, iban):
        assert validate_iban(iban) is False

    @pytest.mark.parametrize(
        "bic",
        [
            "DEUTDEFF",
            "AGRIFRPPXXX",
            "CAIXESBBXXX",
            "deutdeff500",
        ],
    )
    def test_validate_bic_accepts_eight_or_eleven_character_codes(self, bic):
        assert validate_bic(bic) is True

    @pytest.mark.parametrize(
        "bic",
        [
            "DEUTDEFF1",
            "DEU1DEFF",
            "DEUTD3FF",
            "DEUTDEFF-XX",
        ],
    )
    def test_validate_bic_rejects_wrong_length_or_structure(self, bic):
        assert validate_bic(bic) is False


class TestFinancialIdentifierDetection:
    """Financial ID patterns are inherited by every language."""

    @pytest.mark.parametrize(
        ("lang", "text", "expected"),
        [
            (
                "en",
                "Billing note: IBAN GB82 WEST 1234 5698 7654 32 and BIC DEUTDEFF.",
                {
                    ("iban", 19, 46, "GB82 WEST 1234 5698 7654 32"),
                    ("bic", 55, 63, "DEUTDEFF"),
                },
            ),
            (
                "es",
                "Informe: IBAN ES91 2100 0418 4502 0005 1332 y SWIFT CAIXESBBXXX.",
                {
                    ("iban", 14, 43, "ES91 2100 0418 4502 0005 1332"),
                    ("bic", 52, 63, "CAIXESBBXXX"),
                },
            ),
            (
                "fr",
                "Note: IBAN FR14 2004 1010 0505 0001 3M02 606 et BIC AGRIFRPPXXX.",
                {
                    ("iban", 11, 44, "FR14 2004 1010 0505 0001 3M02 606"),
                    ("bic", 52, 63, "AGRIFRPPXXX"),
                },
            ),
        ],
    )
    def test_iban_and_bic_detect_with_offsets(self, lang, text, expected):
        units = find_semantic_units(text, get_patterns_for_language(lang))
        actual = {
            (entity_type, start, end, text[start:end])
            for start, end, entity_type, _score, _pattern, validated in units
            if entity_type in {"iban", "bic"} and validated
        }

        assert actual == expected

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_surrogate_iban_and_bic_round_trip_validators(self, seed):
        anonymizer = Anonymizer(lang="en", consistent=True, seed=seed)

        iban = anonymizer.surrogate("GB82 WEST 1234 5698 7654 32", "IBAN")
        bic = anonymizer.surrogate("DEUTDEFF", "BIC")

        assert validate_iban(iban), f"Invalid IBAN surrogate: {iban!r}"
        assert validate_bic(bic), f"Invalid BIC surrogate: {bic!r}"

    def test_financial_id_golden_fixture_deidentifies_without_leakage(self):
        from datetime import datetime
        from unittest.mock import patch

        from openmed.core.pii import deidentify
        from openmed.eval.golden import GoldenFixture
        from openmed.processing.outputs import PredictionResult

        fixture_path = Path("openmed/eval/golden/financial_ids.jsonl")
        rows = [
            json.loads(line)
            for line in fixture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert {row["language"] for row in rows} == {"en", "es", "fr"}
        for row in rows:
            fixture = GoldenFixture.from_mapping(row)
            with patch("openmed.core.pii.extract_pii") as mock_extract:
                mock_extract.return_value = PredictionResult(
                    text=fixture.text,
                    entities=[],
                    model_name="stub",
                    timestamp=datetime.now().isoformat(),
                )
                result = deidentify(
                    fixture.text,
                    method="mask",
                    lang=fixture.language,
                )

            assert result.metadata["safety_sweep"]["spans_added"] == 2
            for span in fixture.gold_spans:
                assert fixture.text[span.start : span.end] == span.text
                assert span.text not in result.deidentified_text


# ---------------------------------------------------------------------------
# French NIR Validator Tests
# ---------------------------------------------------------------------------


class TestValidateDutchBSN:
    """Tests for validate_dutch_bsn()."""

    def test_valid_bsn(self):
        assert validate_dutch_bsn("123456782") is True

    def test_valid_bsn_with_spaces(self):
        assert validate_dutch_bsn("123 456 782") is True

    def test_invalid_bsn_wrong_checksum(self):
        assert validate_dutch_bsn("123456789") is False

    def test_invalid_bsn_wrong_length(self):
        assert validate_dutch_bsn("1234567") is False


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

    def test_valid_nir_corsica_departments(self):
        assert validate_french_nir("291032A03396109") is True
        assert validate_french_nir("291032B03396136") is True

    def test_invalid_nir_corsica_wrong_checksum(self):
        assert validate_french_nir("291032B03396137") is False


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


class TestValidatePortugueseCPF:
    """Tests for validate_portuguese_cpf()."""

    def test_valid_cpf(self):
        assert validate_portuguese_cpf("123.456.789-09") is True

    def test_valid_cpf_without_punctuation(self):
        assert validate_portuguese_cpf("93541134780") is True

    def test_invalid_cpf_wrong_checksum(self):
        assert validate_portuguese_cpf("123.456.789-00") is False

    def test_invalid_cpf_repeated_digits(self):
        assert validate_portuguese_cpf("111.111.111-11") is False

    def test_invalid_cpf_wrong_length(self):
        assert validate_portuguese_cpf("123456789") is False


class TestValidatePortugueseCNPJ:
    """Tests for validate_portuguese_cnpj()."""

    def test_valid_cnpj(self):
        assert validate_portuguese_cnpj("11.222.333/0001-81") is True

    def test_valid_cnpj_without_punctuation(self):
        assert validate_portuguese_cnpj("04252011000110") is True

    def test_invalid_cnpj_wrong_checksum(self):
        assert validate_portuguese_cnpj("11.222.333/0001-80") is False

    def test_invalid_cnpj_repeated_digits(self):
        assert validate_portuguese_cnpj("11.111.111/1111-11") is False

    def test_invalid_cnpj_wrong_length(self):
        assert validate_portuguese_cnpj("112223330001") is False


class TestValidatePortugueseNIF:
    """Tests for validate_portuguese_nif() (European Portuguese)."""

    def test_valid_nif_individual(self):
        # 123456789 is the canonical valid NIF (leading digit 1).
        assert validate_portuguese_nif("123456789") is True

    def test_valid_nif_company_and_prefix(self):
        assert validate_portuguese_nif("500123403") is True  # company (5)
        assert validate_portuguese_nif("800987608") is True  # sole trader (8)
        assert validate_portuguese_nif("451234561") is True  # prefix 45

    def test_valid_nif_with_spaces(self):
        assert validate_portuguese_nif("123 456 789") is True

    def test_invalid_nif_wrong_checksum(self):
        assert validate_portuguese_nif("123456780") is False

    def test_invalid_nif_bad_leading_digit(self):
        # 400123401 has a valid checksum but an invalid leading digit.
        assert validate_portuguese_nif("400123401") is False

    def test_invalid_nif_wrong_length(self):
        assert validate_portuguese_nif("12345678") is False
        assert validate_portuguese_nif("1234567890") is False

    def test_generated_nif_round_trips_validator(self):
        rng = random.Random(1234)
        for _ in range(200):
            assert validate_portuguese_nif(generate_portuguese_nif(rng=rng))


class TestPortugueseLocaleIdSplit:
    """pt_PT surrogates draw NIF while pt_BR keeps CPF (issue #818)."""

    def test_pt_pt_surrogate_is_valid_nif(self):
        anon = Anonymizer(lang="pt", consistent=True, seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            surrogate = anon.surrogate("123456789", "national_id", locale="pt_PT")
        assert validate_portuguese_nif(surrogate) is True

    def test_pt_br_surrogate_is_valid_cpf(self):
        anon = Anonymizer(lang="pt", consistent=True, seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            surrogate = anon.surrogate("456.378.921-64", "national_id", locale="pt_BR")
        assert validate_portuguese_cpf(surrogate) is True


def test_portuguese_nif_pattern_detects_with_context():
    text = "O doente tem NIF 123456789 (número de contribuinte)."
    observed = set()
    for pattern in get_patterns_for_language("pt"):
        if pattern.entity_type != "national_id":
            continue
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is not None and not pattern.validator(value):
                continue
            observed.add(value)

    assert "123456789" in observed


def test_portuguese_nif_golden_fixture_offsets():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/pt.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    row = next(row for row in rows if row["id"] == "golden-i18n-pt-nif")
    assert row["language"] == "pt"
    assert row["metadata"]["locale"] == "pt_PT"
    assert row["metadata"]["identifier_type"] == "nif"

    text = row["text"]
    for span in row["gold_spans"]:
        assert text[span["start"] : span["end"]] == span["text"], span

    ids_by_type = {
        span["metadata"]["identifier_type"]: span["text"]
        for span in row["gold_spans"]
        if span["label"] == "ID_NUM"
    }
    assert validate_portuguese_nif(ids_by_type["nif"])


def test_portuguese_nif_golden_fixture_deidentifies_with_no_leakage_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.processing.outputs import PredictionResult

    fixture_path = Path("openmed/eval/golden/fixtures/i18n/pt.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    row = next(row for row in rows if row["id"] == "golden-i18n-pt-nif")
    empty_result = PredictionResult(
        text=row["text"],
        entities=[],
        model_name="offline-safety-sweep",
        timestamp="2026-07-03T00:00:00Z",
        metadata={},
    )

    swept_result, added_count = _apply_safety_sweep_to_result(
        row["text"],
        empty_result,
        lang="pt",
    )
    result = _build_deidentification_result(
        row["text"],
        swept_result,
        effective_method="mask",
        keep_year=False,
        date_shift_days=None,
        keep_mapping=False,
        lang="pt",
        consistent=False,
        seed=None,
        locale="pt_PT",
        use_safety_sweep=True,
    )

    assert added_count == len(row["gold_spans"])
    for span in row["gold_spans"]:
        assert span["text"] not in result.deidentified_text


class TestValidateTurkishTCKN:
    """Tests for validate_turkish_tckn()."""

    def test_valid_tckn(self):
        assert validate_turkish_tckn("10000000146") is True

    def test_valid_tckn_with_spaces(self):
        assert validate_turkish_tckn("100 000 001 46") is True

    def test_invalid_tckn_first_digit_zero(self):
        assert validate_turkish_tckn("00000000146") is False

    def test_invalid_tckn_wrong_checksum(self):
        assert validate_turkish_tckn("10000000147") is False

    def test_invalid_tckn_wrong_length(self):
        assert validate_turkish_tckn("1000000014") is False


# -------------------------------------------------------------------
# Korean RRN Validator Tests
# -------------------------------------------------------------------
class TestValidateKoreanRRN:
    """Tests for validate_korean_rrn()."""

    def test_valid_rrn(self):
        assert validate_korean_rrn("940315-1234567") is True

    def test_valid_rrn_without_hyphen(self):
        assert validate_korean_rrn("9403151234567") is True

    def test_invalid_rrn_wrong_checksum(self):
        assert validate_korean_rrn("940315-1234568") is False

    def test_invalid_rrn_wrong_length(self):
        assert validate_korean_rrn("940315-123456") is False


class TestValidateIsraeliTeudatZehut:
    """Tests for validate_israeli_teudat_zehut()."""

    def test_valid_teudat_zehut(self):
        assert validate_israeli_teudat_zehut("123456782") is True

    def test_valid_teudat_zehut_with_spaces(self):
        assert validate_israeli_teudat_zehut("123 456 782") is True

    def test_valid_teudat_zehut_zero_padded(self):
        assert validate_israeli_teudat_zehut("18") is True

    def test_invalid_teudat_zehut_wrong_checksum(self):
        assert validate_israeli_teudat_zehut("123456783") is False

    def test_invalid_teudat_zehut_all_zero(self):
        assert validate_israeli_teudat_zehut("000000000") is False

    def test_invalid_teudat_zehut_wrong_length(self):
        assert validate_israeli_teudat_zehut("1234567890") is False


class TestHebrewLocaleSurrogates:
    """Tests for Hebrew locale and Teudat Zehut surrogate wiring."""

    def test_hebrew_locale_and_national_id_surrogate(self):
        assert LANG_TO_LOCALE["he"] == "he_IL"

        anonymizer = Anonymizer(lang="he", consistent=True, seed=42)
        surrogate = anonymizer.surrogate("123456782", "national_id")

        assert validate_israeli_teudat_zehut(surrogate) is True


class TestValidateIndonesianNIK:
    """Tests for validate_indonesian_nik()."""

    def test_valid_male_nik(self):
        assert validate_indonesian_nik("3174051708850001") is True

    def test_valid_female_nik(self):
        assert validate_indonesian_nik("3174055708850001") is True

    def test_valid_nik_with_spaces(self):
        assert validate_indonesian_nik("317405 570885 0001") is True

    def test_invalid_nik_impossible_birth_date(self):
        assert validate_indonesian_nik("3174057102850001") is False

    def test_invalid_nik_bad_prefix_shape(self):
        assert validate_indonesian_nik("0074051708850001") is False

    def test_invalid_nik_zero_serial(self):
        assert validate_indonesian_nik("3174051708850000") is False

    def test_invalid_nik_wrong_length(self):
        assert validate_indonesian_nik("317405170885000") is False


class TestValidateThaiNationalId:
    """Tests for validate_thai_national_id()."""

    def test_valid_thai_national_id(self):
        assert validate_thai_national_id("1101700203450") is True

    def test_valid_thai_national_id_with_hyphens(self):
        assert validate_thai_national_id("1-1017-00203-45-0") is True

    def test_invalid_thai_national_id_wrong_checksum(self):
        assert validate_thai_national_id("1101700203451") is False

    def test_invalid_thai_national_id_wrong_length(self):
        assert validate_thai_national_id("110170020345") is False

    def test_invalid_thai_national_id_first_digit_zero(self):
        assert validate_thai_national_id("0101700203458") is False

    def test_generated_thai_surrogate_passes_validator(self):
        assert LANG_TO_LOCALE["th"] == "th_TH"

        anonymizer = Anonymizer(lang="th", consistent=True, seed=7)
        surrogate = anonymizer.surrogate("1101700203450", "national_id")

        assert validate_thai_national_id(surrogate) is True


class TestValidateMalaysianMyKad:
    """Tests for validate_malaysian_mykad()."""

    def test_valid_mykad_with_dashes(self):
        assert validate_malaysian_mykad("850817-14-5678") is True

    def test_valid_mykad_without_dashes(self):
        assert validate_malaysian_mykad("850817145678") is True

    def test_invalid_mykad_impossible_embedded_date(self):
        assert validate_malaysian_mykad("850230-14-5678") is False

    def test_invalid_mykad_wrong_length(self):
        assert validate_malaysian_mykad("850817-14-567") is False

    def test_invalid_mykad_zero_place_code(self):
        assert validate_malaysian_mykad("850817-00-5678") is False

    def test_invalid_mykad_zero_serial(self):
        assert validate_malaysian_mykad("850817-14-0000") is False

    def test_generated_mykad_surrogate_passes_validator(self):
        assert LANG_TO_LOCALE["ms"] == "ms_MY"

        anonymizer = Anonymizer(lang="ms", consistent=True, seed=42)
        surrogate = anonymizer.surrogate("850817-14-5678", "national_id")

        assert validate_malaysian_mykad(surrogate) is True


class TestValidatePhilippineIds:
    """Tests for Philippine PhilSys and PhilHealth validators."""

    def test_valid_philsys_psn_with_dashes(self):
        assert validate_philsys_psn("1234-5678-9012") is True

    def test_valid_philsys_psn_without_dashes(self):
        assert validate_philsys_psn("123456789012") is True

    def test_invalid_philsys_psn_wrong_grouping(self):
        assert validate_philsys_psn("98-765432109-8") is False

    def test_invalid_philsys_psn_trivial_digits(self):
        assert validate_philsys_psn("0000-0000-0000") is False

    def test_invalid_philsys_psn_wrong_length(self):
        assert validate_philsys_psn("1234-5678-901") is False

    def test_valid_philhealth_pin_with_dashes(self):
        assert validate_philhealth_pin("98-765432109-8") is True

    def test_valid_philhealth_pin_without_dashes(self):
        assert validate_philhealth_pin("987654321098") is True

    def test_invalid_philhealth_pin_wrong_grouping(self):
        assert validate_philhealth_pin("1234-5678-9012") is False

    def test_invalid_philhealth_pin_zero_groups(self):
        assert validate_philhealth_pin("00-000000000-0") is False

    def test_invalid_philhealth_pin_wrong_length(self):
        assert validate_philhealth_pin("98-765432109") is False

    def test_generated_tl_surrogate_passes_philsys_validator(self):
        assert LANG_TO_LOCALE["tl"] == "fil_PH"

        anonymizer = Anonymizer(lang="tl", consistent=True, seed=42)
        surrogate = anonymizer.surrogate("1234-5678-9012", "national_id")

        assert validate_philsys_psn(surrogate) is True

    def test_generated_philhealth_provider_passes_validator(self):
        surrogate = generate_philhealth_pin()

        assert validate_philhealth_pin(surrogate) is True


class TestValidateDanishCPR:
    """Tests for validate_danish_cpr()."""

    def test_valid_cpr_with_dash(self):
        assert validate_danish_cpr("170885-1234") is True

    def test_valid_cpr_without_dash(self):
        assert validate_danish_cpr("1708851234") is True

    def test_valid_modern_cpr_without_mod11_requirement(self):
        assert validate_danish_cpr("010101-4001") is True

    def test_invalid_cpr_impossible_birth_date(self):
        assert validate_danish_cpr("320185-1234") is False

    def test_invalid_cpr_wrong_grouping(self):
        assert validate_danish_cpr("170885-12-34") is False

    def test_invalid_cpr_zero_serial(self):
        assert validate_danish_cpr("170885-0000") is False

    def test_invalid_cpr_wrong_length(self):
        assert validate_danish_cpr("170885-123") is False

    def test_generated_danish_surrogate_passes_validator(self):
        assert LANG_TO_LOCALE["da"] == "da_DK"

        anonymizer = Anonymizer(lang="da", consistent=True, seed=42)
        surrogate = anonymizer.surrogate("170885-1234", "national_id")

        assert validate_danish_cpr(surrogate) is True


class TestValidateSwedishPersonnummer:
    """Tests for Swedish personnummer structure and Luhn validation."""

    @pytest.mark.parametrize(
        "value",
        [
            "510312-1140",
            "5103121140",
            "19510312-1140",
            "510312+1140",
            "850877-1238",
            "850861-1236",
            "850891-1230",
        ],
    )
    def test_accepts_checksum_valid_values(self, value):
        assert validate_swedish_personnummer(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "510312-1141",
            "19510229-1140",
            "510312-11-40",
            "510312-114",
            "850860-1237",
            "850892-1239",
        ],
    )
    def test_rejects_corrupt_or_malformed_values(self, value):
        assert validate_swedish_personnummer(value) is False

    def test_generated_surrogate_passes_validator(self):
        assert LANG_TO_LOCALE["sv"] == "sv_SE"
        anonymizer = Anonymizer(lang="sv", consistent=True, seed=42)

        surrogate = anonymizer.surrogate("510312-1140", "national_id")

        assert validate_swedish_personnummer(surrogate) is True


class TestValidateNorwegianFodselsnummer:
    """Tests for Norwegian fødselsnummer double modulus-11 validation."""

    @pytest.mark.parametrize(
        "value",
        [
            "12035101460",
            "03088608002",
            "06114737664",
            "52035101454",
            "41035100065",
            "71035100088",
        ],
    )
    def test_accepts_checksum_valid_values(self, value):
        assert validate_norwegian_fodselsnummer(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "12035101461",
            "32018512345",
            "120351-01460",
            "1203510146",
            "40035100026",
            "72035100017",
        ],
    )
    def test_rejects_corrupt_or_malformed_values(self, value):
        assert validate_norwegian_fodselsnummer(value) is False

    def test_generated_surrogate_passes_validator(self):
        assert LANG_TO_LOCALE["no"] == "no_NO"
        anonymizer = Anonymizer(lang="no", consistent=True, seed=42)

        surrogate = anonymizer.surrogate("12035101460", "national_id")

        assert validate_norwegian_fodselsnummer(surrogate) is True


@pytest.mark.parametrize(
    ("lang", "value"),
    [
        ("sv", "510312-1140"),
        ("sv", "850877-1238"),
        ("no", "12035101460"),
        ("no", "52035101454"),
    ],
)
def test_checksum_strong_nordic_ids_detect_without_context(lang, value):
    """Bare checksum-valid IDs must not leak when model detection misses."""
    from openmed.core.safety_sweep import safety_sweep

    entities = safety_sweep(value, [], lang=lang)

    assert any(
        entity.label == "national_id" and entity.text == value for entity in entities
    )


def test_structural_danish_cpr_still_requires_context():
    """Avoid treating arbitrary date-shaped ten-digit numbers as Danish CPRs."""
    from openmed.core.safety_sweep import safety_sweep

    assert safety_sweep("170885-1234", [], lang="da") == []
    assert any(
        entity.label == "national_id" and entity.text == "170885-1234"
        for entity in safety_sweep("CPR 170885-1234", [], lang="da")
    )


class TestValidateHungarianTaj:
    """Tests for validate_hungarian_taj()."""

    @pytest.mark.parametrize(
        "value",
        ["123456788", "123 456 788", "123-456-788", "000000017"],
    )
    def test_accepts_valid_taj(self, value):
        assert validate_hungarian_taj(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "123456789",  # wrong check digit
            "12345678",  # too short
            "1234567880",  # too long
            "123 456-788",  # mixed separators
            "123.456.788",  # unsupported separator
            "１２３４５６７８８",  # non-ASCII digits
            "",
        ],
    )
    def test_rejects_invalid_taj(self, value):
        assert validate_hungarian_taj(value) is False

    def test_rejects_non_string(self):
        assert validate_hungarian_taj(None) is False

    def test_public_generator_round_trips_validator(self):
        assert HungarianTAJProvider is not None
        for seed in range(50):
            value = generate_hungarian_taj(rng=random.Random(seed))
            assert validate_hungarian_taj(value), (seed, value)

    def test_public_generator_skips_reserved_unknown_person_code(self):
        class SequenceRandom:
            def __init__(self):
                self.values = iter([9, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

            def randint(self, _start, _end):
                return next(self.values)

        value = generate_hungarian_taj(rng=SequenceRandom())

        assert value != "900000007"
        assert validate_hungarian_taj(value)


class TestValidateCzechoslovakRodneCislo:
    """Tests for validate_czechoslovak_rodne_cislo()."""

    def test_valid_slovak_rodne_cislo(self):
        assert validate_czechoslovak_rodne_cislo("850505/1236") is True

    def test_valid_slovak_female_rodne_cislo(self):
        assert validate_czechoslovak_rodne_cislo("855505/1230") is True

    def test_valid_slovak_overflow_series(self):
        assert validate_czechoslovak_rodne_cislo("047521/1231") is True

    def test_valid_slovak_rodne_cislo_without_slash(self):
        assert validate_czechoslovak_rodne_cislo("8505051236") is True

    def test_valid_legacy_nine_digit_rodne_cislo(self):
        assert validate_czechoslovak_rodne_cislo("510505/123") is True

    def test_czech_alias_is_the_shared_validator(self):
        assert validate_czech_rodne_cislo is validate_czechoslovak_rodne_cislo

    def test_invalid_slovak_rodne_cislo_wrong_checksum(self):
        assert validate_czechoslovak_rodne_cislo("850505/1237") is False

    def test_invalid_slovak_rodne_cislo_impossible_date(self):
        assert validate_czechoslovak_rodne_cislo("850231/0003") is False

    def test_rejects_legacy_zero_serial(self):
        assert validate_czechoslovak_rodne_cislo("510505/000") is False

    def test_rejects_remainder_ten_shortcut(self):
        assert int("850505006") % 11 == 10
        assert validate_czechoslovak_rodne_cislo("850505/0060") is False

    @pytest.mark.parametrize(
        "value",
        [None, "", "751116.0008", "xx751116/0008", "751116/0008xx"],
    )
    def test_rejects_unsupported_shapes(self, value):
        assert validate_czechoslovak_rodne_cislo(value) is False

    def test_generated_slovak_surrogate_passes_validator(self):
        assert LANG_TO_LOCALE["sk"] == "sk_SK"

        anonymizer = Anonymizer(lang="sk", consistent=True, seed=42)
        surrogate = anonymizer.surrogate("850505/1236", "national_id")

        assert validate_czechoslovak_rodne_cislo(surrogate) is True


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

    def test_portuguese_patterns_exist(self):
        assert "pt" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["pt"]) > 0

    def test_dutch_patterns_exist(self):
        assert "nl" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["nl"]) > 0

    def test_hindi_patterns_exist(self):
        assert "hi" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["hi"]) > 0

    def test_telugu_patterns_exist(self):
        assert "te" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["te"]) > 0

    def test_arabic_patterns_exist(self):
        assert "ar" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["ar"]) > 0

    def test_hebrew_patterns_exist(self):
        assert "he" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["he"]) > 0

    def test_japanese_patterns_exist(self):
        assert "ja" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["ja"]) > 0

    def test_turkish_patterns_exist(self):
        assert "tr" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["tr"]) > 0

    def test_thai_patterns_exist(self):
        assert "th" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["th"]) > 0

    def test_indonesian_patterns_exist(self):
        assert "id" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["id"]) > 0

    def test_slovak_patterns_exist(self):
        assert "sk" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["sk"]) > 0

    def test_malay_patterns_exist(self):
        assert "ms" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["ms"]) > 0

    def test_tagalog_patterns_exist(self):
        assert "tl" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["tl"]) > 0

    def test_danish_patterns_exist(self):
        assert "da" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["da"]) > 0

    def test_swedish_patterns_exist(self):
        assert "sv" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["sv"]) > 0

    def test_norwegian_patterns_exist(self):
        assert "no" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["no"]) > 0

    def test_korean_patterns_exist(self):
        assert "ko" in LANGUAGE_PII_PATTERNS
        assert len(LANGUAGE_PII_PATTERNS["ko"]) > 0

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

    # Portuguese date patterns
    def test_portuguese_date_slash_or_dash(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["pt"] if p.entity_type == "date"]
        texts = ["15/03/1985", "15-03-1985"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Portuguese date pattern should match '{text}'"

    def test_portuguese_date_month_name(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["pt"] if p.entity_type == "date"]
        text = "15 de mar\u00e7o de 1985"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Portuguese date pattern should match '{text}'"

    # French phone patterns
    def test_french_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["fr"] if p.entity_type == "phone_number"
        ]
        texts = ["+33 6 12 34 56 78", "06 12 34 56 78"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"French phone pattern should match '{text}'"

    # German phone patterns
    def test_german_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["de"] if p.entity_type == "phone_number"
        ]
        texts = ["+49 30 1234567", "030 1234567"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"German phone pattern should match '{text}'"

    # Italian phone patterns
    def test_italian_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["it"] if p.entity_type == "phone_number"
        ]
        texts = ["+39 333 123 4567", "333 123 4567"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Italian phone pattern should match '{text}'"

    # Spanish phone patterns
    def test_spanish_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["es"] if p.entity_type == "phone_number"
        ]
        texts = ["+34 612 345 678", "612 345 678"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Spanish phone pattern should match '{text}'"

    # Portuguese phone patterns
    def test_portuguese_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["pt"] if p.entity_type == "phone_number"
        ]
        texts = ["+351 912 345 678", "+55 11 91234-5678"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Portuguese phone pattern should match '{text}'"

    # National ID patterns
    def test_french_nir_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["fr"] if p.entity_type == "national_id"
        ]
        assert len(patterns) >= 1
        text = "1 85 05 78 006 084 36"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "French NIR pattern should match"

    def test_german_steuer_id_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["de"] if p.entity_type == "national_id"
        ]
        assert len(patterns) >= 1

    def test_italian_codice_fiscale_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["it"] if p.entity_type == "national_id"
        ]
        assert len(patterns) >= 1
        text = "RSSMRA85M01H501Z"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Italian Codice Fiscale pattern should match"

    def test_spanish_dni_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["es"] if p.entity_type == "national_id"
        ]
        assert len(patterns) >= 1
        text = "12345678Z"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Spanish DNI pattern should match"

    def test_spanish_nie_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["es"] if p.entity_type == "national_id"
        ]
        text = "X1234567L"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Spanish NIE pattern should match"

    def test_portuguese_cpf_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["pt"] if p.entity_type == "national_id"
        ]
        text = "123.456.789-09"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Portuguese CPF pattern should match"

    def test_portuguese_cnpj_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["pt"] if p.entity_type == "national_id"
        ]
        text = "11.222.333/0001-81"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Portuguese CNPJ pattern should match"

    def test_portuguese_address_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["pt"] if p.entity_type == "street_address"
        ]
        text = "Rua das Flores 25"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Portuguese address pattern should match"

    def test_portuguese_postcode_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["pt"] if p.entity_type == "postcode"
        ]
        texts = ["1200-195", "01310-100"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Portuguese postcode pattern should match '{text}'"

    def test_dutch_date_month_name(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["nl"] if p.entity_type == "date"]
        text = "15 januari 2020"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Dutch date pattern should match '{text}'"

    def test_hindi_date_month_name(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["hi"] if p.entity_type == "date"]
        text = "15 जनवरी 2020"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Hindi date pattern should match '{text}'"

    def test_telugu_date_month_name(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["te"] if p.entity_type == "date"]
        text = "15 జనవరి 2020"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Telugu date pattern should match '{text}'"

    def test_dutch_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["nl"] if p.entity_type == "phone_number"
        ]
        texts = ["+31 6 12345678", "06 12345678"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Dutch phone pattern should match '{text}'"

    def test_hindi_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["hi"] if p.entity_type == "phone_number"
        ]
        texts = ["+91 9876543210", "9876543210"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Hindi phone pattern should match '{text}'"

    def test_telugu_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["te"] if p.entity_type == "phone_number"
        ]
        texts = ["+91 9876543210", "9988776655"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Telugu phone pattern should match '{text}'"

    def test_dutch_bsn_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["nl"] if p.entity_type == "national_id"
        ]
        text = "123456782"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Dutch BSN pattern should match"

    def test_hindi_pin_code_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["hi"] if p.entity_type == "postcode"
        ]
        text = "110001"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Hindi PIN code pattern should match"

    def test_telugu_pin_code_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["te"] if p.entity_type == "postcode"
        ]
        text = "500001"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Telugu PIN code pattern should match"

    def test_arabic_date_month_name(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["ar"] if p.entity_type == "date"]
        text = "15 \u064a\u0646\u0627\u064a\u0631 2020"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Arabic date pattern should match '{text}'"

    def test_hebrew_date_slash(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["he"] if p.entity_type == "date"]
        text = "15/03/1985"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Hebrew date pattern should match '{text}'"

    def test_hebrew_date_month_name(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["he"] if p.entity_type == "date"]
        text = "15 \u05de\u05e8\u05e5 1985"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Hebrew date pattern should match '{text}'"

    def test_japanese_date_kanji(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["ja"] if p.entity_type == "date"]
        text = "1985\u5e743\u670815\u65e5"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Japanese date pattern should match '{text}'"

    def test_turkish_date_month_name(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["tr"] if p.entity_type == "date"]
        text = "15 Mart 1985"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Turkish date pattern should match '{text}'"

    def test_indonesian_date_slash(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["id"] if p.entity_type == "date"]
        text = "17/08/1985"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Indonesian date pattern should match '{text}'"

    def test_indonesian_date_month_name(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["id"] if p.entity_type == "date"]
        text = "17 Agustus 1985"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Indonesian date pattern should match '{text}'"

    def test_arabic_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ar"] if p.entity_type == "phone_number"
        ]
        text = "+20 10 1234 5678"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Arabic phone pattern should match '{text}'"

    def test_hebrew_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["he"] if p.entity_type == "phone_number"
        ]
        texts = ["+972 54-123-4567", "054-123-4567"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Hebrew phone pattern should match '{text}'"

    def test_japanese_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ja"] if p.entity_type == "phone_number"
        ]
        text = "+81 90 1234 5678"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Japanese phone pattern should match '{text}'"

    def test_turkish_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["tr"] if p.entity_type == "phone_number"
        ]
        text = "+90 532 123 45 67"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Turkish phone pattern should match '{text}'"

    def test_indonesian_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["id"] if p.entity_type == "phone_number"
        ]
        texts = ["+62 812 3456 7890", "0812-3456-7890"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Indonesian phone pattern should match '{text}'"

    def test_arabic_national_id_pattern(self):
        patterns = [
            p for p in get_patterns_for_language("ar") if p.entity_type == "national_id"
        ]
        text = "29801011234567"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Arabic national ID pattern should match"

    def test_hebrew_teudat_zehut_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["he"] if p.entity_type == "national_id"
        ]
        text = "123456782"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Hebrew Teudat Zehut pattern should match"

    def test_hebrew_postcode_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["he"] if p.entity_type == "postcode"
        ]
        texts = ["64239", "6423905"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Hebrew postcode pattern should match '{text}'"

    def test_hebrew_address_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["he"] if p.entity_type == "street_address"
        ]
        text = "\u05e8\u05d7\u05d5\u05d1 \u05d4\u05e8\u05e6\u05dc 12"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Hebrew address pattern should match"

    def test_hebrew_rtl_sample_expected_offsets(self):
        text = (
            "\u05de\u05d8\u05d5\u05e4\u05dc\u05ea: "
            "\u05d3\u05e0\u05d4 \u05db\u05d4\u05df. "
            "\u05ea\u05d0\u05e8\u05d9\u05da \u05dc\u05d9\u05d3\u05d4 "
            "15/03/1985, "
            "\u05d8\u05dc\u05e4\u05d5\u05df +972 54-123-4567, "
            "\u05ea\u05e2\u05d5\u05d3\u05ea \u05d6\u05d4\u05d5\u05ea "
            "123456782, "
            "\u05de\u05d9\u05e7\u05d5\u05d3 6423905, "
            "\u05db\u05ea\u05d5\u05d1\u05ea "
            "\u05e8\u05d7\u05d5\u05d1 \u05d4\u05e8\u05e6\u05dc 12 "
            "\u05ea\u05dc \u05d0\u05d1\u05d9\u05d1."
        )
        expected = {
            "15/03/1985": (28, 38, "date"),
            "+972 54-123-4567": (46, 62, "phone_number"),
            "123456782": (75, 84, "national_id"),
            "6423905": (92, 99, "postcode"),
            "\u05e8\u05d7\u05d5\u05d1 \u05d4\u05e8\u05e6\u05dc 12": (
                107,
                119,
                "street_address",
            ),
        }

        units = find_semantic_units(text, get_patterns_for_language("he"))
        by_text = {
            text[start:end]: (start, end, label) for start, end, label, *_ in units
        }

        for span_text, expected_row in expected.items():
            assert by_text[span_text] == expected_row

    def test_japanese_my_number_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ja"] if p.entity_type == "national_id"
        ]
        text = "1234 5678 9012"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Japanese My Number pattern should match"

    def test_turkish_tckn_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["tr"] if p.entity_type == "national_id"
        ]
        text = "10000000146"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Turkish TCKN pattern should match"

    def test_indonesian_nik_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["id"] if p.entity_type == "national_id"
        ]
        text = "3174055708850001"
        matched = any(
            re.search(p.pattern, text, p.flags) and p.validator(text) for p in patterns
        )
        assert matched, "Indonesian NIK pattern should match and validate"

    def test_indonesian_address_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["id"] if p.entity_type == "street_address"
        ]
        text = "Jl. Merdeka No. 10"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Indonesian address pattern should match"

    def test_indonesian_postcode_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["id"] if p.entity_type == "postcode"
        ]
        text = "40123"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Indonesian postcode pattern should match"

    def test_malay_date_month_name(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["ms"] if p.entity_type == "date"]
        text = "17 Ogos 1985"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Malay date pattern should match '{text}'"

    def test_malay_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ms"] if p.entity_type == "phone_number"
        ]
        texts = ["+60 12-345 6789", "012-345 6789"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Malay phone pattern should match '{text}'"

    def test_malay_mykad_patterns(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ms"] if p.entity_type == "national_id"
        ]
        texts = ["850817-14-5678", "850817145678"]
        for text in texts:
            matched = any(
                re.search(p.pattern, text, p.flags) and p.validator(text)
                for p in patterns
            )
            assert matched, f"Malay MyKad pattern should match and validate '{text}'"

    def test_malay_address_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ms"] if p.entity_type == "street_address"
        ]
        texts = ["Jalan Merdeka 10", "Lorong Damai 5", "Taman Sentosa 12"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Malay address pattern should match '{text}'"

    def test_tagalog_date_month_name(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["tl"] if p.entity_type == "date"]
        text = "17 Agosto 1985"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Tagalog date pattern should match '{text}'"

    def test_tagalog_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["tl"] if p.entity_type == "phone_number"
        ]
        texts = ["+63 917 123 4567", "0917-987-6543"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Tagalog phone pattern should match '{text}'"

    def test_tagalog_philippine_id_patterns(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["tl"] if p.entity_type == "national_id"
        ]
        examples = {
            "1234-5678-9012": validate_philsys_psn,
            "98-765432109-8": validate_philhealth_pin,
        }
        for text, validator in examples.items():
            matched = any(
                re.search(p.pattern, text, p.flags) and validator(text)
                for p in patterns
            )
            assert matched, f"Tagalog national ID pattern should match '{text}'"

    def test_tagalog_address_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["tl"] if p.entity_type == "street_address"
        ]
        texts = ["Barangay Maligaya", "Kalye Rizal 12", "Purok Sampaguita"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Tagalog address pattern should match '{text}'"

    def test_danish_date_month_name(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["da"] if p.entity_type == "date"]
        text = "17 august 1985"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Danish date pattern should match '{text}'"

    def test_danish_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["da"] if p.entity_type == "phone_number"
        ]
        texts = ["+45 20 12 34 56", "30 45 67 89"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Danish phone pattern should match '{text}'"

    @pytest.mark.parametrize("lang", ["da", "no"])
    def test_nordic_phone_patterns_do_not_match_iso_dates(self, lang):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS[lang] if p.entity_type == "phone_number"
        ]

        assert not any(re.search(p.pattern, "1985-08-17", p.flags) for p in patterns)

    def test_danish_cpr_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["da"] if p.entity_type == "national_id"
        ]
        texts = ["170885-1234", "1708851234"]
        for text in texts:
            matched = any(
                re.search(p.pattern, text, p.flags) and p.validator(text)
                for p in patterns
            )
            assert matched, f"Danish CPR pattern should match and validate '{text}'"

    def test_danish_address_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["da"] if p.entity_type == "street_address"
        ]
        texts = ["Bredgade 12", "Roskildevej 45"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Danish address pattern should match '{text}'"

    def test_danish_postcode_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["da"] if p.entity_type == "postcode"
        ]
        texts = ["1260", "DK-8000"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Danish postcode pattern should match '{text}'"

    def test_indonesian_clinical_sample_expected_spans(self):
        text = (
            "Pasien Siti Aminah lahir 17/08/1985. Telepon +62 812 3456 7890. "
            "NIK 3174055708850001. Alamat Jl. Merdeka No. 10, kode pos 40123."
        )
        matches = set()
        for pattern in get_patterns_for_language("id"):
            for match in re.finditer(pattern.pattern, text, pattern.flags):
                value = match.group(0)
                if pattern.validator is not None and not pattern.validator(value):
                    continue
                matches.add((pattern.entity_type, match.start(), match.end(), value))

        assert {
            ("date", 25, 35, "17/08/1985"),
            ("phone_number", 45, 62, "+62 812 3456 7890"),
            ("national_id", 68, 84, "3174055708850001"),
            ("street_address", 93, 111, "Jl. Merdeka No. 10"),
            ("postcode", 122, 127, "40123"),
        } <= matches

    def test_malay_clinical_sample_expected_spans(self):
        text = (
            "Pesakit Nur Aisyah lahir 17/08/1985. Telefon +60 12-345 6789. "
            "MyKad 850817-14-5678. Alamat Jalan Merdeka 10."
        )
        expected = {
            ("date", 25, 35, "17/08/1985"),
            ("phone_number", 45, 60, "+60 12-345 6789"),
            ("national_id", 68, 82, "850817-14-5678"),
            ("street_address", 91, 107, "Jalan Merdeka 10"),
        }
        observed = set()
        for pattern in get_patterns_for_language("ms"):
            for match in re.finditer(pattern.pattern, text, pattern.flags):
                value = match.group(0)
                if pattern.validator is not None and not pattern.validator(value):
                    continue
                observed.add((pattern.entity_type, match.start(), match.end(), value))

        assert expected <= observed

    def test_tagalog_clinical_sample_expected_spans(self):
        text = (
            "Pasyente Maria Santos ipinanganak 17/08/1985. Telepono "
            "+63 917 123 4567. PSN 1234-5678-9012. PhilHealth "
            "98-765432109-8. Tirahan Barangay Maligaya."
        )
        expected = {
            ("date", 34, 44, "17/08/1985"),
            ("phone_number", 55, 71, "+63 917 123 4567"),
            ("national_id", 77, 91, "1234-5678-9012"),
            ("national_id", 104, 118, "98-765432109-8"),
            ("street_address", 128, 145, "Barangay Maligaya"),
        }
        observed = set()
        for pattern in get_patterns_for_language("tl"):
            for match in re.finditer(pattern.pattern, text, pattern.flags):
                value = match.group(0)
                if pattern.validator is not None and not pattern.validator(value):
                    continue
                observed.add((pattern.entity_type, match.start(), match.end(), value))

        assert expected <= observed

    @pytest.mark.parametrize(
        ("lang", "text", "expected"),
        [
            (
                "sv",
                "Patient Anna Andersson född 1985-08-17. Telefon "
                "+46 70 123 45 67. Personnummer 510312-1140. Adress "
                "Södra Förstadsgatan 12, 211 43 Malmö.",
                {
                    ("date", 28, 38, "1985-08-17"),
                    ("phone_number", 48, 64, "+46 70 123 45 67"),
                    ("national_id", 79, 90, "510312-1140"),
                    ("street_address", 99, 121, "Södra Förstadsgatan 12"),
                    ("postcode", 123, 129, "211 43"),
                },
            ),
            (
                "da",
                "Patient Anna Nielsen født 1985-08-17. Telefon "
                "+45 20 12 34 56. CPR 170885-1234. Adresse "
                "Nørrebrogade 12, 2200 København.",
                {
                    ("date", 26, 36, "1985-08-17"),
                    ("phone_number", 46, 61, "+45 20 12 34 56"),
                    ("national_id", 67, 78, "170885-1234"),
                    ("street_address", 88, 103, "Nørrebrogade 12"),
                    ("postcode", 105, 109, "2200"),
                },
            ),
            (
                "no",
                "Pasient Ingrid Hansen født 1985-08-17. Telefon "
                "+47 91 23 45 67. Fødselsnummer 12035101460. Adresse "
                "Dronning Eufemias gate 16, 0154 Oslo.",
                {
                    ("date", 27, 37, "1985-08-17"),
                    ("phone_number", 47, 62, "+47 91 23 45 67"),
                    ("national_id", 78, 89, "12035101460"),
                    ("street_address", 99, 124, "Dronning Eufemias gate 16"),
                    ("postcode", 126, 130, "0154"),
                },
            ),
        ],
    )
    def test_nordic_clinical_samples_expected_spans(self, lang, text, expected):
        observed = set()
        for pattern in get_patterns_for_language(lang):
            for match in re.finditer(pattern.pattern, text, pattern.flags):
                value = match.group(0)
                if pattern.validator is not None and not pattern.validator(value):
                    continue
                observed.add((pattern.entity_type, match.start(), match.end(), value))

        assert expected <= observed

    def test_turkish_address_with_turkish_letters(self):
        # Ş, ı, İ, ğ live in Latin Extended-A; the regex must accept them
        # or real Turkish street names won't match.
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["tr"] if p.entity_type == "street_address"
        ]
        samples = [
            "Cadde Şehit Pilot 5",  # "Şehit"
            "Sokak İnönü 12",  # "İnönü"
            "Mahalle Yıldız 3",  # "Yıldız"
            "Atatürk Caddesi 12",
            "İstiklal Sokak 45",
        ]
        for text in samples:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Turkish address pattern should match '{text}'"

    def test_thai_date_month_name(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["th"] if p.entity_type == "date"]
        text = "15 มกราคม 2567"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Thai date pattern should match '{text}'"

    def test_thai_phone(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["th"] if p.entity_type == "phone_number"
        ]
        texts = ["+66 81 234 5678", "081-234-5678"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Thai phone pattern should match '{text}'"

    def test_thai_national_id_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["th"] if p.entity_type == "national_id"
        ]
        text = "1101700203450"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Thai national ID pattern should match"

    def test_thai_address_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["th"] if p.entity_type == "street_address"
        ]
        text = "123 ถนนสุขุมวิท แขวงคลองตัน เขตคลองเตย กรุงเทพฯ"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Thai address pattern should match"

    def test_thai_postcode_pattern(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["th"] if p.entity_type == "postcode"
        ]
        text = "10110"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, "Thai postcode pattern should match"

    def test_thai_jsonl_fixture_matches_expected_offsets(self):
        fixture_path = (
            Path(__file__).resolve().parents[2]
            / "openmed/eval/golden/fixtures/i18n/th.jsonl"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8").strip())
        text = fixture["text"]

        expected = {
            (span["label"], span["start"], span["end"], span["text"])
            for span in fixture["gold_spans"]
        }
        observed = set()
        for pattern in LANGUAGE_PII_PATTERNS["th"]:
            for match in re.finditer(pattern.pattern, text, pattern.flags):
                if pattern.validator and not pattern.validator(match.group(0)):
                    continue
                observed.add(
                    (
                        {
                            "date": "DATE",
                            "phone_number": "PHONE",
                            "national_id": "ID_NUM",
                            "street_address": "STREET_ADDRESS",
                            "postcode": "ZIPCODE",
                        }[pattern.entity_type],
                        match.start(),
                        match.end(),
                        match.group(0),
                    )
                )

        assert expected <= observed

    def test_arabic_phone_rejects_bare_digit_strings(self):
        # The old pattern would match the 14-digit national-ID and any other
        # 5–13-digit number. The tightened pattern requires +CC or a leading 0.
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ar"] if p.entity_type == "phone_number"
        ]
        non_phone_samples = [
            "29801011234567",  # Egyptian national_id format
            "1234567890",  # generic 10-digit string
            "20101234 5678",  # missing the required '+'
        ]
        for text in non_phone_samples:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert not matched, f"Arabic phone pattern should NOT match '{text}'"

    def test_arabic_phone_accepts_local_leading_zero(self):
        # Egyptian local mobile format starts with 0 (no +20 prefix).
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ar"] if p.entity_type == "phone_number"
        ]
        text = "010 1234 5678"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Arabic phone pattern should match local format '{text}'"

    def test_slovak_clinical_sample_expected_spans(self):
        text = (
            "Pacientka: Jana Kovacova. Datum narodenia 05.05.1985, "
            "telefon +421 903 123 456, rodne cislo 855505/1230, "
            "adresa Hlavna ulica 12, PSC 81101."
        )
        expected = {
            ("date", 42, 52, "05.05.1985"),
            ("phone_number", 62, 78, "+421 903 123 456"),
            ("national_id", 92, 103, "855505/1230"),
            ("street_address", 112, 127, "Hlavna ulica 12"),
            ("postcode", 133, 138, "81101"),
        }
        observed = set()
        for pattern in get_patterns_for_language("sk"):
            for match in re.finditer(pattern.pattern, text, pattern.flags):
                value = match.group(0)
                if pattern.validator is not None and not pattern.validator(value):
                    continue
                observed.add((pattern.entity_type, match.start(), match.end(), value))

        assert expected <= observed

    def test_hungarian_clinical_sample_expected_spans(self):
        text = (
            "Beteg: Kovács Anna. Születési dátum: 1985. május 5. "
            "Telefon: +36 30 123 4567. TAJ-szám: 123 456 788. "
            "Lakcím: Kossuth Lajos utca 12, irányítószám: 1051 Budapest."
        )
        expected = {
            ("date", 37, 51, "1985. május 5."),
            ("phone_number", 61, 76, "+36 30 123 4567"),
            ("national_id", 88, 99, "123 456 788"),
            ("street_address", 109, 130, "Kossuth Lajos utca 12"),
            ("postcode", 146, 150, "1051"),
        }
        observed = set()
        for pattern in get_patterns_for_language("hu"):
            for match in re.finditer(pattern.pattern, text, pattern.flags):
                value = match.group(0)
                if pattern.validator is not None and not pattern.validator(value):
                    continue
                observed.add((pattern.entity_type, match.start(), match.end(), value))

        assert expected <= observed

    def test_hungarian_taj_pattern_rejects_bad_checksum(self):
        patterns = [
            pattern
            for pattern in LANGUAGE_PII_PATTERNS["hu"]
            if pattern.entity_type == "national_id"
        ]
        assert patterns, "Hungarian pack must expose a national_id pattern"
        for pattern in patterns:
            matches = list(
                re.finditer(pattern.pattern, "TAJ: 123 456 789", pattern.flags)
            )
            assert matches, "Hungarian TAJ pattern must recognize formatted candidates"
            for match in matches:
                assert pattern.validator is not None
                assert not pattern.validator(match.group(0))

    def test_hungarian_taj_safety_sweep_requires_context(self):
        patterns = [
            pattern
            for pattern in LANGUAGE_PII_PATTERNS["hu"]
            if pattern.entity_type == "national_id"
        ]

        assert patterns
        assert all(pattern.safety_sweep_requires_context for pattern in patterns)

    def test_romanian_clinical_sample_expected_spans(self):
        text = (
            "Pacient: Ana Popescu, nascuta 12 martie 1985. "
            "Telefon +40 721 234 567. CNP 1800101400181. "
            "Adresa Str. Mihai Eminescu 12, cod postal 010011 Bucuresti."
        )
        expected = {
            ("date", 30, 44, "12 martie 1985"),
            ("phone_number", 54, 69, "+40 721 234 567"),
            ("national_id", 75, 88, "1800101400181"),
            ("street_address", 97, 119, "Str. Mihai Eminescu 12"),
            ("postcode", 132, 138, "010011"),
        }
        observed = set()
        for pattern in get_patterns_for_language("ro"):
            for match in re.finditer(pattern.pattern, text, pattern.flags):
                value = match.group(0)
                if pattern.validator is not None and not pattern.validator(value):
                    continue
                observed.add((pattern.entity_type, match.start(), match.end(), value))

        assert expected <= observed

    def test_romanian_diacritic_address_matches(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ro"] if p.entity_type == "street_address"
        ]
        samples = [
            "Șoseaua Ștefan cel Mare 15",
            "Şoseaua Ştefan cel Mare 15",
            "S\u0326oseaua S\u0326tefan cel Mare 15",
            "Str. Gheorghe Doja 7",
            "Bulevardul Dacia 100",
            "Calea Moșilor 24",
        ]
        for sample in samples:
            matched = any(re.search(p.pattern, sample, p.flags) for p in patterns)
            assert matched, f"Romanian address pattern should match '{sample}'"

    def test_romanian_cnp_pattern_rejects_bad_checksum(self):
        # The 13-digit pattern only survives the validator gate for valid CNPs.
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ro"] if p.entity_type == "national_id"
        ]
        assert patterns, "Romanian pack must expose a national_id pattern"
        corrupted = "1800101400182"  # last digit off by one from a valid CNP
        for pattern in patterns:
            for match in re.finditer(pattern.pattern, corrupted, pattern.flags):
                assert pattern.validator is not None
                assert not pattern.validator(match.group(0))

    ### Korean language specific PII Pattern test

    # ── Date patterns ──────────────────────────────────────────────────────

    def test_korean_date_native_format(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["ko"] if p.entity_type == "date"]
        texts = ["1994년 3월 15일", "2000년 1월 1일", "1985년 12월 31일"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Korean date pattern should match '{text}'"

    def test_korean_date_numeric_dot(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["ko"] if p.entity_type == "date"]
        texts = ["1994.03.15", "2000.1.1"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Korean numeric date pattern should match '{text}'"

    def test_korean_date_numeric_hyphen(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["ko"] if p.entity_type == "date"]
        texts = ["1994-03-15", "2000-1-1"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Korean hyphen date pattern should match '{text}'"

    def test_korean_date_numeric_slash(self):
        patterns = [p for p in LANGUAGE_PII_PATTERNS["ko"] if p.entity_type == "date"]
        texts = ["1994/03/15", "2000/1/1"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Korean slash date pattern should match '{text}'"

    # ── Phone patterns ─────────────────────────────────────────────────────

    def test_korean_phone_mobile(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ko"] if p.entity_type == "phone_number"
        ]
        texts = ["010-1234-5678", "010 1234 5678", "01012345678"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Korean mobile phone pattern should match '{text}'"

    def test_korean_phone_plus82(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ko"] if p.entity_type == "phone_number"
        ]
        texts = ["+82-10-1234-5678", "+82 10 1234 5678"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Korean +82 phone pattern should match '{text}'"

    def test_korean_phone_landline(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ko"] if p.entity_type == "phone_number"
        ]
        texts = ["02-1234-5678", "031-123-4567"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Korean landline pattern should match '{text}'"

    # ── RRN / National ID patterns ─────────────────────────────────────────

    def test_korean_rrn_with_hyphen(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ko"] if p.entity_type == "national_id"
        ]
        text = "940315-1234567"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Korean RRN pattern should match '{text}'"

    def test_korean_rrn_without_hyphen(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ko"] if p.entity_type == "national_id"
        ]
        text = "9403151234567"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Korean RRN without hyphen should match '{text}'"

    def test_korean_rrn_validator_wired(self):
        # validator=validate_korean_rrn must be set on the national_id pattern
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ko"] if p.entity_type == "national_id"
        ]
        assert len(patterns) >= 1
        assert any(p.validator is not None for p in patterns), (
            "Korean national_id pattern must have a validator wired"
        )

    def test_korean_rrn_invalid_checksum_rejected(self):
        from openmed.core.pii_i18n import validate_korean_rrn

        assert validate_korean_rrn("940315-1234568") is False

    # ── Street address patterns ────────────────────────────────────────────

    def test_korean_street_address_ro(self):
        # 로 (ro) = road suffix
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ko"] if p.entity_type == "street_address"
        ]
        text = "서울시 강남구 테헤란로 123"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Korean street address pattern should match '{text}'"

    def test_korean_street_address_gil(self):
        # 길 (gil) = street/alley suffix
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ko"] if p.entity_type == "street_address"
        ]
        text = "부산시 해운대구 해운대길 45"
        matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Korean gil address pattern should match '{text}'"

    def test_korean_street_address_dong(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ko"] if p.entity_type == "street_address"
        ]
        text = "서울특별시 강남구 역삼동 123-45"
        matched = any(re.fullmatch(p.pattern, text, p.flags) for p in patterns)
        assert matched, f"Korean dong address pattern should match '{text}'"

    def test_korean_street_address_dong_requires_administrative_context(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ko"] if p.entity_type == "street_address"
        ]
        text = "역삼동 123-45"
        matched = any(re.fullmatch(p.pattern, text, p.flags) for p in patterns)
        assert not matched, "A standalone dong and number must not match an address"

    # ── Postcode patterns ──────────────────────────────────────────────────

    def test_korean_postcode(self):
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ko"] if p.entity_type == "postcode"
        ]
        texts = ["06292", "12345", "00100"]
        for text in texts:
            matched = any(re.search(p.pattern, text, p.flags) for p in patterns)
            assert matched, f"Korean postcode pattern should match '{text}'"

    def test_korean_postcode_not_six_digits(self):
        # 6-digit numbers should not match the 5-digit postcode pattern
        patterns = [
            p for p in LANGUAGE_PII_PATTERNS["ko"] if p.entity_type == "postcode"
        ]
        text = "123456"
        matched = any(re.fullmatch(p.pattern, text, p.flags) for p in patterns)
        assert not matched, "Korean postcode pattern should not match 6-digit number"


# ---------------------------------------------------------------------------
# get_patterns_for_language Tests
# ---------------------------------------------------------------------------


class TestGetPatternsForLanguage:
    """Tests for get_patterns_for_language()."""

    def test_english_returns_base_patterns(self):
        patterns = get_patterns_for_language("en")
        assert len(patterns) == (
            len(PII_PATTERNS)
            + len(MRZ_PII_PATTERNS)
            + len(USCC_PII_PATTERNS)
            + len(AADHAAR_PII_PATTERNS)
            + len(INDIA_HEALTH_ID_PII_PATTERNS)
        )

    @pytest.mark.parametrize(
        "lang",
        sorted((SUPPORTED_LANGUAGES | NATIONAL_ID_ONLY_LANGUAGES) - {"en"}),
    )
    def test_language_includes_universal_base_and_language_patterns(self, lang):
        patterns = get_patterns_for_language(lang)
        base_count = (
            len(PII_PATTERNS)
            + len(MRZ_PII_PATTERNS)
            + len(USCC_PII_PATTERNS)
            + len(AADHAAR_PII_PATTERNS)
            + len(INDIA_HEALTH_ID_PII_PATTERNS)
        )
        universal_patterns = (
            MRZ_PII_PATTERNS
            + USCC_PII_PATTERNS
            + AADHAAR_PII_PATTERNS
            + INDIA_HEALTH_ID_PII_PATTERNS
        )
        language_count = sum(
            not any(pattern is universal for universal in universal_patterns)
            for pattern in LANGUAGE_PII_PATTERNS[lang]
        )
        locale_patterns = LOCALE_PII_PATTERNS.get(lang, [])
        locale_count = sum(
            not any(pattern is existing for existing in LANGUAGE_PII_PATTERNS[lang])
            for pattern in locale_patterns
        )
        assert len(patterns) == base_count + language_count + locale_count

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            get_patterns_for_language("xx")

    @pytest.mark.parametrize("lang", ("bn", "ta", "zh"))
    def test_v2_language_patterns_are_discoverable(self, lang):
        patterns = get_patterns_for_language(lang)

        assert patterns
        assert all(isinstance(pattern, PIIPattern) for pattern in patterns)

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

    @pytest.mark.parametrize("lang", ("bn", "ta", "zh"))
    def test_v2_language_fake_data_is_synthetic_and_script_specific(self, lang):
        data = LANGUAGE_FAKE_DATA[lang]

        assert data["NAME"]
        assert data["PHONE"]
        assert data["DATE"]
        assert data["LOCATION"]
        assert all("example" in value for value in data["EMAIL"])

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

    def test_portuguese_names_are_portuguese(self):
        names = LANGUAGE_FAKE_DATA["pt"]["NAME"]
        assert any("Silva" in n or "Almeida" in n for n in names)

    def test_dutch_names_are_dutch(self):
        names = LANGUAGE_FAKE_DATA["nl"]["NAME"]
        assert any("de Vries" in n or "Jansen" in n for n in names)

    def test_hindi_names_are_hindi(self):
        names = LANGUAGE_FAKE_DATA["hi"]["NAME"]
        assert any(
            "\u0936\u0930\u094d\u092e\u093e" in n
            or "\u0915\u0941\u092e\u093e\u0930" in n
            for n in names
        )

    def test_telugu_names_are_telugu(self):
        names = LANGUAGE_FAKE_DATA["te"]["NAME"]
        assert any(
            "\u0c30\u0c46\u0c21\u0c4d\u0c21\u0c3f" in n
            or "\u0c15\u0c41\u0c2e\u0c3e\u0c30\u0c4d" in n
            for n in names
        )

    def test_arabic_names_are_arabic(self):
        names = LANGUAGE_FAKE_DATA["ar"]["NAME"]
        assert any(
            "\u062d\u0633\u0646" in n or "\u0639\u0644\u064a" in n for n in names
        )

    def test_hebrew_names_are_hebrew(self):
        names = LANGUAGE_FAKE_DATA["he"]["NAME"]
        assert any(
            "\u05db\u05d4\u05df" in n or "\u05dc\u05d5\u05d9" in n for n in names
        )

    def test_japanese_names_are_japanese(self):
        names = LANGUAGE_FAKE_DATA["ja"]["NAME"]
        assert any("\u4f50\u85e4" in n or "\u7530\u4e2d" in n for n in names)

    def test_turkish_names_are_turkish(self):
        names = LANGUAGE_FAKE_DATA["tr"]["NAME"]
        assert any("Y\u0131lmaz" in n or "Kaya" in n for n in names)

    def test_thai_names_are_thai(self):
        names = LANGUAGE_FAKE_DATA["th"]["NAME"]
        assert any("ใจดี" in n or "แก้วใส" in n for n in names)

    def test_indonesian_names_are_indonesian(self):
        names = LANGUAGE_FAKE_DATA["id"]["NAME"]
        assert any("Siti" in n or "Santoso" in n for n in names)

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

    def test_portuguese_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["pt"]["PHONE"]
        assert any("+351" in p or "+55" in p for p in phones)

    def test_dutch_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["nl"]["PHONE"]
        assert any("+31" in p or p.startswith("06") for p in phones)

    def test_hindi_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["hi"]["PHONE"]
        assert any("+91" in p or len(p) == 10 for p in phones)

    def test_telugu_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["te"]["PHONE"]
        assert any("+91" in p or len(p) == 10 for p in phones)

    def test_arabic_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["ar"]["PHONE"]
        assert any("+20" in p or "+966" in p for p in phones)

    def test_hebrew_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["he"]["PHONE"]
        assert any("+972" in p or p.startswith("05") for p in phones)

    def test_japanese_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["ja"]["PHONE"]
        assert any("+81" in p or p.startswith("03") for p in phones)

    def test_turkish_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["tr"]["PHONE"]
        assert any("+90" in p or p.startswith("0") for p in phones)

    def test_thai_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["th"]["PHONE"]
        assert any("+66" in p or p.startswith("0") for p in phones)

    def test_indonesian_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["id"]["PHONE"]
        assert any("+62" in p or p.startswith("0") for p in phones)

    def test_malay_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["ms"]["PHONE"]
        assert any("+60" in p or p.startswith("0") for p in phones)

    def test_tagalog_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["tl"]["PHONE"]
        assert any("+63" in p or p.startswith("0") for p in phones)

    def test_danish_phones_have_country_code(self):
        phones = LANGUAGE_FAKE_DATA["da"]["PHONE"]
        assert any("+45" in p or len(re.sub(r"[^0-9]", "", p)) == 8 for p in phones)

    # Korean fake data test
    def test_korean_names_are_korean(self):
        names = LANGUAGE_FAKE_DATA["ko"]["NAME"]
        assert any("김" in n or "이" in n or "박" in n for n in names)

    def test_korean_phones_have_country_code_or_local(self):
        phones = LANGUAGE_FAKE_DATA["ko"]["PHONE"]
        assert any(
            "+82" in p or p.startswith("010") or p.startswith("02") for p in phones
        )


class TestIndonesianLocaleAndFixture:
    """Tests for Indonesian locale and golden fixture wiring."""

    def test_locale_and_surrogate_nik_round_trip(self):
        assert LANG_TO_LOCALE["id"] == "id_ID"
        anon = Anonymizer(lang="id", consistent=True, seed=42)

        surrogate = anon.surrogate("3174055708850001", "national_id")

        assert validate_indonesian_nik(surrogate) is True

    def test_i18n_golden_fixture_offsets(self):
        fixture_path = Path("openmed/eval/golden/fixtures/i18n/id.jsonl")
        rows = [
            json.loads(line)
            for line in fixture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert len(rows) == 1
        row = rows[0]
        assert row["language"] == "id"
        assert row["metadata"]["synthetic"] is True
        assert row["metadata"]["category"] == "multilingual"

        text = row["text"]
        expected = {
            ("DATE", 25, 35, "17/08/1985"),
            ("PHONE", 45, 62, "+62 812 3456 7890"),
            ("ID_NUM", 68, 84, "3174055708850001"),
            ("STREET_ADDRESS", 93, 111, "Jl. Merdeka No. 10"),
            ("ZIPCODE", 122, 127, "40123"),
        }
        actual = {
            (span["label"], span["start"], span["end"], span["text"])
            for span in row["gold_spans"]
        }
        assert actual == expected
        for label, start, end, value in actual:
            assert text[start:end] == value, label


class TestVietnameseLanguagePack:
    """Tests for Vietnamese rule-based PII and synthetic data wiring."""

    @pytest.mark.parametrize(
        "value",
        (
            "001203123456",
            "001 203 123 456",
            "079-199-654-321",
            "003203123456",
        ),
    )
    def test_valid_cccd_structures(self, value):
        assert validate_vietnamese_cccd(value)

    @pytest.mark.parametrize(
        "value",
        (
            "00120312345",
            "001 203 123 45",
            "001.203.123.456",
            "001 203-123 456",
        ),
    )
    def test_invalid_cccd_lengths_or_groupings(self, value):
        assert not validate_vietnamese_cccd(value)

    @pytest.mark.parametrize("value", ("123456789", "123 456 789", "123-456-789"))
    def test_valid_cmnd_structures(self, value):
        assert validate_vietnamese_cmnd(value)

    @pytest.mark.parametrize("value", ("111111111", "000000000", "12345678"))
    def test_invalid_cmnd_structures(self, value):
        assert not validate_vietnamese_cmnd(value)

    def test_locale_registry_and_cccd_surrogate_round_trip(self):
        assert "vi" in NATIONAL_ID_ONLY_LANGUAGES
        assert "vi" not in DEFAULT_PII_MODELS
        assert LANG_TO_LOCALE["vi"] == "vi_VN"
        assert LANGUAGE_PII_PATTERNS["vi"]

        anon = Anonymizer(lang="vi", consistent=True, seed=42)
        surrogate = anon.surrogate("001203123456", "national_id")
        assert validate_vietnamese_cccd(surrogate)

    def test_synthetic_generators_cover_cccd_and_cmnd(self):
        assert validate_vietnamese_cccd(generate_vietnamese_cccd())
        assert validate_vietnamese_cmnd(generate_vietnamese_cmnd())

    def test_vietnamese_fake_data_has_native_values(self):
        fake_data = LANGUAGE_FAKE_DATA["vi"]
        assert any("Nguyễn" in name for name in fake_data["NAME"])
        assert any(phone.startswith(("+84", "0")) for phone in fake_data["PHONE"])
        assert all(len(postcode) == 5 for postcode in fake_data["ZIPCODE"])

    def test_patterns_cover_required_vietnamese_shapes(self):
        samples = {
            "date": ("17/08/1985", "ngày 1 tháng 1 năm 2000"),
            "phone_number": ("+84 912 345 678", "0912-345-678", "028 3822 1234"),
            "national_id": ("001203123456", "123456789"),
            "street_address": ("12 đường Nguyễn Trãi", "45 pho Tran Hung Dao"),
            "postcode": ("10000",),
        }

        for entity_type, values in samples.items():
            patterns = [
                pattern
                for pattern in LANGUAGE_PII_PATTERNS["vi"]
                if pattern.entity_type == entity_type
            ]
            for value in values:
                assert any(
                    match
                    and (pattern.validator is None or pattern.validator(match.group(0)))
                    for pattern in patterns
                    if (match := re.search(pattern.pattern, value, pattern.flags))
                ), f"Vietnamese {entity_type} pattern should match {value!r}"

    def test_context_gates_cccd_cmnd_postcode_and_landline(self):
        from openmed.core.safety_sweep import safety_sweep

        contextual = (
            "CCCD: 003203123456; CMND: 123456789; "
            "điện thoại 028 3822 1234; mã bưu chính 10000"
        )
        entities = safety_sweep(contextual, [], lang="vi")
        values = {entity.text for entity in entities}
        assert {
            "003203123456",
            "123456789",
            "028 3822 1234",
            "10000",
        } <= values

        hard_negative = (
            "Mã bệnh án 003203123456; mã xét nghiệm 123456789; "
            "mã thuốc 10000; lô sản xuất 02838221234."
        )
        negative_entities = safety_sweep(hard_negative, [], lang="vi")
        assert not any(
            entity.text in {"003203123456", "123456789", "10000", "02838221234"}
            for entity in negative_entities
        )

    def test_golden_fixture_offsets_and_offline_deidentification(self):
        from openmed.core.pii import (
            _apply_safety_sweep_to_result,
            _build_deidentification_result,
        )
        from openmed.eval.golden import GoldenFixture
        from openmed.processing.outputs import PredictionResult

        fixture_path = Path("openmed/eval/golden/fixtures/i18n/vi.jsonl")
        row = json.loads(fixture_path.read_text(encoding="utf-8").strip())
        fixture = GoldenFixture.from_mapping(row)
        assert fixture.language == "vi"

        for span in fixture.gold_spans:
            assert fixture.text[span.start : span.end] == span.text

        empty_result = PredictionResult(
            text=fixture.text,
            entities=[],
            model_name="offline-safety-sweep",
            timestamp="2026-07-15T00:00:00Z",
            metadata={},
        )
        swept_result, added_count = _apply_safety_sweep_to_result(
            fixture.text,
            empty_result,
            lang="vi",
        )
        result = _build_deidentification_result(
            fixture.text,
            swept_result,
            effective_method="mask",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang="vi",
            consistent=False,
            seed=None,
            locale=None,
            use_safety_sweep=True,
        )

        assert added_count == len(fixture.gold_spans)
        for span in fixture.gold_spans:
            assert span.text not in result.deidentified_text


class TestMalayLocaleAndFixture:
    """Tests for Malay locale and golden fixture wiring."""

    def test_locale_and_surrogate_mykad_round_trip(self):
        assert LANG_TO_LOCALE["ms"] == "ms_MY"
        anon = Anonymizer(lang="ms", consistent=True, seed=42)

        surrogate = anon.surrogate("850817-14-5678", "national_id")

        assert validate_malaysian_mykad(surrogate) is True

    def test_i18n_golden_fixture_offsets(self):
        fixture_path = Path("openmed/eval/golden/fixtures/i18n/ms.jsonl")
        rows = [
            json.loads(line)
            for line in fixture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert len(rows) == 1
        row = rows[0]
        assert row["language"] == "ms"
        assert row["metadata"]["synthetic"] is True
        assert row["metadata"]["category"] == "multilingual"

        text = row["text"]
        expected = {
            ("DATE", 25, 35, "17/08/1985"),
            ("PHONE", 45, 60, "+60 12-345 6789"),
            ("ID_NUM", 68, 82, "850817-14-5678"),
            ("STREET_ADDRESS", 91, 107, "Jalan Merdeka 10"),
        }
        actual = {
            (span["label"], span["start"], span["end"], span["text"])
            for span in row["gold_spans"]
        }
        assert actual == expected
        for label, start, end, value in actual:
            assert text[start:end] == value, label


class TestTagalogLocaleAndFixture:
    """Tests for Tagalog/Filipino locale and golden fixture wiring."""

    def test_locale_and_surrogate_philsys_round_trip(self):
        assert LANG_TO_LOCALE["tl"] == "fil_PH"
        anon = Anonymizer(lang="tl", consistent=True, seed=42)

        surrogate = anon.surrogate("1234-5678-9012", "national_id")

        assert validate_philsys_psn(surrogate) is True

    def test_i18n_golden_fixture_offsets(self):
        fixture_path = Path("openmed/eval/golden/fixtures/i18n/tl.jsonl")
        rows = [
            json.loads(line)
            for line in fixture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert len(rows) == 1
        row = rows[0]
        assert row["language"] == "tl"
        assert row["metadata"]["synthetic"] is True
        assert row["metadata"]["category"] == "multilingual"

        text = row["text"]
        expected = {
            ("DATE", 34, 44, "17/08/1985"),
            ("PHONE", 55, 71, "+63 917 123 4567"),
            ("ID_NUM", 77, 91, "1234-5678-9012"),
            ("ID_NUM", 104, 118, "98-765432109-8"),
            ("STREET_ADDRESS", 128, 145, "Barangay Maligaya"),
        }
        actual = {
            (span["label"], span["start"], span["end"], span["text"])
            for span in row["gold_spans"]
        }
        assert actual == expected
        for label, start, end, value in actual:
            assert text[start:end] == value, label


class TestNordicLocaleAndFixtures:
    """Tests for Nordic locale, surrogate, and golden fixture wiring."""

    @pytest.mark.parametrize(
        ("lang", "locale", "original", "validator"),
        [
            ("sv", "sv_SE", "510312-1140", validate_swedish_personnummer),
            ("da", "da_DK", "170885-1234", validate_danish_cpr),
            ("no", "no_NO", "12035101460", validate_norwegian_fodselsnummer),
        ],
    )
    def test_locale_and_national_id_surrogate_round_trip(
        self,
        lang,
        locale,
        original,
        validator,
    ):
        assert LANG_TO_LOCALE[lang] == locale
        anonymizer = Anonymizer(lang=lang, consistent=True, seed=42)

        surrogate = anonymizer.surrogate(original, "national_id")

        assert validator(surrogate) is True

    @pytest.mark.parametrize("lang", ["sv", "da", "no"])
    def test_i18n_golden_fixture_offsets_and_patterns(self, lang):
        fixture_path = Path(f"openmed/eval/golden/fixtures/i18n/{lang}.jsonl")
        rows = [
            json.loads(line)
            for line in fixture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert len(rows) == 1
        row = rows[0]
        assert row["language"] == lang
        assert row["metadata"]["synthetic"] is True
        assert row["metadata"]["category"] == "multilingual"

        text = row["text"]
        observed = set()
        for pattern in get_patterns_for_language(lang):
            for match in re.finditer(pattern.pattern, text, pattern.flags):
                value = match.group(0)
                if pattern.validator is not None and not pattern.validator(value):
                    continue
                observed.add((pattern.entity_type, match.start(), match.end(), value))

        label_to_entity = {
            "DATE": "date",
            "PHONE": "phone_number",
            "ID_NUM": "national_id",
            "STREET_ADDRESS": "street_address",
            "ZIPCODE": "postcode",
        }
        for span in row["gold_spans"]:
            value = span["text"]
            assert text[span["start"] : span["end"]] == value
            assert (
                label_to_entity[span["label"]],
                span["start"],
                span["end"],
                value,
            ) in observed


def test_validate_latvian_personas_kods():
    assert validate_latvian_personas_kods("161175-19997")
    assert validate_latvian_personas_kods("010101-12343")
    assert validate_latvian_personas_kods("32867300679")
    assert validate_latvian_personas_kods("328673-00679")

    assert not validate_latvian_personas_kods("161175-19998")
    assert not validate_latvian_personas_kods("32867300677")
    assert not validate_latvian_personas_kods("161375-19997")
    assert not validate_latvian_personas_kods("abcdef")
    assert not validate_latvian_personas_kods("123")


def test_generated_latvian_surrogate_passes_validator():
    assert LANG_TO_LOCALE["lv"] == "lv_LV"

    anonymizer = Anonymizer(lang="lv", consistent=True, seed=42)
    surrogate = anonymizer.surrogate("161175-19997", "national_id")

    assert validate_latvian_personas_kods(surrogate) is True


def test_latvian_clinical_sample_expected_spans():
    text = (
        "Pacients: Anna Kalnina. Dzimsanas datums 16.11.1975, "
        "telefons +371 2123 4567, personas kods 161175-19997, "
        "adrese Brivibas iela 12, pasta indekss LV-1010."
    )
    expected = {
        ("date", 41, 51, "16.11.1975"),
        ("phone_number", 62, 76, "+371 2123 4567"),
        ("national_id", 92, 104, "161175-19997"),
        ("street_address", 113, 129, "Brivibas iela 12"),
        ("postcode", 145, 152, "LV-1010"),
    }
    observed = set()
    for pattern in get_patterns_for_language("lv"):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is not None and not pattern.validator(value):
                continue
            observed.add((pattern.entity_type, match.start(), match.end(), value))

    assert expected <= observed


def test_latvian_i18n_golden_fixture_offsets():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/lv.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    row = rows[0]
    assert row["language"] == "lv"
    assert row["metadata"]["synthetic"] is True
    assert row["metadata"]["category"] == "multilingual"

    text = row["text"]
    expected = {
        ("DATE", 41, 51, "16.11.1975"),
        ("PHONE", 62, 76, "+371 2123 4567"),
        ("ID_NUM", 92, 104, "161175-19997"),
        ("STREET_ADDRESS", 113, 129, "Brivibas iela 12"),
        ("ZIPCODE", 145, 152, "LV-1010"),
    }
    actual = {
        (span["label"], span["start"], span["end"], span["text"])
        for span in row["gold_spans"]
    }
    assert actual == expected
    for label, start, end, value in actual:
        assert text[start:end] == value, label


def test_validate_estonian_isikukood():
    # Standard first-pass checksums.
    assert validate_estonian_isikukood("47511160002")
    assert validate_estonian_isikukood("38205210123")
    assert validate_estonian_isikukood("60409032208")
    # Second-pass weight fallback (first pass leaves remainder 10).
    assert validate_estonian_isikukood("47511160214")
    assert validate_estonian_isikukood("18503070027")
    # Both passes leave remainder 10, so the check digit falls back to 0.
    assert validate_estonian_isikukood("47511160080")
    assert validate_estonian_isikukood("18503071660")

    # Bad check digit.
    assert not validate_estonian_isikukood("47511160003")
    # Checksum-valid values with impossible embedded dates.
    assert not validate_estonian_isikukood("47511320008")
    assert not validate_estonian_isikukood("47502290008")
    # Checksum-valid values with invalid century/sex first digits.
    assert not validate_estonian_isikukood("77511160005")
    assert not validate_estonian_isikukood("87511160006")
    assert not validate_estonian_isikukood("97511160007")
    # Isikukood has no formatted representation; separators are invalid.
    assert not validate_estonian_isikukood("475-111-60002")
    assert not validate_estonian_isikukood(None)
    assert not validate_estonian_isikukood("abcdef")
    assert not validate_estonian_isikukood("123")


def test_generated_estonian_isikukood_round_trips_validator():
    rng = random.Random(1234)
    for _ in range(200):
        assert validate_estonian_isikukood(generate_estonian_isikukood(rng=rng))


def test_generated_estonian_surrogate_passes_validator():
    assert LANG_TO_LOCALE["et"] == "et_EE"

    anonymizer = Anonymizer(lang="et", consistent=True, seed=42)
    surrogate = anonymizer.surrogate("47511160002", "national_id")

    assert validate_estonian_isikukood(surrogate) is True


def test_estonian_clinical_sample_expected_spans():
    text = (
        "Patsient: Mari Tamm. Sunniaeg 16.11.1975, "
        "telefon +372 5123 4567, isikukood 47511160002, "
        "aadress Pikk tanav 12, postiindeks 10115."
    )
    expected = {
        ("date", 30, 40, "16.11.1975"),
        ("phone_number", 50, 64, "+372 5123 4567"),
        ("national_id", 76, 87, "47511160002"),
        ("street_address", 97, 110, "Pikk tanav 12"),
        ("postcode", 124, 129, "10115"),
    }
    observed = set()
    for pattern in get_patterns_for_language("et"):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is not None and not pattern.validator(value):
                continue
            observed.add((pattern.entity_type, match.start(), match.end(), value))

    assert expected <= observed


def test_estonian_street_address_supports_native_diacritics():
    text = "Aadress: Jõe tänav 5, Tallinn."
    observed = {
        match.group(0)
        for pattern in get_patterns_for_language("et")
        if pattern.entity_type == "street_address"
        for match in re.finditer(pattern.pattern, text, pattern.flags)
    }

    assert "Jõe tänav 5" in observed


def test_estonian_i18n_golden_fixture_offsets():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/et.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    row = rows[0]
    assert row["language"] == "et"
    assert row["metadata"]["synthetic"] is True
    assert row["metadata"]["category"] == "multilingual"

    text = row["text"]
    expected = {
        ("DATE", 30, 40, "16.11.1975"),
        ("PHONE", 50, 64, "+372 5123 4567"),
        ("ID_NUM", 76, 87, "47511160002"),
        ("STREET_ADDRESS", 97, 110, "Pikk tanav 12"),
        ("ZIPCODE", 124, 129, "10115"),
    }
    actual = {
        (span["label"], span["start"], span["end"], span["text"])
        for span in row["gold_spans"]
    }
    assert actual == expected
    for label, start, end, value in actual:
        assert text[start:end] == value, label

    ids_by_type = {
        span["metadata"]["identifier_type"]: span["text"]
        for span in row["gold_spans"]
        if span["label"] == "ID_NUM"
    }
    assert validate_estonian_isikukood(ids_by_type["isikukood"])


def test_estonian_i18n_golden_fixture_deidentifies_with_no_leakage_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.processing.outputs import PredictionResult

    fixture_path = Path("openmed/eval/golden/fixtures/i18n/et.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    row = rows[0]
    empty_result = PredictionResult(
        text=row["text"],
        entities=[],
        model_name="offline-safety-sweep",
        timestamp="2026-07-03T00:00:00Z",
        metadata={},
    )

    swept_result, added_count = _apply_safety_sweep_to_result(
        row["text"],
        empty_result,
        lang="et",
    )
    result = _build_deidentification_result(
        row["text"],
        swept_result,
        effective_method="mask",
        keep_year=False,
        date_shift_days=None,
        keep_mapping=False,
        lang="et",
        consistent=False,
        seed=None,
        locale=None,
        use_safety_sweep=True,
    )

    assert added_count == len(row["gold_spans"])
    for span in row["gold_spans"]:
        assert span["text"] not in result.deidentified_text


def test_validate_jmbg():
    # Standard checksum path.
    assert validate_jmbg("1611975710028")
    assert validate_jmbg("2105982710174")
    # A 2000s birth date (YYY below 800).
    assert validate_jmbg("0309004715013")
    # Special rule: remainder 10 or 11 yields check digit 0.
    assert validate_jmbg("0703985710130")
    assert validate_jmbg("0703985710040")

    # Bad check digit.
    assert not validate_jmbg("1611975710029")
    # Checksum-valid values with impossible embedded dates.
    assert not validate_jmbg(_jmbg_with_valid_check("3211975710"))  # day 32
    assert not validate_jmbg(_jmbg_with_valid_check("3002988715"))  # 30 Feb
    assert not validate_jmbg(_jmbg_with_valid_check("1613975710"))  # month 13
    assert not validate_jmbg("161197-5710028")
    assert not validate_jmbg("JMBG 1611975710028")
    assert not validate_jmbg(None)
    assert not validate_jmbg("abcdef")
    assert not validate_jmbg("123")


def _jmbg_with_valid_check(first_ten: str) -> str:
    """Complete a 12-digit JMBG body with a valid check digit for tests."""
    from openmed.core.pii_i18n import _jmbg_check_digit

    body = [int(digit) for digit in first_ten + "00"]
    return first_ten + "00" + str(_jmbg_check_digit(body))


def test_generated_jmbg_round_trips_validator():
    rng = random.Random(1234)
    for _ in range(200):
        assert validate_jmbg(generate_jmbg(rng=rng))


def test_generated_serbian_surrogate_passes_validator():
    assert LANG_TO_LOCALE["sr"] == "sr_RS"

    anonymizer = Anonymizer(lang="sr", consistent=True, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surrogate = anonymizer.surrogate("1611975710028", "national_id")

    assert validate_jmbg(surrogate) is True


def test_serbian_clinical_sample_expected_spans():
    text = (
        "Pacijent: Marko Petrovic. Datum rodjenja 16.11.1975, "
        "telefon +381 64 123 4567, JMBG 1611975710028, "
        "adresa Bulevar Oslobodjenja 45, postanski broj 21000."
    )
    expected = {
        ("date", 41, 51, "16.11.1975"),
        ("phone_number", 61, 77, "+381 64 123 4567"),
        ("national_id", 84, 97, "1611975710028"),
        ("street_address", 106, 129, "Bulevar Oslobodjenja 45"),
        ("postcode", 146, 151, "21000"),
    }
    observed = set()
    for pattern in get_patterns_for_language("sr"):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is not None and not pattern.validator(value):
                continue
            observed.add((pattern.entity_type, match.start(), match.end(), value))

    assert expected <= observed


def test_serbian_cyrillic_sample_expected_spans():
    text = (
        "Пацијент: Јелена Јовановић. Датум рођења 05.03.1988, "
        "телефон +381 63 987 6543, ЈМБГ 0503988715010, "
        "адреса Булевар Ослобођења 45, поштански број 21000."
    )
    expected = {
        ("date", 41, 51, "05.03.1988"),
        ("phone_number", 61, 77, "+381 63 987 6543"),
        ("national_id", 84, 97, "0503988715010"),
        ("street_address", 106, 127, "Булевар Ослобођења 45"),
        ("postcode", 144, 149, "21000"),
    }
    observed = set()
    for pattern in get_patterns_for_language("sr"):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is not None and not pattern.validator(value):
                continue
            observed.add((pattern.entity_type, match.start(), match.end(), value))

    assert expected <= observed


def test_serbian_i18n_golden_fixture_offsets():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/sr.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 2
    scripts = {row["metadata"]["script"] for row in rows}
    assert scripts == {"latin", "cyrillic"}

    for row in rows:
        assert row["language"] == "sr"
        assert row["metadata"]["synthetic"] is True
        assert row["metadata"]["category"] == "multilingual"

        text = row["text"]
        labels = set()
        for span in row["gold_spans"]:
            assert text[span["start"] : span["end"]] == span["text"], span
            labels.add(span["label"])
        assert labels == {"DATE", "PHONE", "ID_NUM", "STREET_ADDRESS", "ZIPCODE"}

        ids_by_type = {
            span["metadata"]["identifier_type"]: span["text"]
            for span in row["gold_spans"]
            if span["label"] == "ID_NUM"
        }
        assert validate_jmbg(ids_by_type["jmbg"])


def test_serbian_i18n_golden_fixture_deidentifies_with_no_leakage_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.processing.outputs import PredictionResult

    fixture_path = Path("openmed/eval/golden/fixtures/i18n/sr.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 2
    for row in rows:
        empty_result = PredictionResult(
            text=row["text"],
            entities=[],
            model_name="offline-safety-sweep",
            timestamp="2026-07-14T00:00:00Z",
            metadata={},
        )

        swept_result, added_count = _apply_safety_sweep_to_result(
            row["text"],
            empty_result,
            lang="sr",
        )
        result = _build_deidentification_result(
            row["text"],
            swept_result,
            effective_method="mask",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang="sr",
            consistent=False,
            seed=None,
            locale=None,
            use_safety_sweep=True,
        )

        assert added_count == len(row["gold_spans"]), row["metadata"]["script"]
        for span in row["gold_spans"]:
            assert span["text"] not in result.deidentified_text, span


def test_validate_croatian_oib():
    assert validate_croatian_oib("12345678903")
    assert validate_croatian_oib("06185927341")
    assert validate_croatian_oib("97531864205")
    assert validate_croatian_oib("55512345672")

    # Bad ISO 7064 MOD 11,10 check digit.
    assert not validate_croatian_oib("12345678904")
    assert not validate_croatian_oib("97531864206")
    # Wrong length or shape.
    assert not validate_croatian_oib("1234567890")
    assert not validate_croatian_oib("123456789031")
    assert not validate_croatian_oib("abcdef")
    assert not validate_croatian_oib("123")
    assert not validate_croatian_oib("123 456 789 03")
    assert not validate_croatian_oib("OIB 12345678903")
    assert not validate_croatian_oib(None)


def test_croatian_national_id_safety_sweep_requires_context():
    patterns = get_patterns_for_language("hr")
    national_id_patterns = [
        pattern for pattern in patterns if pattern.entity_type == "national_id"
    ]

    assert national_id_patterns
    assert all(
        pattern.safety_sweep_requires_context for pattern in national_id_patterns
    )


def test_faker_native_hr_ssn_round_trips_oib_validator():
    faker = Faker("hr_HR")
    faker.seed_instance(1234)
    for _ in range(200):
        assert validate_croatian_oib(faker.ssn())


def test_generated_croatian_surrogate_passes_validator():
    assert LANG_TO_LOCALE["hr"] == "hr_HR"

    anonymizer = Anonymizer(lang="hr", consistent=True, seed=42)
    surrogate = anonymizer.surrogate("97531864205", "national_id")

    assert validate_croatian_oib(surrogate) is True


def test_croatian_clinical_sample_expected_spans():
    text = (
        "Pacijent: Ivan Horvat. Datum rodjenja 16.11.1975, "
        "telefon +385 91 234 5678, OIB 97531864205, "
        "adresa Savska ulica 12, postanski broj 10000."
    )
    expected = {
        ("date", 38, 48, "16.11.1975"),
        ("phone_number", 58, 74, "+385 91 234 5678"),
        ("national_id", 80, 91, "97531864205"),
        ("street_address", 100, 115, "Savska ulica 12"),
        ("postcode", 132, 137, "10000"),
    }
    observed = set()
    for pattern in get_patterns_for_language("hr"):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is not None and not pattern.validator(value):
                continue
            observed.add((pattern.entity_type, match.start(), match.end(), value))

    assert expected <= observed


def test_croatian_i18n_golden_fixture_offsets():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/hr.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    row = rows[0]
    assert row["language"] == "hr"
    assert row["metadata"]["synthetic"] is True
    assert row["metadata"]["category"] == "multilingual"

    text = row["text"]
    expected = {
        ("DATE", 38, 48, "16.11.1975"),
        ("PHONE", 58, 74, "+385 91 234 5678"),
        ("ID_NUM", 80, 91, "97531864205"),
        ("STREET_ADDRESS", 100, 115, "Savska ulica 12"),
        ("ZIPCODE", 132, 137, "10000"),
    }
    actual = {
        (span["label"], span["start"], span["end"], span["text"])
        for span in row["gold_spans"]
    }
    assert actual == expected
    for label, start, end, value in actual:
        assert text[start:end] == value, label

    ids_by_type = {
        span["metadata"]["identifier_type"]: span["text"]
        for span in row["gold_spans"]
        if span["label"] == "ID_NUM"
    }
    assert validate_croatian_oib(ids_by_type["oib"])


def test_croatian_i18n_golden_fixture_deidentifies_with_no_leakage_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.processing.outputs import PredictionResult

    fixture_path = Path("openmed/eval/golden/fixtures/i18n/hr.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    row = rows[0]
    empty_result = PredictionResult(
        text=row["text"],
        entities=[],
        model_name="offline-safety-sweep",
        timestamp="2026-07-14T00:00:00Z",
        metadata={},
    )

    swept_result, added_count = _apply_safety_sweep_to_result(
        row["text"],
        empty_result,
        lang="hr",
    )
    result = _build_deidentification_result(
        row["text"],
        swept_result,
        effective_method="mask",
        keep_year=False,
        date_shift_days=None,
        keep_mapping=False,
        lang="hr",
        consistent=False,
        seed=None,
        locale=None,
        use_safety_sweep=True,
    )

    assert added_count == len(row["gold_spans"])
    for span in row["gold_spans"]:
        assert span["text"] not in result.deidentified_text


@pytest.mark.parametrize(
    "value",
    [
        "12345-6789012-3",
        "1234567890123",
        "۱۲۳۴۵-۶۷۸۹۰۱۲-۳",
        "۱۲۳۴۵۶۷۸۹۰۱۲۳",
    ],
)
def test_validate_pakistani_cnic(value):
    assert validate_pakistani_cnic(value)


def test_validate_pakistani_cnic_rejects_invalid_shapes():
    assert not validate_pakistani_cnic("1234-56789012-3")
    assert not validate_pakistani_cnic("123456789012")
    assert not validate_pakistani_cnic("abcdefghijklm")
    assert not validate_pakistani_cnic(None)


def test_urdu_national_id_safety_sweep_requires_context():
    from openmed.core.safety_sweep import safety_sweep

    patterns = get_patterns_for_language("ur")
    national_id_patterns = [
        pattern for pattern in patterns if pattern.entity_type == "national_id"
    ]
    assert national_id_patterns
    assert all(
        pattern.safety_sweep_requires_context for pattern in national_id_patterns
    )
    assert safety_sweep("12345-6789012-3", [], lang="ur") == []


def test_generated_urdu_surrogate_passes_validator():
    assert LANG_TO_LOCALE["ur"] == "ur_PK"
    anonymizer = Anonymizer(lang="ur", consistent=True, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surrogate = anonymizer.surrogate("12345-6789012-3", "national_id")
    assert validate_pakistani_cnic(surrogate) is True


@pytest.mark.parametrize(
    "value",
    [
        "12345-6789012-3",
        "1234567890123",
        "۱۲۳۴۵-۶۷۸۹۰۱۲-۳",
        "۱۲۳۴۵۶۷۸۹۰۱۲۳",
    ],
)
def test_urdu_cnic_safety_sweep_preserves_exact_offsets(value):
    from openmed.core.safety_sweep import safety_sweep

    text = f"شناختی کارڈ: {value}"
    expected_start = text.index(value)
    matches = [
        entity
        for entity in safety_sweep(text, [], lang="ur")
        if entity.label == "national_id"
    ]

    assert len(matches) == 1
    match = matches[0]
    assert (match.start, match.end, match.text) == (
        expected_start,
        expected_start + len(value),
        value,
    )


def test_urdu_i18n_golden_fixture_offsets():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/ur.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    row = next(row for row in rows if row["id"] == "golden-i18n-ur-clinical-pii")
    assert row["language"] == "ur"
    assert row["metadata"]["synthetic"] is True
    assert row["metadata"]["category"] == "multilingual"

    text = row["text"]
    for span in row["gold_spans"]:
        assert text[span["start"] : span["end"]] == span["text"], span

    ids_by_type = {
        span["metadata"]["identifier_type"]: span["text"]
        for span in row["gold_spans"]
        if span["label"] == "ID_NUM"
    }
    assert validate_pakistani_cnic(ids_by_type["cnic"])


def test_urdu_i18n_golden_fixture_deidentifies_with_no_leakage_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.processing.outputs import PredictionResult

    fixture_path = Path("openmed/eval/golden/fixtures/i18n/ur.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    row = rows[0]

    empty_result = PredictionResult(
        text=row["text"],
        entities=[],
        model_name="offline-safety-sweep",
        timestamp="2026-07-03T00:00:00Z",
        metadata={},
    )

    swept_result, added_count = _apply_safety_sweep_to_result(
        row["text"],
        empty_result,
        lang="ur",
    )
    result = _build_deidentification_result(
        row["text"],
        swept_result,
        effective_method="mask",
        keep_year=False,
        date_shift_days=None,
        keep_mapping=False,
        lang="ur",
        consistent=False,
        seed=None,
        locale=None,
        use_safety_sweep=True,
    )

    assert added_count == len(row["gold_spans"])
    label_map = {
        "DATE": "date",
        "PHONE": "phone_number",
        "ID_NUM": "national_id",
        "STREET_ADDRESS": "street_address",
        "ZIPCODE": "postcode",
    }
    expected_spans = {
        (label_map[span["label"]], span["start"], span["end"], span["text"])
        for span in row["gold_spans"]
    }
    actual_spans = {
        (entity.label, entity.start, entity.end, entity.text)
        for entity in swept_result.entities
    }
    assert actual_spans == expected_spans
    canonicalized_text = result.deidentified_text
    for canonical_label, internal_label in label_map.items():
        canonicalized_text = canonicalized_text.replace(
            f"[{internal_label}]",
            f"[{canonical_label}]",
        )
    assert canonicalized_text == row["metadata"]["expected_output"]["text"]
    for span in row["gold_spans"]:
        assert span["text"] not in result.deidentified_text


def test_validate_bulgarian_egn():
    # All three century encodings: 1900s (month 01-12), 1800s (month
    # 21-32), and 2000s (month 41-52).
    assert validate_bulgarian_egn("7511168208")
    assert validate_bulgarian_egn("8523071005")
    assert validate_bulgarian_egn("0449035017")
    # Weighted-sum remainder of 10 yields check digit 0.
    assert validate_bulgarian_egn("8503070010")

    # Bad check digit.
    assert validate_bulgarian_egn("7511168209") is False
    # Checksum-valid values with impossible embedded dates.
    assert not validate_bulgarian_egn("7511328203")  # day 32
    assert not validate_bulgarian_egn("7502308204")  # 30 February
    assert not validate_bulgarian_egn("7515328201")  # month code 15
    assert not validate_bulgarian_egn("abcdef")
    assert not validate_bulgarian_egn("123")
    assert not validate_bulgarian_egn("751116-8208")
    assert not validate_bulgarian_egn("ЕГН 7511168208")
    assert not validate_bulgarian_egn(None)


def test_generated_bulgarian_egn_round_trips_validator():
    rng = random.Random(1234)
    for _ in range(200):
        assert validate_bulgarian_egn(generate_bulgarian_egn(rng=rng))


def test_generated_bulgarian_surrogate_passes_validator():
    assert LANG_TO_LOCALE["bg"] == "bg_BG"

    anonymizer = Anonymizer(lang="bg", consistent=True, seed=42)
    surrogate = anonymizer.surrogate("7511168208", "national_id")

    assert validate_bulgarian_egn(surrogate) is True


def test_bulgarian_clinical_sample_expected_spans():
    text = (
        "Пациент: Иван Петров. Дата на раждане 16.11.1975, "
        "телефон +359 88 123 4567, ЕГН 7511168208, "
        "адрес улица Раковски 35, пощенски код 1000."
    )
    expected = {
        ("date", 38, 48, "16.11.1975"),
        ("phone_number", 58, 74, "+359 88 123 4567"),
        ("national_id", 80, 90, "7511168208"),
        ("street_address", 98, 115, "улица Раковски 35"),
        ("postcode", 130, 134, "1000"),
    }
    observed = set()
    for pattern in get_patterns_for_language("bg"):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is not None and not pattern.validator(value):
                continue
            observed.add((pattern.entity_type, match.start(), match.end(), value))

    assert expected <= observed


def test_bulgarian_i18n_golden_fixture_offsets():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/bg.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    row = rows[0]
    assert row["language"] == "bg"
    assert row["metadata"]["synthetic"] is True
    assert row["metadata"]["category"] == "multilingual"

    text = row["text"]
    expected = {
        ("DATE", 38, 48, "16.11.1975"),
        ("PHONE", 58, 74, "+359 88 123 4567"),
        ("ID_NUM", 80, 90, "7511168208"),
        ("STREET_ADDRESS", 98, 115, "улица Раковски 35"),
        ("ZIPCODE", 130, 134, "1000"),
    }
    actual = {
        (span["label"], span["start"], span["end"], span["text"])
        for span in row["gold_spans"]
    }
    assert actual == expected
    for label, start, end, value in actual:
        assert text[start:end] == value, label

    ids_by_type = {
        span["metadata"]["identifier_type"]: span["text"]
        for span in row["gold_spans"]
        if span["label"] == "ID_NUM"
    }
    assert validate_bulgarian_egn(ids_by_type["egn"])


def test_bulgarian_i18n_golden_fixture_deidentifies_with_no_leakage_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.processing.outputs import PredictionResult

    fixture_path = Path("openmed/eval/golden/fixtures/i18n/bg.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    row = rows[0]
    empty_result = PredictionResult(
        text=row["text"],
        entities=[],
        model_name="offline-safety-sweep",
        timestamp="2026-07-14T00:00:00Z",
        metadata={},
    )

    swept_result, added_count = _apply_safety_sweep_to_result(
        row["text"],
        empty_result,
        lang="bg",
    )
    result = _build_deidentification_result(
        row["text"],
        swept_result,
        effective_method="mask",
        keep_year=False,
        date_shift_days=None,
        keep_mapping=False,
        lang="bg",
        consistent=False,
        seed=None,
        locale=None,
        use_safety_sweep=True,
    )

    assert added_count == len(row["gold_spans"])
    for span in row["gold_spans"]:
        assert span["text"] not in result.deidentified_text


def test_validate_russian_snils():
    # Checksum-valid SNILS values (grouped and plain-digit forms).
    assert validate_russian_snils("112-233-445 95")
    assert validate_russian_snils("11223344595")
    assert validate_russian_snils("46257310279")
    assert validate_russian_snils("08765430300")

    # Transposed/incremented digits break the mod-101 checksum.
    assert validate_russian_snils("11223344559") is False
    assert validate_russian_snils("12123344595") is False
    assert validate_russian_snils("11223345495") is False

    assert not validate_russian_snils("abcdef")
    assert not validate_russian_snils("123")
    assert not validate_russian_snils("112233445")
    assert not validate_russian_snils("112x233x445x95")
    assert not validate_russian_snils(None)


def test_validate_russian_oms():
    # Checksum-valid 16-digit ENP/OMS policy number.
    assert validate_russian_oms("1234567890123452")

    # Transposed/incremented digits break the Luhn-style check digit.
    assert validate_russian_oms("1234567890123425") is False
    assert validate_russian_oms("2134567890123452") is False
    assert validate_russian_oms("1234567890123453") is False

    assert not validate_russian_oms("abcdef")
    assert not validate_russian_oms("123")
    assert not validate_russian_oms("123456789012345")
    assert not validate_russian_oms(None)


def test_generated_russian_snils_round_trips_validator():
    rng = random.Random(1234)
    for _ in range(200):
        assert validate_russian_snils(generate_russian_snils(rng=rng))


def test_generated_russian_oms_round_trips_validator():
    rng = random.Random(1234)
    for _ in range(200):
        assert validate_russian_oms(generate_russian_oms(rng=rng))


def test_generated_russian_surrogate_passes_validator():
    assert LANG_TO_LOCALE["ru"] == "ru_RU"

    anonymizer = Anonymizer(lang="ru", consistent=True, seed=42)
    surrogate = anonymizer.surrogate("112-233-445 95", "national_id")

    assert validate_russian_snils(surrogate) is True


def test_russian_clinical_sample_expected_spans():
    text = (
        "Пациент: Иван Петров. Дата рождения 16.11.1975, "
        "телефон +7 916 123-45-67, СНИЛС 112-233-445 95, "
        "полис ОМС 1234567890123452, "
        "адрес: улица Ленина, дом 12, индекс 101000."
    )
    expected = {
        ("date", 36, 46, "16.11.1975"),
        ("phone_number", 56, 72, "+7 916 123-45-67"),
        ("national_id", 80, 94, "112-233-445 95"),
        ("national_id", 106, 122, "1234567890123452"),
        ("street_address", 131, 151, "улица Ленина, дом 12"),
        ("postcode", 160, 166, "101000"),
    }
    observed = set()
    for pattern in get_patterns_for_language("ru"):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is not None and not pattern.validator(value):
                continue
            observed.add((pattern.entity_type, match.start(), match.end(), value))

    assert expected <= observed


def test_russian_phone_pattern_requires_balanced_parentheses():
    patterns = [
        pattern
        for pattern in get_patterns_for_language("ru")
        if pattern.entity_type == "phone_number"
    ]

    assert any(
        re.search(pattern.pattern, "телефон 8 (495) 123-45-67") for pattern in patterns
    )
    assert not any(
        re.search(pattern.pattern, "телефон 8 (495 123-45-67") for pattern in patterns
    )


def test_russian_i18n_golden_fixture_offsets():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/ru.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 2
    for row in rows:
        assert row["language"] == "ru"
        assert row["metadata"]["synthetic"] is True
        assert row["metadata"]["category"] == "multilingual"

        text = row["text"]
        for span in row["gold_spans"]:
            assert text[span["start"] : span["end"]] == span["text"], span["label"]

        ids_by_type = {
            span["metadata"]["identifier_type"]: span["text"]
            for span in row["gold_spans"]
            if span["label"] == "ID_NUM"
        }
        if "snils" in ids_by_type:
            assert validate_russian_snils(ids_by_type["snils"])
        if "oms" in ids_by_type:
            assert validate_russian_oms(ids_by_type["oms"])

    labels = {span["label"] for row in rows for span in row["gold_spans"]}
    assert labels == {"DATE", "PHONE", "ID_NUM", "STREET_ADDRESS", "ZIPCODE"}


def test_russian_i18n_golden_fixture_deidentifies_with_no_leakage_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.processing.outputs import PredictionResult

    fixture_path = Path("openmed/eval/golden/fixtures/i18n/ru.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 2
    for row in rows:
        empty_result = PredictionResult(
            text=row["text"],
            entities=[],
            model_name="offline-safety-sweep",
            timestamp="2026-07-14T00:00:00Z",
            metadata={},
        )

        swept_result, added_count = _apply_safety_sweep_to_result(
            row["text"],
            empty_result,
            lang="ru",
        )
        result = _build_deidentification_result(
            row["text"],
            swept_result,
            effective_method="mask",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang="ru",
            consistent=False,
            seed=None,
            locale=None,
            use_safety_sweep=True,
        )

        assert added_count == len(row["gold_spans"])
        for span in row["gold_spans"]:
            assert span["text"] not in result.deidentified_text


def test_validate_finnish_hetu():
    # 1900s '-' sign, including letter check characters.
    assert validate_finnish_hetu("161175-802D")
    assert validate_finnish_hetu("110174-237M")
    # 1800s '+' sign.
    assert validate_finnish_hetu("070385+1003")
    # 2000s 'A' sign and 2023-reform signs.
    assert validate_finnish_hetu("030904A406A")
    assert validate_finnish_hetu("161175Y802D")
    assert validate_finnish_hetu("030904B406A")

    # Bad check character.
    assert validate_finnish_hetu("161175-802E") is False
    # Check-valid values with impossible embedded dates.
    assert not validate_finnish_hetu("300275-8026")  # 30 February
    assert not validate_finnish_hetu("320175-802N")  # day 32
    # Invalid century sign or check char outside the mod-31 alphabet.
    assert not validate_finnish_hetu("161175G802D")
    assert not validate_finnish_hetu("161175-802I")
    assert not validate_finnish_hetu("161175-802")
    assert not validate_finnish_hetu("abcdef")
    assert not validate_finnish_hetu("123")
    # Individual numbers 000-001 and 900-999 are not ordinary HETU values.
    assert not validate_finnish_hetu("161175-000J")
    assert not validate_finnish_hetu("161175-001K")
    assert not validate_finnish_hetu("161175-900K")
    assert not validate_finnish_hetu(" 161175-802D ")
    assert not validate_finnish_hetu(None)


def test_faker_native_fi_ssn_round_trips_hetu_validator():
    faker = Faker("fi_FI")
    faker.seed_instance(1234)
    for _ in range(200):
        assert validate_finnish_hetu(faker.ssn())


def test_generated_finnish_surrogate_passes_validator():
    assert LANG_TO_LOCALE["fi"] == "fi_FI"

    anonymizer = Anonymizer(lang="fi", consistent=True, seed=42)
    surrogate = anonymizer.surrogate("161175-802D", "national_id")

    assert validate_finnish_hetu(surrogate) is True


def test_finnish_clinical_sample_expected_spans():
    text = (
        "Potilas: Matti Virtanen. Syntymäaika 16.11.1975, "
        "puhelin +358 40 123 4567, henkilötunnus 161175-802D, "
        "osoite Mannerheimintie 12, postinumero 00100."
    )
    expected = {
        ("date", 37, 47, "16.11.1975"),
        ("phone_number", 57, 73, "+358 40 123 4567"),
        ("national_id", 89, 100, "161175-802D"),
        ("street_address", 109, 127, "Mannerheimintie 12"),
        ("postcode", 141, 146, "00100"),
    }
    observed = set()
    for pattern in get_patterns_for_language("fi"):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is not None and not pattern.validator(value):
                continue
            observed.add((pattern.entity_type, match.start(), match.end(), value))

    assert expected <= observed


def test_finnish_i18n_golden_fixture_offsets():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/fi.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    row = rows[0]
    assert row["language"] == "fi"
    assert row["metadata"]["synthetic"] is True
    assert row["metadata"]["category"] == "multilingual"

    text = row["text"]
    expected = {
        ("DATE", 37, 47, "16.11.1975"),
        ("PHONE", 57, 73, "+358 40 123 4567"),
        ("ID_NUM", 89, 100, "161175-802D"),
        ("STREET_ADDRESS", 109, 127, "Mannerheimintie 12"),
        ("ZIPCODE", 141, 146, "00100"),
    }
    actual = {
        (span["label"], span["start"], span["end"], span["text"])
        for span in row["gold_spans"]
    }
    assert actual == expected
    for label, start, end, value in actual:
        assert text[start:end] == value, label

    ids_by_type = {
        span["metadata"]["identifier_type"]: span["text"]
        for span in row["gold_spans"]
        if span["label"] == "ID_NUM"
    }
    assert validate_finnish_hetu(ids_by_type["hetu"])


def test_finnish_i18n_golden_fixture_deidentifies_with_no_leakage_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.processing.outputs import PredictionResult

    fixture_path = Path("openmed/eval/golden/fixtures/i18n/fi.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    row = rows[0]
    empty_result = PredictionResult(
        text=row["text"],
        entities=[],
        model_name="offline-safety-sweep",
        timestamp="2026-07-14T00:00:00Z",
        metadata={},
    )

    swept_result, added_count = _apply_safety_sweep_to_result(
        row["text"],
        empty_result,
        lang="fi",
    )
    result = _build_deidentification_result(
        row["text"],
        swept_result,
        effective_method="mask",
        keep_year=False,
        date_shift_days=None,
        keep_mapping=False,
        lang="fi",
        consistent=False,
        seed=None,
        locale=None,
        use_safety_sweep=True,
    )

    assert added_count == len(row["gold_spans"])
    for span in row["gold_spans"]:
        assert span["text"] not in result.deidentified_text


def test_validate_ukrainian_rnokpp():
    assert validate_ukrainian_rnokpp("2974281300")
    assert validate_ukrainian_rnokpp("3695007088")

    assert validate_ukrainian_rnokpp("2974281301") is False
    assert validate_ukrainian_rnokpp("297428130") is False
    assert validate_ukrainian_rnokpp("29742 81300") is False
    assert validate_ukrainian_rnokpp("abcdefghij") is False


def test_generated_ukrainian_surrogate_passes_validator():
    assert LANG_TO_LOCALE["uk"] == "uk_UA"

    anonymizer = Anonymizer(lang="uk", consistent=True, seed=42)
    surrogate = anonymizer.surrogate("2974281300", "national_id")

    assert validate_ukrainian_rnokpp(surrogate) is True
    assert surrogate != "2974281300"


def test_ukrainian_clinical_sample_expected_spans():
    text = (
        "Пацієнт: Олена Коваль. Дата народження 16.11.1975, "
        "телефон +380 67 123 45 67, РНОКПП 2974281300, "
        "адреса вулиця Хрещатик 22, поштовий індекс 01001."
    )
    expected = {
        ("date", 39, 49, "16.11.1975"),
        ("phone_number", 59, 76, "+380 67 123 45 67"),
        ("national_id", 85, 95, "2974281300"),
        ("street_address", 104, 122, "вулиця Хрещатик 22"),
        ("postcode", 140, 145, "01001"),
    }
    observed = set()
    for pattern in get_patterns_for_language("uk"):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is not None and not pattern.validator(value):
                continue
            observed.add((pattern.entity_type, match.start(), match.end(), value))

    assert expected <= observed


def test_ukrainian_textual_date_and_street_patterns():
    text = "Дата народження 16 листопада 1975, адреса проспект Свободи 15."
    observed = {
        (pattern.entity_type, match.group(0))
        for pattern in get_patterns_for_language("uk")
        for match in re.finditer(pattern.pattern, text, pattern.flags)
    }

    assert ("date", "16 листопада 1975") in observed
    assert ("street_address", "проспект Свободи 15") in observed


def test_ukrainian_i18n_golden_fixture_offsets():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/uk.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    row = rows[0]
    assert row["language"] == "uk"
    assert row["metadata"]["synthetic"] is True
    assert row["metadata"]["category"] == "multilingual"

    text = row["text"]
    expected = {
        ("DATE", 39, 49, "16.11.1975"),
        ("PHONE", 59, 76, "+380 67 123 45 67"),
        ("ID_NUM", 85, 95, "2974281300"),
        ("STREET_ADDRESS", 104, 122, "вулиця Хрещатик 22"),
        ("ZIPCODE", 140, 145, "01001"),
    }
    actual = {
        (span["label"], span["start"], span["end"], span["text"])
        for span in row["gold_spans"]
    }
    assert actual == expected
    for label, start, end, value in actual:
        assert text[start:end] == value, label

    identifier = next(
        span["text"] for span in row["gold_spans"] if span["label"] == "ID_NUM"
    )
    assert validate_ukrainian_rnokpp(identifier)


def test_ukrainian_i18n_golden_fixture_deidentifies_with_no_leakage_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.processing.outputs import PredictionResult

    fixture_path = Path("openmed/eval/golden/fixtures/i18n/uk.jsonl")
    row = json.loads(fixture_path.read_text(encoding="utf-8").strip())
    empty_result = PredictionResult(
        text=row["text"],
        entities=[],
        model_name="offline-safety-sweep",
        timestamp="2026-07-19T00:00:00Z",
        metadata={},
    )

    swept_result, added_count = _apply_safety_sweep_to_result(
        row["text"],
        empty_result,
        lang="uk",
    )
    result = _build_deidentification_result(
        row["text"],
        swept_result,
        effective_method="mask",
        keep_year=False,
        date_shift_days=None,
        keep_mapping=False,
        lang="uk",
        consistent=False,
        seed=None,
        locale=None,
        use_safety_sweep=True,
    )

    assert added_count == len(row["gold_spans"])
    for span in row["gold_spans"]:
        assert span["text"] not in result.deidentified_text


def test_validate_czech_rodne_cislo():
    # Modern ten-digit form (delegated to the Czechoslovak validator).
    assert validate_czech_rodne_cislo("751116/0008")
    assert validate_czech_rodne_cislo("7511160008")
    # Female +50 month offset and post-2004 +20 overflow series.
    assert validate_czech_rodne_cislo("756116/0002")
    assert validate_czech_rodne_cislo("083116/0000")
    # Legacy pre-1954 nine-digit form (no checksum).
    assert validate_czech_rodne_cislo("485305/123")
    assert validate_czech_rodne_cislo("531116/456")
    assert validate_czech_rodne_cislo("850307/789")
    # Year suffixes above 53 decode to the 1800s (here 1854), so every
    # nine-digit suffix maps to a valid pre-1954 century.
    assert validate_czech_rodne_cislo("541116/123")

    # Modern form with a broken modulo-11 check.
    assert validate_czech_rodne_cislo("751116/0009") is False
    # Legacy form must not use the post-2004 overflow month series.
    assert not validate_czech_rodne_cislo("483105/123")
    # Legacy form with an impossible embedded date.
    assert not validate_czech_rodne_cislo("480230/123")
    assert not validate_czech_rodne_cislo("12345678")
    assert not validate_czech_rodne_cislo("abcdef")


def test_generated_czech_surrogate_passes_validator():
    assert LANG_TO_LOCALE["cs"] == "cs_CZ"

    anonymizer = Anonymizer(lang="cs", consistent=True, seed=42)
    surrogate = anonymizer.surrogate("751116/0008", "national_id")

    assert validate_czech_rodne_cislo(surrogate) is True
    # The shared Czechoslovak provider output stays valid for Slovak too.
    assert validate_czechoslovak_rodne_cislo(surrogate) is True


def test_czech_format_preserving_dates_are_day_first():
    anonymizer = Anonymizer(lang="cs", consistent=True, seed=42)

    with patch(
        "openmed.core.anonymizer.engine.preserve_date_format",
        return_value="14/05/1951",
    ) as preserve:
        surrogate = anonymizer.format_preserving_surrogate("05/06/2020", "date")

    assert surrogate == "14/05/1951"
    assert preserve.call_args.kwargs["day_first"] is True


def test_czech_clinical_sample_expected_spans():
    text = (
        "Pacient: Jan Novak. Datum narozeni 16.11.1975, "
        "telefon +420 601 234 567, rodne cislo 751116/0008, "
        "adresa Vodickova ulice 12, PSC 110 00."
    )
    expected = {
        ("date", 35, 45, "16.11.1975"),
        ("phone_number", 55, 71, "+420 601 234 567"),
        ("national_id", 85, 96, "751116/0008"),
        ("street_address", 105, 123, "Vodickova ulice 12"),
        ("postcode", 129, 135, "110 00"),
    }
    observed = set()
    for pattern in get_patterns_for_language("cs"):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is not None and not pattern.validator(value):
                continue
            observed.add((pattern.entity_type, match.start(), match.end(), value))

    assert expected <= observed


def test_czech_textual_date_and_diacritic_address_patterns():
    text = "Datum narození 16. listopadu 1975, adresa Náměstí Míru 5."
    observed = {
        (pattern.entity_type, match.group(0))
        for pattern in get_patterns_for_language("cs")
        for match in re.finditer(pattern.pattern, text, pattern.flags)
    }

    assert ("date", "16. listopadu 1975") in observed
    assert ("street_address", "Náměstí Míru 5") in observed


@pytest.mark.parametrize("language", ["cs", "sk"])
def test_czech_legacy_rodne_cislo_pattern_matches(language):
    text = "Pacientka, rodne cislo 485305/123, prijata k hospitalizaci."
    observed = set()
    for pattern in get_patterns_for_language(language):
        if pattern.entity_type != "national_id":
            continue
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is not None and not pattern.validator(value):
                continue
            observed.add(value)

    assert "485305/123" in observed


def test_czech_phone_patterns_cover_fixed_and_mobile_without_trunk_zero():
    phone_patterns = [
        pattern
        for pattern in get_patterns_for_language("cs")
        if pattern.entity_type == "phone_number"
    ]

    for value in ["+420 212 345 678", "212 345 678", "+420 601 234 567"]:
        assert any(
            re.fullmatch(pattern.pattern, value, pattern.flags)
            for pattern in phone_patterns
        )

    assert not any(
        re.search(pattern.pattern, "telefon 0601 234 567", pattern.flags)
        for pattern in phone_patterns
    )


def test_czech_i18n_golden_fixture_offsets():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/cs.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    row = rows[0]
    assert row["language"] == "cs"
    assert row["metadata"]["synthetic"] is True
    assert row["metadata"]["category"] == "multilingual"

    text = row["text"]
    expected = {
        ("DATE", 35, 45, "16.11.1975"),
        ("PHONE", 55, 71, "+420 601 234 567"),
        ("ID_NUM", 85, 96, "751116/0008"),
        ("STREET_ADDRESS", 105, 123, "Vodickova ulice 12"),
        ("ZIPCODE", 129, 135, "110 00"),
    }
    actual = {
        (span["label"], span["start"], span["end"], span["text"])
        for span in row["gold_spans"]
    }
    assert actual == expected
    for label, start, end, value in actual:
        assert text[start:end] == value, label

    ids_by_type = {
        span["metadata"]["identifier_type"]: span["text"]
        for span in row["gold_spans"]
        if span["label"] == "ID_NUM"
    }
    assert validate_czech_rodne_cislo(ids_by_type["rodne_cislo"])


def test_czech_i18n_golden_fixture_deidentifies_with_no_leakage_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.processing.outputs import PredictionResult

    fixture_path = Path("openmed/eval/golden/fixtures/i18n/cs.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    row = rows[0]
    empty_result = PredictionResult(
        text=row["text"],
        entities=[],
        model_name="offline-safety-sweep",
        timestamp="2026-07-14T00:00:00Z",
        metadata={},
    )

    swept_result, added_count = _apply_safety_sweep_to_result(
        row["text"],
        empty_result,
        lang="cs",
    )
    result = _build_deidentification_result(
        row["text"],
        swept_result,
        effective_method="mask",
        keep_year=False,
        date_shift_days=None,
        keep_mapping=False,
        lang="cs",
        consistent=False,
        seed=None,
        locale=None,
        use_safety_sweep=True,
    )

    assert added_count == len(row["gold_spans"])
    for span in row["gold_spans"]:
        assert span["text"] not in result.deidentified_text


def test_validate_greek_amka():
    # Synthetic valid AMKA (DDMMYY prefix + Luhn check).
    assert validate_greek_amka("16117508024")
    assert validate_greek_amka("21058200177")
    assert validate_greek_amka("03090450119")

    # Broken Luhn check digit.
    assert validate_greek_amka("16117508025") is False
    # Luhn-valid values with impossible embedded birth dates.
    assert not validate_greek_amka("32017508022")  # day 32
    assert not validate_greek_amka("30027508024")  # 30 February
    assert not validate_greek_amka("16137508020")  # month 13
    # Wrong length or shape.
    assert not validate_greek_amka("1611758024")
    assert not validate_greek_amka("abcdefghijk")
    assert not validate_greek_amka("123")


def test_faker_native_el_ssn_round_trips_amka_validator():
    faker = Faker("el_GR")
    faker.seed_instance(1234)
    for _ in range(200):
        assert validate_greek_amka(faker.ssn())


def test_generated_greek_surrogate_passes_validator():
    assert LANG_TO_LOCALE["el"] == "el_GR"

    anonymizer = Anonymizer(lang="el", consistent=True, seed=42)
    surrogate = anonymizer.surrogate("16117508024", "national_id")

    assert validate_greek_amka(surrogate) is True


def test_greek_clinical_sample_expected_spans():
    text = (
        "Ασθενής: Γιώργος Παπαδόπουλος. Ημερομηνία γέννησης 16.11.1975, "
        "τηλέφωνο +30 691 234 5678, ΑΜΚΑ 16117508024, "
        "διεύθυνση οδός Ερμού 15, ταχυδρομικός κώδικας 104 31."
    )
    expected = {
        ("date", 51, 61, "16.11.1975"),
        ("phone_number", 72, 88, "+30 691 234 5678"),
        ("national_id", 95, 106, "16117508024"),
        ("street_address", 118, 131, "οδός Ερμού 15"),
        ("postcode", 154, 160, "104 31"),
    }
    observed = set()
    for pattern in get_patterns_for_language("el"):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is not None and not pattern.validator(value):
                continue
            observed.add((pattern.entity_type, match.start(), match.end(), value))

    assert expected <= observed


def test_greek_i18n_golden_fixture_offsets():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/el.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    row = rows[0]
    assert row["language"] == "el"
    assert row["metadata"]["synthetic"] is True
    assert row["metadata"]["category"] == "multilingual"

    text = row["text"]
    expected = {
        ("DATE", 51, 61, "16.11.1975"),
        ("PHONE", 72, 88, "+30 691 234 5678"),
        ("ID_NUM", 95, 106, "16117508024"),
        ("STREET_ADDRESS", 118, 131, "οδός Ερμού 15"),
        ("ZIPCODE", 154, 160, "104 31"),
    }
    actual = {
        (span["label"], span["start"], span["end"], span["text"])
        for span in row["gold_spans"]
    }
    assert actual == expected
    for label, start, end, value in actual:
        assert text[start:end] == value, label

    ids_by_type = {
        span["metadata"]["identifier_type"]: span["text"]
        for span in row["gold_spans"]
        if span["label"] == "ID_NUM"
    }
    assert validate_greek_amka(ids_by_type["amka"])


def test_greek_i18n_golden_fixture_deidentifies_with_no_leakage_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.processing.outputs import PredictionResult

    fixture_path = Path("openmed/eval/golden/fixtures/i18n/el.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    row = rows[0]
    empty_result = PredictionResult(
        text=row["text"],
        entities=[],
        model_name="offline-safety-sweep",
        timestamp="2026-07-14T00:00:00Z",
        metadata={},
    )

    swept_result, added_count = _apply_safety_sweep_to_result(
        row["text"],
        empty_result,
        lang="el",
    )
    result = _build_deidentification_result(
        row["text"],
        swept_result,
        effective_method="mask",
        keep_year=False,
        date_shift_days=None,
        keep_mapping=False,
        lang="el",
        consistent=False,
        seed=None,
        locale=None,
        use_safety_sweep=True,
    )

    assert added_count == len(row["gold_spans"])
    for span in row["gold_spans"]:
        assert span["text"] not in result.deidentified_text


class TestSlovakLocaleAndFixture:
    """Tests for Slovak locale and golden fixture wiring."""

    def test_locale_and_surrogate_rodne_cislo_round_trip(self):
        assert LANG_TO_LOCALE["sk"] == "sk_SK"
        anon = Anonymizer(lang="sk", consistent=True, seed=42)

        surrogate = anon.surrogate("850505/1236", "national_id")

        assert validate_czechoslovak_rodne_cislo(surrogate) is True

    def test_i18n_golden_fixture_offsets(self):
        fixture_path = Path("openmed/eval/golden/fixtures/i18n/sk.jsonl")
        rows = [
            json.loads(line)
            for line in fixture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert len(rows) == 1
        row = rows[0]
        assert row["language"] == "sk"
        assert row["metadata"]["synthetic"] is True
        assert row["metadata"]["category"] == "multilingual"

        text = row["text"]
        expected = {
            ("DATE", 42, 52, "05.05.1985"),
            ("PHONE", 62, 78, "+421 903 123 456"),
            ("ID_NUM", 92, 103, "855505/1230"),
            ("STREET_ADDRESS", 112, 127, "Hlavna ulica 12"),
            ("ZIPCODE", 133, 138, "81101"),
        }
        actual = {
            (span["label"], span["start"], span["end"], span["text"])
            for span in row["gold_spans"]
        }
        assert actual == expected
        for label, start, end, value in actual:
            assert text[start:end] == value, label


class TestHungarianLocaleAndFixture:
    """Tests for Hungarian locale, TAJ surrogate, and golden fixture wiring."""

    def test_locale_and_surrogate_taj_round_trip(self):
        assert LANG_TO_LOCALE["hu"] == "hu_HU"
        anon = Anonymizer(lang="hu", consistent=True, seed=42)

        surrogate = anon.surrogate("123456788", "national_id")

        assert validate_hungarian_taj(surrogate) is True

    def test_i18n_golden_fixture_offsets_and_patterns(self):
        fixture_path = Path("openmed/eval/golden/fixtures/i18n/hu.jsonl")
        rows = [
            json.loads(line)
            for line in fixture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert len(rows) == 1
        row = rows[0]
        assert row["language"] == "hu"
        assert row["metadata"]["synthetic"] is True
        assert row["metadata"]["category"] == "multilingual"

        text = row["text"]
        expected = {
            (span["label"], span["start"], span["end"], span["text"])
            for span in row["gold_spans"]
        }
        labels = {
            "date": "DATE",
            "national_id": "ID_NUM",
            "phone_number": "PHONE",
            "postcode": "ZIPCODE",
            "street_address": "STREET_ADDRESS",
        }
        observed = set()
        for pattern in LANGUAGE_PII_PATTERNS["hu"]:
            for match in re.finditer(pattern.pattern, text, pattern.flags):
                value = match.group(0)
                if pattern.validator is not None and not pattern.validator(value):
                    continue
                observed.add(
                    (
                        labels[pattern.entity_type],
                        match.start(),
                        match.end(),
                        value,
                    )
                )

        assert expected <= observed
        for label, start, end, value in expected:
            assert text[start:end] == value, label


class TestKoreanLocaleAndFixture:
    """Tests for Korean locale and golden fixture wiring."""

    def test_locale_and_surrogate_rrn_round_trip(self):
        assert LANG_TO_LOCALE["ko"] == "ko_KR"
        anon = Anonymizer(lang="ko", consistent=True, seed=42)
        surrogate = anon.surrogate("940315-1234567", "national_id")
        assert validate_korean_rrn(surrogate) is True

    def test_i18n_golden_fixture_offsets(self):
        fixture_path = Path("openmed/eval/golden/fixtures/i18n/ko.jsonl")
        rows = [
            json.loads(line)
            for line in fixture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert len(rows) == 2

        # ── clinical fixture ──────────────────────────────────────────────
        row = next(r for r in rows if r["id"] == "golden-multilingual-ko-clinical")
        assert row["language"] == "ko"
        assert row["metadata"]["synthetic"] is True
        assert row["metadata"]["category"] == "multilingual"

        text = row["text"]
        expected = {
            ("DATE", 8, 20, "1994년 3월 15일"),
            ("PHONE", 34, 47, "010-1234-5678"),
            ("ID_NUM", 58, 72, "940315-1234567"),
            ("ZIPCODE", 81, 86, "06292"),
            ("STREET_ADDRESS", 93, 109, "서울시 강남구 테헤란로 123"),
        }
        actual = {
            (span["label"], span["start"], span["end"], span["text"])
            for span in row["gold_spans"]
        }
        assert actual == expected
        for label, start, end, value in actual:
            assert text[start:end] == value, label

        labels = {
            "date": "DATE",
            "national_id": "ID_NUM",
            "phone_number": "PHONE",
            "postcode": "ZIPCODE",
            "street_address": "STREET_ADDRESS",
        }
        observed = set()
        for pattern in LANGUAGE_PII_PATTERNS["ko"]:
            for match in re.finditer(pattern.pattern, text, pattern.flags):
                value = match.group(0)
                if pattern.validator is not None and not pattern.validator(value):
                    continue
                observed.add(
                    (
                        labels[pattern.entity_type],
                        match.start(),
                        match.end(),
                        value,
                    )
                )

        assert expected <= observed

        checksum_row = next(r for r in rows if r["id"] == "golden-checksum-ko-rrn")
        assert checksum_row["language"] == "ko"
        assert checksum_row["metadata"]["synthetic"] is True
        assert checksum_row["metadata"]["category"] == "checksum_ids"

        checksum_text = checksum_row["text"]
        checksum_span = checksum_row["gold_spans"][0]
        assert (
            checksum_text[checksum_span["start"] : checksum_span["end"]]
            == (checksum_span["text"])
        )
        assert validate_korean_rrn(checksum_span["text"])

        hard_negative = checksum_row["metadata"]["hard_negatives"][0]
        assert (
            checksum_text[hard_negative["start"] : hard_negative["end"]]
            == (hard_negative["text"])
        )
        assert not validate_korean_rrn(hard_negative["text"])


class TestNigerianIdentifiers:
    """Validator, pattern, surrogate, and synthetic-fixture coverage for NG."""

    @staticmethod
    def _faker(seed: int, locale: str = "en_NG") -> Faker:
        faker = Faker(locale)
        register_clinical_providers(faker)
        faker.seed_instance(seed)
        return faker

    @pytest.mark.parametrize(
        "value",
        ["36787753186", "12345678901"],
    )
    def test_nin_validator_accepts_non_trivial_values(self, value):
        assert validate_nigeria_nin(value)

    @pytest.mark.parametrize(
        "value",
        [
            "00000000000",
            "99999999999",
            "01234567890",
            "98765432109",
            "1234567890",
            "123456789012",
            "１２３４５６７８９０１",
            None,
        ],
    )
    def test_nin_validator_rejects_trivial_or_malformed_values(self, value):
        assert not validate_nigeria_nin(value)

    def test_bvn_validator_accepts_non_trivial_eleven_digit_values(self):
        assert validate_nigeria_bvn("24107152688")
        assert validate_nigeria_bvn("12345678901")
        assert validate_nigeria_bvn("36787753186")
        assert validate_nigeria_bvn("04107152688")
        assert not validate_nigeria_bvn("00000000000")
        assert not validate_nigeria_bvn("99999999999")
        assert not validate_nigeria_bvn("2410715268")
        assert not validate_nigeria_bvn("241071526880")

    def test_one_thousand_nin_and_bvn_surrogates_pass_validators(self):
        faker = self._faker(840)
        for _ in range(1_000):
            assert validate_nigeria_nin(faker.nigeria_nin())
            assert validate_nigeria_bvn(faker.nigeria_bvn())

    def test_provider_is_seed_deterministic_and_preserves_phone_prefix_class(self):
        first = self._faker(840)
        second = self._faker(840)
        assert first.nigeria_nin() == second.nigeria_nin()
        assert first.nigeria_bvn() == second.nigeria_bvn()

        for phone in (
            "+234 803 123 4567",
            "07051234567",
            "08039999999",
            "08121234567",
            "09031234567",
            "09159999999",
        ):
            surrogate = first.ng_mobile_number(phone)
            source_digits = re.sub(r"[^0-9]", "", phone)
            surrogate_digits = re.sub(r"[^0-9]", "", surrogate)
            source_national = (
                "0" + source_digits[3:]
                if source_digits.startswith("234")
                else source_digits
            )
            surrogate_national = (
                "0" + surrogate_digits[3:]
                if surrogate_digits.startswith("234")
                else surrogate_digits
            )
            assert surrogate_national[:3] == source_national[:3]
            assert surrogate_national != source_national

    def test_locale_aliases_share_patterns_and_language_lookup(self):
        aliases = ("en_ng", "ha", "ig", "yo")
        assert all(
            LOCALE_PII_PATTERNS[alias] is LOCALE_PII_PATTERNS["en_ng"]
            for alias in aliases
        )
        for language in ("ha", "ig", "yo"):
            patterns = get_patterns_for_language(language)
            assert all(pattern in patterns for pattern in LOCALE_PII_PATTERNS[language])

    def test_nin_and_bvn_patterns_require_context_in_safety_sweep(self):
        from openmed.core.safety_sweep import safety_sweep

        patterns = LOCALE_PII_PATTERNS["en_ng"]
        bare = safety_sweep("12345678901", [], patterns=patterns)
        labeled_nin = safety_sweep("NIN: 12345678901", [], patterns=patterns)
        labeled_bvn = safety_sweep("BVN: 24107152688", [], patterns=patterns)

        assert bare == []
        assert [(entity.label, entity.text) for entity in labeled_nin] == [
            ("NG_NIN", "12345678901")
        ]
        assert [(entity.label, entity.text) for entity in labeled_bvn] == [
            ("NG_BVN", "24107152688")
        ]

    def test_phone_patterns_match_required_forms_and_id_priority_wins(self):
        from openmed.core.safety_sweep import safety_sweep

        patterns = LOCALE_PII_PATTERNS["en_ng"]
        phone_patterns = [
            pattern for pattern in patterns if pattern.entity_type == "NG_PHONE"
        ]
        for phone in ("+234 803 123 4567", "08039999999", "09159999999"):
            assert any(
                re.fullmatch(pattern.pattern, phone, pattern.flags)
                for pattern in phone_patterns
            )

        for identifier in ("12345678901", "24107152688"):
            assert not any(
                re.search(pattern.pattern, identifier, pattern.flags)
                for pattern in phone_patterns
            )

        mobile_shaped_nin = safety_sweep(
            "NIN: 08039999999",
            [],
            patterns=patterns,
        )
        assert [(entity.label, entity.text) for entity in mobile_shaped_nin] == [
            ("NG_NIN", "08039999999")
        ]

    def test_source_labels_route_to_distinct_identifier_surrogates(self):
        from openmed.core.labels import ID_NUM, PHONE, id_subtype_for, normalize_label

        assert normalize_label("NG_NIN") == ID_NUM
        assert normalize_label("NG_BVN") == ID_NUM
        assert normalize_label("NG_PHONE") == PHONE
        assert id_subtype_for("NG_NIN") == "national_id"
        assert id_subtype_for("NG_BVN") == "national_id"

        anonymizer = Anonymizer(locale="en_NG", consistent=True, seed=840)
        nin = anonymizer.surrogate("36787753186", "NG_NIN")
        bvn = anonymizer.surrogate("24107152688", "NG_BVN")
        phone = anonymizer.surrogate("+234 803 123 4567", "NG_PHONE")

        assert nin != "36787753186"
        assert bvn != "24107152688"
        assert phone != "+234 803 123 4567"
        assert validate_nigeria_nin(nin)
        assert validate_nigeria_bvn(bvn)
        assert re.fullmatch(r"\+234 80\d \d{3} \d{4}", phone)

    def test_synthetic_fixture_replace_round_trip_has_zero_leakage(self):
        from openmed.core.pii import (
            _apply_safety_sweep_to_result,
            _build_deidentification_result,
        )
        from openmed.processing.outputs import PredictionResult

        rows = [
            json.loads(line)
            for line in Path("tests/fixtures/pii/ng_synthetic_notes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert {row["id"] for row in rows} == {
            "ng-synthetic-registration-nin",
            "ng-synthetic-payment-bvn",
            "ng-synthetic-registration-phone",
            "ng-synthetic-payment-phone",
        }

        for row in rows:
            assert row["metadata"]["synthetic"] is True
            assert row["metadata"]["generated_only"] is True
            entity = row["entities"][0]
            assert row["text"][entity["start"] : entity["end"]] == entity["text"]

            empty_result = PredictionResult(
                text=row["text"],
                entities=[],
                model_name="offline-safety-sweep",
                timestamp="2026-07-16T00:00:00Z",
                metadata={},
            )
            swept_result, added_count = _apply_safety_sweep_to_result(
                row["text"],
                empty_result,
                lang="en",
                locale="en_NG",
            )
            result = _build_deidentification_result(
                row["text"],
                swept_result,
                effective_method="replace",
                keep_year=False,
                date_shift_days=None,
                keep_mapping=False,
                lang="en",
                consistent=True,
                seed=840,
                locale="en_NG",
                use_safety_sweep=True,
            )

            assert added_count == 1
            assert len(result.pii_entities) == 1
            assert result.pii_entities[0].label == entity["label"]
            assert entity["text"] not in result.deidentified_text
            assert result.pii_entities[0].redacted_text != entity["text"]


class TestGhanaKenyaIdentifiers:
    """Validator, pattern, provider, and fixture coverage for Ghana and Kenya."""

    @staticmethod
    def _faker(seed: int, locale: str = "en_KE") -> Faker:
        faker = Faker(locale)
        register_clinical_providers(faker)
        faker.seed_instance(seed)
        return faker

    @staticmethod
    def _fixture_rows():
        return [
            json.loads(line)
            for line in Path("tests/fixtures/pii/gh_ke_synthetic_notes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]

    def test_ghana_card_validator_accepts_ghana_and_resident_prefixes(self):
        assert validate_ghana_card_pin("GHA-012345678-9")
        assert validate_ghana_card_pin("NGA-123456789-0")
        assert validate_ghana_card_pin("nga-123456789-0")

    @pytest.mark.parametrize(
        "value",
        (
            "GHA6899581872",
            "GH-689958187-2",
            "GHA-68995818-2",
            "GHA-689958187-A",
            "GHA-689958187-*",
            "GHA-６８９９５８１８７-2",
            None,
        ),
    )
    def test_ghana_card_validator_rejects_bad_shape(self, value):
        assert not validate_ghana_card_pin(value)

    def test_one_thousand_ghana_surrogates_are_structurally_valid(self):
        faker = self._faker(841)
        for _ in range(1_000):
            surrogate = faker.ghana_card_pin()
            assert re.fullmatch(r"GHA-[0-9]{9}-[0-9]", surrogate)
            assert validate_ghana_card_pin(surrogate)

    def test_providers_are_deterministic_format_preserving_and_distinct(self):
        first = self._faker(1418)
        second = self._faker(1418)

        ghana_source = "GHA-012345678-9"
        ghana_first = first.ghana_card_pin(ghana_source)
        ghana_second = second.ghana_card_pin(ghana_source)
        assert ghana_first == ghana_second
        assert ghana_first != ghana_source
        assert re.fullmatch(r"GHA-[0-9]{9}-[0-9]", ghana_first)
        assert validate_ghana_card_pin(ghana_first)

        for source, method, validator in (
            ("7654321", "kenya_national_id", validate_kenya_national_id),
            ("12345678", "kenya_national_id", validate_kenya_national_id),
            ("246813579", "kenya_maisha_namba", validate_kenya_maisha_namba),
        ):
            surrogate = getattr(first, method)(source)
            assert surrogate != source
            assert len(surrogate) == len(source)
            assert validator(surrogate)

    @pytest.mark.parametrize("value", ("1234567", "12345678"))
    def test_kenya_national_id_validator_accepts_required_lengths(self, value):
        assert validate_kenya_national_id(value)

    @pytest.mark.parametrize(
        "value",
        ("123456", "123456789", "1234 5678", "１２３４５６７", None),
    )
    def test_kenya_national_id_validator_rejects_other_shapes(self, value):
        assert not validate_kenya_national_id(value)

    def test_kenya_maisha_validator_is_strictly_nine_ascii_digits(self):
        assert validate_kenya_maisha_namba("246813579")
        assert not validate_kenya_maisha_namba("24681357")
        assert not validate_kenya_maisha_namba("2468135790")
        assert not validate_kenya_maisha_namba("２４６８１３５７９")
        assert not validate_kenya_maisha_namba(None)

    def test_kenyan_patterns_require_english_or_swahili_identity_context(self):
        from openmed.core.safety_sweep import safety_sweep

        patterns = LOCALE_PII_PATTERNS["en_ke"]
        assert safety_sweep("12345678", [], patterns=patterns) == []
        assert (
            safety_sweep("Lab result 7654321; MRN-87654321.", [], patterns=patterns)
            == []
        )
        assert safety_sweep("246813579", [], patterns=patterns) == []

        cases = (
            ("ID No: 12345678", "KE_NATIONAL_ID", "12345678"),
            ("Nambari ya kitambulisho 7654321", "KE_NATIONAL_ID", "7654321"),
            ("Maisha Namba: 246813579", "KE_MAISHA_NAMBA", "246813579"),
            (
                "Nambari ya Maisha 975318642",
                "KE_MAISHA_NAMBA",
                "975318642",
            ),
        )
        for text, label, expected in cases:
            entities = safety_sweep(text, [], patterns=patterns)
            assert [(entity.label, entity.text) for entity in entities] == [
                (label, expected)
            ]

    def test_ghana_pattern_requires_explicit_card_context(self):
        from openmed.core.safety_sweep import safety_sweep

        patterns = LOCALE_PII_PATTERNS["en_gh"]
        assert safety_sweep("GHA-012345678-9", [], patterns=patterns) == []
        entities = safety_sweep(
            "Ghana Card PIN: GHA-012345678-9",
            [],
            patterns=patterns,
        )
        assert [(entity.label, entity.text) for entity in entities] == [
            ("GH_GHANA_CARD", "GHA-012345678-9")
        ]

    def test_source_labels_normalize_and_route_to_distinct_surrogates(self):
        from openmed.core.labels import ID_NUM, id_subtype_for, normalize_label

        for label in ("GH_GHANA_CARD", "KE_NATIONAL_ID", "KE_MAISHA_NAMBA"):
            assert normalize_label(label) == ID_NUM
            assert id_subtype_for(label) == "national_id"

        ghana = Anonymizer(locale="en_GH", consistent=True, seed=841).surrogate(
            "GHA-012345678-9",
            "GH_GHANA_CARD",
        )
        kenya_id = Anonymizer(locale="en_KE", consistent=True, seed=841).surrogate(
            "12345678",
            "KE_NATIONAL_ID",
        )
        maisha = Anonymizer(locale="en_KE", consistent=True, seed=841).surrogate(
            "246813579",
            "KE_MAISHA_NAMBA",
        )
        assert validate_ghana_card_pin(ghana)
        assert validate_kenya_national_id(kenya_id)
        assert validate_kenya_maisha_namba(maisha)

    def test_locale_aliases_share_kenyan_patterns(self):
        assert all(
            pattern in LOCALE_PII_PATTERNS["sw"]
            for pattern in LOCALE_PII_PATTERNS["en_ke"]
        )
        swahili_patterns = get_patterns_for_language("sw")
        assert all(pattern in swahili_patterns for pattern in LOCALE_PII_PATTERNS["sw"])
        assert LOCALE_PII_PATTERNS["en_gh"] is not LOCALE_PII_PATTERNS["en_ke"]

    def test_synthetic_fixture_offsets_and_hard_negatives(self):
        from openmed.core.safety_sweep import safety_sweep

        rows = self._fixture_rows()
        assert {row["id"] for row in rows} == {
            "gh-synthetic-card-en",
            "ke-synthetic-national-id-en",
            "ke-synthetic-national-id-sw",
            "ke-synthetic-maisha-en",
            "ke-synthetic-maisha-sw",
            "ke-synthetic-hard-negatives",
        }

        for row in rows:
            assert row["metadata"]["synthetic"] is True
            assert row["metadata"]["generated_only"] is True
            for entity in row["entities"]:
                assert row["text"][entity["start"] : entity["end"]] == entity["text"]

        hard_negative = next(
            row for row in rows if row["id"] == "ke-synthetic-hard-negatives"
        )
        assert (
            safety_sweep(
                hard_negative["text"],
                [],
                patterns=LOCALE_PII_PATTERNS["en_ke"],
            )
            == []
        )

    def test_synthetic_fixture_replace_round_trip_has_zero_leakage(self):
        from openmed.core.pii import (
            _apply_safety_sweep_to_result,
            _build_deidentification_result,
        )
        from openmed.processing.outputs import PredictionResult

        gold_count = 0
        leaked_count = 0
        for row in self._fixture_rows():
            if not row["entities"]:
                continue
            language = row["language"]
            lang = "en" if language.startswith("en_") else language
            locale = language if language.startswith("en_") else None
            empty_result = PredictionResult(
                text=row["text"],
                entities=[],
                model_name="offline-safety-sweep",
                timestamp="2026-07-16T00:00:00Z",
                metadata={},
            )
            swept_result, added_count = _apply_safety_sweep_to_result(
                row["text"],
                empty_result,
                lang=lang,
                locale=locale,
            )
            result = _build_deidentification_result(
                row["text"],
                swept_result,
                effective_method="replace",
                keep_year=False,
                date_shift_days=None,
                keep_mapping=False,
                lang=lang,
                consistent=True,
                seed=841,
                locale=locale,
                use_safety_sweep=True,
            )

            assert added_count == len(row["entities"])
            for entity in row["entities"]:
                gold_count += 1
                leaked_count += int(entity["text"] in result.deidentified_text)
            assert result.pii_entities[0].redacted_text != row["entities"][0]["text"]

            replacement = result.pii_entities[0].redacted_text
            if language == "en_GH":
                assert validate_ghana_card_pin(replacement)
            elif row["entities"][0]["label"] == "KE_MAISHA_NAMBA":
                assert validate_kenya_maisha_namba(replacement)
            else:
                assert validate_kenya_national_id(replacement)
                assert len(replacement) == len(row["entities"][0]["text"])

        assert gold_count == 5
        assert leaked_count / gold_count == 0


class TestEgyptMoroccoIdentifiers:
    """Validator, pattern, provider, and fixture coverage for Egypt/Morocco."""

    @staticmethod
    def _faker(seed: int) -> Faker:
        faker = Faker("ar_EG")
        register_clinical_providers(faker)
        faker.seed_instance(seed)
        return faker

    @staticmethod
    def _fixture_rows():
        return [
            json.loads(line)
            for line in Path("tests/fixtures/pii/eg_ma_synthetic_notes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]

    def test_arabic_indic_normalization_is_length_preserving(self):
        original = "٠١٢٣٤٥٦٧٨٩ ۰۱۲۳۴۵۶۷۸۹ 0123456789"
        normalized = normalize_arabic_indic_digits(original)

        assert normalized == "0123456789 0123456789 0123456789"
        assert len(normalized) == len(original)
        assert normalize_arabic_indic_digits(normalized) == normalized

    @pytest.mark.parametrize(
        "value",
        (
            "29801011234567",
            "٣٠٠٠٢٢٩٨٨١٢٣٤٥",
            "۳۰۰۰۲۲۹۸۸۱۲۳۴۵",
        ),
    )
    def test_egyptian_validator_accepts_ascii_and_arabic_digits(self, value):
        assert validate_egyptian_national_id(value)

    @pytest.mark.parametrize(
        "governorate",
        ("00", "05", "10", "20", "30", "36", "50", "87", "89", "99"),
    )
    def test_egyptian_validator_rejects_unpublished_governorates(
        self,
        governorate,
    ):
        assert not validate_egyptian_national_id(f"2980101{governorate}34567")

    @pytest.mark.parametrize(
        "value",
        (
            "19801011234567",
            "29802301234567",
            "29902291234567",
            "2980101123456",
            "298010112345678",
            "29801011234X67",
            None,
        ),
    )
    def test_egyptian_validator_rejects_bad_century_date_or_shape(self, value):
        assert not validate_egyptian_national_id(value)

    def test_one_thousand_generated_egyptian_ids_validate(self):
        faker = self._faker(842)
        for _ in range(1_000):
            assert validate_egyptian_national_id(faker.egyptian_national_id())

    def test_egyptian_surrogate_preserves_fields_and_changes_serial(self):
        source = "29801011234567"
        surrogate = generate_egyptian_national_id(
            source,
            rng=random.Random(842),
        )

        assert validate_egyptian_national_id(surrogate)
        assert surrogate != source
        assert surrogate[0] == source[0]
        assert surrogate[1:9] == source[1:9]
        assert int(surrogate[12]) % 2 == int(source[12]) % 2
        assert surrogate[9:13] != source[9:13]

    @pytest.mark.parametrize(
        "value",
        ("A12345", "BK123456", "z1234567", "BK١٢٣٤٥٦", "A۱۲۳۴۵"),
    )
    def test_moroccan_cin_validator_accepts_region_prefix_and_digit_width(
        self,
        value,
    ):
        assert validate_moroccan_cin(value)

    @pytest.mark.parametrize(
        "value",
        (
            "123456",
            "ABC123456",
            "BK1234",
            "BK12345678",
            "B-123456",
            "B12A456",
            None,
        ),
    )
    def test_moroccan_cin_validator_rejects_bad_shape(self, value):
        assert not validate_moroccan_cin(value)

    def test_moroccan_surrogate_preserves_exact_prefix_and_digit_width(self):
        source = "BK123456"
        surrogate = generate_moroccan_cin(source, rng=random.Random(842))

        assert validate_moroccan_cin(surrogate)
        assert surrogate != source
        assert surrogate.startswith("BK")
        assert len(surrogate) == len(source)

    def test_moroccan_pattern_requires_identity_context_in_both_engines(self):
        from openmed.core.safety_sweep import safety_sweep

        patterns = LOCALE_PII_PATTERNS["ar_ma"]
        for text in (
            "Dose A12345 administered; specimen BK765432 queued.",
            "Medicine BK765432 was logged as a specimen code.",
            "الجرعة A١٢٣٤٥؛ العينة BK٧٦٥٤٣٢.",
        ):
            assert safety_sweep(text, [], patterns=patterns) == []
            assert find_semantic_units(text, patterns) == []

        for text, expected in (
            ("CIN: BK123456", "BK123456"),
            ("Carte nationale A7654321", "A7654321"),
            ("بطاقة التعريف الوطنية BK١٢٣٤٥٦", "BK١٢٣٤٥٦"),
            ("Bitaqa watania A۱۲۳۴۵", "A۱۲۳۴۵"),
        ):
            swept = safety_sweep(text, [], patterns=patterns)
            assert [(entity.label, entity.text) for entity in swept] == [
                ("national_id", expected)
            ]
            semantic = find_semantic_units(text, patterns)
            assert [(item[2], text[item[0] : item[1]]) for item in semantic] == [
                ("national_id", expected)
            ]

    def test_locale_aliases_select_country_specific_patterns(self):
        assert LOCALE_PII_PATTERNS["ar_eg"] is not LOCALE_PII_PATTERNS["ar_ma"]
        assert all(
            pattern in get_patterns_for_language("ar", locale="ar_EG")
            for pattern in LOCALE_PII_PATTERNS["ar_eg"]
        )
        assert all(
            pattern in get_patterns_for_language("ar_MA")
            for pattern in LOCALE_PII_PATTERNS["ar_ma"]
        )
        assert all(
            pattern in get_patterns_for_language("ar")
            for pattern in LOCALE_PII_PATTERNS["ar"]
        )

    def test_ascii_and_arabic_egyptian_ids_have_span_and_mask_parity(self):
        from openmed.core.pii import (
            _apply_safety_sweep_to_result,
            _build_deidentification_result,
        )
        from openmed.processing.outputs import PredictionResult

        outputs = []
        for identifier in ("29801011234567", "٢٩٨٠١٠١١٢٣٤٥٦٧"):
            text = f"الرقم القومي: {identifier}"
            empty = PredictionResult(
                text=text,
                entities=[],
                model_name="offline-safety-sweep",
                timestamp="2026-07-16T00:00:00Z",
                metadata={},
            )
            swept, added = _apply_safety_sweep_to_result(
                text,
                empty,
                lang="ar",
                locale="ar_EG",
            )
            assert added == 1
            masked = _build_deidentification_result(
                text,
                swept,
                effective_method="mask",
                keep_year=False,
                date_shift_days=None,
                keep_mapping=False,
                lang="ar",
                consistent=False,
                seed=None,
                locale="ar_EG",
                use_safety_sweep=True,
            )
            outputs.append(
                (
                    [
                        (entity.start, entity.end, entity.label)
                        for entity in swept.entities
                    ],
                    masked.deidentified_text,
                )
            )

        assert outputs[0] == outputs[1]

    def test_arabic_extract_normalization_preserves_original_surface(self, monkeypatch):
        import openmed
        from openmed.processing.outputs import PredictionResult

        original_id = "٢٩٨٠١٠١١٢٣٤٥٦٧"
        original = f"الرقم القومي: {original_id}"

        def fake_analyze_text(text, **_kwargs):
            assert "29801011234567" in text
            return PredictionResult(
                text=text,
                entities=[],
                model_name="fixture-pii-model",
                timestamp="2026-07-16T00:00:00Z",
            )

        monkeypatch.setattr(openmed, "analyze_text", fake_analyze_text)
        result = openmed.extract_pii(
            original,
            model_name="fixture-pii-model",
            lang="ar",
            locale="ar_EG",
        )

        national_ids = [
            entity for entity in result.entities if entity.label == "national_id"
        ]
        assert len(national_ids) == 1
        assert national_ids[0].text == original_id
        assert original[national_ids[0].start : national_ids[0].end] == original_id

    def test_synthetic_fixture_offsets_pairs_and_hard_negatives(self):
        from openmed.core.safety_sweep import safety_sweep

        rows = self._fixture_rows()
        assert len(rows) == 6
        assert {row["metadata"]["digit_script"] for row in rows} == {
            "ascii",
            "arabic_indic",
        }

        paired_values = {}
        for row in rows:
            assert row["metadata"]["synthetic"] is True
            assert row["metadata"]["generated_only"] is True
            for entity in row["entities"]:
                assert row["text"][entity["start"] : entity["end"]] == entity["text"]
                paired_values.setdefault(row["metadata"]["pair"], set()).add(
                    normalize_arabic_indic_digits(entity["text"])
                )

            if not row["entities"]:
                assert (
                    safety_sweep(
                        row["text"],
                        [],
                        patterns=LOCALE_PII_PATTERNS["ar_ma"],
                    )
                    == []
                )

        assert paired_values == {"eg-1": {"29801011234567"}, "ma-1": {"BK123456"}}

    def test_synthetic_fixture_replace_round_trip_has_zero_leakage(self):
        from openmed.core.pii import (
            _apply_safety_sweep_to_result,
            _build_deidentification_result,
        )
        from openmed.processing.outputs import PredictionResult

        gold_count = 0
        leaked_count = 0
        for row in self._fixture_rows():
            if not row["entities"]:
                continue
            empty = PredictionResult(
                text=row["text"],
                entities=[],
                model_name="offline-safety-sweep",
                timestamp="2026-07-16T00:00:00Z",
                metadata={},
            )
            swept, added = _apply_safety_sweep_to_result(
                row["text"],
                empty,
                lang="ar",
                locale=row["language"],
            )
            result = _build_deidentification_result(
                row["text"],
                swept,
                effective_method="replace",
                keep_year=False,
                date_shift_days=None,
                keep_mapping=False,
                lang="ar",
                consistent=True,
                seed=842,
                locale=row["language"],
                use_safety_sweep=True,
            )

            assert added == len(row["entities"])
            for entity in row["entities"]:
                gold_count += 1
                leaked_count += int(
                    normalize_arabic_indic_digits(entity["text"])
                    in normalize_arabic_indic_digits(result.deidentified_text)
                )

            replacement = result.pii_entities[0].redacted_text
            if row["language"] == "ar_EG":
                assert validate_egyptian_national_id(replacement)
            else:
                assert validate_moroccan_cin(replacement)

        assert gold_count == 4
        assert leaked_count == 0


def _south_african_id_with_luhn(body: str) -> str:
    """Return a 13-digit test value with a valid Luhn digit."""
    assert len(body) == 12 and body.isdigit()
    total = 0
    for index, value in enumerate(int(digit) for digit in body):
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return body + str((10 - total % 10) % 10)


def _decoded_south_african_birth_year(value: str) -> int:
    modern = date(2000 + int(value[:2]), int(value[2:4]), int(value[4:6]))
    if modern <= date.today():
        return modern.year
    return modern.year - 100


class TestSouthAfricanIdentifiers:
    """Validator, pattern, surrogate, and synthetic-fixture coverage for ZA."""

    @staticmethod
    def _faker(seed: int) -> Faker:
        faker = Faker("zu_ZA")
        register_clinical_providers(faker)
        faker.seed_instance(seed)
        return faker

    def test_validator_rejects_bad_date_luhn_citizenship_and_shape(self):
        faker = self._faker(839)
        valid = faker.south_african_id()
        assert validate_south_african_id(valid)

        assert not validate_south_african_id(
            _south_african_id_with_luhn("991332500001")
        )
        assert not validate_south_african_id(
            _south_african_id_with_luhn("900101500021")
        )
        assert not validate_south_african_id(
            valid[:-1] + str((int(valid[-1]) + 1) % 10)
        )
        assert not validate_south_african_id(valid[:-1])
        assert not validate_south_african_id(f"{valid[:6]} {valid[6:]}")

    def test_one_thousand_fresh_surrogates_pass_and_check_mutations_fail(self):
        faker = self._faker(1416)
        for _ in range(1_000):
            surrogate = faker.south_african_id()
            assert validate_south_african_id(surrogate)

            mutated = surrogate[:-1] + str((int(surrogate[-1]) + 1) % 10)
            assert not validate_south_african_id(mutated)

    def test_surrogates_preserve_decade_gender_and_citizenship(self):
        source_faker = self._faker(1416)
        surrogate_faker = self._faker(839)

        for _ in range(100):
            original = source_faker.south_african_id()
            surrogate = surrogate_faker.south_african_id(original)

            assert validate_south_african_id(surrogate)
            assert surrogate != original
            assert _decoded_south_african_birth_year(surrogate) // 10 == (
                _decoded_south_african_birth_year(original) // 10
            )
            assert (int(surrogate[6:10]) >= 5000) == (int(original[6:10]) >= 5000)
            assert surrogate[10] == original[10]

    def test_provider_is_seed_deterministic_and_preserves_phone_prefix(self):
        source = self._faker(1416).south_african_id()
        first = self._faker(839)
        second = self._faker(839)
        assert first.south_african_id(source) == second.south_african_id(source)

        for phone in ("+27 82 123 4567", "0827654321", "27829999999"):
            surrogate = first.za_mobile_number(phone)
            source_digits = re.sub(r"[^0-9]", "", phone)
            surrogate_digits = re.sub(r"[^0-9]", "", surrogate)
            source_national = (
                "0" + source_digits[2:]
                if source_digits.startswith("27")
                else source_digits
            )
            surrogate_national = (
                "0" + surrogate_digits[2:]
                if surrogate_digits.startswith("27")
                else surrogate_digits
            )
            assert surrogate_national[:3] == source_national[:3]

    def test_locale_aliases_match_fixture_phones_and_not_id_runs(self):
        assert LOCALE_PII_PATTERNS["en_za"] is LOCALE_PII_PATTERNS["af"]
        assert LOCALE_PII_PATTERNS["af"] is LOCALE_PII_PATTERNS["zu"]

        rows = [
            json.loads(line)
            for line in Path("tests/fixtures/pii/za_synthetic_notes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        phone_patterns = [
            pattern
            for pattern in LOCALE_PII_PATTERNS["en_za"]
            if pattern.entity_type == "phone_number"
        ]

        assert [row["entities"][1]["text"] for row in rows] == [
            "+27 82 123 4567",
            "0827654321",
            "27829999999",
        ]
        for row in rows:
            id_value, phone = (entity["text"] for entity in row["entities"])
            assert validate_south_african_id(id_value)
            assert any(
                re.fullmatch(pattern.pattern, phone, pattern.flags)
                for pattern in phone_patterns
            )
            assert not any(
                re.search(pattern.pattern, id_value, pattern.flags)
                for pattern in phone_patterns
            )

    def test_synthetic_fixture_replace_round_trip_has_zero_leakage(self):
        from openmed.core.pii import (
            _apply_safety_sweep_to_result,
            _build_deidentification_result,
        )
        from openmed.processing.outputs import PredictionResult

        rows = [
            json.loads(line)
            for line in Path("tests/fixtures/pii/za_synthetic_notes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        for row in rows:
            empty_result = PredictionResult(
                text=row["text"],
                entities=[],
                model_name="offline-safety-sweep",
                timestamp="2026-07-16T00:00:00Z",
                metadata={},
            )
            swept_result, added_count = _apply_safety_sweep_to_result(
                row["text"],
                empty_result,
                lang="en",
                locale="en_ZA",
            )
            result = _build_deidentification_result(
                row["text"],
                swept_result,
                effective_method="replace",
                keep_year=False,
                date_shift_days=None,
                keep_mapping=False,
                lang="en",
                consistent=True,
                seed=839,
                locale="en_ZA",
                use_safety_sweep=True,
            )

            assert added_count == len(row["entities"])
            assert len(result.pii_entities) == len(row["entities"])
            for entity in result.pii_entities:
                source_digits = re.sub(r"[^0-9]", "", entity.text)
                replacement_digits = re.sub(r"[^0-9]", "", entity.redacted_text)
                assert entity.text not in result.deidentified_text
                assert all(
                    source_digits[index : index + 6] not in replacement_digits
                    for index in range(len(source_digits) - 5)
                )


class TestAfricanMobilePlans:
    """Data-driven African phone detection and anonymization regressions."""

    VALID_NUMBERS = {
        "EG": ("+20 10 1234 5678", "0020 11 2468 1357", "012 8765 4321"),
        "GH": ("+233 24 123 4567", "00233 50 246 8135", "055 765 4321"),
        "ET": ("+251 91 234 5678", "00251 72 246 8135", "092 876 5432"),
        "TZ": ("+255 71 234 5678", "00255 68 246 8135", "065 876 5432"),
        "UG": ("+256 772 123 456", "00256 782 246 813", "0752 876 543"),
        "RW": ("+250 78 123 4567", "00250 79 246 8135", "073 876 5432"),
    }

    INVALID_PREFIX_NUMBERS = {
        "EG": "+20 13 1234 5678",
        "GH": "+233 30 123 4567",
        "ET": "+251 11 234 5678",
        "TZ": "+255 22 234 5678",
        "UG": "+256 200 123 456",
        "RW": "+250 25 123 4567",
    }

    @staticmethod
    def _fullmatch(country: str, value: str) -> bool:
        pattern = AFRICAN_MOBILE_PII_PATTERNS[country]
        return re.fullmatch(pattern.pattern, value, pattern.flags) is not None

    def test_plan_table_encodes_all_six_country_contracts(self):
        assert set(AFRICAN_MOBILE_PLANS) == {"EG", "GH", "ET", "TZ", "UG", "RW"}
        assert AFRICAN_MOBILE_PLANS["EG"].mobile_prefixes == (
            "10",
            "11",
            "12",
            "15",
        )
        assert AFRICAN_MOBILE_PLANS["UG"].mobile_prefixes == ("7xx",)
        assert AFRICAN_MOBILE_PLANS["RW"].mobile_prefixes == (
            "72",
            "73",
            "78",
            "79",
        )

    @pytest.mark.parametrize("country", VALID_NUMBERS)
    def test_all_three_renderings_are_detected(self, country):
        for value in self.VALID_NUMBERS[country]:
            assert self._fullmatch(country, value), value

    @pytest.mark.parametrize("country", VALID_NUMBERS)
    def test_invalid_prefixes_and_lengths_are_rejected(self, country):
        valid = self.VALID_NUMBERS[country][0]
        compact = valid.replace(" ", "")

        assert not self._fullmatch(country, self.INVALID_PREFIX_NUMBERS[country])
        assert not self._fullmatch(country, compact[:-1])
        assert not self._fullmatch(country, compact + "0")

    def test_builder_compiles_a_dummy_plan_without_code_changes(self):
        dummy = AfricanMobilePlan(
            country_code="999",
            nsn_length=6,
            mobile_prefixes=("4x",),
            locale_aliases=("zz",),
        )
        pattern = build_african_mobile_pattern(dummy)

        for value in ("+999 42 3456", "00999-47-6543", "049 8765"):
            assert re.fullmatch(pattern, value), value
        assert re.fullmatch(pattern, "+999 32 3456") is None
        assert re.fullmatch(pattern, "+999 42 345") is None

    def test_plan_entries_populate_every_locale_alias(self):
        for country, plan in AFRICAN_MOBILE_PLANS.items():
            expected = AFRICAN_MOBILE_PII_PATTERNS[country]
            for alias in plan.locale_aliases:
                assert expected in LOCALE_PII_PATTERNS[alias]

        for lang in ("am", "sw", "rw"):
            patterns = get_patterns_for_language(lang)
            assert any(
                pattern in patterns for pattern in AFRICAN_MOBILE_PII_PATTERNS.values()
            )

    def test_other_african_national_id_fixture_values_are_not_phones(self):
        fixture_paths = (
            "eg_ma_synthetic_notes.jsonl",
            "ng_synthetic_notes.jsonl",
            "gh_ke_synthetic_notes.jsonl",
            "za_synthetic_notes.jsonl",
            "east_africa_synthetic_notes.jsonl",
        )
        national_ids: set[str] = set()
        for fixture_name in fixture_paths:
            rows = [
                json.loads(line)
                for line in Path("tests/fixtures/pii", fixture_name)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            for row in rows:
                national_ids.update(row.get("identifiers", ()))
                national_ids.update(
                    entity["text"]
                    for entity in row.get("entities", ())
                    if "PHONE" not in entity["label"].upper()
                )

        assert national_ids
        phone_patterns = tuple(AFRICAN_MOBILE_PII_PATTERNS.values())

        for identifier in national_ids:
            assert not any(
                re.search(pattern.pattern, identifier, pattern.flags)
                for pattern in phone_patterns
            ), identifier

    def test_fixture_has_every_country_rendering_and_exact_spans(self):
        fixture_path = Path("tests/fixtures/pii/africa_phones_synthetic.jsonl")
        rows = [
            json.loads(line)
            for line in fixture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert {row["country"] for row in rows} == set(AFRICAN_MOBILE_PLANS)
        assert all(row["synthetic"] is True for row in rows)
        for row in rows:
            assert "Appointment reminder" in row["text"]
            assert "Billing contact" in row["text"]
            assert len(row["phones"]) == 3
            assert any(phone.startswith("+") for phone in row["phones"])
            assert any(phone.startswith("00") for phone in row["phones"])
            assert any(
                phone.startswith("0") and not phone.startswith("00")
                for phone in row["phones"]
            )
            for phone in row["phones"]:
                assert phone in row["text"]
                assert self._fullmatch(row["country"], phone)

    def test_fixture_round_trip_has_zero_phone_leakage(self):
        from openmed.core.pii import (
            _apply_safety_sweep_to_result,
            _build_deidentification_result,
        )
        from openmed.processing.outputs import PredictionResult

        fixture_path = Path("tests/fixtures/pii/africa_phones_synthetic.jsonl")
        rows = [
            json.loads(line)
            for line in fixture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        for row in rows:
            empty_result = PredictionResult(
                text=row["text"],
                entities=[],
                model_name="offline-safety-sweep",
                timestamp="2026-07-18T00:00:00Z",
                metadata={},
            )
            swept_result, added_count = _apply_safety_sweep_to_result(
                row["text"],
                empty_result,
                lang=row["language"],
                locale=row["locale"],
            )
            result = _build_deidentification_result(
                row["text"],
                swept_result,
                effective_method="replace",
                keep_year=False,
                date_shift_days=None,
                keep_mapping=False,
                lang=row["language"],
                consistent=True,
                seed=858,
                locale=row["locale"],
                use_safety_sweep=True,
            )

            assert added_count == len(row["phones"])
            assert all(phone not in result.deidentified_text for phone in row["phones"])

    @pytest.mark.parametrize(
        ("original", "preserved_prefix"),
        [
            ("+251 91 234 5678", "+251 91"),
            ("00233 24 123 4567", "00233 24"),
            ("0752 876 543", "0752"),
        ],
    )
    def test_generator_preserves_operator_prefix_and_changes_input(
        self,
        original,
        preserved_prefix,
    ):
        surrogate = generate_african_phone(original, rng=random.Random(858))

        assert surrogate is not None
        assert surrogate.startswith(preserved_prefix)
        assert surrogate != original


_MPESA_FIXTURE_PATH = Path("tests/fixtures/pii/mpesa_synthetic_receipts.jsonl")


def _mpesa_fixture_rows():
    return [
        json.loads(line)
        for line in _MPESA_FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mpesa_fixture_entities(row):
    from openmed.core.safety_sweep import safety_sweep

    return [
        entity
        for entity in safety_sweep(
            row["text"],
            [],
            lang=row["language"],
            locale=row.get("locale"),
        )
        if entity.label == "mpesa_tx_code"
    ]


class TestMpesaTransactionCodes:
    """M-Pesa receipt codes stay precise and redact without leakage."""

    @pytest.mark.parametrize(
        "code",
        ("TB17CVOCY9", "UC34HJKLM8", "VD56NPQRS7", "AB12CDEFG3"),
    )
    def test_validator_accepts_valid_codes(self, code):
        assert validate_mpesa_transaction_code(code)

    @pytest.mark.parametrize(
        "code",
        (
            "TB1ACVOCY9",
            "tb17cvocy9",
            "TB17CVOCY",
            "TB17CVOCY90",
            "1234567890",
            "ABCDEFGHIJ",
            " TB17CVOCY9",
            "TB17CVOCY9 ",
            None,
        ),
    )
    def test_validator_rejects_invalid_structure(self, code):
        assert not validate_mpesa_transaction_code(code)

    def test_one_thousand_seeded_surrogates_are_valid_and_reproducible(self):
        first_rng = random.Random(859)
        second_rng = random.Random(859)
        generated = [
            generate_mpesa_transaction_code(rng=first_rng) for _ in range(1000)
        ]

        assert generated == [
            generate_mpesa_transaction_code(rng=second_rng) for _ in range(1000)
        ]
        assert all(validate_mpesa_transaction_code(code) for code in generated)

    def test_surrogate_preserves_leading_date_encoding_character(self):
        originals = ("TB17CVOCY9", "UC34HJKLM8", "7D89TUVWX6")
        rng = random.Random(859)

        for original in originals:
            surrogate = generate_mpesa_transaction_code(original, rng=rng)
            assert surrogate[0] == original[0]
            assert surrogate != original
            assert validate_mpesa_transaction_code(surrogate)

    def test_fixture_keyword_gating_and_hard_negatives(self):
        rows = _mpesa_fixture_rows()

        for row in rows:
            observed = {entity.text for entity in _mpesa_fixture_entities(row)}
            assert observed == set(row["expected_codes"]), row["id"]

        ungated = next(row for row in rows if row["id"] == "mpesa-ungated-ids")
        gated = next(row for row in rows if row["id"] == "mpesa-gated-ids")
        assert ungated["candidate_codes"] == gated["candidate_codes"]
        assert not ungated["expected_codes"]
        assert gated["expected_codes"] == gated["candidate_codes"]

    def test_swahili_and_english_receipts_detect_identically(self):
        rows = _mpesa_fixture_rows()
        english = next(row for row in rows if row["id"] == "mpesa-en-shared")
        swahili = next(row for row in rows if row["id"] == "mpesa-sw-shared")

        english_codes = {entity.text for entity in _mpesa_fixture_entities(english)}
        swahili_codes = {entity.text for entity in _mpesa_fixture_entities(swahili)}

        assert english_codes == swahili_codes == {"TB17CVOCY9"}

    def test_anonymizer_round_trip_has_zero_code_leakage(self):
        from openmed.core.pii import (
            _apply_safety_sweep_to_result,
            _build_deidentification_result,
        )
        from openmed.processing.outputs import PredictionResult

        leakage = 0
        for row in _mpesa_fixture_rows():
            if not row["expected_codes"]:
                continue
            empty_result = PredictionResult(
                text=row["text"],
                entities=[],
                model_name="offline-safety-sweep",
                timestamp="2026-07-18T00:00:00Z",
                metadata={},
            )
            swept_result, added_count = _apply_safety_sweep_to_result(
                row["text"],
                empty_result,
                lang=row["language"],
                locale=row.get("locale"),
            )
            result = _build_deidentification_result(
                row["text"],
                swept_result,
                effective_method="replace",
                keep_year=False,
                date_shift_days=None,
                keep_mapping=False,
                lang=row["language"],
                consistent=True,
                seed=859,
                locale=row.get("locale"),
                use_safety_sweep=True,
            )

            # Africa context sweeps may also protect a named facility in the
            # same synthetic receipt; every planted transaction code must
            # still be detected and removed.
            assert added_count >= len(row["expected_codes"])
            leakage += sum(
                original in result.deidentified_text
                for original in row["expected_codes"]
            )
            for entity in result.pii_entities:
                if entity.entity_type != "mpesa_tx_code":
                    continue
                assert entity.surrogate is not None
                assert entity.surrogate[0] == entity.original_text[0]
                assert validate_mpesa_transaction_code(entity.surrogate)

        assert leakage == 0

    def test_repeated_code_has_referential_integrity_with_fixed_seed(self):
        row = next(
            row for row in _mpesa_fixture_rows() if row["id"] == "mpesa-repeated-code"
        )
        entities = _mpesa_fixture_entities(row)
        anonymizer = Anonymizer(lang="en", consistent=True, seed=859)
        surrogates = [
            anonymizer.surrogate(entity.text, entity.label) for entity in entities
        ]

        assert len(surrogates) == 2
        assert len(set(surrogates)) == 1
        assert surrogates[0][0] == row["expected_codes"][0][0]
        assert validate_mpesa_transaction_code(surrogates[0])


_MOBILE_MONEY_FIXTURE_PATH = Path(
    "tests/fixtures/pii/mobile_money_synthetic_billing.jsonl"
)
_MOBILE_MONEY_LABELS = {
    "mobile_money_account",
    "mobile_money_agent",
    "mobile_money_paybill",
    "mobile_money_till",
    "momo_reference",
}


def _mobile_money_fixture_rows():
    return [
        json.loads(line)
        for line in _MOBILE_MONEY_FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mobile_money_fixture_entities(row):
    from openmed.core.safety_sweep import safety_sweep

    return [
        entity
        for entity in safety_sweep(
            row["text"],
            [],
            lang=row["language"],
            locale=row.get("locale"),
        )
        if entity.label in _MOBILE_MONEY_LABELS
    ]


class TestMobileMoneyBillingIdentifiers:
    """Mobile-money billing identifiers are precise and safely replaceable."""

    @pytest.mark.parametrize("value", ("12345", "542542", "7654321"))
    def test_paybill_validator_accepts_five_to_seven_ascii_digits(self, value):
        assert validate_mobile_money_paybill(value)

    @pytest.mark.parametrize("value", ("83290", "832909", "1832909"))
    def test_till_validator_accepts_five_to_seven_ascii_digits(self, value):
        assert validate_mobile_money_till(value)

    @pytest.mark.parametrize(
        "value",
        ("8123456789", "81234567890", "812345678901"),
    )
    def test_momo_validator_accepts_ten_to_twelve_ascii_digits(self, value):
        assert validate_momo_reference(value)

    @pytest.mark.parametrize(
        ("validator", "value"),
        (
            (validate_mobile_money_paybill, "1234"),
            (validate_mobile_money_paybill, "12345678"),
            (validate_mobile_money_paybill, "１２３４５"),
            (validate_mobile_money_till, "123-45"),
            (validate_mobile_money_till, None),
            (validate_momo_reference, "123456789"),
            (validate_momo_reference, "1234567890123"),
            (validate_momo_reference, "12345ABCDE"),
        ),
    )
    def test_validators_reject_wrong_length_or_non_ascii_digits(
        self,
        validator,
        value,
    ):
        assert not validator(value)

    @pytest.mark.parametrize(
        ("method", "validator", "originals"),
        (
            (
                "mobile_money_paybill",
                validate_mobile_money_paybill,
                ("12345", "542542", "7654321"),
            ),
            (
                "mobile_money_till",
                validate_mobile_money_till,
                ("83290", "832909", "1832909"),
            ),
            (
                "mobile_money_agent",
                validate_mobile_money_paybill,
                ("55443", "554433", "5544331"),
            ),
            (
                "momo_reference",
                validate_momo_reference,
                ("8123456789", "81234567890", "812345678901"),
            ),
        ),
    )
    def test_seeded_provider_surrogates_validate_preserve_length_and_differ(
        self,
        method,
        validator,
        originals,
    ):
        for seed, original in enumerate(originals, start=860):
            first = Faker("en_KE")
            second = Faker("en_KE")
            register_clinical_providers(first)
            register_clinical_providers(second)
            first.seed_instance(seed)
            second.seed_instance(seed)

            first_surrogate = getattr(first, method)(original)
            second_surrogate = getattr(second, method)(original)

            assert first_surrogate == second_surrogate
            assert first_surrogate != original
            assert len(first_surrogate) == len(original)
            assert validator(first_surrogate)

    @pytest.mark.parametrize(
        ("label", "original", "validator"),
        (
            ("mobile_money_paybill", "542542", validate_mobile_money_paybill),
            ("mobile_money_till", "83290", validate_mobile_money_till),
            ("mobile_money_agent", "5544331", validate_mobile_money_paybill),
            ("momo_reference", "812345678901", validate_momo_reference),
        ),
    )
    def test_source_labels_route_to_shape_preserving_generators(
        self,
        label,
        original,
        validator,
    ):
        first = Anonymizer(lang="sw", consistent=True, seed=860)
        second = Anonymizer(lang="sw", consistent=True, seed=860)

        first_surrogate = first.surrogate(original, label)
        second_surrogate = second.surrogate(original, label)

        assert first_surrogate == second_surrogate
        assert first_surrogate != original
        assert len(first_surrogate) == len(original)
        assert validator(first_surrogate)

    def test_fixture_detects_expected_values_and_keyword_gates_negatives(self):
        rows = _mobile_money_fixture_rows()
        assert {row["country"] for row in rows} == {
            "Ghana",
            "Kenya",
            "Tanzania",
            "Uganda",
        }

        for row in rows:
            observed = {
                (entity.label, entity.text)
                for entity in _mobile_money_fixture_entities(row)
            }
            expected = {(entity["label"], entity["text"]) for entity in row["expected"]}
            assert observed == expected, row["id"]

        negative = next(
            row for row in rows if row["id"] == "mobile-money-hard-negatives"
        )
        assert negative["candidate_numbers"]
        assert not _mobile_money_fixture_entities(negative)

    def test_account_suffix_requires_paybill_and_captures_only_reference(self):
        from openmed.core.safety_sweep import safety_sweep

        for positive, reference in (
            (
                "Paybill 542542 Account AMINA OTIENO; invoice paid.",
                "AMINA OTIENO",
            ),
            ("Paybill: 542542 Acc: PATIENT-2041. Invoice paid.", "PATIENT-2041"),
        ):
            entities = safety_sweep(positive, [], lang="sw")
            account_entities = [
                entity for entity in entities if entity.label == "mobile_money_account"
            ]

            assert [
                (entity.text, entity.start, entity.end) for entity in account_entities
            ] == [
                (
                    reference,
                    positive.index(reference),
                    positive.index(reference) + len(reference),
                )
            ]

        negative = "Invoice Account AMINA OTIENO; balance paid in cash."
        assert not any(
            entity.label == "mobile_money_account"
            for entity in safety_sweep(negative, [], lang="sw")
        )

    @pytest.mark.parametrize(
        "alias",
        ("sw", "sw_TZ", "en_KE", "en_TZ", "en_GH", "en_UG"),
    )
    def test_locale_aliases_expose_mobile_money_patterns(self, alias):
        patterns = get_patterns_for_language(alias)
        labels = {pattern.entity_type for pattern in patterns}

        assert _MOBILE_MONEY_LABELS <= labels

    def test_anonymizer_round_trip_has_zero_fixture_leakage(self):
        from openmed.core.pii import (
            _apply_safety_sweep_to_result,
            _build_deidentification_result,
        )
        from openmed.processing.outputs import PredictionResult

        leakage = 0
        for row in _mobile_money_fixture_rows():
            if not row["expected"]:
                continue

            empty_result = PredictionResult(
                text=row["text"],
                entities=[],
                model_name="offline-safety-sweep",
                timestamp="2026-07-18T00:00:00Z",
                metadata={},
            )
            swept_result, _added_count = _apply_safety_sweep_to_result(
                row["text"],
                empty_result,
                lang=row["language"],
                locale=row.get("locale"),
            )
            result = _build_deidentification_result(
                row["text"],
                swept_result,
                effective_method="replace",
                keep_year=False,
                date_shift_days=None,
                keep_mapping=False,
                lang=row["language"],
                consistent=True,
                seed=860,
                locale=row.get("locale"),
                use_safety_sweep=True,
            )

            for expected in row["expected"]:
                leakage += expected["text"] in result.deidentified_text

            for entity in result.pii_entities:
                if entity.entity_type not in {
                    "mobile_money_agent",
                    "mobile_money_paybill",
                    "mobile_money_till",
                    "momo_reference",
                }:
                    continue
                assert entity.surrogate is not None
                assert entity.surrogate != entity.original_text
                assert len(entity.surrogate) == len(entity.original_text)
                validator = (
                    validate_momo_reference
                    if entity.entity_type == "momo_reference"
                    else validate_mobile_money_paybill
                )
                assert validator(entity.surrogate)

        assert leakage == 0


_HEALTH_FACILITY_FIXTURE_PATH = Path(
    "tests/fixtures/pii/health_facility_synthetic_referrals.jsonl"
)


def _health_facility_fixture_rows():
    return [
        json.loads(line)
        for line in _HEALTH_FACILITY_FIXTURE_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def _health_facility_entities(row):
    from openmed.core.safety_sweep import safety_sweep

    return [
        entity
        for entity in safety_sweep(
            row["text"],
            [],
            lang=row["language"],
            locale=row.get("locale"),
        )
        if entity.label == "FACILITY_ID"
    ]


class TestAfricanHealthFacilityCodes:
    """Kenya KMHFL and Nigeria HFR identifiers remain precise and synthetic."""

    @pytest.mark.parametrize("value", ("12345", "48321", "99999"))
    def test_kenya_validator_accepts_five_ascii_digits(self, value):
        assert validate_kenya_mfl_code(value)

    @pytest.mark.parametrize(
        "value",
        ("1234", "123456", "１２３４５", "123-45", " 12345", None),
    )
    def test_kenya_validator_rejects_invalid_shapes(self, value):
        assert not validate_kenya_mfl_code(value)

    @pytest.mark.parametrize("value", ("0101110001", "1301220008", "3744239999"))
    def test_nigeria_validator_accepts_structural_ranges(self, value):
        assert validate_nigeria_hfr_code(value)

    @pytest.mark.parametrize(
        "value",
        (
            "0001110001",
            "3801110001",
            "0100110001",
            "0145110001",
            "0101010001",
            "0101310001",
            "0101100001",
            "0101140001",
            "0101110000",
            "010111001",
            "01011100011",
            "０１０１１１０００１",
            None,
        ),
    )
    def test_nigeria_validator_rejects_invalid_fields(self, value):
        assert not validate_nigeria_hfr_code(value)

    def test_facility_source_label_normalizes_as_identifier(self):
        from openmed.core.labels import ID_NUM, normalize_label

        assert normalize_label("FACILITY_ID") == ID_NUM

    @pytest.mark.parametrize("alias", ("sw", "sw_TZ", "en_KE", "en_NG"))
    def test_locale_aliases_expose_facility_id_patterns(self, alias):
        patterns = get_patterns_for_language(alias)

        assert any(pattern.entity_type == "FACILITY_ID" for pattern in patterns)

    def test_fixture_detects_context_gated_facility_codes(self):
        rows = _health_facility_fixture_rows()
        assert {row["country"] for row in rows} == {"Kenya", "Nigeria", "Mixed"}

        for row in rows:
            observed = {entity.text for entity in _health_facility_entities(row)}
            assert observed == set(row["facility_codes"]), row["id"]

    @pytest.mark.parametrize(
        ("text", "lang", "locale"),
        (
            ("Platelet count 12345 and batch 234567 were recorded.", "en", "en_KE"),
            ("Dozi 12345 na hesabu 234567 zilirekodiwa.", "sw", None),
            ("Archive 3744238123 and count 12345 were recorded.", "en", "en_NG"),
        ),
    )
    def test_bare_clinical_numbers_do_not_match(self, text, lang, locale):
        from openmed.core.safety_sweep import safety_sweep

        entities = safety_sweep(text, [], lang=lang, locale=locale)
        assert not [entity for entity in entities if entity.label == "FACILITY_ID"]

    def test_seeded_provider_uses_reserved_bands_disjoint_from_fixtures(self):
        fixture_codes = {
            code
            for row in _health_facility_fixture_rows()
            for code in row["facility_codes"]
        }
        generated_mfl: set[str] = set()
        generated_hfr: set[str] = set()

        for seed in range(861, 961):
            first = Faker("en_US")
            second = Faker("en_US")
            register_clinical_providers(first)
            register_clinical_providers(second)
            first.seed_instance(seed)
            second.seed_instance(seed)

            mfl = first.kmhfl_code()
            hfr = first.hfr_facility_code()
            assert (mfl, hfr) == (
                second.kmhfl_code(),
                second.hfr_facility_code(),
            )
            assert validate_kenya_mfl_code(mfl)
            assert KENYA_MFL_SYNTHETIC_MIN <= int(mfl) <= KENYA_MFL_SYNTHETIC_MAX
            assert validate_nigeria_hfr_code(hfr)
            assert (
                NIGERIA_HFR_SYNTHETIC_SERIAL_MIN
                <= int(hfr[-4:])
                <= NIGERIA_HFR_SYNTHETIC_SERIAL_MAX
            )
            generated_mfl.add(mfl)
            generated_hfr.add(hfr)

        assert generated_mfl.isdisjoint(fixture_codes)
        assert generated_hfr.isdisjoint(fixture_codes)

    @pytest.mark.parametrize(
        ("original", "validator"),
        (
            ("48321", validate_kenya_mfl_code),
            ("3744238123", validate_nigeria_hfr_code),
        ),
    )
    def test_anonymizer_routes_facility_source_label(self, original, validator):
        first = Anonymizer(lang="en", consistent=True, seed=861)
        second = Anonymizer(lang="en", consistent=True, seed=861)

        first_surrogate = first.surrogate(original, "FACILITY_ID")
        second_surrogate = second.surrogate(original, "FACILITY_ID")

        assert first_surrogate == second_surrogate
        assert first_surrogate != original
        assert validator(first_surrogate)

    def test_patient_and_repeated_facility_ids_are_replaced_in_one_pass(self):
        from openmed.core.pii import (
            _apply_safety_sweep_to_result,
            _build_deidentification_result,
        )
        from openmed.processing.outputs import PredictionResult

        leakage = 0
        for row in _health_facility_fixture_rows():
            if not row["facility_codes"]:
                continue

            empty_result = PredictionResult(
                text=row["text"],
                entities=[],
                model_name="offline-safety-sweep",
                timestamp="2026-07-18T00:00:00Z",
                metadata={},
            )
            swept_result, _added_count = _apply_safety_sweep_to_result(
                row["text"],
                empty_result,
                lang=row["language"],
                locale=row.get("locale"),
            )
            result = _build_deidentification_result(
                row["text"],
                swept_result,
                effective_method="replace",
                keep_year=False,
                date_shift_days=None,
                keep_mapping=False,
                lang=row["language"],
                consistent=True,
                seed=861,
                locale=row.get("locale"),
                use_safety_sweep=True,
            )

            for source in row["facility_codes"] + row["patient_identifiers"]:
                leakage += source in result.deidentified_text

            facility_entities = [
                entity
                for entity in result.pii_entities
                if entity.entity_type == "FACILITY_ID"
            ]
            patient_entities = [
                entity
                for entity in result.pii_entities
                if entity.entity_type == "medical_record_number"
            ]
            assert len(facility_entities) == 2
            assert len(patient_entities) == 1
            assert len({entity.surrogate for entity in facility_entities}) == 1

            validator = (
                validate_kenya_mfl_code
                if row["country"] == "Kenya"
                else validate_nigeria_hfr_code
            )
            assert all(
                entity.surrogate is not None and validator(entity.surrogate)
                for entity in facility_entities
            )
            assert all(
                entity.surrogate is not None
                and entity.surrogate != entity.original_text
                for entity in patient_entities
            )

        assert leakage == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
