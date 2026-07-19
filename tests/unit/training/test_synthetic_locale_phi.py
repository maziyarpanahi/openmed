from __future__ import annotations

import pytest

from openmed.core.anonymizer.locales import NATIONAL_ID_PROVIDERS
from openmed.core.anonymizer.providers import clinical_ids
from openmed.core.labels import CANONICAL_LABELS, ID_NUM
from openmed.core.pii_i18n import (
    SUPPORTED_LANGUAGES,
    validate_aadhaar,
    validate_dutch_bsn,
    validate_french_nir,
    validate_german_steuer_id,
    validate_indonesian_nik,
    validate_israeli_teudat_zehut,
    validate_italian_codice_fiscale,
    validate_japanese_my_number,
    validate_korean_rrn,
    validate_portuguese_cpf,
    validate_romanian_cnp,
    validate_spanish_nie,
    validate_thai_national_id,
    validate_turkish_tckn,
)
from openmed.training.synthetic import (
    LOCALE_PHI_LABELS,
    SUPPORTED_LOCALE_PHI_LANGUAGES,
    LocalePhiGenerator,
    generate_locale_phi_examples,
)

_ID_VALIDATORS = {
    "en": clinical_ids.validate_ssn,
    "fr": validate_french_nir,
    "de": validate_german_steuer_id,
    "it": validate_italian_codice_fiscale,
    "es": validate_spanish_nie,
    "nl": validate_dutch_bsn,
    "hi": validate_aadhaar,
    "te": validate_aadhaar,
    "pt": validate_portuguese_cpf,
    "tr": validate_turkish_tckn,
    "he": validate_israeli_teudat_zehut,
    "ja": validate_japanese_my_number,
    "id": validate_indonesian_nik,
    "th": validate_thai_national_id,
    "ko": validate_korean_rrn,
    "ro": validate_romanian_cnp,
}

_SCRIPT_RANGES = {
    "ar": ("\u0600", "\u06ff"),
    "he": ("\u0590", "\u05ff"),
    "hi": ("\u0900", "\u097f"),
    "ja": ("\u3040", "\u9fff"),
    "ko": ("\uac00", "\ud7a3"),
    "te": ("\u0c00", "\u0c7f"),
    "th": ("\u0e00", "\u0e7f"),
}


def test_supported_languages_match_wired_language_set():
    assert set(SUPPORTED_LOCALE_PHI_LANGUAGES) == SUPPORTED_LANGUAGES


@pytest.mark.parametrize("language", SUPPORTED_LOCALE_PHI_LANGUAGES)
def test_locale_phi_example_emits_text_and_canonical_spans(language):
    example = LocalePhiGenerator(seed=17).generate(language)

    assert example.text
    assert example.metadata["synthetic"] is True
    assert example.metadata["augmentation_only"] is True
    assert example.metadata["contains_real_phi"] is False
    assert {span.label for span in example.gold_spans} == set(LOCALE_PHI_LABELS)
    assert all(span.label in CANONICAL_LABELS for span in example.gold_spans)

    for span in example.gold_spans:
        assert span.start < span.end
        assert example.text[span.start : span.end] == span.text
        assert span.metadata["synthetic"] is True

    item = example.to_training_item()
    assert item["is_synthetic"] is True
    assert item["synthetic_source"] == "locale_phi"
    assert item["labels"] == [span.to_dict() for span in example.gold_spans]


@pytest.mark.parametrize("language", SUPPORTED_LOCALE_PHI_LANGUAGES)
def test_locale_phi_identifier_is_checksum_valid(language):
    example = LocalePhiGenerator(seed=23).generate(language)
    id_span = next(span for span in example.gold_spans if span.label == ID_NUM)

    if language in NATIONAL_ID_PROVIDERS:
        assert id_span.metadata["id_subtype"] == "national_id"
        assert _ID_VALIDATORS[language](id_span.text)
    else:
        assert id_span.metadata["id_subtype"] == "luhn"
        assert clinical_ids.validate_luhn(id_span.text)


def test_locale_phi_generation_is_deterministic_per_seed():
    first = generate_locale_phi_examples(seed=101)
    second = generate_locale_phi_examples(seed=101)

    assert first == second


@pytest.mark.parametrize("language", ("ar", "he", "hi", "ja", "te", "th"))
def test_non_latin_locale_templates_render_target_script(language):
    example = LocalePhiGenerator(seed=29).generate(language)
    low, high = _SCRIPT_RANGES[language]

    assert any(low <= char <= high for char in example.text)


def test_locale_phi_generator_rejects_unsupported_language():
    with pytest.raises(ValueError, match="unsupported locale PHI language"):
        LocalePhiGenerator(seed=1).generate("zh")
