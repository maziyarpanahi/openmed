"""Locale PHI notes for national-ID-only and user-supplied-model languages."""

from __future__ import annotations

import re
import unicodedata
import warnings

import pytest

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.locales import LANG_TO_LOCALE, NATIONAL_ID_PROVIDERS
from openmed.core.anonymizer.providers import clinical_ids
from openmed.core.labels import CANONICAL_LABELS, ID_NUM, PHONE
from openmed.core.pii_i18n import (
    LANGUAGE_FAKE_DATA,
    NATIONAL_ID_ONLY_LANGUAGES,
    SUPPORTED_LANGUAGES,
    USER_SUPPLIED_MODEL_LANGUAGES,
    validate_nigeria_nin,
    validate_philsys_psn,
    validate_rwanda_id,
)
from openmed.training.synthetic import locale_phi
from openmed.training.synthetic.locale_phi import (
    AUGMENTATION_ONLY_LOCALE_PHI_LANGUAGES,
    LOCALE_PHI_LABELS,
    NATIONAL_ID_ONLY_LOCALE_PHI_LANGUAGES,
    SUPPORTED_LOCALE_PHI_LANGUAGES,
    USER_SUPPLIED_MODEL_LOCALE_PHI_LANGUAGES,
    LocalePhiGenerator,
    generate_locale_phi_examples,
)

_SEEDS = (0, 1, 17, 23, 101, 3801)

_ID_VALIDATORS = {
    "ha": (validate_nigeria_nin, "pii_i18n.validate_nigeria_nin", "nigeria_nin"),
    "ig": (validate_nigeria_nin, "pii_i18n.validate_nigeria_nin", "nigeria_nin"),
    "yo": (validate_nigeria_nin, "pii_i18n.validate_nigeria_nin", "nigeria_nin"),
    "rw": (validate_rwanda_id, "pii_i18n.validate_rwanda_id", "rwanda_id"),
    "tl": (validate_philsys_psn, "pii_i18n.validate_philsys_psn", "philsys_psn"),
}

_ROUTES = {
    "ha": "national_id_only",
    "ig": "national_id_only",
    "yo": "national_id_only",
    "rw": "national_id_only",
    "tl": "national_id_only",
    "ne": "user_supplied_model",
}

_LEGACY_EXAMPLE_METADATA_KEYS = {
    "augmentation_only",
    "contains_real_phi",
    "language",
    "locale",
    "seed",
    "synthetic",
    "synthetic_source",
}

_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")


def _without_spans(text, spans):
    """Return ``text`` with every gold span cut out."""

    pieces = []
    cursor = 0
    for span in sorted(spans, key=lambda item: item.start):
        pieces.append(text[cursor : span.start])
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def test_augmentation_only_languages_are_explicit_and_wired():
    assert NATIONAL_ID_ONLY_LOCALE_PHI_LANGUAGES == ("ha", "ig", "yo", "rw", "tl")
    assert USER_SUPPLIED_MODEL_LOCALE_PHI_LANGUAGES == ("ne",)
    assert AUGMENTATION_ONLY_LOCALE_PHI_LANGUAGES == (
        *NATIONAL_ID_ONLY_LOCALE_PHI_LANGUAGES,
        *USER_SUPPLIED_MODEL_LOCALE_PHI_LANGUAGES,
    )
    assert set(NATIONAL_ID_ONLY_LOCALE_PHI_LANGUAGES) <= NATIONAL_ID_ONLY_LANGUAGES
    assert set(USER_SUPPLIED_MODEL_LOCALE_PHI_LANGUAGES) <= set(
        USER_SUPPLIED_MODEL_LANGUAGES
    )
    # None of them ships a bundled model, and none overlaps the existing set.
    assert not set(AUGMENTATION_ONLY_LOCALE_PHI_LANGUAGES) & SUPPORTED_LANGUAGES
    assert not set(AUGMENTATION_ONLY_LOCALE_PHI_LANGUAGES) & set(
        SUPPORTED_LOCALE_PHI_LANGUAGES
    )


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("language", AUGMENTATION_ONLY_LOCALE_PHI_LANGUAGES)
def test_seeded_generation_is_deterministic_with_exact_span_truth(language, seed):
    example = LocalePhiGenerator(seed=seed).generate(language)
    again = LocalePhiGenerator(seed=seed).generate(language)

    assert example == again
    assert example.language == language
    assert example.locale == LANG_TO_LOCALE[language]
    assert [span.label for span in example.gold_spans] == list(LOCALE_PHI_LABELS)
    assert all(span.label in CANONICAL_LABELS for span in example.gold_spans)

    previous_end = 0
    for span in example.gold_spans:
        assert previous_end <= span.start < span.end <= len(example.text)
        assert example.text[span.start : span.end] == span.text
        assert span.text == span.text.strip()
        assert not _PLACEHOLDER.fullmatch(span.text)
        assert span.metadata["synthetic"] is True
        previous_end = span.end

    item = example.to_training_item()
    assert item["is_synthetic"] is True
    assert item["synthetic_source"] == "locale_phi"
    assert item["labels"] == [span.to_dict() for span in example.gold_spans]


@pytest.mark.parametrize("language", AUGMENTATION_ONLY_LOCALE_PHI_LANGUAGES)
def test_examples_without_bundled_model_are_marked_augmentation_only(language):
    example = LocalePhiGenerator(seed=7).generate(language)

    assert example.metadata["augmentation_only"] is True
    assert example.metadata["bundled_model"] is False
    assert example.metadata["language_route"] == _ROUTES[language]
    assert example.metadata["contains_real_phi"] is False
    assert example.metadata["synthetic"] is True
    assert example.metadata["seed"] == 7


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("language", NATIONAL_ID_ONLY_LOCALE_PHI_LANGUAGES)
def test_national_ids_use_the_validator_backed_provider(language, seed):
    example = LocalePhiGenerator(seed=seed).generate(language)
    id_span = next(span for span in example.gold_spans if span.label == ID_NUM)
    validator, validator_name, method = _ID_VALIDATORS[language]

    assert validator(id_span.text)
    assert id_span.metadata["id_subtype"] == "national_id"
    assert id_span.metadata["id_method"] == method
    assert id_span.metadata["id_method"] == NATIONAL_ID_PROVIDERS[language][1]
    assert id_span.metadata["id_locale"] == NATIONAL_ID_PROVIDERS[language][0]
    assert id_span.metadata["validator"] == validator_name


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("language", USER_SUPPLIED_MODEL_LOCALE_PHI_LANGUAGES)
def test_languages_without_id_provider_record_the_luhn_fallback(language, seed):
    assert language not in NATIONAL_ID_PROVIDERS

    example = LocalePhiGenerator(seed=seed).generate(language)
    id_span = next(span for span in example.gold_spans if span.label == ID_NUM)

    assert clinical_ids.validate_luhn(id_span.text)
    assert id_span.metadata["id_subtype"] == "luhn"
    assert id_span.metadata["id_method"] == "generate_luhn_identifier"
    assert id_span.metadata["id_locale"] == LANG_TO_LOCALE[language]
    assert id_span.metadata["validator"] == "clinical_ids.validate_luhn"


@pytest.mark.parametrize("language", ("xx", "", "EN", "af", "ms", "lv", "fi"))
def test_unregistered_codes_still_fail_with_the_existing_error(language):
    # ``af`` .. ``fi`` are wired national-ID-only routes without a reviewed
    # template, so they stay unsupported rather than passing a catch-all.
    with pytest.raises(ValueError, match="unsupported locale PHI language"):
        LocalePhiGenerator(seed=1).generate(language)


@pytest.mark.parametrize(
    ("language", "registry"),
    (
        ("ha", "NATIONAL_ID_ONLY_LANGUAGES"),
        ("ne", "USER_SUPPLIED_MODEL_LANGUAGES"),
    ),
)
def test_unwired_route_fails_closed(monkeypatch, language, registry):
    monkeypatch.setattr(locale_phi, registry, frozenset())

    with pytest.raises(ValueError, match="is not wired in OpenMed"):
        LocalePhiGenerator(seed=1).generate(language)


def test_placeholder_surrogate_never_becomes_a_gold_span(monkeypatch):
    real_surrogate = Anonymizer.surrogate

    def phone_fallback(self, original, label, *, lang=None, locale=None):
        if label == PHONE:
            return f"[{label}]"
        return real_surrogate(self, original, label, lang=lang, locale=locale)

    monkeypatch.setattr(Anonymizer, "surrogate", phone_fallback)

    with pytest.raises(RuntimeError, match="placeholder for 'tl' field 'phone'"):
        LocalePhiGenerator(seed=1).generate("tl")


@pytest.mark.parametrize("language", AUGMENTATION_ONLY_LOCALE_PHI_LANGUAGES)
def test_templates_contain_no_names_addresses_or_identifiers(language):
    literals = locale_phi._AUGMENTATION_ONLY_TEMPLATES[language]
    skeleton = "".join(literals)

    assert len(literals) == len(LOCALE_PHI_LABELS) + 1
    assert all(unicodedata.normalize("NFC", part) == part for part in literals)
    assert not any(char.isdigit() for char in skeleton)
    assert "@" not in skeleton
    assert "http" not in skeleton.casefold()

    # Curated names and places for the language never appear as whole words.
    fake_data = LANGUAGE_FAKE_DATA.get(language, {})
    for key in ("NAME", "FIRST_NAME", "LAST_NAME", "STREET_ADDRESS", "LOCATION"):
        for value in fake_data.get(key, ()):
            word = rf"(?<!\w){re.escape(value.casefold())}(?!\w)"
            assert not re.search(word, skeleton.casefold()), (language, key)


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("language", AUGMENTATION_ONLY_LOCALE_PHI_LANGUAGES)
def test_phi_values_come_only_from_surrogate_providers(monkeypatch, language, seed):
    provided: list[str] = []
    real_surrogate = Anonymizer.surrogate
    real_luhn = clinical_ids.generate_luhn_identifier

    def recording_surrogate(self, *args, **kwargs):
        value = real_surrogate(self, *args, **kwargs)
        provided.append(" ".join(str(value).split()))
        return value

    def recording_luhn(*args, **kwargs):
        value = real_luhn(*args, **kwargs)
        provided.append(value)
        return value

    monkeypatch.setattr(Anonymizer, "surrogate", recording_surrogate)
    monkeypatch.setattr(clinical_ids, "generate_luhn_identifier", recording_luhn)

    example = LocalePhiGenerator(seed=seed).generate(language)
    literals = locale_phi._AUGMENTATION_ONLY_TEMPLATES[language]

    # Every character outside a gold span is template text, and every gold
    # span is a value a surrogate or identifier provider returned.
    assert _without_spans(example.text, example.gold_spans) == "".join(literals)
    assert len(provided) == len(LOCALE_PHI_LABELS)
    assert sorted(span.text for span in example.gold_spans) == sorted(provided)


def test_nepali_template_renders_devanagari():
    example = LocalePhiGenerator(seed=29).generate("ne")
    skeleton = _without_spans(example.text, example.gold_spans)

    assert any("ऀ" <= char <= "ॿ" for char in skeleton)


@pytest.mark.parametrize("locale", ("fil_PH", "tl_PH", "en_PH"))
def test_philippine_phone_surrogate_is_a_mobile_number_not_a_placeholder(locale):
    anonymizer = Anonymizer(lang="tl", consistent=True, seed=5)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        value = anonymizer.surrogate("", PHONE, lang="tl", locale=locale)

    assert re.fullmatch(r"(?:\+63|0)9\d{2}[\s-]?\d{3}[\s-]?\d{4}", value)


def test_generate_all_output_for_existing_languages_is_unchanged():
    examples = LocalePhiGenerator(seed=101).generate_all()

    assert tuple(example.language for example in examples) == (
        SUPPORTED_LOCALE_PHI_LANGUAGES
    )
    assert examples == generate_locale_phi_examples(seed=101)
    for example in examples:
        assert set(example.metadata) == _LEGACY_EXAMPLE_METADATA_KEYS
        assert _without_spans(example.text, example.gold_spans) == "".join(
            locale_phi._TEMPLATES[example.language]
        )

    # Requesting the augmentation-only languages appends them without
    # perturbing the existing examples, even after they were generated first.
    generator = LocalePhiGenerator(seed=101)
    for language in AUGMENTATION_ONLY_LOCALE_PHI_LANGUAGES:
        generator.generate(language)
    extended = generator.generate_all(include_augmentation_only=True)

    assert extended[: len(examples)] == examples
    assert tuple(example.language for example in extended[len(examples) :]) == (
        AUGMENTATION_ONLY_LOCALE_PHI_LANGUAGES
    )
    assert all(
        example.metadata["bundled_model"] is False
        for example in extended[len(examples) :]
    )


def test_unseeded_generate_all_keeps_the_existing_language_order():
    examples = LocalePhiGenerator().generate_all(include_augmentation_only=True)

    assert tuple(example.language for example in examples) == (
        *SUPPORTED_LOCALE_PHI_LANGUAGES,
        *AUGMENTATION_ONLY_LOCALE_PHI_LANGUAGES,
    )
