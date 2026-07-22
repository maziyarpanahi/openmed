"""Acceptance tests for locale-correct synthetic Indian surrogates."""

from __future__ import annotations

import random
import re
import warnings

import pytest

from openmed.core import labels as L
from openmed.core.anonymizer import (
    Anonymizer,
    IndiaSurrogateProvider,
    register_clinical_provider,
)
from openmed.core.anonymizer.engine import _FORMAT_PRESERVE_DAY_FIRST_LOCALES
from openmed.core.anonymizer.locales import locale_coherence_report, resolve_locale
from openmed.core.anonymizer.providers import clinical_ids
from openmed.core.anonymizer.providers.clinical_ids import (
    generate_abha,
    generate_gstin,
    generate_pan,
    validate_abha,
    validate_gstin,
    validate_indian_phone,
    validate_indian_pin,
    validate_pan,
)
from openmed.core.anonymizer.providers.registry_ids import get_national_id
from openmed.core.anonymizer.registry import (
    _DAY_FIRST_LOCALES,
    LABEL_GENERATORS,
    _gen_india_id_num,
    _gen_india_person,
    _gen_india_phone,
    _gen_india_street_address,
    _gen_india_zipcode,
)
from openmed.core.pii_i18n import validate_aadhaar


def _valid_aadhaar(seed: int) -> str:
    anonymizer = Anonymizer(lang="te")
    faker = anonymizer._get_faker("en_IN")
    faker.seed_instance(seed)
    return faker.aadhaar()


@pytest.mark.parametrize(
    ("lang", "expected_locale"),
    (("hi", "hi_IN"), ("te", "en_IN")),
)
def test_hindi_and_telugu_use_india_locale_surrogates(
    lang: str,
    expected_locale: str,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anonymizer = Anonymizer(lang=lang, consistent=True, seed=675)
        person = anonymizer.surrogate("उदाहरण व्यक्ति", "PERSON")
        address = anonymizer.surrogate("42 परीक्षण मार्ग 123456", "STREET_ADDRESS")
        phone = anonymizer.surrogate("+91 98765 43210", "PHONE")
        pin = anonymizer.surrogate("123456", "ZIPCODE")

    assert resolve_locale(lang) == expected_locale
    assert anonymizer._get_faker(expected_locale).locales == [expected_locale]
    assert person and person != "उदाहरण व्यक्ति"
    if lang == "hi":
        assert re.search(r"[\u0900-\u097f]", person)
    else:
        assert re.search(r"[A-Za-z]", person)

    address_pin = re.search(r"(?<!\d)([1-9]\d{5})(?!\d)", address)
    assert address_pin is not None
    assert validate_indian_pin(address_pin.group(1))
    assert validate_indian_phone(phone)
    assert validate_indian_pin(pin)


@pytest.mark.parametrize(("lang", "locale"), (("hi", "hi_IN"), ("te", "en_IN")))
def test_indian_ids_round_trip_through_engine(lang: str, locale: str) -> None:
    originals = {
        "aadhaar": _valid_aadhaar(11),
        "pan": generate_pan(rng=random.Random(12)),
        "gstin": generate_gstin(rng=random.Random(13)),
        "abha": generate_abha(rng=random.Random(14)),
    }
    validators = {
        "aadhaar": validate_aadhaar,
        "pan": validate_pan,
        "gstin": validate_gstin,
        "abha": validate_abha,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anonymizer = Anonymizer(lang=lang, consistent=True, seed=675)
        generated = {
            id_type: anonymizer.surrogate(original, id_type, locale=locale)
            for id_type, original in originals.items()
        }

    for id_type, value in generated.items():
        assert validators[id_type](value), f"invalid {id_type} surrogate: {value!r}"
        assert value != originals[id_type]


@pytest.mark.parametrize("alias", ("in", "hi", "te", "en_IN", "hi_IN"))
def test_indian_registry_specs_generate_values_accepted_by_their_validators(
    alias: str,
) -> None:
    faker = Anonymizer(lang="te")._get_faker("en_IN")
    faker.seed_instance(675)
    for id_type in ("aadhaar", "pan", "gstin", "abha"):
        spec = get_national_id(alias, id_type)
        assert spec is not None
        value = getattr(faker, spec.faker_method)()
        assert spec.validate(value), f"invalid {alias}/{id_type}: {value!r}"


def test_locale_coherence_report_has_no_india_id_gaps() -> None:
    rows = {row["language"]: row for row in locale_coherence_report()}
    expected_types = ["aadhaar", "pan", "gstin", "abha"]
    expected_methods = ["aadhaar", "pan", "gstin", "abha"]

    for lang, locale in (("hi", "hi_IN"), ("te", "en_IN")):
        assert rows[lang]["id_types"] == expected_types
        assert rows[lang]["id_providers"] == expected_methods
        assert rows[lang]["id_locale"] == locale


@pytest.mark.parametrize("lang", ("hi", "te"))
def test_seeded_consistent_indian_surrogates_are_stable(lang: str) -> None:
    originals = {
        "PERSON": "उदाहरण व्यक्ति",
        "STREET_ADDRESS": "42 परीक्षण मार्ग 123456",
        "PHONE": "+91 98765 43210",
        "PAN": generate_pan(rng=random.Random(21)),
        "GSTIN": generate_gstin(rng=random.Random(22)),
        "ABHA": generate_abha(rng=random.Random(23)),
    }

    def generate() -> dict[str, str]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            anonymizer = Anonymizer(lang=lang, consistent=True, seed=675)
            return {
                label: anonymizer.surrogate(original, label)
                for label, original in originals.items()
            }

    assert generate() == generate()


@pytest.mark.parametrize("lang", ("hi", "te"))
def test_indian_surrogate_leakage_gate(lang: str) -> None:
    originals = {
        "PERSON": "उदाहरण व्यक्ति",
        "ZIPCODE": "123456",
        "PAN": generate_pan(rng=random.Random(31)),
        "GSTIN": generate_gstin(rng=random.Random(32)),
        "ABHA": generate_abha(rng=random.Random(33)),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anonymizer = Anonymizer(lang=lang, consistent=True, seed=675)
        surrogates = {
            label: anonymizer.surrogate(original, label)
            for label, original in originals.items()
        }

    serialized = "\n".join(surrogates.values()).casefold()
    for original in originals.values():
        assert original.casefold() not in serialized
    for name_token in originals["PERSON"].split():
        assert name_token.casefold() not in surrogates["PERSON"].casefold()
    assert originals["PAN"][5:9] not in surrogates["PAN"]
    assert originals["GSTIN"][7:11] not in surrogates["GSTIN"]


def test_register_clinical_provider_applies_india_bundle_to_fresh_instances() -> None:
    register_clinical_provider(IndiaSurrogateProvider)
    try:
        first = Anonymizer(lang="hi")._get_faker("hi_IN")
        second = Anonymizer(lang="te")._get_faker("en_IN")
        for faker in (first, second):
            assert validate_indian_pin(faker.indian_pin())
            assert validate_indian_phone(faker.indian_phone_number())
            assert validate_pan(faker.pan())
            assert validate_gstin(faker.gstin())
            assert validate_abha(faker.abha())
    finally:
        clinical_ids._extra_providers.remove(IndiaSurrogateProvider)


def test_india_label_generators_and_day_first_locales_are_registered() -> None:
    assert LABEL_GENERATORS[L.PERSON] is _gen_india_person
    assert LABEL_GENERATORS[L.PHONE] is _gen_india_phone
    assert LABEL_GENERATORS[L.STREET_ADDRESS] is _gen_india_street_address
    assert LABEL_GENERATORS[L.ZIPCODE] is _gen_india_zipcode
    assert LABEL_GENERATORS[L.ID_NUM] is _gen_india_id_num
    assert {"en_IN", "hi_IN"} <= _DAY_FIRST_LOCALES
    assert {"en_IN", "hi_IN"} <= _FORMAT_PRESERVE_DAY_FIRST_LOCALES
