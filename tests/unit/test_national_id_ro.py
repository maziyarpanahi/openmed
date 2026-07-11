"""Tests for the Romanian CNP (Cod Numeric Personal) national-ID pack."""

from __future__ import annotations

import random
import unicodedata
from datetime import date

import pytest

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.locales import (
    LANG_TO_LOCALE,
    locale_coherence_report,
)
from openmed.core.pii_entity_merger import PIIPattern
from openmed.core.pii_i18n import (
    get_patterns_for_language,
    validate_romanian_cnp,
)
from openmed.core.safety_sweep import SAFETY_SWEEP_SOURCE, safety_sweep

# Documented CNP control-digit weights ("279146358279").
_CNP_WEIGHTS = (2, 7, 9, 1, 4, 6, 3, 5, 8, 2, 7, 9)


def _cnp_from_body(body12: str) -> str:
    """Return ``body12`` plus its computed CNP control digit."""
    assert len(body12) == 12
    total = sum(w * int(d) for w, d in zip(_CNP_WEIGHTS, body12))
    control = total % 11
    if control == 10:
        control = 1
    return body12 + str(control)


def _cnp_birth_date(cnp: str) -> date:
    """Decode a documented 1-6 CNP birth date."""
    century = {1: 1900, 2: 1900, 3: 1800, 4: 1800, 5: 2000, 6: 2000}[int(cnp[0])]
    return date(century + int(cnp[1:3]), int(cnp[3:5]), int(cnp[5:7]))


# ---------------------------------------------------------------------------
# validate_romanian_cnp
# ---------------------------------------------------------------------------


class TestValidateRomanianCnp:
    """Tests for :func:`validate_romanian_cnp`."""

    def test_valid_1900s_male_cnp(self):
        # 1980-01-01, county 40 (Bucuresti), serial 018 -> control 1.
        assert validate_romanian_cnp("1800101400181") is True

    def test_valid_2000s_female_cnp(self):
        # S=6 (2000s female), 2005-07-22, county 12, serial 345.
        cnp = _cnp_from_body("605072212345")
        assert validate_romanian_cnp(cnp) is True

    def test_accepts_remainder_ten_control_digit_one(self):
        # A CNP whose weighted sum mod 11 == 10 must map to control digit 1.
        cnp = "1800101400181"
        total = sum(w * int(d) for w, d in zip(_CNP_WEIGHTS, cnp[:12]))
        assert total % 11 == 10
        assert cnp[-1] == "1"
        assert validate_romanian_cnp(cnp) is True

    def test_rejects_bad_checksum(self):
        valid = "1800101400181"
        bad = valid[:-1] + str((int(valid[-1]) + 1) % 10)
        assert validate_romanian_cnp(bad) is False

    def test_rejects_too_short(self):
        assert validate_romanian_cnp("180010140018") is False

    def test_rejects_too_long(self):
        assert validate_romanian_cnp("18001014001810") is False

    @pytest.mark.parametrize("gender_code", [0, 7, 8, 9])
    def test_rejects_undocumented_gender_or_century_codes(self, gender_code):
        cnp = _cnp_from_body(f"{gender_code}80010140018")
        assert validate_romanian_cnp(cnp) is False

    def test_rejects_impossible_date_feb30(self):
        # 1980-02-30 is not a real date.
        cnp = _cnp_from_body("180023040018")
        assert validate_romanian_cnp(cnp) is False

    def test_rejects_impossible_month_13(self):
        cnp = _cnp_from_body("181301040018")
        assert validate_romanian_cnp(cnp) is False

    def test_rejects_checksum_valid_future_birth_date(self):
        cnp = _cnp_from_body("599123140001")
        assert _cnp_birth_date(cnp) == date(2099, 12, 31)
        assert validate_romanian_cnp(cnp) is False

    @pytest.mark.parametrize("county", [0, 47, 48, 49, 50, 53, 99])
    def test_rejects_unassigned_county_codes(self, county):
        cnp = _cnp_from_body(f"1800101{county:02d}018")
        assert validate_romanian_cnp(cnp) is False

    @pytest.mark.parametrize("county", [1, 40, 46, 51, 52, 70])
    def test_accepts_assigned_county_codes(self, county):
        cnp = _cnp_from_body(f"1800101{county:02d}018")
        assert validate_romanian_cnp(cnp) is True

    def test_rejects_zero_serial(self):
        cnp = _cnp_from_body("180010140000")
        assert validate_romanian_cnp(cnp) is False

    @pytest.mark.parametrize(
        "candidate",
        [
            "1 800101 400181",
            "1-800101-400181",
            "1.800101.400181",
            "1/800101/400181",
            "1x800101x400181",
        ],
    )
    def test_rejects_internal_separators(self, candidate):
        assert validate_romanian_cnp(candidate) is False

    def test_ignores_surrounding_whitespace_only(self):
        assert validate_romanian_cnp("  1800101400181\n") is True

    def test_non_digit_input(self):
        assert validate_romanian_cnp("abcdefghijklm") is False

    def test_empty_string(self):
        assert validate_romanian_cnp("") is False


# ---------------------------------------------------------------------------
# Faker provider round-trip
# ---------------------------------------------------------------------------


class TestFakerProviderRoundTrip:
    """Generated CNP surrogates must pass the registered validator."""

    def test_cnp_generator_is_seeded_valid_and_never_future(self):
        from openmed.core.anonymizer.providers.clinical_ids import (
            generate_romanian_cnp,
        )

        first_rng = random.Random(472)
        second_rng = random.Random(472)
        generated = [generate_romanian_cnp(rng=first_rng) for _ in range(1000)]

        assert generated == [generate_romanian_cnp(rng=second_rng) for _ in range(1000)]
        for cnp in generated:
            assert len(cnp) == 13
            assert cnp[0] in "123456"
            assert _cnp_birth_date(cnp) <= date.today()
            assert validate_romanian_cnp(cnp) is True

    def test_cnp_registered_anonymizer_path_round_trips(self):
        anon = Anonymizer(lang="ro", consistent=True, seed=42)

        surrogate = anon.surrogate("1800101400181", "national_id")

        assert validate_romanian_cnp(surrogate) is True


# ---------------------------------------------------------------------------
# Language-pack wiring
# ---------------------------------------------------------------------------


class TestRomanianPackWiring:
    """The Romanian pack is discoverable through the shared framework."""

    def test_locale_maps_to_ro_ro(self):
        assert LANG_TO_LOCALE["ro"] == "ro_RO"

    def test_cnp_pattern_is_national_id_only(self):
        patterns = [
            pattern
            for pattern in get_patterns_for_language("ro")
            if pattern.entity_type == "national_id"
            and pattern.validator is validate_romanian_cnp
        ]

        assert len(patterns) == 1
        assert isinstance(patterns[0], PIIPattern)

    def test_locale_coherence_report_includes_cnp_provider(self):
        rows = {row["language"]: row for row in locale_coherence_report()}

        assert rows["ro"]["locale"] == "ro_RO"
        assert rows["ro"]["id_locale"] == "ro_RO"
        assert rows["ro"]["id_providers"] == ["romanian_cnp"]

    @pytest.mark.parametrize("id_type", ["cnp"])
    def test_registry_lookup_returns_validating_spec(self, id_type):
        from faker import Faker

        from openmed.core.anonymizer.providers.clinical_ids import (
            register_clinical_providers,
        )
        from openmed.core.anonymizer.providers.registry_ids import get_national_id

        spec = get_national_id("ro", id_type)
        assert spec is not None
        assert spec.validate is validate_romanian_cnp

        faker = Faker("ro_RO")
        register_clinical_providers(faker)
        faker.seed_instance(7)
        surrogate = getattr(faker, spec.faker_method)()

        assert spec.validate(surrogate)


@pytest.mark.parametrize(
    "address",
    (
        "Șoseaua Ștefan cel Mare 15",
        "Şoseaua Ştefan cel Mare 15",
        unicodedata.normalize("NFD", "Șoseaua Ștefan cel Mare 15"),
    ),
)
def test_romanian_safety_sweep_preserves_unicode_address_offsets(address):
    text = f"Adresă {address}, cod poștal 010101 București."
    entities = safety_sweep(text, [], lang="ro")
    address_entities = [
        entity for entity in entities if entity.label == "street_address"
    ]

    assert len(address_entities) == 1
    entity = address_entities[0]
    assert entity.text == address
    assert entity.start == text.index(address)
    assert entity.end == entity.start + len(address)
    assert text[entity.start : entity.end] == address
    assert entity.metadata["source"] == SAFETY_SWEEP_SOURCE


@pytest.mark.parametrize(
    "context",
    (
        "cod poștal",
        "cod poştal",
        unicodedata.normalize("NFD", "cod poștal"),
    ),
)
def test_romanian_safety_sweep_accepts_unicode_postcode_context(context):
    text = f"{context} 010101"
    entities = safety_sweep(text, [], lang="ro")

    assert [(entity.text, entity.label) for entity in entities] == [
        ("010101", "postcode")
    ]
