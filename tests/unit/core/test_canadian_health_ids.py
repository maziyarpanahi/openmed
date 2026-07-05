"""Tests for Canadian SIN and provincial health-card ID support.

All identifiers used here are synthetic, checksum-constructed values; none are
real Social Insurance Numbers or health-card numbers.
"""

from __future__ import annotations

import pytest
from faker import Faker

from openmed.core.anonymizer.providers.clinical_ids import (
    register_clinical_providers,
    validate_bc_phn,
    validate_canadian_sin,
    validate_ontario_health_card,
)
from openmed.core.anonymizer.providers.registry_ids import get_national_id
from openmed.core.pii import extract_pii
from openmed.processing.outputs import PredictionResult

# Synthetic, Luhn-valid SIN with a non-zero province prefix.
VALID_SIN = "130692544"
# Same digits with the leading pair transposed: no longer Luhn-valid.
TRANSPOSED_SIN = "310692544"
# Synthetic, Luhn-valid 10-digit Ontario (OHIP) health-card core.
VALID_ONTARIO_CARD = "6317048459"
# Synthetic, mod-11-valid BC Personal Health Number (leading 9).
VALID_BC_PHN = "9291417779"


def test_validate_canadian_sin_accepts_luhn_valid_number():
    assert validate_canadian_sin(VALID_SIN)
    assert validate_canadian_sin("130 692 544")
    assert validate_canadian_sin("130-692-544")


def test_validate_canadian_sin_rejects_transposed_and_malformed():
    assert not validate_canadian_sin(TRANSPOSED_SIN)
    # A zero-prefixed SIN is not assigned to a person.
    assert not validate_canadian_sin("046454286")
    assert not validate_canadian_sin("13069254")  # too short
    assert not validate_canadian_sin("1306925440")  # too long


def test_validate_ontario_health_card_accepts_core_and_version_code():
    assert validate_ontario_health_card(VALID_ONTARIO_CARD)
    assert validate_ontario_health_card("6317-048-459")
    # Optional two-letter version code, spaced or hyphenated.
    assert validate_ontario_health_card(f"{VALID_ONTARIO_CARD}-QC")
    assert validate_ontario_health_card("6317 048 459 QC")


def test_validate_ontario_health_card_rejects_corrupted_and_bad_version():
    # Flip the check digit.
    assert not validate_ontario_health_card("6317048458")
    # Single-letter version code is malformed.
    assert not validate_ontario_health_card(f"{VALID_ONTARIO_CARD}-Q")
    # Three-letter version code is malformed.
    assert not validate_ontario_health_card(f"{VALID_ONTARIO_CARD}-QCX")
    assert not validate_ontario_health_card("631704845")  # too short


def test_validate_bc_phn_accepts_valid_and_rejects_corrupted():
    assert validate_bc_phn(VALID_BC_PHN)
    assert validate_bc_phn("9291 417 779")
    # Corrupt the check digit.
    assert not validate_bc_phn("9291417770")
    # PHNs always start with 9.
    assert not validate_bc_phn("1291417779")
    assert not validate_bc_phn("929141777")  # too short


@pytest.mark.parametrize(
    ("id_type", "faker_method"),
    (
        ("sin", "canadian_sin"),
        ("on_health_card", "ontario_health_card"),
        ("bc_phn", "bc_phn"),
    ),
)
def test_canadian_registry_specs_generate_valid_surrogates(id_type, faker_method):
    faker = Faker("en_CA")
    register_clinical_providers(faker)
    faker.seed_instance(42)

    spec = get_national_id("en", id_type)
    assert spec is not None
    assert spec.faker_method == faker_method

    surrogate = getattr(faker, spec.faker_method)()
    assert spec.validate(surrogate), f"{id_type} generated {surrogate!r}"


@pytest.mark.parametrize("locale", ("en_CA", "fr_CA", "ca"))
def test_canadian_ids_discoverable_from_locale_aliases(locale):
    for id_type in ("sin", "on_health_card", "bc_phn"):
        assert get_national_id(locale, id_type) is not None


def test_extract_pii_en_ca_locale_yields_canadian_identifier_spans(monkeypatch):
    text = (
        "SIN 130 692 544 on file; Ontario health card 6317048459-QC "
        "and BC PHN 9291 417 779 were verified."
    )

    def fake_analyze_text(inference_text, **kwargs):
        return PredictionResult(
            text=inference_text,
            entities=[],
            model_name=kwargs["model_name"],
            timestamp="2026-01-01T00:00:00",
        )

    import openmed

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze_text)

    result = extract_pii(
        text,
        model_name="fixture-pii-model",
        lang="en",
        locale="en_CA",
    )

    spans = {
        (entity.text, entity.label, entity.start, entity.end)
        for entity in result.entities
    }

    for fragment in ("130 692 544", "6317048459-QC", "9291 417 779"):
        start = text.index(fragment)
        assert (
            fragment,
            "national_id",
            start,
            start + len(fragment),
        ) in spans, f"missing span for {fragment!r}"
