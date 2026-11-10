"""Tests for UK NHS Number and National Insurance ID support."""

from __future__ import annotations

from faker import Faker

from openmed.core.anonymizer.providers.clinical_ids import (
    register_clinical_providers,
    validate_uk_nhs_number,
    validate_uk_nino,
)
from openmed.core.anonymizer.providers.registry_ids import get_national_id
from openmed.core.pii import extract_pii
from openmed.processing.outputs import PredictionResult


def test_validate_uk_nhs_number_accepts_published_test_number():
    assert validate_uk_nhs_number("943 476 5919")
    assert validate_uk_nhs_number("9434765919")


def test_validate_uk_nhs_number_rejects_remainder_10_body():
    # First nine digits sum to 12, so sum % 11 == 1 and check digit would be 10.
    assert not validate_uk_nhs_number("000 020 0000")


def test_validate_uk_nino_accepts_and_rejects_structural_cases():
    assert validate_uk_nino("AB 12 34 56 C")

    assert not validate_uk_nino("BG 12 34 56 C")
    assert not validate_uk_nino("AB 12 34 56 E")
    assert not validate_uk_nino("AO 12 34 56 C")


def test_uk_registry_specs_generate_valid_surrogates():
    faker = Faker("en_GB")
    register_clinical_providers(faker)
    faker.seed_instance(42)

    for id_type in ("nhs_number", "nino"):
        spec = get_national_id("en", id_type)
        assert spec is not None
        surrogate = getattr(faker, spec.faker_method)()
        assert spec.validate(surrogate), f"{id_type} generated {surrogate!r}"


def test_extract_pii_en_gb_locale_yields_uk_identifier_spans(monkeypatch):
    text = (
        "NHS Number 943 476 5919 and National Insurance number "
        "AB 12 34 56 C were verified."
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
        locale="en_GB",
    )

    spans = {
        (entity.text, entity.label, entity.start, entity.end)
        for entity in result.entities
    }
    nhs_start = text.index("943 476 5919")
    nino_start = text.index("AB 12 34 56 C")

    assert (
        "943 476 5919",
        "national_id",
        nhs_start,
        nhs_start + len("943 476 5919"),
    ) in spans
    assert (
        "AB 12 34 56 C",
        "national_id",
        nino_start,
        nino_start + len("AB 12 34 56 C"),
    ) in spans
