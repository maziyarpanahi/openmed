"""Tests for Australian Medicare card number and Tax File Number support.

All identifiers used here are synthetic values constructed to satisfy (or
deliberately fail) the published checksums; none correspond to a real person.
"""

from __future__ import annotations

import random
import re

import pytest
from faker import Faker

from openmed.core.anonymizer.providers.clinical_ids import (
    generate_australian_medicare,
    generate_australian_tfn,
    register_clinical_providers,
    validate_australian_medicare,
    validate_australian_tfn,
)
from openmed.core.anonymizer.providers.registry_ids import get_national_id
from openmed.core.pii import extract_pii
from openmed.core.pii_i18n import LOCALE_PII_PATTERNS
from openmed.core.safety_sweep import SAFETY_SWEEP_SOURCE, safety_sweep
from openmed.processing.outputs import PredictionResult

# Synthetic fixtures constructed to satisfy the documented format and checksum
# rules; none are asserted to be issued identifiers.
VALID_MEDICARE = "2123456701"
VALID_MEDICARE_SPACED = "2123 45670 1"
VALID_TFN_9 = "123456782"


class TestMedicareChecksum:
    def test_accepts_checksum_valid_card_number(self):
        assert validate_australian_medicare(VALID_MEDICARE)
        assert validate_australian_medicare(VALID_MEDICARE_SPACED)

    def test_rejects_digit_flipped_variant(self):
        # Flip the ninth digit (the check digit) so the checksum no longer holds.
        corrupted = VALID_MEDICARE[:8] + "1" + VALID_MEDICARE[9]
        assert corrupted != VALID_MEDICARE
        assert not validate_australian_medicare(corrupted)

    def test_issue_reference_digit_does_not_affect_checksum(self):
        # The tenth digit is the issue / reference number and is outside the
        # checksum, so varying it keeps a valid card valid.
        for issue in "123456789":
            candidate = VALID_MEDICARE[:9] + issue
            assert validate_australian_medicare(candidate), candidate

    def test_accepts_separate_individual_reference_number(self):
        for candidate in (
            "21234567012",
            "2123 45670 1/2",
            "2123 45670 1 / 2",
            "2123 45670 1 2",
            "2123 45670 1-2",
        ):
            assert validate_australian_medicare(candidate), candidate

    def test_rejects_wrong_length_and_bad_leading_digit(self):
        assert not validate_australian_medicare("212345670")  # 9 digits
        assert not validate_australian_medicare("212345670123")  # 12 digits
        assert not validate_australian_medicare("21234567010")  # invalid IRN
        # Leading digit outside the 2-6 issuing range.
        assert not validate_australian_medicare("1123456701")
        assert not validate_australian_medicare("7123456701")

    def test_rejects_decorated_or_noncanonical_formats(self):
        assert not validate_australian_medicare("Medicare=2123456701!")
        assert not validate_australian_medicare("2123-45670-1")
        assert not validate_australian_medicare("2123 45670-1")


class TestTfnChecksum:
    def test_accepts_weighted_mod11_valid_tfn(self):
        assert validate_australian_tfn(VALID_TFN_9)
        assert validate_australian_tfn("123 456 782")

    def test_rejects_corrupted_tfn(self):
        # Increment the last digit so the weighted sum is no longer mod-11 zero.
        corrupted = VALID_TFN_9[:-1] + str((int(VALID_TFN_9[-1]) + 1) % 10)
        assert corrupted != VALID_TFN_9
        assert not validate_australian_tfn(corrupted)

    def test_rejects_wrong_length(self):
        assert not validate_australian_tfn("1234567")  # 7 digits
        assert not validate_australian_tfn("47041532")  # 8 digits
        assert not validate_australian_tfn("1234567890")  # 10 digits
        assert not validate_australian_tfn("000000000")

    def test_rejects_decorated_or_noncanonical_formats(self):
        assert not validate_australian_tfn("TFN=123456782!")
        assert not validate_australian_tfn("123-456-782")
        assert not validate_australian_tfn("123 456-782")


def test_en_au_patterns_accept_only_exact_valid_identifiers():
    patterns = {pattern.validator: pattern for pattern in LOCALE_PII_PATTERNS["en_au"]}
    medicare_pattern = patterns[validate_australian_medicare]
    tfn_pattern = patterns[validate_australian_tfn]

    def accepted(pattern, value):
        match = re.fullmatch(pattern.pattern, value, pattern.flags)
        return match is not None and pattern.validator(value)

    for supported_shape in (
        VALID_MEDICARE_SPACED,
        "2123 45670 1 2",
        "2123 45670 1/2",
        "21234567012",
    ):
        assert accepted(medicare_pattern, supported_shape)
    for impossible_shape in ("2123 45670 1/0", "212345670123"):
        assert not accepted(medicare_pattern, impossible_shape)
    assert not accepted(medicare_pattern, "2123 45671 1")
    assert accepted(tfn_pattern, "123 456 782")
    assert not accepted(tfn_pattern, "123 456 783")


class TestGenerators:
    def test_generated_medicare_round_trips(self):
        rng = random.Random(1234)
        for _ in range(200):
            value = generate_australian_medicare(rng=rng)
            assert len(value) == 10
            assert validate_australian_medicare(value), value
            assert validate_australian_medicare(f"{value}/1"), value

    def test_generated_tfn_round_trips(self):
        rng = random.Random(4321)
        for _ in range(200):
            value = generate_australian_tfn(rng=rng)
            assert len(value) == 9
            assert value != "000000000"
            assert validate_australian_tfn(value), value


class TestRegistry:
    @pytest.mark.parametrize("id_type", ["medicare", "tfn"])
    def test_registry_specs_generate_valid_surrogates(self, id_type):
        faker = Faker("en_AU")
        register_clinical_providers(faker)
        faker.seed_instance(42)

        spec = get_national_id("en", id_type)
        assert spec is not None
        for _ in range(40):
            surrogate = getattr(faker, spec.faker_method)()
            assert spec.validate(surrogate), f"{id_type} generated {surrogate!r}"

    def test_registry_lookup_aliases(self):
        for id_type in ("medicare", "tfn"):
            base = get_national_id("en", id_type)
            assert base is not None
            # Locale and country aliases resolve to equivalent specs (each
            # alias is registered as its own spec sharing the validator and
            # Faker method).
            for alias in ("en_AU", "AU"):
                spec = get_national_id(alias, id_type)
                assert spec is not None
                assert spec.validate is base.validate
                assert spec.faker_method == base.faker_method


def test_extract_pii_en_au_locale_yields_medicare_and_tfn_spans(monkeypatch):
    text = (
        "Patient Medicare number 2123 45670 1 and Tax File Number "
        "123 456 782 were recorded on intake."
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
        locale="en_AU",
    )

    spans = {
        (entity.text, entity.label, entity.start, entity.end)
        for entity in result.entities
    }
    medicare_value = "2123 45670 1"
    tfn_value = "123 456 782"
    medicare_start = text.index(medicare_value)
    tfn_start = text.index(tfn_value)

    assert (
        medicare_value,
        "national_id",
        medicare_start,
        medicare_start + len(medicare_value),
    ) in spans
    assert (
        tfn_value,
        "national_id",
        tfn_start,
        tfn_start + len(tfn_value),
    ) in spans


def test_safety_sweep_protects_medicare_card_and_irn_as_one_span():
    text = "Medicare number 2123 45670 1/2 was verified."
    entities = safety_sweep(text, [], lang="en", locale="en_AU")

    assert len(entities) == 1
    entity = entities[0]
    assert entity.text == "2123 45670 1/2"
    assert entity.start == text.index(entity.text)
    assert entity.end == entity.start + len(entity.text)
    assert entity.metadata["source"] == SAFETY_SWEEP_SOURCE
