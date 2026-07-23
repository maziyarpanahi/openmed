"""Acceptance tests for the Indian multi-identifier recognizer pack."""

from __future__ import annotations

import random
from datetime import datetime

import pytest
from faker import Faker

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.providers.clinical_ids import (
    IndianIdentifierProvider,
    generate_gstin,
    generate_pan,
    register_clinical_providers,
)
from openmed.core.anonymizer.providers.registry_ids import get_national_id
from openmed.core.detector_plugins import (
    INDIAN_MULTI_ID_DETECTOR,
    INDIAN_MULTI_ID_SUBTYPES,
    detect_indian_identifiers,
    iter_detectors,
)
from openmed.core.labels import (
    DIRECT_IDENTIFIER,
    HIPAA_UNIQUE_IDENTIFIER,
    HIPAA_VEHICLE_IDENTIFIER,
    ID_NUM,
    ID_SUBTYPE_ABHA,
    ID_SUBTYPE_GSTIN,
    ID_SUBTYPE_IFSC,
    ID_SUBTYPE_INDIAN_DRIVING_LICENCE,
    ID_SUBTYPE_INDIAN_PASSPORT,
    ID_SUBTYPE_NATIONAL_ID,
    ID_SUBTYPE_PAN,
    ID_SUBTYPE_VOTER_ID_EPIC,
    ID_SUBTYPES,
    VEHICLE_REGISTRATION,
    hipaa_class_for,
    id_subtype_for,
    normalize_label,
    policy_label_for,
)
from openmed.core.pii_i18n import (
    INDIAN_MULTI_ID_PII_PATTERNS,
    LANGUAGE_PII_PATTERNS,
    get_patterns_for_language,
    gstin_check_char,
    validate_abha,
    validate_gstin,
    validate_ifsc,
    validate_indian_driving_licence,
    validate_indian_passport,
    validate_pan,
    validate_vehicle_registration,
    validate_voter_id_epic,
)
from openmed.core.pipeline import Pipeline
from openmed.core.surrogate_vault import SurrogateVault
from openmed.processing.outputs import PredictionResult

VALID_CASES = (
    ("pan", "pan", "OMDBR7117R", validate_pan),
    ("voter_id_epic", "voter_id_epic", "VNL5203138", validate_voter_id_epic),
    (
        "indian_driving_licence",
        "indian_driving_licence",
        "LF1220108382646",
        validate_indian_driving_licence,
    ),
    (
        "indian_passport",
        "indian_passport",
        "K4789111",
        validate_indian_passport,
    ),
    ("ifsc", "ifsc", "FVQO0ZQ3X83", validate_ifsc),
    ("gstin", "gstin", "06OMDBX7342DTZM", validate_gstin),
    (
        "vehicle_registration",
        "indian_vehicle_registration",
        "ZE65ZRR5256",
        validate_vehicle_registration,
    ),
    ("abha", "abha", "28665452177536", validate_abha),
)

INVALID_CASES = (
    ("OMDBR7117A", validate_pan),
    ("VNL0000000", validate_voter_id_epic),
    ("LF0020108382646", validate_indian_driving_licence),
    ("K0789111", validate_indian_passport),
    ("FVQO1ZQ3X83", validate_ifsc),
    ("06OMDBX7342DTZ0", validate_gstin),
    ("ZE00ZRR5256", validate_vehicle_registration),
    ("00000000000000", validate_abha),
)


@pytest.mark.parametrize(
    ("_id_type", "_entity_type", "value", "validator"), VALID_CASES
)
def test_valid_synthetic_cases_pass_structure_validation(
    _id_type,
    _entity_type,
    value,
    validator,
):
    assert validator(value) is True


@pytest.mark.parametrize(("value", "validator"), INVALID_CASES)
def test_structurally_invalid_cases_are_rejected(value, validator):
    assert validator(value) is False


def test_pan_public_structure_and_synthetic_checksum_behavior():
    assert validate_pan("ALWPG5809L") is True
    assert validate_pan("ALWZG5809L") is False  # unsupported holder type
    assert validate_pan("ALWPG0000L") is False

    generated = generate_pan(rng=random.Random(667))
    wrong_check = generated[:-1] + ("A" if generated[-1] != "A" else "B")
    assert validate_pan(generated) is True
    assert validate_pan(wrong_check) is False


def test_gstin_validates_checksum_embedded_pan_and_issue_state_range():
    assert validate_gstin("27AAPFU0939F1ZV") is True

    pan = generate_pan(rng=random.Random(1493))
    out_of_range_body = f"38{pan}1Z"
    out_of_range = out_of_range_body + gstin_check_char(out_of_range_body)
    assert validate_gstin(out_of_range) is False

    invalid_pan = pan[:3] + "Z" + pan[4:]
    invalid_pan_body = f"06{invalid_pan}1Z"
    invalid_pan_gstin = invalid_pan_body + gstin_check_char(invalid_pan_body)
    assert validate_gstin(invalid_pan_gstin) is False


def test_ifsc_uses_official_eleven_character_shape_without_registry_data():
    assert validate_ifsc("SBIN0004337") is True
    assert validate_ifsc("SBIN1004337") is False
    assert validate_ifsc("SBI0004337") is False


def test_pan_and_gstin_generators_pass_one_thousand_round_trips():
    rng = random.Random(667)
    for _ in range(1_000):
        assert validate_pan(generate_pan(rng=rng)) is True
        assert validate_gstin(generate_gstin(rng=rng)) is True


def test_every_faker_provider_method_round_trips_through_its_validator():
    faker = Faker("en_IN")
    faker.add_provider(IndianIdentifierProvider)
    faker.seed_instance(667)

    for id_type, _entity_type, _original, validator in VALID_CASES:
        method = (
            "indian_vehicle_registration"
            if id_type == "vehicle_registration"
            else id_type
        )
        for _ in range(100):
            assert validator(getattr(faker, method)()) is True


@pytest.mark.parametrize("alias", ("in", "india", "hi", "te", "en_IN", "hi_IN"))
@pytest.mark.parametrize(
    "id_type",
    (
        "pan",
        "gstin",
        "ifsc",
        "voter_id_epic",
        "indian_driving_licence",
        "indian_passport",
        "vehicle_registration",
        "abha",
    ),
)
def test_registry_exposes_all_india_aliases(alias, id_type):
    spec = get_national_id(alias, id_type)
    assert spec is not None

    faker = Faker("en_IN")
    register_clinical_providers(faker)
    faker.seed_instance(1493)
    assert spec.validate(getattr(faker, spec.faker_method)()) is True


def test_label_subtypes_policy_and_hipaa_resolution():
    compatibility_subtypes = {
        "pan": ID_SUBTYPE_NATIONAL_ID,
        "gstin": ID_SUBTYPE_NATIONAL_ID,
        "ifsc": ID_SUBTYPE_IFSC,
        "voter_id_epic": ID_SUBTYPE_VOTER_ID_EPIC,
        "indian_driving_licence": ID_SUBTYPE_INDIAN_DRIVING_LICENCE,
        "indian_passport": ID_SUBTYPE_INDIAN_PASSPORT,
        "abha": ID_SUBTYPE_NATIONAL_ID,
    }
    assert {
        ID_SUBTYPE_ABHA,
        ID_SUBTYPE_GSTIN,
        ID_SUBTYPE_IFSC,
        ID_SUBTYPE_INDIAN_DRIVING_LICENCE,
        ID_SUBTYPE_INDIAN_PASSPORT,
        ID_SUBTYPE_PAN,
        ID_SUBTYPE_VOTER_ID_EPIC,
    }.issubset(ID_SUBTYPES)
    assert INDIAN_MULTI_ID_SUBTYPES == {
        "abha": ID_SUBTYPE_ABHA,
        "gstin": ID_SUBTYPE_GSTIN,
        "ifsc": ID_SUBTYPE_IFSC,
        "indian_driving_licence": ID_SUBTYPE_INDIAN_DRIVING_LICENCE,
        "indian_passport": ID_SUBTYPE_INDIAN_PASSPORT,
        "pan": ID_SUBTYPE_PAN,
        "voter_id_epic": ID_SUBTYPE_VOTER_ID_EPIC,
    }
    for entity_type, subtype in compatibility_subtypes.items():
        assert normalize_label(entity_type) == ID_NUM
        assert id_subtype_for(entity_type) == subtype
        assert policy_label_for(entity_type) == DIRECT_IDENTIFIER
        assert hipaa_class_for(entity_type) == HIPAA_UNIQUE_IDENTIFIER

    assert normalize_label("indian_vehicle_registration") == VEHICLE_REGISTRATION
    assert hipaa_class_for("indian_vehicle_registration") == HIPAA_VEHICLE_IDENTIFIER


def test_patterns_are_wired_for_india_languages_and_locale():
    entity_types = {pattern.entity_type for pattern in INDIAN_MULTI_ID_PII_PATTERNS}
    assert entity_types == {
        "abha",
        "gstin",
        "ifsc",
        "indian_driving_licence",
        "indian_passport",
        "indian_vehicle_registration",
        "pan",
        "voter_id_epic",
    }
    for language in ("hi", "te"):
        assert entity_types.issubset(
            {pattern.entity_type for pattern in LANGUAGE_PII_PATTERNS[language]}
        )
    assert entity_types.issubset(
        {
            pattern.entity_type
            for pattern in get_patterns_for_language("en", locale="en_IN")
        }
    )

    all_context = {
        context.casefold()
        for pattern in INDIAN_MULTI_ID_PII_PATTERNS
        for context in pattern.context_words
    }
    assert {"pan number", "पैन नंबर", "పాన్ నంబర్"}.issubset(all_context)


def test_builtin_detector_is_discoverable_and_rejects_all_invalid_cases():
    names = {spec.name for spec in iter_detectors("deterministic", "hi")}
    assert INDIAN_MULTI_ID_DETECTOR in names

    invalid_text = "; ".join(
        (
            "PAN OMDBR7117A",
            "Voter ID VNL0000000",
            "DL LF0020108382646",
            "passport K0789111",
            "IFSC FVQO1ZQ3X83",
            "GSTIN 06OMDBX7342DTZ0",
            "vehicle registration ZE00ZRR5256",
            "ABHA 00000000000000",
        )
    )
    assert detect_indian_identifiers(invalid_text, lang="hi") == ()


def test_builtin_detector_emits_granular_subtypes_without_changing_aliases():
    text = "पैन नंबर OMDBR7117R; GSTIN 06OMDBX7342DTZM."

    spans = detect_indian_identifiers(text, lang="hi")

    assert {span.entity_type: span.metadata["identifier_type"] for span in spans} == {
        "gstin": ID_SUBTYPE_GSTIN,
        "pan": ID_SUBTYPE_PAN,
    }


def _empty_prediction(text: str, model_name: str = "stub") -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
    )


def test_pipeline_detector_masks_ids_and_keeps_only_safe_evidence():
    text = "पैन नंबर OMDBR7117R; GSTIN 06OMDBX7342DTZM; गाड़ी नंबर ZE65ZRR5256."
    result = Pipeline(
        model_detector=lambda text, **kwargs: _empty_prediction(
            text,
            kwargs["model_name"],
        ),
        lang="hi",
        use_safety_sweep=False,
    ).run(text, method="mask")

    assert "OMDBR7117R" not in result.redacted_text
    assert "06OMDBX7342DTZM" not in result.redacted_text
    assert "ZE65ZRR5256" not in result.redacted_text
    assert "[pan]" in result.redacted_text
    assert "[gstin]" in result.redacted_text
    assert "[indian_vehicle_registration]" in result.redacted_text
    assert all(span.text_hash.startswith("hmac-sha256:") for span in result.spans)
    assert all("text" not in span.metadata for span in result.spans)
    assert all("surface" not in span.evidence for span in result.spans)


def test_anonymizer_and_surrogate_vault_preserve_each_identifier_structure():
    anonymizer = Anonymizer(lang="hi", consistent=True, seed=667)
    vault = SurrogateVault.in_memory("indian-id-test-secret")

    for id_type, entity_type, original, validator in VALID_CASES:
        label = normalize_label(entity_type)

        def create_surrogate(_attempt, *, source=original, source_label=entity_type):
            return anonymizer.surrogate(
                source,
                source_label,
                lang="hi",
                locale="hi_IN",
            )

        first = vault.get_or_create(
            original,
            label=label,
            lang="hi",
            create_surrogate=create_surrogate,
        )
        second = vault.get_or_create(
            original,
            label=label,
            lang="hi",
            create_surrogate=create_surrogate,
        )
        assert first == second, id_type
        assert first != original, id_type
        assert validator(first) is True, id_type
