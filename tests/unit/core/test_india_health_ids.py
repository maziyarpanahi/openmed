"""India health-ID recognition, policy, surrogate, and leakage tests."""

from __future__ import annotations

import json
import random

import pytest

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.providers.clinical_ids import (
    generate_abha_number,
)
from openmed.core.anonymizer.providers.registry_ids import get_national_id
from openmed.core.pii_entity_merger import find_semantic_units
from openmed.core.pii_i18n import (
    get_patterns_for_language,
    validate_abha_address,
    validate_abha_number,
    validate_indian_ration_card,
    validate_upi_id,
)
from openmed.core.pipeline import Pipeline
from openmed.core.policy import PolicyName, list_policies, load_policy
from openmed.core.surrogate_vault import SurrogateVault
from openmed.eval.suites import (
    INDIA_HEALTH_ID_LEAKAGE,
    assert_india_health_id_leakage_gate,
    load_india_health_id_fixtures,
    load_suite_fixtures,
    suite_metadata,
)
from openmed.processing.outputs import PredictionResult


def _empty_prediction(text: str, **_kwargs) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name="offline-india-health-id-test",
        timestamp="2026-07-18T00:00:00Z",
        metadata={},
    )


@pytest.mark.parametrize("seed", range(20))
def test_abha_number_generator_passes_validator_and_mutation_fails(seed: int) -> None:
    value = generate_abha_number(rng=random.Random(seed))
    invalid = f"{value[:-1]}{(int(value[-1]) + 1) % 10}"

    assert len(value) == 14
    assert validate_abha_number(value)
    assert not validate_abha_number(invalid)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "1234567890123",
        "123456789012345",
        "00000000000000",
        "1111 1111 1111 11",
        "7604-8764-7593-82",
    ],
)
def test_abha_number_rejects_invalid_structure_or_checksum(value: str) -> None:
    assert not validate_abha_number(value)


def test_address_upi_and_ration_validators_are_precise() -> None:
    assert validate_abha_address("patient.482901@abdm")
    assert validate_abha_address("member123@sbx")
    assert validate_upi_id("refund.593104@okaxis")
    assert validate_indian_ration_card("DL-4829013756")

    assert not validate_abha_address("records@example.org")
    assert not validate_abha_address("1patient@abdm")
    assert not validate_upi_id("records@example.org")
    assert not validate_upi_id("patient.482901@abdm")
    assert not validate_indian_ration_card("ABC")


def test_code_mixed_fixture_recognizes_all_health_ids_and_not_negatives() -> None:
    fixture = load_india_health_id_fixtures()[0]
    units = find_semantic_units(
        fixture.text,
        get_patterns_for_language(fixture.language),
    )
    validated = {
        (start, end, entity_type)
        for start, end, entity_type, _score, _pattern, is_valid in units
        if is_valid
    }

    expected = {
        (span.start, span.end, str(span.metadata["identifier_type"]))
        for span in fixture.gold_spans
    }
    assert expected <= validated

    india_types = {"abha_number", "abha_address", "upi_id", "ration_card"}
    for negative in fixture.metadata["hard_negatives"]:
        assert not any(
            start < int(negative["end"])
            and end > int(negative["start"])
            and entity_type in india_types
            and is_valid
            for start, end, entity_type, _score, _pattern, is_valid in units
        )


def test_non_health_at_string_requires_upi_context() -> None:
    units = find_semantic_units(
        "Ward routing token records@ward has no payment meaning.",
        get_patterns_for_language("en"),
    )

    assert all(entity_type != "upi_id" for _, _, entity_type, *_ in units)


@pytest.mark.parametrize(
    ("id_type", "validator"),
    [
        ("abha_number", validate_abha_number),
        ("abha_address", validate_abha_address),
        ("upi_id", validate_upi_id),
        ("ration_card", validate_indian_ration_card),
    ],
)
def test_registry_generates_validator_compatible_india_ids(id_type, validator) -> None:
    spec = get_national_id("hi_IN", id_type)
    assert spec is not None

    anonymizer = Anonymizer(lang="hi", consistent=True, seed=17)
    faker = anonymizer._get_faker("hi_IN")
    value = spec.format(getattr(faker, spec.faker_method)())

    assert validator(value)


@pytest.mark.parametrize(
    ("original", "label", "validator"),
    [
        ("7604 8764 7593 81", "ABHA_NUMBER", validate_abha_number),
        ("patient.482901@abdm", "ABHA_ADDRESS", validate_abha_address),
        ("refund.593104@okaxis", "UPI_ID", validate_upi_id),
        ("DL-4829013756", "RATION_CARD", validate_indian_ration_card),
    ],
)
def test_anonymizer_dispatches_india_health_id_surrogates(
    original,
    label,
    validator,
) -> None:
    anonymizer = Anonymizer(lang="hi", consistent=True, seed=23)

    first = anonymizer.surrogate(original, label)
    second = anonymizer.surrogate(original, label)

    assert first == second
    assert first != original
    assert validator(first)


def test_surrogate_vault_reuses_abha_and_upi_without_raw_audit_or_logs(caplog) -> None:
    abha = "7604 8764 7593 81"
    upi = "refund.593104@okaxis"
    first_text = f"आभा नंबर {abha}; यूपीआई आईडी {upi}."
    second_text = f"Follow-up ABHA {abha}; refund UPI ID {upi}."
    vault = SurrogateVault.in_memory("india-health-id-unit-test-secret")
    pipeline = Pipeline(
        model_detector=_empty_prediction,
        lang="hi",
        use_smart_merging=False,
    )

    first = pipeline.run(
        first_text,
        method="replace",
        surrogate_vault=vault,
        audit=True,
    ).deidentification_result
    second = pipeline.run(
        second_text,
        method="replace",
        surrogate_vault=vault,
        audit=True,
    ).deidentification_result

    first_surrogates = {
        entity.text: entity.redacted_text for entity in first.pii_entities
    }
    second_surrogates = {
        entity.text: entity.redacted_text for entity in second.pii_entities
    }
    assert first_surrogates[abha] == second_surrogates[abha]
    assert first_surrogates[upi] == second_surrogates[upi]
    assert validate_abha_number(first_surrogates[abha] or "")
    assert validate_upi_id(first_surrogates[upi] or "")

    serialized_audits = json.dumps(
        [first.audit_report.to_dict(), second.audit_report.to_dict()],
        sort_keys=True,
    )
    serialized_vault = json.dumps(vault.to_payload(), sort_keys=True)
    for raw in (abha, upi):
        assert raw not in serialized_audits
        assert raw not in serialized_vault
        assert raw not in caplog.text
    assert all(
        span.text_hash.startswith("sha256:") for span in first.audit_report.spans
    )


def test_india_health_id_policy_masks_sweep_entities_even_with_replace_method() -> None:
    fixture = load_india_health_id_fixtures()[0]
    policy = load_policy("india_health_id")
    pipeline = Pipeline(
        model_detector=_empty_prediction,
        lang=fixture.language,
        policy=policy.name,
        use_smart_merging=False,
    )

    result = pipeline.run(fixture.text, method="replace").deidentification_result

    for span in fixture.gold_spans:
        assert span.text not in result.deidentified_text
    india_entities = [
        entity
        for entity in result.pii_entities
        if ((entity.metadata or {}).get("safety_sweep") or {}).get("entity_type")
        in {"abha_number", "abha_address", "upi_id", "ration_card"}
    ]
    assert len(india_entities) == 4
    assert all(entity.action == "mask" for entity in india_entities)
    assert all(
        (entity.metadata or {})["policy_action"]["policy"] == policy.name
        for entity in india_entities
    )


def test_policy_and_eval_registries_include_india_health_id_mode() -> None:
    profile = load_policy("abha")

    assert profile.name == PolicyName.INDIA_HEALTH_ID.value
    assert profile.safety_sweep_mandatory is True
    assert profile.strict_no_leak is True
    assert profile.action_for("ABHA_NUMBER") == "mask"
    assert profile.action_for("UPI_ID") == "mask"
    assert profile.name in list_policies()

    fixtures = load_suite_fixtures(INDIA_HEALTH_ID_LEAKAGE)
    metadata = suite_metadata(INDIA_HEALTH_ID_LEAKAGE)
    assert fixtures
    assert metadata["policy"] == profile.name
    assert metadata["synthetic"] is True


def test_india_health_id_leakage_gate_passes_with_zero_leakage_and_false_accepts() -> (
    None
):
    result = assert_india_health_id_leakage_gate()

    assert result.passed is True
    assert result.expected_entity_count == 4
    assert result.detected_entity_count == 4
    assert result.entity_leakage == 0.0
    assert result.false_accept_count == 0
    assert result.failures == ()
