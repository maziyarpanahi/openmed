from __future__ import annotations

from openmed.core.anonymizer.providers import clinical_ids
from openmed.training import load_preset
from openmed.training.hard_negatives import (
    HARD_NEGATIVE_CATEGORIES,
    STRUCTURALLY_VALID_FAKE_ID,
    HardNegativeGenerator,
    count_hard_negatives,
    requires_hard_negative_sampler,
    sample_hard_negatives,
    sampler_for_recipe,
)


def test_generator_emits_all_required_categories():
    generator = HardNegativeGenerator(seed=13)
    examples = generator.generate_all_categories()

    assert {example.category for example in examples} == set(HARD_NEGATIVE_CATEGORIES)
    assert all(example.text for example in examples)
    assert all(example.to_training_item()["labels"] == [] for example in examples)
    assert all(
        example.to_training_item()["is_hard_negative"] is True for example in examples
    )


def test_structurally_valid_fake_ids_use_clinical_id_validators():
    generator = HardNegativeGenerator(seed=7)
    examples = [generator.generate(STRUCTURALLY_VALID_FAKE_ID) for _ in range(6)]
    by_subtype = {example.subtype: example.value for example in examples}

    assert clinical_ids.validate_npi(by_subtype["npi"])
    assert clinical_ids.validate_luhn(by_subtype["luhn"])
    assert clinical_ids.validate_ssn(by_subtype["ssn"])


def test_clinical_id_generator_helpers_validate_their_outputs():
    assert clinical_ids.validate_npi(clinical_ids.generate_npi())
    assert clinical_ids.validate_luhn(clinical_ids.generate_luhn_identifier())
    assert clinical_ids.validate_ssn(clinical_ids.generate_ssn())


def test_sampler_guarantees_nonzero_hard_negatives_per_batch():
    config = load_preset("A")
    sampler = sampler_for_recipe(config, min_hard_negatives_per_batch=2)
    fixture_batch = (
        {"text": "Patient has improved oxygenation.", "labels": []},
        {"text": "Discharge planning started.", "labels": []},
    )

    sampled = sampler.sample_batch(fixture_batch, recipe_config=config)

    assert len(sampled) == len(fixture_batch) + 2
    assert count_hard_negatives(sampled) == 2
    assert all("hard_negative_category" in item for item in sampled[-2:])


def test_sampler_does_not_duplicate_existing_hard_negatives():
    fixture_batch = (
        {"text": "Original training row.", "labels": []},
        {
            "text": "ASA 81 mg daily.",
            "labels": [],
            "is_hard_negative": True,
            "hard_negative_category": "clinical_abbreviation_or_drug_name",
        },
    )

    sampled = sample_hard_negatives(fixture_batch, min_hard_negatives_per_batch=1)

    assert len(sampled) == len(fixture_batch)
    assert count_hard_negatives(sampled) == 1


def test_sample_hard_negatives_noops_when_recipe_flag_is_false():
    fixture_batch = ({"text": "Plain negative row.", "labels": []},)

    sampled = sample_hard_negatives(
        fixture_batch,
        recipe_config={"hard_negatives_required": False},
    )

    assert sampled == fixture_batch
    assert count_hard_negatives(sampled) == 0


def test_recipe_flag_controls_sampler_requirement():
    config = load_preset("B")

    assert requires_hard_negative_sampler(config) is True
    assert requires_hard_negative_sampler(config.to_dict()) is True
    assert requires_hard_negative_sampler({"hard_negatives_required": False}) is False
