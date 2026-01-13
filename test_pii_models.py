#!/usr/bin/env python3
"""Test script to validate PII model registry integration."""

from openmed.core.model_registry import (
    get_models_by_category,
    get_model_info,
    get_models_by_size,
    find_models_by_entity_type,
)


def test_pii_model_registry():
    """Test that all PII models are properly registered."""
    print("=" * 70)
    print("OpenMed PII Model Registry Test")
    print("=" * 70)

    # Test 1: Count total PII models
    all_pii = get_models_by_category('Privacy')
    print(f"\n✓ Total PII models registered: {len(all_pii)}")
    assert len(all_pii) == 34, f"Expected 34 PII models, found {len(all_pii)}"

    # Test 2: Verify specific models can be retrieved
    test_models = [
        'pii_superclinical_large',
        'pii_superclinical_base',
        'pii_superclinical_small',
        'pii_lite_clinical',
        'pii_fast_clinical',
        'pii_qwen_med_xlarge',
        'pii_biomed_bert_large',
        'pii_clinical_e5_small',
    ]

    print("\n✓ Testing specific model retrieval:")
    for model_key in test_models:
        model = get_model_info(model_key)
        assert model is not None, f"Model {model_key} not found"
        print(f"  • {model.display_name:40} {model.size_mb:4}MB  ({model.size_category})")

    # Test 3: Verify size categories
    print("\n✓ PII models by size category:")
    size_counts = {}
    for size in ['Tiny', 'Small', 'Medium', 'Large', 'XLarge']:
        models = [m for m in get_models_by_size(size) if m.category == 'Privacy']
        size_counts[size] = len(models)
        print(f"  • {size:8}: {len(models):2} models")

    assert size_counts['Tiny'] == 1, "Expected 1 Tiny model"
    assert size_counts['Small'] == 3, "Expected 3 Small models"
    assert size_counts['Medium'] == 14, "Expected 14 Medium models"
    assert size_counts['Large'] == 11, "Expected 11 Large models"
    assert size_counts['XLarge'] == 5, "Expected 5 XLarge models"

    # Test 4: Verify entity types
    print("\n✓ Testing entity type search:")
    ssn_models = find_models_by_entity_type('ssn')
    email_models = find_models_by_entity_type('email')
    print(f"  • Models detecting SSN: {len(ssn_models)}")
    print(f"  • Models detecting EMAIL: {len(email_models)}")

    # Test 5: Display all model IDs
    print("\n✓ All registered PII model IDs:")
    for i, model in enumerate(all_pii, 1):
        print(f"  {i:2}. {model.model_id}")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_pii_model_registry()
