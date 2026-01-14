#!/usr/bin/env python3
"""
Example: Comparing Different PII Detection Models

This script demonstrates how to use different PII models from the OpenMed
collection to detect and de-identify sensitive information in clinical text.

The OpenMed PII collection includes 33 specialized models ranging from:
- Tiny (33MB) for fast inference
- Small (44-82MB) for balanced performance
- Medium (109-210MB) for general use
- Large (278-434MB) for high accuracy
- XLarge (560-600MB) for maximum coverage
"""

from openmed import extract_pii, deidentify
from openmed.core.model_registry import get_models_by_category, get_model_info


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def compare_pii_models():
    """Compare different PII models on the same clinical text."""

    # Sample clinical text with various PII entities
    sample_text = """
    DISCHARGE SUMMARY

    Patient: Dr. Sarah Johnson
    DOB: 03/15/1985
    MRN: 12345678
    SSN: 987-65-4321

    Contact Information:
    Phone: (555) 234-5678
    Email: sarah.j@healthmail.com
    Address: 456 Oak Avenue, Boston, MA 02115

    Admission Date: 01/10/2024
    Discharge Date: 01/15/2024

    The patient was admitted for evaluation of chest pain.
    Treatment was successful and patient was discharged in stable condition.

    Follow-up: Contact clinic at (555) 876-5432
    """

    print_header("PII Model Comparison Demo")

    # Select diverse models for comparison
    models_to_test = [
        ('pii_clinical_e5_small', 'Fastest (33MB)'),
        ('pii_superclinical_small', 'Fast (44MB)'),
        ('pii_superclinical_base', 'Balanced (184MB)'),
        ('pii_superclinical_large', 'Accurate (434MB)'),
    ]

    print(f"\nTesting {len(models_to_test)} different PII models on clinical text...")
    print(f"Text length: {len(sample_text)} characters\n")

    # Compare each model
    for model_key, description in models_to_test:
        model_info = get_model_info(model_key)

        print(f"\n{'-' * 70}")
        print(f"Model: {model_info.display_name}")
        print(f"Description: {description}")
        print(f"Model ID: {model_info.model_id}")
        print(f"{'-' * 70}")

        # Extract PII with smart merging
        result = extract_pii(
            sample_text,
            model_name=model_key,
            use_smart_merging=True,
            confidence_threshold=0.3
        )

        print(f"\nFound {len(result.entities)} PII entities:")

        # Group by entity type
        by_type = {}
        for entity in result.entities:
            if entity.label not in by_type:
                by_type[entity.label] = []
            by_type[entity.label].append(entity)

        for entity_type, entities in sorted(by_type.items()):
            print(f"\n  {entity_type} ({len(entities)}):")
            for entity in entities:
                print(f"    • {entity.text:30} (confidence: {entity.confidence:.3f})")


def demonstrate_deidentification():
    """Show de-identification with different methods."""

    print_header("De-identification Methods Comparison")

    sample_text = "Patient John Doe (DOB: 01/15/1970, SSN: 123-45-6789) was seen on 03/20/2024."

    print(f"\nOriginal text:")
    print(f"  {sample_text}")

    # Use the flagship model for de-identification
    model = 'pii_superclinical_large'

    methods = [
        ('mask', 'Replace with placeholders'),
        ('remove', 'Remove PII completely'),
        ('hash', 'Cryptographic hashing'),
        ('shift_dates', 'Shift dates by 180 days'),
    ]

    for method, description in methods:
        kwargs = {'date_shift_days': 180} if method == 'shift_dates' else {}

        result = deidentify(
            sample_text,
            method=method,
            model_name=model,
            use_smart_merging=True,
            **kwargs
        )

        print(f"\n{method.upper()} ({description}):")
        print(f"  {result.deidentified_text}")


def list_all_pii_models():
    """List all available PII models organized by size."""

    print_header("All Available PII Models")

    all_pii = get_models_by_category('Privacy')

    # Organize by size
    by_size = {}
    for model in all_pii:
        size = model.size_category
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(model)

    # Display by size category
    size_order = ['Tiny', 'Small', 'Medium', 'Large', 'XLarge']

    for size in size_order:
        if size in by_size:
            models = sorted(by_size[size], key=lambda m: m.size_mb or 0)
            print(f"\n{size.upper()} Models ({len(models)}):")
            for model in models:
                size_str = f"{model.size_mb}MB" if model.size_mb else "N/A"
                print(f"  • {model.display_name:45} {size_str:>7}")

    print(f"\n{'=' * 70}")
    print(f"Total: {len(all_pii)} PII detection models available")
    print(f"{'=' * 70}")


def show_usage_recommendations():
    """Show recommended models for different use cases."""

    print_header("Model Recommendations by Use Case")

    recommendations = [
        ("Real-time Processing", "pii_clinical_e5_small",
         "Ultra-fast inference for production systems"),
        ("Balanced Performance", "pii_superclinical_base",
         "Good accuracy with reasonable speed"),
        ("Maximum Accuracy", "pii_superclinical_large",
         "Best performance for critical applications"),
        ("Long Documents", "pii_clinical_longformer",
         "Optimized for lengthy clinical notes"),
        ("European Data", "pii_euro_med",
         "GDPR-compliant for European healthcare"),
        ("Multilingual", "pii_msuper_clinical",
         "Cross-language PII detection"),
    ]

    for use_case, model_key, description in recommendations:
        model = get_model_info(model_key)
        print(f"\n{use_case}:")
        print(f"  Model: {model.display_name}")
        print(f"  Size: {model.size_mb}MB ({model.size_category})")
        print(f"  Use for: {description}")


if __name__ == "__main__":
    # Run all demonstrations
    try:
        print("\n" + "🔒 " * 35)
        print("OpenMed PII Detection Model Collection Demo")
        print("🔒 " * 35)

        list_all_pii_models()
        show_usage_recommendations()

        print("\n\nNOTE: The following demonstrations require model downloads.")
        print("To run the full comparison, uncomment the lines below.\n")

        # Uncomment to test models (requires downloading):
        # compare_pii_models()
        # demonstrate_deidentification()

    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: This demo requires the openmed package with transformers.")
        print("Install with: pip install openmed[hf]")
