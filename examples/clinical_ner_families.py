#!/usr/bin/env python3
"""Worked example for OpenMed biomedical and clinical NER families.

The example uses synthetic clinical text and discovers models through the
registry helpers instead of hard-coding Hugging Face repository ids.

Run:
    python examples/clinical_ner_families.py
"""

from __future__ import annotations

from collections.abc import Iterable

from openmed import analyze_text, get_models_by_category
from openmed.core.model_registry import ModelInfo, get_recommended_models


FAMILY_CATEGORIES = ("Disease", "Pharmaceutical", "Oncology")
SYNTHETIC_TEXT = (
    "A synthetic patient with chronic myeloid leukemia started imatinib "
    "after the oncology team reviewed BCR-ABL testing."
)
TIER_ORDER = {
    "Tiny": 0,
    "Small": 1,
    "Medium": 2,
    "Large": 3,
    "XLarge": 4,
    "Unknown": 5,
}


def _recommended_candidates() -> Iterable[ModelInfo]:
    for profile in ("fast", "balanced", "accurate"):
        yield from get_recommended_models(profile)


def select_model_for_category(category: str) -> ModelInfo:
    """Pick a representative model for a registry category."""

    for model in _recommended_candidates():
        if model.category == category:
            return model

    models = get_models_by_category(category)
    if not models:
        raise LookupError(f"No registered models found for {category}")

    return sorted(
        models,
        key=lambda item: (TIER_ORDER.get(item.size_category, 99), item.display_name),
    )[0]


def print_entities(category: str, model: ModelInfo) -> None:
    """Run one family and print labelled spans, skipping unavailable models."""

    print()
    print(f"## {category}")
    print(f"Model: {model.model_id}")
    print(f"Tier: {model.size_category}")
    print(f"Recommended confidence: {model.recommended_confidence:.2f}")

    try:
        result = analyze_text(
            SYNTHETIC_TEXT,
            model_name=model.model_id,
            confidence_threshold=model.recommended_confidence,
        )
    except Exception as exc:
        print(f"Model unavailable offline: {exc}")
        return

    entities = getattr(result, "entities", [])
    if not entities:
        print("No entities returned.")
        return

    for entity in entities:
        label = getattr(entity, "label", "UNKNOWN")
        text = getattr(entity, "text", "")
        confidence = getattr(entity, "confidence", 0.0)
        print(f"- {label:<24} {text!r} ({confidence:.3f})")


def main() -> None:
    print("Clinical NER Families Example")
    print("Synthetic text:")
    print(SYNTHETIC_TEXT)

    for category in FAMILY_CATEGORIES:
        model = select_model_for_category(category)
        print_entities(category, model)


if __name__ == "__main__":
    main()
