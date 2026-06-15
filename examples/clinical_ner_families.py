#!/usr/bin/env python3
"""Worked example for biomedical and clinical NER model families."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from openmed import analyze_text
from openmed.core.model_registry import (
    ModelInfo,
    get_models_by_category,
    get_recommended_models,
)


NER_FAMILIES = ("Disease", "Pharmaceutical", "Oncology")

SYNTHETIC_CLINICAL_TEXT = (
    "Synthetic note: metastatic melanoma was treated with pembrolizumab, "
    "and follow-up imaging evaluated lymph node involvement."
)


@dataclass(frozen=True)
class FamilySelection:
    """Selected model for one NER family."""

    family: str
    model: ModelInfo
    source: str


def select_model_for_family(
    family: str,
    *,
    tier: str = "balanced",
) -> FamilySelection | None:
    """Select a model for *family*, preferring the requested recommendation tier."""
    family_models = get_models_by_category(family)
    if not family_models:
        return None

    recommended_ids = {model.model_id for model in get_recommended_models(tier)}
    for model in family_models:
        if model.model_id in recommended_ids:
            return FamilySelection(family=family, model=model, source=f"{tier} tier")

    return FamilySelection(family=family, model=family_models[0], source="category fallback")


def selected_families(
    families: Iterable[str] = NER_FAMILIES,
    *,
    tier: str = "balanced",
) -> list[FamilySelection]:
    """Return model selections for each requested family."""
    selections: list[FamilySelection] = []
    for family in families:
        selection = select_model_for_family(family, tier=tier)
        if selection is not None:
            selections.append(selection)
    return selections


def iter_entities(result: Any) -> list[Any]:
    """Normalize result shapes returned by analyze_text for display."""
    if hasattr(result, "entities"):
        return list(result.entities)
    if isinstance(result, dict):
        return list(result.get("entities", []))
    return list(result or [])


def entity_value(entity: Any, key: str, default: Any = "") -> Any:
    """Read an entity field from dicts or dataclass-like objects."""
    if isinstance(entity, dict):
        return entity.get(key, default)
    return getattr(entity, key, default)


def print_entities(family: str, entities: Iterable[Any]) -> None:
    """Print labelled spans for one family."""
    print(f"\n{family}")
    rows = list(entities)
    if not rows:
        print("  No entities returned.")
        return

    for entity in rows:
        label = str(entity_value(entity, "label", "UNKNOWN"))
        text = str(entity_value(entity, "text", ""))
        confidence = entity_value(entity, "confidence", None)
        if confidence is None:
            print(f"  - {label:<24} {text!r}")
        else:
            print(f"  - {label:<24} {text!r} ({float(confidence):.3f})")


def run_family_extraction(
    text: str = SYNTHETIC_CLINICAL_TEXT,
    *,
    families: Iterable[str] = NER_FAMILIES,
    tier: str = "balanced",
    analyzer: Callable[..., Any] = analyze_text,
) -> None:
    """Run selected family models over synthetic clinical text."""
    print("OpenMed biomedical NER families")
    print(f"Text: {text}")

    for selection in selected_families(families, tier=tier):
        model = selection.model
        print(f"\nUsing {model.display_name} ({selection.source})")
        print(f"Model: {model.model_id}")
        try:
            result = analyzer(
                text,
                model_name=model.model_id,
                confidence_threshold=model.recommended_confidence,
                group_entities=True,
            )
        except Exception as exc:
            print(f"{selection.family}: model unavailable locally ({exc})")
            continue

        print_entities(selection.family, iter_entities(result))


def main() -> None:
    run_family_extraction()


if __name__ == "__main__":
    main()
