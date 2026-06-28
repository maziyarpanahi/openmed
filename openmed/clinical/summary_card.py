"""PHI-free aggregate summary cards for clinical document extraction.

The summary card is an aggregate overview for review and operational status
surfaces. It is not a clinical decision artifact and must not be used to make,
rank, or automate clinical decisions.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

ENTITY_CATEGORY_LABELS = ("problems", "medications", "labs", "procedures", "other")
CODING_COUNT_LABELS = ("coded_entities", "uncoded_entities", "distinct_codes")
SUMMARY_CARD_NOTE = (
    "Clinical summary cards are aggregate overviews, not clinical decision artifacts."
)

_TOP_LEVEL_KEYS = ("entity_counts", "coding_counts", "section_count")
_CATEGORY_FIELD_NAMES = (
    "category",
    "entity_category",
    "clinical_category",
    "semantic_type",
    "kind",
    "type",
    "entity_type",
    "entity_group",
    "label",
    "canonical_label",
)
_ENTITY_CONTAINER_NAMES = ("entities", "clinical_entities", "spans")
_SECTION_CONTAINER_NAMES = ("sections", "clinical_sections")
_CODING_CONTAINER_NAMES = (
    "coding",
    "codings",
    "codes",
    "codeable_concept",
    "codeableConcept",
    "concept",
)

_CATEGORY_ALIASES = {
    "problem": "problems",
    "problems": "problems",
    "condition": "problems",
    "conditions": "problems",
    "diagnosis": "problems",
    "diagnoses": "problems",
    "disease": "problems",
    "diseases": "problems",
    "disorder": "problems",
    "disorders": "problems",
    "finding": "problems",
    "findings": "problems",
    "symptom": "problems",
    "symptoms": "problems",
    "problem_list": "problems",
    "medication": "medications",
    "medications": "medications",
    "medicine": "medications",
    "medicines": "medications",
    "drug": "medications",
    "drugs": "medications",
    "rx": "medications",
    "rxnorm": "medications",
    "antibiotic": "medications",
    "antibiotics": "medications",
    "lab": "labs",
    "labs": "labs",
    "laboratory": "labs",
    "laboratory_test": "labs",
    "laboratory_tests": "labs",
    "lab_result": "labs",
    "lab_results": "labs",
    "test_result": "labs",
    "test_results": "labs",
    "loinc": "labs",
    "microorganism": "labs",
    "susceptibility": "labs",
    "procedure": "procedures",
    "procedures": "procedures",
    "surgery": "procedures",
    "surgeries": "procedures",
    "operation": "procedures",
    "operations": "procedures",
    "intervention": "procedures",
    "interventions": "procedures",
    "proc": "procedures",
}


@dataclass(frozen=True)
class ClinicalSummaryCard:
    """Typed PHI-free aggregate summary for a clinical document."""

    problems: int = 0
    medications: int = 0
    labs: int = 0
    procedures: int = 0
    other: int = 0
    coded_entities: int = 0
    uncoded_entities: int = 0
    distinct_codes: int = 0
    section_count: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            *ENTITY_CATEGORY_LABELS,
            *CODING_COUNT_LABELS,
            "section_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer count")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-compatible card representation."""

        payload: dict[str, Any] = {
            "entity_counts": {
                "problems": self.problems,
                "medications": self.medications,
                "labs": self.labs,
                "procedures": self.procedures,
                "other": self.other,
            },
            "coding_counts": {
                "coded_entities": self.coded_entities,
                "uncoded_entities": self.uncoded_entities,
                "distinct_codes": self.distinct_codes,
            },
            "section_count": self.section_count,
        }
        _assert_card_payload_is_log_safe(payload)
        return payload

    def to_json(self) -> str:
        """Serialize the card with fixed key order and compact separators."""

        return json.dumps(self.to_dict(), separators=(",", ":"))


def build_summary_card(
    entities: object | None = None,
    sections: object | None = None,
) -> ClinicalSummaryCard:
    """Build a PHI-free summary card from clinical entities and sections.

    The builder reads category labels and code presence only. It never copies
    entity text, offsets, dates, display strings, or source section labels into
    the returned card.

    Args:
        entities: Sequence of clinical entity mappings or objects. A document
            mapping with an ``entities``, ``clinical_entities``, or ``spans``
            field is also accepted.
        sections: Optional sequence of detected sections, or a section metadata
            mapping with a ``sections`` or ``clinical_sections`` field.

    Returns:
        A :class:`ClinicalSummaryCard` containing only aggregate counts.
    """

    entity_source = entities
    if isinstance(entities, Mapping):
        if sections is None:
            sections = _first_field_value(entities, _SECTION_CONTAINER_NAMES)
        nested_entities = _first_field_value(entities, _ENTITY_CONTAINER_NAMES)
        if nested_entities is not None:
            entity_source = nested_entities

    category_counts = dict.fromkeys(ENTITY_CATEGORY_LABELS, 0)
    coded_entities = 0
    uncoded_entities = 0
    distinct_codes: set[tuple[str, str]] = set()

    for entity in _iter_items(entity_source):
        category = _category_for_entity(entity)
        category_counts[category] += 1

        code_pairs = tuple(_code_pairs_for(entity))
        if code_pairs:
            coded_entities += 1
            distinct_codes.update(code_pairs)
        else:
            uncoded_entities += 1

    return ClinicalSummaryCard(
        problems=category_counts["problems"],
        medications=category_counts["medications"],
        labs=category_counts["labs"],
        procedures=category_counts["procedures"],
        other=category_counts["other"],
        coded_entities=coded_entities,
        uncoded_entities=uncoded_entities,
        distinct_codes=len(distinct_codes),
        section_count=_count_sections(sections),
    )


def _category_for_entity(entity: object) -> str:
    for field_name in _CATEGORY_FIELD_NAMES:
        value = _field_value(entity, field_name)
        if value is None:
            continue
        category = _normalize_category(value)
        if category != "other":
            return category
    return "other"


def _normalize_category(value: object) -> str:
    if not isinstance(value, str):
        return "other"
    key = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return _CATEGORY_ALIASES.get(key, "other")


def _code_pairs_for(entity: object) -> Iterable[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    for pair in _iter_code_pairs(entity):
        if pair not in seen:
            seen.add(pair)
            yield pair


def _iter_code_pairs(value: object) -> Iterable[tuple[str, str]]:
    if value is None or isinstance(value, (str, bytes)):
        return

    if isinstance(value, Mapping):
        code = value.get("code")
        if _has_code_value(code):
            system = value.get("system")
            yield (_safe_code_part(system), _safe_code_part(code))

        for field_name in _CODING_CONTAINER_NAMES:
            if field_name in value:
                yield from _iter_code_pairs(value[field_name])
        return

    if isinstance(value, Iterable):
        for item in value:
            if isinstance(item, str):
                if _has_code_value(item):
                    yield ("", item)
            elif isinstance(item, (Mapping, Iterable)) and not isinstance(item, bytes):
                yield from _iter_code_pairs(item)
        return

    code = getattr(value, "code", None)
    if _has_code_value(code):
        yield (_safe_code_part(getattr(value, "system", None)), _safe_code_part(code))

    for field_name in _CODING_CONTAINER_NAMES:
        nested = getattr(value, field_name, None)
        if nested is not None:
            yield from _iter_code_pairs(nested)


def _has_code_value(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _safe_code_part(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _count_sections(sections: object | None) -> int:
    if sections is None or isinstance(sections, (str, bytes)):
        return 0

    if isinstance(sections, Mapping):
        nested = _first_field_value(sections, _SECTION_CONTAINER_NAMES)
        if nested is not None:
            return _count_sections(nested)
        if any(key in sections for key in ("start", "end", "label", "name", "title")):
            return 1
        return 0

    if isinstance(sections, Iterable):
        return sum(1 for section in sections if section is not None)

    return 0


def _iter_items(value: object | None) -> Iterable[object]:
    if value is None or isinstance(value, (str, bytes)):
        return

    if isinstance(value, Mapping):
        nested = _first_field_value(value, _ENTITY_CONTAINER_NAMES)
        if nested is not None:
            yield from _iter_items(nested)
            return
        yield value
        return

    if isinstance(value, Iterable):
        yield from value
        return

    yield value


def _first_field_value(source: object, field_names: Sequence[str]) -> object | None:
    for field_name in field_names:
        value = _field_value(source, field_name)
        if value is not None:
            return value
    return None


def _field_value(source: object, field_name: str) -> object | None:
    if isinstance(source, Mapping):
        return source.get(field_name)
    return getattr(source, field_name, None)


def _assert_card_payload_is_log_safe(payload: Mapping[str, Any]) -> None:
    if tuple(payload.keys()) != _TOP_LEVEL_KEYS:
        raise ValueError("summary card payload keys must be fixed")

    entity_counts = payload["entity_counts"]
    coding_counts = payload["coding_counts"]
    if (
        not isinstance(entity_counts, Mapping)
        or tuple(entity_counts.keys()) != ENTITY_CATEGORY_LABELS
    ):
        raise ValueError("summary card entity count keys must be fixed")
    if (
        not isinstance(coding_counts, Mapping)
        or tuple(coding_counts.keys()) != CODING_COUNT_LABELS
    ):
        raise ValueError("summary card coding count keys must be fixed")

    for label, value in (
        *entity_counts.items(),
        *coding_counts.items(),
        ("section_count", payload["section_count"]),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"summary card field {label!r} must be an integer count")


__all__ = [
    "CODING_COUNT_LABELS",
    "ENTITY_CATEGORY_LABELS",
    "SUMMARY_CARD_NOTE",
    "ClinicalSummaryCard",
    "build_summary_card",
]
