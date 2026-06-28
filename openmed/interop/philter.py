"""Adapter between Philter PHI spans and OpenMed PII entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.labels import normalize_label
from openmed.core.pii import PIIEntity
from openmed.core.pii_entity_merger import merge_entities_with_semantic_units

from ._pii import (
    coerce_confidence,
    coerce_int,
    make_entity,
    merge_with_openmed_entities,
    surface_text,
    value,
)

_PHILTER_TO_CANONICAL = {
    "AGE": "AGE",
    "CONTACT": "PHONE",
    "DATE": "DATE",
    "DOCTOR": "PERSON",
    "EMAIL": "EMAIL",
    "HOSPITAL": "ORGANIZATION",
    "ID": "ID_NUM",
    "LOCATION": "LOCATION",
    "NAME": "PERSON",
    "PATIENT": "PERSON",
    "PHONE": "PHONE",
    "PROFESSION": "OCCUPATION",
    "SSN": "SSN",
    "STREET": "STREET_ADDRESS",
    "URL": "URL",
    "ZIP": "ZIPCODE",
}
_CANONICAL_TO_PHILTER = {
    "AGE": "AGE",
    "DATE": "DATE",
    "DATE_OF_BIRTH": "DATE",
    "EMAIL": "EMAIL",
    "ID_NUM": "ID",
    "LOCATION": "LOCATION",
    "OCCUPATION": "PROFESSION",
    "ORGANIZATION": "HOSPITAL",
    "PERSON": "NAME",
    "FIRST_NAME": "NAME",
    "LAST_NAME": "NAME",
    "PHONE": "PHONE",
    "SSN": "SSN",
    "STREET_ADDRESS": "STREET",
    "URL": "URL",
    "ZIPCODE": "ZIP",
}


@dataclass(frozen=True)
class PhilterAdapterConfig:
    """Runtime conversion options for Philter interoperability."""

    source: str = "philter"
    default_confidence: float = 1.0
    preserve_philter_labels: bool = True
    allow_semantic_only_matches: bool = False


def to_canonical(
    result: Any,
    *,
    text: str | None = None,
    config: PhilterAdapterConfig | None = None,
) -> list[PIIEntity]:
    """Convert Philter ``phi`` records or XML-tag-like records to entities."""

    cfg = config or PhilterAdapterConfig()
    records = _phi_records(result)
    return [_record_to_entity(record, text=text, config=cfg) for record in records]


def from_canonical(
    entities: Sequence[PIIEntity],
    *,
    config: PhilterAdapterConfig | None = None,
) -> list[dict[str, Any]]:
    """Convert canonical PII entities back to Philter-style ``phi`` records."""

    cfg = config or PhilterAdapterConfig()
    return [_entity_to_record(entity, config=cfg) for entity in entities]


def merge_with_openmed(
    openmed_entities: Sequence[PIIEntity],
    philter_output: Any,
    *,
    text: str,
    config: PhilterAdapterConfig | None = None,
    use_semantic_patterns: bool = True,
) -> list[PIIEntity]:
    """Merge OpenMed and Philter spans through OpenMed's PII merger path."""

    cfg = config or PhilterAdapterConfig()
    return merge_with_openmed_entities(
        openmed_entities,
        to_canonical(philter_output, text=text, config=cfg),
        text=text,
        source=cfg.source,
        merger=merge_entities_with_semantic_units,
        use_semantic_patterns=use_semantic_patterns,
        allow_semantic_only_matches=cfg.allow_semantic_only_matches,
    )


def _record_to_entity(
    record: Any,
    *,
    text: str | None,
    config: PhilterAdapterConfig,
) -> PIIEntity:
    start = coerce_int(value(record, ("start", "begin", "start_char")), field="start")
    end = coerce_int(value(record, ("stop", "end", "end_char")), field="end")
    source_label = str(
        value(record, ("phi_type", "TYPE", "type", "label", "entity_type"), "OTHER")
    )
    confidence = coerce_confidence(
        value(record, ("score", "confidence")),
        config.default_confidence,
    )
    surface = surface_text(
        record,
        text,
        start,
        end,
        fields=("word", "text", "phi"),
    )
    metadata = {
        "philter_record": dict(record) if isinstance(record, Mapping) else None,
    }
    if metadata["philter_record"] is None:
        metadata.pop("philter_record")
    return make_entity(
        text=surface,
        label=_canonical_label(source_label),
        confidence=confidence,
        start=start,
        end=end,
        source=config.source,
        source_label_key="philter_phi_type",
        source_label=source_label,
        metadata=metadata,
    )


def _entity_to_record(
    entity: PIIEntity,
    *,
    config: PhilterAdapterConfig,
) -> dict[str, Any]:
    canonical = normalize_label(entity.entity_type or entity.label)
    metadata = entity.metadata or {}
    source_label = None
    if config.preserve_philter_labels:
        source_label = metadata.get("philter_phi_type")
    phi_type = str(source_label or _CANONICAL_TO_PHILTER.get(canonical, canonical))
    return {
        "start": int(entity.start or 0),
        "stop": int(entity.end or 0),
        "word": entity.text,
        "phi_type": phi_type,
        "filepath": str(metadata.get("filepath", "")),
    }


def _phi_records(result: Any) -> list[Any]:
    if result is None:
        return []
    if _looks_like_record(result):
        return [result]
    if isinstance(result, Mapping):
        if "phi" in result and isinstance(result["phi"], Iterable):
            return list(result["phi"])

        records: list[Any] = []
        for value_ in result.values():
            if _looks_like_record(value_):
                records.append(value_)
            elif isinstance(value_, Mapping) and "phi" in value_:
                records.extend(list(value_["phi"]))
            elif isinstance(value_, Iterable) and not isinstance(value_, (str, bytes)):
                records.extend(item for item in value_ if _looks_like_record(item))
        return records
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
        return list(result)
    return [result]


def _looks_like_record(result: Any) -> bool:
    return (
        value(result, ("start", "begin", "start_char")) is not None
        and value(
            result,
            ("stop", "end", "end_char"),
        )
        is not None
    )


def _canonical_label(label: str) -> str:
    mapped = _PHILTER_TO_CANONICAL.get(label.strip().upper())
    return mapped or normalize_label(label)


__all__ = [
    "PhilterAdapterConfig",
    "from_canonical",
    "merge_with_openmed",
    "to_canonical",
]
