"""Adapter between pyDeid PHI spans and OpenMed PII entities."""

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

_CANONICAL_TO_PYDEID = {
    "AGE": "Age",
    "DATE": "Date",
    "DATE_OF_BIRTH": "Date",
    "EMAIL": "Email",
    "ID_NUM": "Medical Record Number",
    "LOCATION": "Location",
    "ORGANIZATION": "Hospital Name",
    "PERSON": "Name",
    "FIRST_NAME": "Name",
    "LAST_NAME": "Name",
    "PHONE": "Telephone/Fax",
    "SSN": "SIN/SSN",
    "STREET_ADDRESS": "Address",
    "ZIPCODE": "Postal Code",
}


@dataclass(frozen=True)
class PyDeidAdapterConfig:
    """Runtime conversion options for pyDeid interoperability."""

    source: str = "pydeid"
    default_confidence: float = 1.0
    preserve_pydeid_types: bool = True
    allow_semantic_only_matches: bool = False


def to_canonical(
    result: Any,
    *,
    text: str | None = None,
    config: PyDeidAdapterConfig | None = None,
) -> list[PIIEntity]:
    """Convert pyDeid ``phi`` records to canonical PII entities."""

    cfg = config or PyDeidAdapterConfig()
    records = _phi_records(result)
    return [_record_to_entity(record, text=text, config=cfg) for record in records]


def from_canonical(
    entities: Sequence[PIIEntity],
    *,
    config: PyDeidAdapterConfig | None = None,
) -> list[dict[str, Any]]:
    """Convert canonical PII entities back to pyDeid-style ``phi`` records."""

    cfg = config or PyDeidAdapterConfig()
    return [_entity_to_record(entity, config=cfg) for entity in entities]


def merge_with_openmed(
    openmed_entities: Sequence[PIIEntity],
    pydeid_output: Any,
    *,
    text: str,
    config: PyDeidAdapterConfig | None = None,
    use_semantic_patterns: bool = True,
) -> list[PIIEntity]:
    """Merge OpenMed and pyDeid spans through OpenMed's PII merger path."""

    cfg = config or PyDeidAdapterConfig()
    return merge_with_openmed_entities(
        openmed_entities,
        to_canonical(pydeid_output, text=text, config=cfg),
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
    config: PyDeidAdapterConfig,
) -> PIIEntity:
    start = coerce_int(
        value(record, ("phi_start", "start", "begin")), field="phi_start"
    )
    end = coerce_int(value(record, ("phi_end", "end", "stop")), field="phi_end")
    source_types = _source_types(record)
    source_label = source_types[0] if source_types else "PHI"
    confidence = coerce_confidence(
        value(record, ("score", "confidence")),
        config.default_confidence,
    )
    surface = surface_text(
        record,
        text,
        start,
        end,
        fields=("phi", "text", "word"),
    )
    metadata: dict[str, Any] = {"pydeid_types": source_types}
    surrogate = value(record, "surrogate")
    if surrogate is not None:
        metadata["surrogate"] = str(surrogate)
    return make_entity(
        text=surface,
        label=_canonical_label(source_types),
        confidence=confidence,
        start=start,
        end=end,
        source=config.source,
        source_label_key="pydeid_type",
        source_label=source_label,
        metadata=metadata,
    )


def _entity_to_record(
    entity: PIIEntity,
    *,
    config: PyDeidAdapterConfig,
) -> dict[str, Any]:
    canonical = normalize_label(entity.entity_type or entity.label)
    metadata = entity.metadata or {}
    source_types = None
    if config.preserve_pydeid_types:
        source_types = metadata.get("pydeid_types")
    if not source_types:
        source_types = [_CANONICAL_TO_PYDEID.get(canonical, canonical)]
    return {
        "phi_start": int(entity.start or 0),
        "phi_end": int(entity.end or 0),
        "phi": entity.text,
        "surrogate_start": None,
        "surrogate_end": None,
        "surrogate": metadata.get("surrogate", "<PHI>"),
        "types": list(source_types),
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
        value(result, ("phi_start", "start", "begin")) is not None
        and value(
            result,
            ("phi_end", "end", "stop"),
        )
        is not None
    )


def _source_types(record: Any) -> list[str]:
    raw = value(record, ("types", "type", "label", "entity_type"), [])
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Iterable):
        return [str(item) for item in raw]
    return [str(raw)]


def _canonical_label(types: Sequence[str]) -> str:
    for source_type in types:
        canonical = _canonical_type(source_type)
        if canonical != "OTHER":
            return canonical
    return "OTHER"


def _canonical_type(source_type: str) -> str:
    key = source_type.lower()
    if "email" in key:
        return "EMAIL"
    if "telephone" in key or "phone" in key or "fax" in key:
        return "PHONE"
    if "postal" in key or "zip" in key:
        return "ZIPCODE"
    if "address" in key or "street" in key:
        return "STREET_ADDRESS"
    if "hospital" in key or "clinic" in key or "facility" in key:
        return "ORGANIZATION"
    if "date" in key or "month" in key or "year" in key or "day" in key:
        return "DATE"
    if "time" in key:
        return "TIME"
    if "age" in key:
        return "AGE"
    if (
        "medical record" in key
        or "mrn" in key
        or "ohip" in key
        or "sin" in key
        or "identifier" in key
        or "id" == key.strip()
    ):
        return "ID_NUM"
    if "ssn" in key or "social security" in key:
        return "SSN"
    if "name" in key or "patient" in key or "doctor" in key:
        return "PERSON"
    return normalize_label(source_type)


__all__ = [
    "PyDeidAdapterConfig",
    "from_canonical",
    "merge_with_openmed",
    "to_canonical",
]
