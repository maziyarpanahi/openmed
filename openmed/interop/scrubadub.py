"""Adapter between scrubadub Filth spans and OpenMed PII entities."""

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

_SCRUBADUB_TO_CANONICAL = {
    "address": "STREET_ADDRESS",
    "credential": "PASSWORD",
    "credit_card": "CREDIT_CARD",
    "date_of_birth": "DATE_OF_BIRTH",
    "drivers_licence": "ID_NUM",
    "email": "EMAIL",
    "location": "LOCATION",
    "name": "PERSON",
    "national_insurance_number": "SSN",
    "organization": "ORGANIZATION",
    "phone": "PHONE",
    "postalcode": "ZIPCODE",
    "skype": "USERNAME",
    "social_security_number": "SSN",
    "tax_reference_number": "ID_NUM",
    "twitter": "USERNAME",
    "url": "URL",
    "vehicle_licence_plate": "VEHICLE_REGISTRATION",
}
_CANONICAL_TO_SCRUBADUB = {
    "STREET_ADDRESS": "address",
    "PASSWORD": "credential",
    "CREDIT_CARD": "credit_card",
    "DATE_OF_BIRTH": "date_of_birth",
    "ID_NUM": "drivers_licence",
    "EMAIL": "email",
    "LOCATION": "location",
    "PERSON": "name",
    "FIRST_NAME": "name",
    "LAST_NAME": "name",
    "SSN": "social_security_number",
    "ORGANIZATION": "organization",
    "PHONE": "phone",
    "ZIPCODE": "postalcode",
    "USERNAME": "twitter",
    "URL": "url",
    "VEHICLE_REGISTRATION": "vehicle_licence_plate",
}


@dataclass(frozen=True)
class ScrubadubAdapterConfig:
    """Runtime conversion options for scrubadub interoperability."""

    source: str = "scrubadub"
    default_confidence: float = 1.0
    preserve_scrubadub_types: bool = True
    allow_semantic_only_matches: bool = False


def to_canonical(
    result: Any,
    *,
    text: str | None = None,
    config: ScrubadubAdapterConfig | None = None,
) -> list[PIIEntity]:
    """Convert one or more scrubadub ``Filth`` records to canonical entities."""

    cfg = config or ScrubadubAdapterConfig()
    entities: list[PIIEntity] = []
    for record in _as_sequence(result):
        entities.extend(_record_to_entities(record, text=text, config=cfg))
    return entities


def from_canonical(
    entities: Sequence[PIIEntity],
    *,
    config: ScrubadubAdapterConfig | None = None,
) -> list[dict[str, Any]]:
    """Convert canonical PII entities back to scrubadub-style Filth records.

    A username/password pair produced by :func:`to_canonical` splitting a
    ``credential`` match is recombined into the single original record here,
    so the round trip loses nothing when both halves are present. A lone
    surviving half (for example if a caller filtered entities beforehand)
    still degrades gracefully to a standalone ``credential`` record.
    """

    cfg = config or ScrubadubAdapterConfig()
    pairs = _index_credential_pairs(entities)
    emitted_pairs: set[tuple[int, int]] = set()
    records: list[dict[str, Any]] = []
    for entity in entities:
        key = _credential_pair_key(entity)
        if key is None:
            records.append(_entity_to_record(entity, config=cfg))
            continue
        if key in emitted_pairs:
            continue
        pair = pairs.get(key, {})
        if "username" in pair and "password" in pair:
            records.append(_credential_pair_to_record(pair, config=cfg))
            emitted_pairs.add(key)
        else:
            records.append(_entity_to_record(entity, config=cfg))
    return records


def merge_with_openmed(
    openmed_entities: Sequence[PIIEntity],
    scrubadub_output: Any,
    *,
    text: str,
    config: ScrubadubAdapterConfig | None = None,
    use_semantic_patterns: bool = True,
) -> list[PIIEntity]:
    """Merge OpenMed and scrubadub spans through OpenMed's PII merger path."""

    cfg = config or ScrubadubAdapterConfig()
    return merge_with_openmed_entities(
        openmed_entities,
        to_canonical(scrubadub_output, text=text, config=cfg),
        text=text,
        source=cfg.source,
        merger=merge_entities_with_semantic_units,
        use_semantic_patterns=use_semantic_patterns,
        allow_semantic_only_matches=cfg.allow_semantic_only_matches,
    )


def _record_to_entities(
    record: Any,
    *,
    text: str | None,
    config: ScrubadubAdapterConfig,
) -> list[PIIEntity]:
    source_type = str(value(record, ("type",), "unknown"))
    if source_type == "credential":
        split = _split_credential_record(record, config=config)
        if split is not None:
            return split
    return [_record_to_entity(record, text=text, config=config)]


def _split_credential_record(
    record: Any,
    *,
    config: ScrubadubAdapterConfig,
) -> list[PIIEntity] | None:
    """Split a real ``CredentialFilth`` match into USERNAME/PASSWORD entities.

    scrubadub's ``CredentialDetector`` regex has named ``username``/``password``
    groups and attaches the full ``re.Match`` to ``Filth.match``, so the exact
    substrings and offsets are available without re-parsing anything. Returns
    ``None`` when no such match is present (e.g. synthetic dict-shaped
    records), so the caller falls back to a single collapsed entity.
    """

    match = value(record, "match")
    if match is None:
        return None
    try:
        groups = match.groupdict()
    except AttributeError:
        return None
    if "username" not in groups or "password" not in groups:
        return None

    detector_name = value(record, "detector_name")
    document_name = value(record, "document_name")
    entities: list[PIIEntity] = []
    for group_name, label in (("username", "USERNAME"), ("password", "PASSWORD")):
        group_text = groups.get(group_name)
        if not group_text:
            continue
        metadata: dict[str, Any] = {
            "scrubadub_type": "credential",
            "scrubadub_credential_field": group_name,
            "scrubadub_credential_beg": match.start(),
            "scrubadub_credential_end": match.end(),
            "scrubadub_credential_text": match.group(),
        }
        if detector_name is not None:
            metadata["detector_name"] = str(detector_name)
        if document_name is not None:
            metadata["document_name"] = str(document_name)
        entities.append(
            make_entity(
                text=group_text,
                label=label,
                confidence=config.default_confidence,
                start=match.start(group_name),
                end=match.end(group_name),
                source=config.source,
                source_label_key="scrubadub_type",
                source_label="credential",
                metadata=metadata,
            )
        )
    return entities or None


def _record_to_entity(
    record: Any,
    *,
    text: str | None,
    config: ScrubadubAdapterConfig,
) -> PIIEntity:
    start = coerce_int(value(record, ("beg", "start")), field="beg")
    end = coerce_int(value(record, ("end", "stop")), field="end")
    source_type = str(value(record, ("type",), "unknown"))
    confidence = coerce_confidence(
        value(record, ("prob", "score", "confidence")),
        config.default_confidence,
    )
    surface = surface_text(
        record,
        text,
        start,
        end,
        fields=("text", "word"),
    )
    metadata: dict[str, Any] = {"scrubadub_type": source_type}
    detector_name = value(record, "detector_name")
    if detector_name is not None:
        metadata["detector_name"] = str(detector_name)
    document_name = value(record, "document_name")
    if document_name is not None:
        metadata["document_name"] = str(document_name)
    return make_entity(
        text=surface,
        label=_canonical_label(source_type),
        confidence=confidence,
        start=start,
        end=end,
        source=config.source,
        source_label_key="scrubadub_type",
        source_label=source_type,
        metadata=metadata,
    )


def _credential_pair_key(entity: PIIEntity) -> tuple[int, int] | None:
    metadata = entity.metadata or {}
    if metadata.get("scrubadub_type") != "credential":
        return None
    beg = metadata.get("scrubadub_credential_beg")
    end = metadata.get("scrubadub_credential_end")
    if beg is None or end is None:
        return None
    return int(beg), int(end)


def _index_credential_pairs(
    entities: Sequence[PIIEntity],
) -> dict[tuple[int, int], dict[str, PIIEntity]]:
    pairs: dict[tuple[int, int], dict[str, PIIEntity]] = {}
    for entity in entities:
        key = _credential_pair_key(entity)
        if key is None:
            continue
        field = (entity.metadata or {}).get("scrubadub_credential_field")
        if field not in ("username", "password"):
            continue
        pairs.setdefault(key, {})[field] = entity
    return pairs


def _credential_pair_to_record(
    pair: Mapping[str, PIIEntity],
    *,
    config: ScrubadubAdapterConfig,
) -> dict[str, Any]:
    metadata = pair["username"].metadata or {}
    beg = int(metadata["scrubadub_credential_beg"])
    end = int(metadata["scrubadub_credential_end"])
    text = metadata.get("scrubadub_credential_text")
    if text is None:
        text = f"{pair['username'].text} {pair['password'].text}"
    return {
        "beg": beg,
        "end": end,
        "text": text,
        "type": "credential",
        "detector_name": metadata.get("detector_name", config.source),
        "document_name": metadata.get("document_name"),
        "replacement_string": None,
    }


def _entity_to_record(
    entity: PIIEntity,
    *,
    config: ScrubadubAdapterConfig,
) -> dict[str, Any]:
    canonical = normalize_label(entity.entity_type or entity.label)
    metadata = entity.metadata or {}
    source_type = None
    if config.preserve_scrubadub_types:
        source_type = metadata.get("scrubadub_type")
    if not source_type:
        source_type = _CANONICAL_TO_SCRUBADUB.get(canonical, canonical.lower())
    return {
        "beg": int(entity.start or 0),
        "end": int(entity.end or 0),
        "text": entity.text,
        "type": source_type,
        "detector_name": metadata.get("detector_name", config.source),
        "document_name": metadata.get("document_name"),
        "replacement_string": None,
    }


def _canonical_label(source_type: str) -> str:
    mapped = _SCRUBADUB_TO_CANONICAL.get(source_type.strip().lower())
    return mapped or normalize_label(source_type)


def _as_sequence(result: Any) -> list[Any]:
    if result is None:
        return []
    if isinstance(result, (str, bytes, Mapping)):
        return [result]
    if _looks_like_record(result):
        return [result]
    if isinstance(result, Iterable):
        return list(result)
    return [result]


def _looks_like_record(result: Any) -> bool:
    return (
        value(result, ("beg", "start")) is not None
        and value(result, ("end", "stop")) is not None
    )


__all__ = [
    "ScrubadubAdapterConfig",
    "from_canonical",
    "merge_with_openmed",
    "to_canonical",
]
