"""Adapter between Presidio recognizer results and OpenMed PII entities."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module as _import_module
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.labels import normalize_label
from openmed.core.pii import PIIEntity
from openmed.core.pii_entity_merger import merge_entities_with_semantic_units

_PRESIDIO_TO_CANONICAL = {
    "PERSON": "PERSON",
    "PHONE_NUMBER": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "LOCATION": "LOCATION",
    "DATE_TIME": "DATE",
    "IP_ADDRESS": "IP_ADDRESS",
    "URL": "URL",
    "US_SSN": "SSN",
    "CREDIT_CARD": "CREDIT_CARD",
    "CRYPTO": "BITCOIN_ADDRESS",
    "IBAN_CODE": "IBAN",
    "US_DRIVER_LICENSE": "ID_NUM",
    "US_PASSPORT": "ID_NUM",
    "US_BANK_NUMBER": "ACCOUNT_NUMBER",
    "MEDICAL_LICENSE": "ID_NUM",
}
_CANONICAL_TO_PRESIDIO = {
    "PERSON": "PERSON",
    "FIRST_NAME": "PERSON",
    "LAST_NAME": "PERSON",
    "MIDDLE_NAME": "PERSON",
    "EMAIL": "EMAIL_ADDRESS",
    "PHONE": "PHONE_NUMBER",
    "LOCATION": "LOCATION",
    "STREET_ADDRESS": "LOCATION",
    "ZIPCODE": "LOCATION",
    "DATE": "DATE_TIME",
    "DATE_OF_BIRTH": "DATE_TIME",
    "TIME": "DATE_TIME",
    "SSN": "US_SSN",
    "CREDIT_CARD": "CREDIT_CARD",
    "IBAN": "IBAN_CODE",
    "IP_ADDRESS": "IP_ADDRESS",
    "URL": "URL",
    "ACCOUNT_NUMBER": "US_BANK_NUMBER",
    "ID_NUM": "MEDICAL_LICENSE",
}


@dataclass(frozen=True)
class PresidioAdapterConfig:
    """Runtime conversion options for Presidio interoperability."""

    source: str = "presidio"
    preserve_presidio_labels: bool = True
    allow_semantic_only_matches: bool = False


def to_canonical(
    result: Any,
    *,
    text: str | None = None,
    config: PresidioAdapterConfig | None = None,
) -> list[PIIEntity]:
    """Convert one or more RecognizerResult-like objects to ``PIIEntity``."""

    cfg = config or PresidioAdapterConfig()
    return [
        _recognizer_result_to_entity(item, text=text, config=cfg)
        for item in _as_sequence(result)
    ]


def from_canonical(
    entities: Sequence[PIIEntity],
    *,
    result_cls: type[Any] | None = None,
    config: PresidioAdapterConfig | None = None,
) -> list[Any]:
    """Convert canonical PII entities back to RecognizerResult-like objects.

    Passing ``result_cls`` keeps tests and lightweight integrations free of the
    optional dependency. Without it, the Presidio extra must be installed.
    """

    cfg = config or PresidioAdapterConfig()
    cls = result_cls or _load_presidio_result_cls()
    return [
        _entity_to_recognizer_result(entity, cls, config=cfg) for entity in entities
    ]


def merge_with_openmed(
    openmed_entities: Sequence[PIIEntity],
    presidio_results: Any,
    *,
    text: str,
    config: PresidioAdapterConfig | None = None,
    use_semantic_patterns: bool = True,
) -> list[PIIEntity]:
    """Merge OpenMed and Presidio spans through OpenMed's PII merger path."""

    cfg = config or PresidioAdapterConfig()
    entities = [
        *_copy_entities(openmed_entities, source="openmed"),
        *to_canonical(presidio_results, text=text, config=cfg),
    ]
    if not entities:
        return []

    merger_input = [_entity_to_merger_dict(entity) for entity in entities]
    merged = merge_entities_with_semantic_units(
        merger_input,
        text,
        use_semantic_patterns=use_semantic_patterns,
        prefer_model_labels=True,
        allow_semantic_only_matches=cfg.allow_semantic_only_matches,
        allow_label_expansion=True,
    )
    return [_merger_dict_to_entity(item, text) for item in _resolve_overlaps(merged)]


def _load_presidio_result_cls() -> type[Any]:
    try:
        module = _import_module("presidio_analyzer")
    except ImportError as exc:
        raise ImportError(
            "Presidio support requires the 'presidio' extra. "
            "Install with `pip install openmed[presidio]`."
        ) from exc
    return module.RecognizerResult


def _recognizer_result_to_entity(
    result: Any,
    *,
    text: str | None,
    config: PresidioAdapterConfig,
) -> PIIEntity:
    entity_type = _result_value(result, "entity_type")
    start = int(_result_value(result, "start"))
    end = int(_result_value(result, "end"))
    score = float(_result_value(result, "score", default=0.0) or 0.0)
    source_label = str(entity_type)
    canonical = _canonical_label(source_label)
    source_text = _surface_text(result, text, start, end)
    metadata = {
        "adapter": config.source,
        "source": config.source,
        "presidio_entity_type": source_label,
    }
    recognition_metadata = _result_value(result, "recognition_metadata", default=None)
    if isinstance(recognition_metadata, Mapping):
        metadata["recognition_metadata"] = dict(recognition_metadata)
    recognizer_name = _result_value(result, "recognizer_name", default=None)
    if recognizer_name is not None:
        metadata["recognizer_name"] = str(recognizer_name)

    return PIIEntity(
        text=source_text,
        label=canonical,
        confidence=score,
        start=start,
        end=end,
        entity_type=canonical,
        original_text=source_text,
        metadata=metadata,
    )


def _entity_to_recognizer_result(
    entity: PIIEntity,
    result_cls: type[Any],
    *,
    config: PresidioAdapterConfig,
) -> Any:
    canonical = normalize_label(entity.entity_type or entity.label)
    source_label = None
    metadata = entity.metadata or {}
    if config.preserve_presidio_labels:
        source_label = metadata.get("presidio_entity_type")
    entity_type = str(source_label or _CANONICAL_TO_PRESIDIO.get(canonical, canonical))
    kwargs = {
        "entity_type": entity_type,
        "start": int(entity.start or 0),
        "end": int(entity.end or 0),
        "score": float(entity.confidence or 0.0),
    }
    return result_cls(**kwargs)


def _entity_to_merger_dict(entity: PIIEntity) -> dict[str, Any]:
    metadata = dict(entity.metadata or {})
    return {
        "entity_type": normalize_label(entity.entity_type or entity.label),
        "score": float(entity.confidence or 0.0),
        "start": int(entity.start or 0),
        "end": int(entity.end or 0),
        "word": entity.text,
        "metadata": metadata,
    }


def _merger_dict_to_entity(item: Mapping[str, Any], text: str) -> PIIEntity:
    start = int(item["start"])
    end = int(item["end"])
    label = normalize_label(str(item["entity_type"]))
    surface = str(item.get("word") or text[start:end])
    metadata = dict(item.get("metadata") or {})
    metadata.setdefault("adapter", "openmed")
    return PIIEntity(
        text=surface,
        label=label,
        confidence=float(item.get("score", 0.0) or 0.0),
        start=start,
        end=end,
        entity_type=label,
        original_text=surface,
        metadata=metadata,
    )


def _resolve_overlaps(items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    winners: list[Mapping[str, Any]] = []
    for item in sorted(
        items, key=lambda value: (int(value["start"]), int(value["end"]))
    ):
        overlaps = [
            existing
            for existing in winners
            if int(item["start"]) < int(existing["end"])
            and int(item["end"]) > int(existing["start"])
        ]
        if not overlaps:
            winners.append(item)
            continue

        candidate = _best_overlap([item, *overlaps])
        winners = [existing for existing in winners if existing not in overlaps]
        winners.append(candidate)

    return sorted(winners, key=lambda value: (int(value["start"]), int(value["end"])))


def _best_overlap(items: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return max(
        items,
        key=lambda item: (
            float(item.get("score", 0.0) or 0.0),
            int(item["end"]) - int(item["start"]),
            1 if (item.get("metadata") or {}).get("adapter") == "openmed" else 0,
            -int(item["start"]),
        ),
    )


def _canonical_label(label: str) -> str:
    mapped = _PRESIDIO_TO_CANONICAL.get(label.strip().upper())
    return mapped or normalize_label(label)


def _surface_text(result: Any, text: str | None, start: int, end: int) -> str:
    direct = _result_value(result, "text", default=None)
    if direct is not None:
        return str(direct)
    if text is not None:
        return text[start:end]
    return ""


def _result_value(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _as_sequence(result: Any) -> list[Any]:
    if result is None:
        return []
    if isinstance(result, (str, bytes, Mapping)):
        return [result]
    if _looks_like_result(result):
        return [result]
    if isinstance(result, Iterable):
        return list(result)
    return [result]


def _looks_like_result(value: Any) -> bool:
    return all(hasattr(value, name) for name in ("entity_type", "start", "end"))


def _copy_entities(entities: Sequence[PIIEntity], *, source: str) -> list[PIIEntity]:
    copied: list[PIIEntity] = []
    for entity in entities:
        metadata = dict(entity.metadata or {})
        metadata.setdefault("adapter", source)
        metadata.setdefault("source", source)
        copied.append(
            PIIEntity(
                text=entity.text,
                label=normalize_label(entity.entity_type or entity.label),
                confidence=float(entity.confidence or 0.0),
                start=entity.start,
                end=entity.end,
                entity_type=normalize_label(entity.entity_type or entity.label),
                redacted_text=entity.redacted_text,
                original_text=entity.original_text or entity.text,
                hash_value=entity.hash_value,
                reversible_id=entity.reversible_id,
                metadata=metadata,
            )
        )
    return copied


__all__ = [
    "PresidioAdapterConfig",
    "from_canonical",
    "merge_with_openmed",
    "to_canonical",
]
