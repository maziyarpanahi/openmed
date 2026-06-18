"""Adapter for GLiNER-BioMed zero-shot entity outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module as _import_module
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.labels import normalize_label
from openmed.core.pii import PIIEntity
from openmed.core.pii_entity_merger import merge_entities_with_semantic_units

from ._pii import (
    coerce_confidence,
    find_offsets,
    make_entity,
    merge_with_openmed_entities,
    value,
)


DEFAULT_MODEL_ID = "Ihor/gliner-biomed-large-v1.0"
DEFAULT_LABELS = (
    "person",
    "patient name",
    "doctor name",
    "hospital",
    "organization",
    "location",
    "street address",
    "date",
    "date of birth",
    "age",
    "phone number",
    "email address",
    "medical record number",
    "social security number",
    "identifier",
)
_GLINER_TO_CANONICAL = {
    "address": "STREET_ADDRESS",
    "birth date": "DATE_OF_BIRTH",
    "date of birth": "DATE_OF_BIRTH",
    "doctor": "PERSON",
    "doctor name": "PERSON",
    "email address": "EMAIL",
    "facility": "ORGANIZATION",
    "full address": "STREET_ADDRESS",
    "hospital": "ORGANIZATION",
    "identifier": "ID_NUM",
    "medical record number": "ID_NUM",
    "mrn": "ID_NUM",
    "patient": "PERSON",
    "patient name": "PERSON",
    "phone number": "PHONE",
    "social security number": "SSN",
    "ssn": "SSN",
}


@dataclass(frozen=True)
class GlinerBioMedAdapterConfig:
    """Runtime conversion options for GLiNER-BioMed interoperability."""

    source: str = "gliner_biomed"
    default_confidence: float = 0.0
    model_id: str = DEFAULT_MODEL_ID
    labels: Sequence[str] = field(default_factory=lambda: DEFAULT_LABELS)
    threshold: float = 0.5
    preserve_gliner_labels: bool = True
    allow_semantic_only_matches: bool = False


def to_canonical(
    result: Any,
    *,
    text: str | None = None,
    config: GlinerBioMedAdapterConfig | None = None,
) -> list[PIIEntity]:
    """Convert GLiNER ``predict_entities`` output to canonical PII entities."""

    cfg = config or GlinerBioMedAdapterConfig()
    entities: list[PIIEntity] = []
    search_from = 0
    for record in _entity_records(result):
        entity = _record_to_entity(record, text=text, config=cfg, search_from=search_from)
        entities.append(entity)
        search_from = int(entity.end or search_from)
    return entities


def predict_to_canonical(
    text: str,
    *,
    model: Any | None = None,
    labels: Sequence[str] | None = None,
    threshold: float | None = None,
    config: GlinerBioMedAdapterConfig | None = None,
) -> list[PIIEntity]:
    """Run a GLiNER-BioMed model and convert predictions to canonical entities."""

    cfg = config or GlinerBioMedAdapterConfig()
    model_ = model or _load_gliner_model(cfg.model_id)
    label_set = list(labels or cfg.labels)
    cutoff = cfg.threshold if threshold is None else threshold
    predictions = model_.predict_entities(text, label_set, threshold=cutoff)
    return to_canonical(predictions, text=text, config=cfg)


def from_canonical(
    entities: Sequence[PIIEntity],
    *,
    config: GlinerBioMedAdapterConfig | None = None,
) -> list[dict[str, Any]]:
    """Convert canonical PII entities back to GLiNER-style dictionaries."""

    cfg = config or GlinerBioMedAdapterConfig()
    return [_entity_to_record(entity, config=cfg) for entity in entities]


def merge_with_openmed(
    openmed_entities: Sequence[PIIEntity],
    gliner_output: Any,
    *,
    text: str,
    config: GlinerBioMedAdapterConfig | None = None,
    use_semantic_patterns: bool = True,
) -> list[PIIEntity]:
    """Merge OpenMed and GLiNER-BioMed spans through OpenMed's PII merger path."""

    cfg = config or GlinerBioMedAdapterConfig()
    return merge_with_openmed_entities(
        openmed_entities,
        to_canonical(gliner_output, text=text, config=cfg),
        text=text,
        source=cfg.source,
        merger=merge_entities_with_semantic_units,
        use_semantic_patterns=use_semantic_patterns,
        allow_semantic_only_matches=cfg.allow_semantic_only_matches,
    )


def _load_gliner_model(model_id: str) -> Any:
    try:
        module = _import_module("gliner")
    except ImportError as exc:
        raise ImportError(
            "GLiNER-BioMed support requires the 'gliner' extra. "
            "Install with `pip install openmed[gliner]`."
        ) from exc
    return module.GLiNER.from_pretrained(model_id)


def _record_to_entity(
    record: Any,
    *,
    text: str | None,
    config: GlinerBioMedAdapterConfig,
    search_from: int,
) -> PIIEntity:
    surface = str(value(record, ("text", "word", "span"), ""))
    raw_start = value(record, ("start", "start_char", "begin"))
    raw_end = value(record, ("end", "end_char", "stop"))
    start, end = find_offsets(
        surface,
        text,
        start=int(raw_start) if raw_start is not None else None,
        end=int(raw_end) if raw_end is not None else None,
        search_from=search_from,
    )
    if not surface and text is not None:
        surface = text[start:end]
    source_label = str(value(record, ("label", "entity_type", "type"), "entity"))
    confidence = coerce_confidence(
        value(record, ("score", "confidence", "probability")),
        config.default_confidence,
    )
    metadata = {
        "gliner_model": config.model_id,
        "gliner_record": dict(record) if isinstance(record, Mapping) else None,
    }
    if metadata["gliner_record"] is None:
        metadata.pop("gliner_record")
    return make_entity(
        text=surface,
        label=_canonical_label(source_label),
        confidence=confidence,
        start=start,
        end=end,
        source=config.source,
        source_label_key="gliner_label",
        source_label=source_label,
        metadata=metadata,
    )


def _entity_to_record(
    entity: PIIEntity,
    *,
    config: GlinerBioMedAdapterConfig,
) -> dict[str, Any]:
    canonical = normalize_label(entity.entity_type or entity.label)
    metadata = entity.metadata or {}
    source_label = None
    if config.preserve_gliner_labels:
        source_label = metadata.get("gliner_label")
    return {
        "text": entity.text,
        "label": str(source_label or canonical.lower().replace("_", " ")),
        "start": int(entity.start or 0),
        "end": int(entity.end or 0),
        "score": float(entity.confidence or 0.0),
    }


def _entity_records(result: Any) -> list[Any]:
    if result is None:
        return []
    if _looks_like_record(result):
        return [result]
    if isinstance(result, Mapping):
        for key in ("entities", "predictions", "spans"):
            records = result.get(key)
            if isinstance(records, Iterable) and not isinstance(records, (str, bytes)):
                return list(records)
        return [result]
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
        return list(result)
    return [result]


def _looks_like_record(result: Any) -> bool:
    return value(result, ("label", "entity_type", "type")) is not None and (
        value(result, ("text", "word", "span")) is not None
        or value(result, ("start", "start_char", "begin")) is not None
    )


def _canonical_label(label: str) -> str:
    key = " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())
    mapped = _GLINER_TO_CANONICAL.get(key)
    return mapped or normalize_label(label)


__all__ = [
    "DEFAULT_LABELS",
    "DEFAULT_MODEL_ID",
    "GlinerBioMedAdapterConfig",
    "from_canonical",
    "merge_with_openmed",
    "predict_to_canonical",
    "to_canonical",
]
