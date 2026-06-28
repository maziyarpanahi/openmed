"""Shared helpers for optional PII interoperability adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from openmed.core.labels import normalize_label
from openmed.core.pii import PIIEntity

Merger = Callable[..., list[dict[str, Any]]]


def value(result: Any, names: str | Sequence[str], default: Any = None) -> Any:
    """Read the first present mapping key or object attribute from *result*."""

    candidates = (names,) if isinstance(names, str) else names
    for name in candidates:
        if isinstance(result, Mapping) and name in result:
            return result[name]
        if hasattr(result, name):
            return getattr(result, name)
    return default


def coerce_int(raw: Any, *, field: str) -> int:
    if raw is None:
        raise ValueError(f"missing required span field {field!r}")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer span field {field!r}: {raw!r}") from exc


def coerce_confidence(raw: Any, default: float) -> float:
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def surface_text(
    result: Any,
    text: str | None,
    start: int,
    end: int,
    *,
    fields: Sequence[str],
) -> str:
    for field in fields:
        direct = value(result, field)
        if direct is not None:
            return str(direct)
    if text is not None:
        return text[start:end]
    return ""


def find_offsets(
    surface: str,
    text: str | None,
    *,
    start: int | None = None,
    end: int | None = None,
    search_from: int = 0,
) -> tuple[int, int]:
    if start is not None and end is not None:
        return start, end
    if not surface or text is None:
        raise ValueError("span offsets are required when source text is unavailable")
    found = text.find(surface, max(search_from, 0))
    if found < 0:
        found = text.find(surface)
    if found < 0:
        raise ValueError(f"could not locate span text {surface!r} in source text")
    return found, found + len(surface)


def make_entity(
    *,
    text: str,
    label: str,
    confidence: float,
    start: int,
    end: int,
    source: str,
    source_label_key: str,
    source_label: str,
    metadata: Mapping[str, Any] | None = None,
) -> PIIEntity:
    canonical = normalize_label(label)
    details = dict(metadata or {})
    details.setdefault("adapter", source)
    details.setdefault("source", source)
    details[source_label_key] = source_label
    return PIIEntity(
        text=text,
        label=canonical,
        confidence=confidence,
        start=start,
        end=end,
        entity_type=canonical,
        original_text=text,
        metadata=details,
    )


def merge_with_openmed_entities(
    openmed_entities: Sequence[PIIEntity],
    adapter_entities: Sequence[PIIEntity],
    *,
    text: str,
    source: str,
    merger: Merger,
    use_semantic_patterns: bool = True,
    allow_semantic_only_matches: bool = False,
) -> list[PIIEntity]:
    """Route first-party and adapter spans through OpenMed's PII merger."""

    entities = [
        *_copy_entities(openmed_entities, source="openmed"),
        *_copy_entities(adapter_entities, source=source),
    ]
    if not entities:
        return []

    source_dicts = [_entity_to_merger_dict(entity) for entity in entities]
    merged = merger(
        source_dicts,
        text,
        use_semantic_patterns=use_semantic_patterns,
        prefer_model_labels=True,
        allow_semantic_only_matches=allow_semantic_only_matches,
        allow_label_expansion=True,
    )
    return [
        _merger_dict_to_entity(item, text)
        for item in _resolve_overlaps(
            [_restore_provenance(item, source_dicts) for item in merged]
        )
    ]


def _copy_entities(
    entities: Sequence[PIIEntity],
    *,
    source: str,
) -> list[PIIEntity]:
    copied: list[PIIEntity] = []
    for entity in entities:
        label = normalize_label(entity.entity_type or entity.label)
        metadata = dict(entity.metadata or {})
        metadata.setdefault("adapter", source)
        metadata.setdefault("source", source)
        copied.append(
            PIIEntity(
                text=entity.text,
                label=label,
                confidence=float(entity.confidence or 0.0),
                start=entity.start,
                end=entity.end,
                entity_type=label,
                redacted_text=entity.redacted_text,
                original_text=entity.original_text or entity.text,
                hash_value=entity.hash_value,
                reversible_id=entity.reversible_id,
                metadata=metadata,
                sources=list(entity.sources or [source]),
            )
        )
    return copied


def _entity_to_merger_dict(entity: PIIEntity) -> dict[str, Any]:
    return {
        "entity_type": normalize_label(entity.entity_type or entity.label),
        "score": float(entity.confidence or 0.0),
        "start": int(entity.start or 0),
        "end": int(entity.end or 0),
        "word": entity.text,
        "metadata": dict(entity.metadata or {}),
        "sources": list(entity.sources or []),
    }


def _merger_dict_to_entity(item: Mapping[str, Any], text: str) -> PIIEntity:
    start = int(item["start"])
    end = int(item["end"])
    label = normalize_label(str(item["entity_type"]))
    surface = str(item.get("word") or text[start:end])
    metadata = dict(item.get("metadata") or {})
    metadata.setdefault("adapter", "openmed")
    metadata.setdefault("source", metadata["adapter"])
    sources = list(item.get("sources") or metadata.get("source_adapters") or [])
    if not sources:
        sources = [metadata["source"]]
    return PIIEntity(
        text=surface,
        label=label,
        confidence=float(item.get("score", 0.0) or 0.0),
        start=start,
        end=end,
        entity_type=label,
        original_text=surface,
        metadata=metadata,
        sources=sources,
    )


def _restore_provenance(
    item: Mapping[str, Any],
    source_entities: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    restored = dict(item)
    if restored.get("metadata") and restored.get("sources"):
        return restored

    overlapping = [
        source
        for source in source_entities
        if int(restored["start"]) < int(source["end"])
        and int(restored["end"]) > int(source["start"])
    ]
    if not overlapping:
        return restored

    metadata = dict(restored.get("metadata") or {})
    adapters: list[str] = []
    sources: list[str] = []
    for source in overlapping:
        source_metadata = dict(source.get("metadata") or {})
        adapter = str(
            source_metadata.get("adapter") or source_metadata.get("source") or "openmed"
        )
        if adapter not in adapters:
            adapters.append(adapter)
        for source_name in source.get("sources") or [adapter]:
            source_text = str(source_name)
            if source_text not in sources:
                sources.append(source_text)
        for key, value_ in source_metadata.items():
            if key not in metadata:
                metadata[key] = value_

    if adapters:
        metadata["adapter"] = adapters[0] if len(adapters) == 1 else "merged"
        metadata["source"] = metadata["adapter"]
        metadata["source_adapters"] = adapters
    restored["metadata"] = metadata
    restored["sources"] = sources or adapters
    return restored


def _resolve_overlaps(
    items: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    winners: list[Mapping[str, Any]] = []
    for item in sorted(
        items, key=lambda entry: (int(entry["start"]), int(entry["end"]))
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

    return sorted(winners, key=lambda entry: (int(entry["start"]), int(entry["end"])))


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
