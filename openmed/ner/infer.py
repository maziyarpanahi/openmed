"""Inference orchestration for zero-shot NER models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openmed.core.config import OpenMedConfig, get_config
from openmed.core.models import ModelLoader

from .exceptions import MissingDependencyError
from .families import ModelFamily, ensure_gliner_available
from .families.gliner import GLiNERHandle, load_gliner_handle
from .indexing import DEFAULT_INDEX_PATH, ModelIndex, ModelRecord, load_index
from .labels import get_default_labels


@dataclass
class NerRequest:
    model_id: str
    text: str
    threshold: float = 0.5
    labels: Optional[List[str]] = None
    domain: Optional[str] = None


@dataclass
class Entity:
    text: str
    start: int
    end: int
    label: str
    score: float
    group: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "score": self.score,
        }
        if self.group is not None:
            payload["group"] = self.group
        if self.extras:
            payload["extras"] = self.extras
        return payload


@dataclass
class NerResponse:
    entities: List[Entity]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "meta": self.meta,
        }


def infer(
    request: NerRequest,
    *,
    index: Optional[ModelIndex] = None,
    index_path: Optional[Path] = None,
    config: Optional[OpenMedConfig] = None,
    loader: Optional[ModelLoader] = None,
) -> NerResponse:
    """Run inference according to ``request`` and return structured entities."""

    model_index = index or _load_index(index_path)
    record = _lookup_model(request.model_id, model_index)

    resolved_labels, resolved_domain = _resolve_labels(request, record)

    if record.family == ModelFamily.GLINER.value:
        entities = _run_gliner_inference(
            record,
            request,
            resolved_labels,
            config=config,
        )
    else:
        entities = _run_other_inference(
            record,
            request,
            loader=loader,
            config=config,
        )

    filtered = _apply_threshold(entities, request.threshold)

    response = NerResponse(
        entities=filtered,
        meta={
            "model_id": record.id,
            "family": record.family,
            "labels_used": resolved_labels,
            "domain_used": resolved_domain,
            "threshold": request.threshold,
        },
    )
    return response


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_index(index_path: Optional[Path]) -> ModelIndex:
    path = index_path or DEFAULT_INDEX_PATH
    return load_index(path)


def _lookup_model(model_id: str, index: ModelIndex) -> ModelRecord:
    for record in index.models:
        if record.id == model_id:
            return record
    raise ValueError(f"Model '{model_id}' not found in index.")


def _resolve_labels(request: NerRequest, record: ModelRecord) -> Tuple[List[str], Optional[str]]:
    if request.labels:
        labels = [label.strip() for label in request.labels if label.strip()]
        return labels, request.domain

    domain = request.domain
    if not domain and record.domains:
        domain = record.domains[0]

    labels = get_default_labels(domain)
    if not labels:
        labels = get_default_labels("generic")
    return labels, domain


def _run_gliner_inference(
    record: ModelRecord,
    request: NerRequest,
    labels: List[str],
    *,
    config: Optional[OpenMedConfig],
) -> List[Entity]:
    ensure_gliner_available()
    effective_config = config or get_config()
    token = effective_config.hf_token
    cache_dir = effective_config.cache_dir
    device = effective_config.device

    handle = load_gliner_handle(
        record.id,
        cache_dir=cache_dir,
        token=token,
        device=device,
    )

    raw_entities = handle.predict_entities(
        request.text,
        labels=labels,
        threshold=request.threshold,
        flat_ner=True,
    )

    return [_convert_gliner_entity(item) for item in raw_entities]


def _convert_gliner_entity(item: Any) -> Entity:
    if isinstance(item, dict):
        start = _extract_position(item, "start", 0)
        end = _extract_position(item, "end", 1)
        label = item.get("label") or item.get("type") or "UNKNOWN"
        score = float(item.get("score", 0.0))
        text = item.get("text") or item.get("span_text") or ""
        group = item.get("group")
        extras = {k: v for k, v in item.items() if k not in {"text", "start", "end", "label", "score", "group", "span"}}
    else:
        raise TypeError("Unexpected GLiNER entity format.")

    return Entity(
        text=text,
        start=int(start),
        end=int(end),
        label=str(label),
        score=score,
        group=group,
        extras=extras,
    )


def _extract_position(item: Dict[str, Any], key: str, span_index: int) -> int:
    if key in item:
        return int(item[key])
    span = item.get("span")
    if isinstance(span, (list, tuple)) and len(span) > span_index:
        return int(span[span_index])
    raise KeyError(f"Missing '{key}' in GLiNER entity: {item}")


def _run_other_inference(
    record: ModelRecord,
    request: NerRequest,
    *,
    loader: Optional[ModelLoader],
    config: Optional[OpenMedConfig],
) -> List[Entity]:
    effective_loader = loader or ModelLoader(config)
    pipeline = effective_loader.create_pipeline(
        request.model_id,
        task="token-classification",
        aggregation_strategy="simple",
    )

    outputs = pipeline(request.text)
    if isinstance(outputs, dict):
        outputs = [outputs]
    return [_convert_hf_entity(item) for item in outputs]


def _convert_hf_entity(item: Dict[str, Any]) -> Entity:
    label = item.get("entity_group") or item.get("entity") or "UNKNOWN"
    score = float(item.get("score", 0.0))
    text = item.get("word") or item.get("text") or ""
    start = int(item.get("start", 0))
    end = int(item.get("end", start + len(text)))
    group = item.get("group")
    extras = {k: v for k, v in item.items() if k not in {"entity", "entity_group", "score", "word", "text", "start", "end", "group"}}

    return Entity(
        text=text,
        start=start,
        end=end,
        label=str(label),
        score=score,
        group=group,
        extras=extras,
    )


def _apply_threshold(entities: Iterable[Entity], threshold: float) -> List[Entity]:
    return [entity for entity in entities if entity.score >= threshold]


__all__ = [
    "NerRequest",
    "Entity",
    "NerResponse",
    "infer",
]
