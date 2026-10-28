"""Inference orchestration for zero-shot NER models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openmed.core.config import OpenMedConfig, get_config
from openmed.core.labels import normalize_label
from openmed.core.model_integrity import prepare_model_reference
from openmed.core.model_registry import get_model_info
from openmed.core.models import ModelLoader
from openmed.core.offline import is_local_only, network_blocked_if_offline
from openmed.utils.validation import validate_confidence_threshold

from .families import (
    ModelFamily,
    ensure_gliner2_available,
    ensure_gliner_available,
    load_gliner2_handle,
)
from .families.gliner import (
    ensure_gliformer_available,
    load_gliformer_handle,
    load_gliner_handle,
)
from .indexing import (
    GLINER_BIOMED_MODEL_ID,
    ModelIndex,
    ModelRecord,
    load_index,
    resolve_model_record,
)
from .labels import get_default_labels


@dataclass
class NerRequest:
    model_id: str
    text: str
    threshold: float = 0.5
    labels: Optional[List[str]] = None
    domain: Optional[str] = None
    family: Optional[str] = None
    flat_ner: bool = True


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
    _validate_native_threshold(request.threshold)
    record = _lookup_model(request.model_id, model_index, family=request.family)

    resolved_labels, resolved_domain = _resolve_labels(request, record)

    if record.family == ModelFamily.GLINER.value:
        entities = _run_gliner_inference(
            record,
            request,
            resolved_labels,
            config=config,
        )
    elif record.family == ModelFamily.GLINER2.value:
        entities = _run_gliner2_inference(
            record,
            request,
            resolved_labels,
            config=config,
        )
    elif record.family == ModelFamily.GLIFORMER.value:
        entities = _run_native_inference(
            record, request, resolved_labels, config=config
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


def infer_biomedical(
    text: str,
    *,
    threshold: float = 0.5,
    labels: Optional[List[str]] = None,
    model_id: str = GLINER_BIOMED_MODEL_ID,
    index: Optional[ModelIndex] = None,
    index_path: Optional[Path] = None,
    config: Optional[OpenMedConfig] = None,
    loader: Optional[ModelLoader] = None,
) -> NerResponse:
    """Run GLiNER-BioMed with the packaged biomedical label preset."""

    return infer(
        NerRequest(
            model_id=model_id,
            text=text,
            threshold=threshold,
            labels=labels,
            domain="biomedical",
        ),
        index=index,
        index_path=index_path,
        config=config,
        loader=loader,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_index(index_path: Optional[Path]) -> ModelIndex:
    return load_index(index_path)


def _lookup_model(
    model_id: str, index: ModelIndex, *, family: Optional[str] = None
) -> ModelRecord:
    record = resolve_model_record(model_id, family=family, index=index)
    if record is not None:
        return record
    raise ValueError(f"Model '{model_id}' not found in index.")


def _resolve_labels(
    request: NerRequest, record: ModelRecord
) -> Tuple[List[str], Optional[str]]:
    if request.labels is not None:
        if (
            not isinstance(request.labels, (list, tuple))
            or not all(
                isinstance(label, str) and label.strip() for label in request.labels
            )
            or not request.labels
        ):
            raise ValueError("labels must be a non-empty list of non-empty strings")
        labels = list(dict.fromkeys(label.strip() for label in request.labels))
        return labels, request.domain

    if request.domain is None and record.default_labels:
        return list(record.default_labels), None

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
        record.path or record.id,
        cache_dir=cache_dir,
        token=token,
        device=device,
        **({"local_files_only": True} if is_local_only(effective_config) else {}),
    )

    raw_entities = handle.predict_entities(
        request.text,
        labels=labels,
        threshold=request.threshold,
        flat_ner=request.flat_ner,
    )

    return [_convert_gliner_entity(item) for item in raw_entities]


def _run_gliner2_inference(
    record: ModelRecord,
    request: NerRequest,
    labels: List[str],
    *,
    config: Optional[OpenMedConfig],
) -> List[Entity]:
    return _run_native_inference(record, request, labels, config=config)


def _load_native_handle(
    record: ModelRecord, config: OpenMedConfig, **options: Any
) -> Any:
    if record.family == ModelFamily.GLIFORMER.value:
        ensure_gliformer_available()
        load_handle = load_gliformer_handle
    elif record.family == ModelFamily.GLINER2.value:
        ensure_gliner2_available()
        load_handle = load_gliner2_handle
    else:
        ensure_gliner_available()
        load_handle = load_gliner_handle
    reference = record.path or record.id
    local_only = (
        is_local_only(config)
        or Path(reference).is_dir()
        or bool(options.pop("local_files_only", False))
    )
    load_options = {
        "cache_dir": config.cache_dir,
        "token": config.hf_token,
        "device": config.device,
        **options,
    }
    with network_blocked_if_offline(config):
        reference = prepare_model_reference(
            reference,
            registry_info=get_model_info(record.id),
            cache_dir=str(load_options["cache_dir"]),
            local_only=local_only,
            token=load_options["token"],
        )
        return load_handle(reference, local_files_only=local_only, **load_options)


def _native_predictions(
    handle: Any,
    text: str,
    labels: List[str],
    threshold: float,
    flat_ner: bool,
    config: OpenMedConfig,
) -> List[Entity]:
    with network_blocked_if_offline(config):
        raw = handle.predict_entities(
            text,
            labels=labels,
            threshold=threshold,
            flat_ner=flat_ner,
        )
    if not isinstance(raw, list) or any(
        not isinstance(item, dict)
        or not {"text", "start", "end", "label", "score"}.issubset(item)
        for item in raw
    ):
        raise ValueError("Native NER must return labeled spans with confidence")
    try:
        entities = [_convert_gliner_entity(item) for item in raw]
    except (TypeError, ValueError, KeyError):
        raise ValueError("Native NER returned malformed entity fields") from None
    for entity in entities:
        if not 0 <= entity.start < entity.end <= len(text):
            raise ValueError("Native NER returned an invalid character span")
        if text[entity.start : entity.end] != entity.text:
            raise ValueError("Native NER span text does not match its offsets")
        if not math.isfinite(entity.score) or not 0.0 <= entity.score <= 1.0:
            raise ValueError("Native NER confidence must be a finite probability")
    return _apply_threshold(entities, threshold)


def _validate_native_threshold(threshold: float) -> float:
    value = validate_confidence_threshold(threshold)
    if not math.isfinite(value):
        raise ValueError("Native NER threshold must be a finite probability")
    return value


def _run_native_inference(
    record: ModelRecord,
    request: NerRequest,
    labels: List[str],
    *,
    config: Optional[OpenMedConfig],
) -> List[Entity]:
    effective_config = config or get_config()
    handle = _load_native_handle(record, effective_config)
    return _native_predictions(
        handle,
        request.text,
        labels,
        request.threshold,
        request.flat_ner,
        effective_config,
    )


class SpanPipeline:
    """Expose native span models through OpenMed's pipeline output contract.

    Text and entity predictions stay in memory. The pipeline accepts a string
    or a list of strings and returns Transformers-shaped span dictionaries so
    existing formatters, sentence offset adjustment, and context stages apply.
    """

    tokenizer = None

    def __init__(
        self, record: ModelRecord, config: OpenMedConfig, **options: Any
    ) -> None:
        self.record = record
        self.config = config
        self.threshold = _validate_native_threshold(options.pop("threshold", 0.5))
        self.flat_ner = options.pop("flat_ner", True)
        request = NerRequest(
            model_id=record.id,
            text="",
            labels=options.pop("labels", None),
            domain=options.pop("domain", None),
        )
        self.labels, self.domain = _resolve_labels(request, record)
        unknown = set(options) - {
            "cache_dir",
            "token",
            "device",
            "local_files_only",
            "revision",
        }
        if unknown:
            raise TypeError(
                "Unsupported native NER loading option: " + ", ".join(sorted(unknown))
            )
        self.handle = _load_native_handle(record, config, **options)
        native_config = getattr(self.handle.model, "config", None)
        max_length = getattr(native_config, "max_len", None)
        self.openmed_max_length = (
            max_length
            if type(max_length) is int and 0 < max_length < 1_000_000
            else 512
        )

    def __call__(self, texts: Any, *, batch_size: int = 1, num_workers: int = 0) -> Any:
        """Return one prediction list per input, with stable source offsets."""

        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if num_workers != 0:
            raise ValueError("Native NER pipelines currently require num_workers=0")
        single = isinstance(texts, str)
        batch = [texts] if single else texts
        if not isinstance(batch, list) or not all(
            isinstance(text, str) for text in batch
        ):
            raise TypeError("Native NER input must be a string or list of strings")
        outputs = []
        for text in batch:
            entities = _native_predictions(
                self.handle,
                text,
                self.labels,
                self.threshold,
                self.flat_ner,
                self.config,
            )
            outputs.append(
                [
                    {
                        "entity_group": entity.label,
                        "word": entity.text,
                        "start": entity.start,
                        "end": entity.end,
                        "score": entity.score,
                    }
                    for entity in entities
                ]
            )
        return outputs[0] if single else outputs


def _convert_gliner_entity(item: Any) -> Entity:
    if isinstance(item, dict):
        start = _extract_position(item, "start", 0)
        end = _extract_position(item, "end", 1)
        source_label = item.get("label") or item.get("type") or "UNKNOWN"
        label = _normalise_gliner_label(source_label)
        score = float(item.get("score", 0.0))
        text = item.get("text") or item.get("span_text") or ""
        group = item.get("group")
        extras = {
            k: v
            for k, v in item.items()
            if k not in {"text", "start", "end", "label", "score", "group", "span"}
        }
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


def _normalise_gliner_label(label: Any) -> str:
    """Return a canonical label while retaining unknown zero-shot labels."""

    if not isinstance(label, str):
        raise ValueError("GLiNER entity labels must be strings")
    source_label = label
    canonical_label = normalize_label(source_label)
    return source_label if canonical_label == "OTHER" else canonical_label


def _extract_position(item: Dict[str, Any], key: str, span_index: int) -> int:
    if key in item:
        value = item[key]
    else:
        span = item.get("span")
        if not isinstance(span, (list, tuple)) or len(span) <= span_index:
            raise KeyError(f"Missing '{key}' in GLiNER entity")
        value = span[span_index]
    if type(value) is not int or value < 0:
        raise ValueError("GLiNER entity offsets must be non-negative integers")
    return value


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
    extras = {
        k: v
        for k, v in item.items()
        if k
        not in {
            "entity",
            "entity_group",
            "score",
            "word",
            "text",
            "start",
            "end",
            "group",
        }
    }

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
    "infer_biomedical",
]
