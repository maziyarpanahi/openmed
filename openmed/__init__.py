"""Top-level interface for the OpenMed library."""

from __future__ import annotations

import logging
import re
import time
from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .__about__ import __version__

if TYPE_CHECKING:
    from .core import ModelLoader, OpenMedConfig
    from .core.pii import (
        DeidentificationResult,
        PIIEntity,
        deidentify,
        extract_pii,
        reidentify,
    )
    from .core.results import AnalyzeResult
    from .processing import BatchProcessor
    from .processing.sentences import SentenceSpan

_LAZY_IMPORTS = {
    "ModelLoader": ".core",
    "OpenMedConfig": ".core",
    "load_model": ".core",
    "LANG_TO_LOCALE": ".core.anonymizer",
    "Anonymizer": ".core.anonymizer",
    "AnonymizerConfig": ".core.anonymizer",
    "IndiaSurrogateProvider": ".core.anonymizer",
    "register_clinical_provider": ".core.anonymizer",
    "register_label_generator": ".core.anonymizer",
    "AttestationReport": ".core.attestation",
    "AttestationTemplateError": ".core.attestation",
    "generate_attestation": ".core.attestation",
    "list_attestation_profiles": ".core.attestation",
    "load_attestation_template": ".core.attestation",
    "AuditReport": ".core.audit",
    "AuditSignature": ".core.audit",
    "AuditSpan": ".core.audit",
    "DetectorInfo": ".core.audit",
    "CustomRecognizer": ".core.custom_recognizer",
    "ExplainReport": ".core.explain",
    "explain": ".core.explain",
    "CachedModel": ".core.hf_hub",
    "clear_cached_model": ".core.hf_hub",
    "list_cached_models": ".core.hf_hub",
    "prefetch_model": ".core.hf_hub",
    "resolve_repo_id": ".core.hf_hub",
    "IndicNameNormalizer": ".core.indic_name_match",
    "canonical_indic_name_key": ".core.indic_name_match",
    "detect_name_script": ".core.indic_name_match",
    "indic_names_match": ".core.indic_name_match",
    "CANONICAL_LABELS": ".core.labels",
    "normalize_label": ".core.labels",
    "get_all_models": ".core.model_registry",
    "get_default_pii_model": ".core.model_registry",
    "get_model_info": ".core.model_registry",
    "get_model_suggestions": ".core.model_registry",
    "get_models_by_category": ".core.model_registry",
    "get_pii_models_by_language": ".core.model_registry",
    "list_model_categories": ".core.model_registry",
    "ModelQuery": ".core.model_search",
    "ModelSearchResult": ".core.model_search",
    "search_models": ".core.model_search",
    "network_blocked_if_offline": ".core.offline",
    "DeidentificationResult": ".core.pii",
    "PIIEntity": ".core.pii",
    "deidentify": ".core.pii",
    "extract_pii": ".core.pii",
    "reidentify": ".core.pii",
    "PII_PATTERNS": ".core.pii_entity_merger",
    "PIIPattern": ".core.pii_entity_merger",
    "calculate_dominant_label": ".core.pii_entity_merger",
    "find_semantic_units": ".core.pii_entity_merger",
    "merge_entities_with_semantic_units": ".core.pii_entity_merger",
    "merge_india_code_mixed_spans": ".core.pii_entity_merger",
    "DEFAULT_PII_MODELS": ".core.pii_i18n",
    "LANGUAGE_PII_PATTERNS": ".core.pii_i18n",
    "SUPPORTED_LANGUAGES": ".core.pii_i18n",
    "get_india_clinical_model_route": ".core.pii_i18n",
    "get_patterns_for_language": ".core.pii_i18n",
    "india_clinical_route_active": ".core.pii_i18n",
    "redaction_preview": ".core.redaction_preview",
    "render_redaction_preview": ".core.redaction_preview",
    "get_result_cache": ".core.result_cache",
    "make_cache_key": ".core.result_cache",
    "AnalyzeResult": ".core.results",
    "StreamingBufferError": ".core.streaming",
    "StreamingDeidentificationEvent": ".core.streaming",
    "StreamingDeidentifier": ".core.streaming",
    "deidentify_stream": ".core.streaming",
    "ENCRYPTION_SCHEME": ".core.surrogate_vault",
    "SUBJECT_SURROGATE_LABEL": ".core.surrogate_vault",
    "SUBJECT_SURROGATE_LANG": ".core.surrogate_vault",
    "InMemorySurrogateStore": ".core.surrogate_vault",
    "JsonFileSurrogateStore": ".core.surrogate_vault",
    "SurrogateEntry": ".core.surrogate_vault",
    "SurrogateKey": ".core.surrogate_vault",
    "SurrogateSource": ".core.surrogate_vault",
    "SurrogateVault": ".core.surrogate_vault",
    "SubjectResolutionError": ".core.surrogate_vault",
    "VaultConsistencyReport": ".core.surrogate_vault",
    "VaultRotationResult": ".core.surrogate_vault",
    "OpenMedMLXLanguageModel": ".mlx.lm",
    "OpenMedPagedKVCache": ".mlx.lm",
    "PagedKVCacheConfig": ".mlx.lm",
    "PagedKVCachePlan": ".mlx.lm",
    "PagedKVCacheStats": ".mlx.lm",
    "TokenRange": ".mlx.lm",
    "generate_text": ".mlx.lm",
    "OnnxEntity": ".onnx.inference",
    "OnnxModel": ".onnx.inference",
    "load_onnx_model": ".onnx.inference",
    "BatchItem": ".processing",
    "BatchItemResult": ".processing",
    "BatchProcessor": ".processing",
    "BatchProgress": ".processing",
    "BatchResult": ".processing",
    "DatasetRedactionResult": ".processing",
    "DatasetRedactionSummary": ".processing",
    "IndicNormalization": ".processing",
    "IndicNormalizer": ".processing",
    "OutputFormatter": ".processing",
    "TextProcessor": ".processing",
    "TokenizationHelper": ".processing",
    "format_predictions": ".processing",
    "postprocess_text": ".processing",
    "preprocess_text": ".processing",
    "process_batch": ".processing",
    "redact_dataset": ".processing",
    "sentence_utils": ".processing",
    "AdvancedNERProcessor": ".processing.advanced_ner",
    "StreamingReplayResult": ".processing.advanced_ner",
    "StreamingTokenClassifier": ".processing.advanced_ner",
    "create_advanced_processor": ".processing.advanced_ner",
    "replay_token_classifier": ".processing.advanced_ner",
    "stream_token_classifier": ".processing.advanced_ner",
    "PredictionResult": ".processing.outputs",
    "PeakRSSMeasurement": ".utils",
    "Profiler": ".utils",
    "ProfileReport": ".utils",
    "Timer": ".utils",
    "disable_profiling": ".utils",
    "enable_profiling": ".utils",
    "get_logger": ".utils",
    "get_peak_rss_bytes": ".utils",
    "get_profile_report": ".utils",
    "measure_peak_rss": ".utils",
    "profile": ".utils",
    "setup_logging": ".utils",
    "timed": ".utils",
    "validate_input": ".utils",
    "validate_model_name": ".utils",
    "sanitize_filename": ".utils.validation",
    "validate_batch_size": ".utils.validation",
    "validate_confidence_threshold": ".utils.validation",
    "validate_output_format": ".utils.validation",
}
_LAZY_ATTRIBUTE_NAMES = {"sentence_utils": "sentences"}
_LAZY_IMPORT_PREREQUISITES = {
    ".core.pii_i18n": (".core.anonymizer",),
}


def __getattr__(name: str) -> Any:
    """Resolve and cache a top-level public export on first access."""

    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for prerequisite in _LAZY_IMPORT_PREREQUISITES.get(module_name, ()):
        import_module(prerequisite, __name__)
    attribute_name = _LAZY_ATTRIBUTE_NAMES.get(name, name)
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return eager and lazy top-level attributes for interactive discovery."""

    return sorted(set(globals()) | set(_LAZY_IMPORTS))


def _resolve_export(name: str) -> Any:
    """Return an overridden, cached, or newly loaded top-level export."""

    try:
        return globals()[name]
    except KeyError:
        return __getattr__(name)


_PLACEHOLDER_SEGMENT_PATTERN = re.compile(r"(?:_{3,}|placeholder|^\W+$)", re.IGNORECASE)
_HARD_LINE_BREAK_PATTERN = re.compile(r"\r\n|[\n\r\v\f\x85\u2028\u2029]")

logger = logging.getLogger(__name__)


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim whitespace from a source span while preserving exact offsets."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_prediction_at_boundaries(
    prediction: Dict[str, Any],
    text: str,
    segments: List[Dict[str, Any]],
    *,
    sentence_detection: bool,
) -> List[Dict[str, Any]]:
    """Split a model span at hard line breaks and detected sentence boundaries."""
    start = prediction.get("start")
    end = prediction.get("end")
    if not (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(text)
    ):
        return []

    line_ranges: List[tuple[int, int]] = []
    cursor = start
    for match in _HARD_LINE_BREAK_PATTERN.finditer(text, start, end):
        fragment_start, fragment_end = _trim_span(text, cursor, match.start())
        if fragment_end > fragment_start:
            line_ranges.append((fragment_start, fragment_end))
        cursor = match.end()
    fragment_start, fragment_end = _trim_span(text, cursor, end)
    if fragment_end > fragment_start:
        line_ranges.append((fragment_start, fragment_end))

    fragments: List[Dict[str, Any]] = []
    for line_start, line_end in line_ranges:
        overlapping_segments = (
            [
                segment
                for segment in segments
                if segment["end"] > line_start and segment["start"] < line_end
            ]
            if sentence_detection
            else []
        )
        ranges = (
            [
                (
                    max(line_start, segment["start"]),
                    min(line_end, segment["end"]),
                    segment,
                )
                for segment in overlapping_segments
            ]
            if overlapping_segments
            else [(line_start, line_end, None)]
        )

        for range_start, range_end, segment in ranges:
            range_start, range_end = _trim_span(text, range_start, range_end)
            if range_end <= range_start:
                continue
            if segment is not None and segment.get("suppress_predictions"):
                continue

            fragment = dict(prediction)
            fragment["start"] = range_start
            fragment["end"] = range_end
            fragment["word"] = text[range_start:range_end]

            if sentence_detection:
                span_metadata = dict(fragment.get("metadata") or {})
                if segment is None:
                    span_metadata.update(
                        {
                            "sentence_index": -1,
                            "sentence_text": "",
                            "sentence_start": range_start,
                            "sentence_end": range_end,
                        }
                    )
                else:
                    span_metadata.update(
                        {
                            "sentence_index": segment["index"],
                            "sentence_text": segment["text"],
                            "sentence_start": segment["start"],
                            "sentence_end": segment["end"],
                        }
                    )
                fragment["metadata"] = span_metadata

            fragments.append(fragment)

    return fragments


def list_models(
    *,
    include_registry: bool = True,
    include_remote: bool = True,
    config: Optional[OpenMedConfig] = None,
) -> List[str]:
    """Return available OpenMed model identifiers.

    Args:
        include_registry: Include entries from the bundled registry in addition to
            entries in the committed manifest.
        include_remote: Retained for compatibility; no live discovery is performed.
        config: Optional custom configuration for model discovery.
    """

    model_loader = _resolve_export("ModelLoader")
    loader = model_loader(config)
    return loader.list_available_models(
        include_registry=include_registry,
        include_remote=include_remote,
    )


def get_model_max_length(
    model_name: str,
    *,
    config: Optional[OpenMedConfig] = None,
    loader: Optional[ModelLoader] = None,
) -> Optional[int]:
    """Return the inferred maximum sequence length for ``model_name``."""

    model_loader = _resolve_export("ModelLoader")
    loader = loader or model_loader(config)
    return loader.get_max_sequence_length(model_name)


def analyze_text(
    text: str,
    model_name: str = "disease_detection_superclinical",
    *,
    model_id: Optional[str] = None,
    config: Optional[OpenMedConfig] = None,
    loader: Optional[ModelLoader] = None,
    aggregation_strategy: Optional[str] = "simple",
    output_format: str = "dict",
    include_confidence: bool = True,
    confidence_threshold: Optional[float] = 0.0,
    group_entities: bool = False,
    formatter_kwargs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    use_fast_tokenizer: bool = True,
    sentence_detection: bool = True,
    sentence_language: str = "en",
    sentence_clean: bool = False,
    sentence_segmenter: Optional[Any] = None,
    cache_results: bool = False,
    max_cache_entries: int = 128,
    **pipeline_kwargs: Any,
) -> Union[AnalyzeResult, str, List[Dict[str, Any]]]:
    """Run a token-classification model on ``text`` and format the predictions.

    Args:
        text: Clinical or biomedical text to analyse.
        model_name: Registry key, fully-qualified Hugging Face model id, or
            local model path.
        model_id: Alias for ``model_name``. Useful for APIs and examples that
            name model identifiers as ``model_id``.
        config: Optional :class:`~openmed.core.config.OpenMedConfig` instance.
        loader: Reuse an existing :class:`~openmed.core.models.ModelLoader`.
        aggregation_strategy: Hugging Face aggregation strategy (``"simple"`` by
            default). Set to ``None`` to work with raw token outputs.
        output_format: ``"dict"`` (default), ``"json"``, ``"html"`` or ``"csv"``.
        include_confidence: Whether to include confidence scores in formatted output.
        confidence_threshold: Minimum confidence for entities. ``None`` keeps all.
        group_entities: Merge adjacent entities of the same label in the formatted
            output.
        formatter_kwargs: Extra keyword arguments forwarded to
            :func:`openmed.processing.format_predictions`.
        metadata: Optional metadata to attach to the result.
        use_fast_tokenizer: Prefer fast tokenizers when available.
        sentence_detection: Enable pySBD-powered sentence detection (default: True).
        sentence_language: Language hint for the sentence detector.
        sentence_clean: Whether to enable pySBD's cleaning heuristics.
        sentence_segmenter: Optional preconstructed pySBD segmenter to reuse.
        cache_results: Whether to cache this result in the in-process LRU cache. Cached results may contain PHI, but are never saved to disk.
        max_cache_entries: Maximum number of cached results.
        **pipeline_kwargs: Additional arguments passed to
            :meth:`openmed.core.models.ModelLoader.create_pipeline`.

    Returns:
        Analyze result for ``"dict"`` output, otherwise the requested rendered
        format.

    Example:
        >>> class FixtureLoader:
        ...     config = None
        ...
        ...     def create_pipeline(self, model_name, **kwargs):
        ...         def pipeline(text, **call_kwargs):
        ...             return [
        ...                 {
        ...                     "entity_group": "CONDITION",
        ...                     "score": 0.99,
        ...                     "start": 11,
        ...                     "end": 17,
        ...                     "word": "asthma",
        ...                 }
        ...             ]
        ...
        ...         return pipeline
        ...
        ...     def get_max_sequence_length(self, model_name, tokenizer=None):
        ...         return 128
        >>> result = analyze_text(
        ...     "History of asthma.",
        ...     model_name="fixture-ner-model",
        ...     loader=FixtureLoader(),
        ...     sentence_detection=False,
        ... )
        >>> next((entity.text, entity.label) for entity in result.entities)
        ('asthma', 'CONDITION')
    """

    model_loader = _resolve_export("ModelLoader")
    network_blocked_if_offline = _resolve_export("network_blocked_if_offline")
    get_result_cache = _resolve_export("get_result_cache")
    make_cache_key = _resolve_export("make_cache_key")
    analyze_result = _resolve_export("AnalyzeResult")
    format_predictions = _resolve_export("format_predictions")
    sentence_utils = _resolve_export("sentence_utils")
    prediction_result = _resolve_export("PredictionResult")
    validate_input = _resolve_export("validate_input")
    validate_model_name = _resolve_export("validate_model_name")
    validate_confidence_threshold = _resolve_export("validate_confidence_threshold")
    validate_output_format = _resolve_export("validate_output_format")

    validated_text = validate_input(text)
    selected_model = model_id if model_id is not None else model_name
    if model_id is not None and model_name != "disease_detection_superclinical":
        raise ValueError("Pass only one of model_name or model_id")

    validated_model = validate_model_name(selected_model)

    if cache_results:
        params = dict(locals())
        cache_key = make_cache_key("analyze_text", params)
        cache = get_result_cache(max_entries=max_cache_entries)
        final_result = cache.get(cache_key)
        if final_result is not None:
            return final_result

    loader = loader or model_loader(config)
    runtime_config = getattr(loader, "config", config)

    pipeline_args = dict(
        task="token-classification",
        aggregation_strategy=aggregation_strategy,
        use_fast_tokenizer=use_fast_tokenizer,
    )

    provided_max_length = pipeline_kwargs.pop("max_length", None)
    truncate_inputs = pipeline_kwargs.pop("truncation", True)

    call_kwargs: Dict[str, Any] = {}
    for key in ("batch_size", "num_workers"):
        if key in pipeline_kwargs:
            call_kwargs[key] = pipeline_kwargs.pop(key)

    pipeline_args.update(pipeline_kwargs)

    ner_pipeline = loader.create_pipeline(validated_model, **pipeline_args)

    effective_max_length: Optional[int] = None
    if truncate_inputs and provided_max_length is not None:
        effective_max_length = provided_max_length
    elif truncate_inputs:
        with network_blocked_if_offline(runtime_config):
            effective_max_length = loader.get_max_sequence_length(
                validated_model,
                tokenizer=getattr(ner_pipeline, "tokenizer", None),
            )

    desired_max_length = (
        provided_max_length if provided_max_length is not None else effective_max_length
    )

    tokenizer = getattr(ner_pipeline, "tokenizer", None)
    if tokenizer is not None:
        if truncate_inputs:
            if desired_max_length is not None:
                try:
                    tokenizer.model_max_length = int(desired_max_length)
                except Exception:
                    pass
        else:
            try:
                tokenizer.model_max_length = 0
            except Exception:
                pass

    raw_segments: List[SentenceSpan] = []
    if sentence_detection:
        try:
            raw_segments = sentence_utils.segment_text(
                validated_text,
                language=sentence_language,
                clean=sentence_clean,
                segmenter=sentence_segmenter,
            )
        except ImportError:
            sentence_detection = False
    if not raw_segments:
        sentence_detection = False

    processed_segments: List[Dict[str, Any]] = []
    if sentence_detection:
        for span in raw_segments:
            span_text = span.text or ""
            base_start = span.start
            base_end = span.end

            leading_ws = len(span_text) - len(span_text.lstrip())
            trailing_ws = len(span_text) - len(span_text.rstrip())

            if leading_ws:
                base_start += leading_ws
            if trailing_ws:
                base_end -= trailing_ws

            trimmed_text = span_text[leading_ws : len(span_text) - trailing_ws]

            if not trimmed_text:
                continue

            suppress_predictions = bool(
                _PLACEHOLDER_SEGMENT_PATTERN.search(trimmed_text)
            )

            processed_segments.append(
                {
                    "index": len(processed_segments),
                    "text": trimmed_text,
                    "start": base_start,
                    "end": base_end,
                    "suppress_predictions": suppress_predictions,
                }
            )

    if not processed_segments:
        processed_segments.append(
            {
                "index": 0,
                "text": validated_text,
                "start": 0,
                "end": len(validated_text),
                "suppress_predictions": False,
            }
        )
        sentence_detection = False

    chunk_descriptors: List[Dict[str, Any]] = []
    if sentence_detection:
        max_chunk_chars = max(480, (desired_max_length or 256) * 4)
        max_chunk_sentences = 6

        current_indices: List[int] = []
        current_start: Optional[int] = None
        current_end: Optional[int] = None

        for seg in processed_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]

            if not current_indices:
                current_indices = [seg["index"]]
                current_start = seg_start
                current_end = seg_end
                continue

            proposed_start = current_start if current_start is not None else seg_start
            proposed_end = seg_end
            span_length = proposed_end - proposed_start

            if (
                len(current_indices) >= max_chunk_sentences
                or span_length > max_chunk_chars
            ):
                if current_start is None or current_end is None:
                    raise RuntimeError("chunk boundary unexpectedly None")
                chunk_descriptors.append(
                    {
                        "text": validated_text[current_start:current_end],
                        "start": current_start,
                        "end": current_end,
                        "segment_indices": current_indices[:],
                    }
                )
                current_indices = [seg["index"]]
                current_start = seg_start
                current_end = seg_end
            else:
                current_indices.append(seg["index"])
                current_end = seg_end

        if current_indices:
            if current_start is None or current_end is None:
                raise RuntimeError("chunk boundary unexpectedly None")
            chunk_descriptors.append(
                {
                    "text": validated_text[current_start:current_end],
                    "start": current_start,
                    "end": current_end,
                    "segment_indices": current_indices[:],
                }
            )
    else:
        chunk_descriptors.append(
            {
                "text": validated_text,
                "start": 0,
                "end": len(validated_text),
                "segment_indices": [seg["index"] for seg in processed_segments],
            }
        )

    if sentence_detection:
        inference_input = [chunk["text"] for chunk in chunk_descriptors]
    else:
        inference_input = validated_text

    with network_blocked_if_offline(runtime_config):
        start_time = time.time()
        raw_predictions = ner_pipeline(inference_input, **call_kwargs)
        processing_time = time.time() - start_time

    def _normalize_predictions(
        predictions: Any,
        segment_count: int,
    ) -> List[List[Dict[str, Any]]]:
        if not isinstance(predictions, list):
            return [[predictions]]

        if segment_count == 1 and predictions and isinstance(predictions[0], dict):
            return [predictions]

        normalized: List[List[Dict[str, Any]]] = []
        for item in predictions:
            if isinstance(item, list):
                normalized.append(item)
            elif item is None:
                normalized.append([])
            elif isinstance(item, dict):
                normalized.append([item])
            else:
                normalized.append(list(item) if item else [])
        return normalized

    normalized_predictions = _normalize_predictions(
        raw_predictions, len(chunk_descriptors)
    )

    flattened_predictions: List[Dict[str, Any]] = []
    for chunk_idx, chunk in enumerate(chunk_descriptors):
        chunk_segments = [
            processed_segments[idx] for idx in chunk.get("segment_indices", [])
        ]
        if chunk_idx < len(normalized_predictions):
            segment_predictions = normalized_predictions[chunk_idx]
        else:
            segment_predictions = []

        for prediction in segment_predictions:
            if not isinstance(prediction, dict):
                continue

            adjusted = dict(prediction)
            start = adjusted.get("start")
            end = adjusted.get("end")

            if isinstance(start, int):
                adjusted["start"] = start + chunk["start"]
            if isinstance(end, int):
                adjusted["end"] = end + chunk["start"]

            for fragment in _split_prediction_at_boundaries(
                adjusted,
                validated_text,
                chunk_segments,
                sentence_detection=sentence_detection,
            ):
                span_slice = validated_text[fragment["start"] : fragment["end"]]
                if _PLACEHOLDER_SEGMENT_PATTERN.search(span_slice):
                    continue
                flattened_predictions.append(fragment)

    base_metadata = dict(metadata) if metadata else {}
    base_metadata.setdefault("sentence_detection", sentence_detection)
    if sentence_detection:
        base_metadata.setdefault("sentence_count", len(processed_segments))
        base_metadata.setdefault("sentence_language", sentence_language)

    # Optional: remap model spans onto medical-friendly tokens (no change to model tokenization).
    active_config = loader.config if hasattr(loader, "config") else config
    if active_config is not None and getattr(
        active_config, "use_medical_tokenizer", False
    ):
        try:
            from .processing.tokenization import (
                DEFAULT_MEDICAL_EXCEPTIONS,
                medical_tokenize,
                remap_predictions_to_tokens,
            )

            extra_exceptions = (
                getattr(active_config, "medical_tokenizer_exceptions", None) or []
            )
            token_exceptions = list(DEFAULT_MEDICAL_EXCEPTIONS) + list(extra_exceptions)
            medical_tokens = medical_tokenize(
                validated_text, exceptions=token_exceptions
            )
            flattened_predictions = remap_predictions_to_tokens(
                flattened_predictions,
                validated_text,
                medical_tokens,
            )
            base_metadata.setdefault("medical_tokenizer", True)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to remap predictions to medical tokens: %s", exc)
            base_metadata.setdefault("medical_tokenizer", False)

    fmt_kwargs: Dict[str, Any] = {
        "include_confidence": include_confidence,
        "group_entities": group_entities,
        "metadata": base_metadata,
        "processing_time": processing_time,
    }

    if effective_max_length is not None:
        fmt_kwargs["metadata"]["max_length"] = effective_max_length

    if confidence_threshold is not None:
        fmt_kwargs["confidence_threshold"] = validate_confidence_threshold(
            confidence_threshold
        )

    if formatter_kwargs:
        fmt_kwargs.update(formatter_kwargs)

    fmt_output = validate_output_format(output_format)

    result = format_predictions(
        flattened_predictions,
        validated_text,
        model_name=validated_model,
        output_format=fmt_output,
        **fmt_kwargs,
    )
    final_result: Union[AnalyzeResult, str, List[Dict[str, Any]]]
    if fmt_output == "dict" and isinstance(result, prediction_result):
        final_result = analyze_result.from_prediction_result(result)
    else:
        final_result = result
    if cache_results:
        cache.set(cache_key, final_result)
    return final_result


__all__ = [
    "__version__",
    "ModelLoader",
    "load_model",
    "OpenMedConfig",
    "OnnxEntity",
    "OnnxModel",
    "load_onnx_model",
    "TextProcessor",
    "IndicNormalization",
    "IndicNormalizer",
    "preprocess_text",
    "postprocess_text",
    "TokenizationHelper",
    "OutputFormatter",
    "format_predictions",
    "BatchProcessor",
    "BatchItem",
    "BatchItemResult",
    "BatchProgress",
    "BatchResult",
    "DatasetRedactionResult",
    "DatasetRedactionSummary",
    "process_batch",
    "redact_dataset",
    "AdvancedNERProcessor",
    "StreamingReplayResult",
    "StreamingTokenClassifier",
    "create_advanced_processor",
    "AnalyzeResult",
    "PredictionResult",
    "setup_logging",
    "get_logger",
    "validate_input",
    "validate_model_name",
    "validate_confidence_threshold",
    "validate_output_format",
    "validate_batch_size",
    "sanitize_filename",
    "get_model_info",
    "get_models_by_category",
    "get_all_models",
    "list_model_categories",
    "get_model_suggestions",
    "get_pii_models_by_language",
    "get_default_pii_model",
    # Hugging Face Hub model-pull helpers
    "prefetch_model",
    "list_cached_models",
    "clear_cached_model",
    "resolve_repo_id",
    "CachedModel",
    "list_models",
    "get_model_max_length",
    "analyze_text",
    "explain",
    "ExplainReport",
    "generate_text",
    "OpenMedMLXLanguageModel",
    "OpenMedPagedKVCache",
    "PagedKVCacheConfig",
    "PagedKVCachePlan",
    "PagedKVCacheStats",
    "TokenRange",
    # Profiling utilities
    "Profiler",
    "ProfileReport",
    "PeakRSSMeasurement",
    "Timer",
    "enable_profiling",
    "disable_profiling",
    "get_profile_report",
    "get_peak_rss_bytes",
    "measure_peak_rss",
    "profile",
    "timed",
    # PII detection and de-identification
    "extract_pii",
    "deidentify",
    "reidentify",
    "PIIEntity",
    "DeidentificationResult",
    "AttestationReport",
    "AttestationTemplateError",
    "generate_attestation",
    "list_attestation_profiles",
    "load_attestation_template",
    "CustomRecognizer",
    "StreamingBufferError",
    "StreamingDeidentificationEvent",
    "StreamingDeidentifier",
    "deidentify_stream",
    "replay_token_classifier",
    "stream_token_classifier",
    "redaction_preview",
    "render_redaction_preview",
    # PII entity merging utilities
    "merge_entities_with_semantic_units",
    "merge_india_code_mixed_spans",
    "find_semantic_units",
    "calculate_dominant_label",
    "PII_PATTERNS",
    "PIIPattern",
    # Multilingual PII support
    "SUPPORTED_LANGUAGES",
    "DEFAULT_PII_MODELS",
    "LANGUAGE_PII_PATTERNS",
    "get_india_clinical_model_route",
    "get_patterns_for_language",
    "india_clinical_route_active",
    # Canonical label taxonomy
    "CANONICAL_LABELS",
    "normalize_label",
    # Anonymization engine
    "Anonymizer",
    "AnonymizerConfig",
    "IndiaSurrogateProvider",
    "LANG_TO_LOCALE",
    "register_clinical_provider",
    "register_label_generator",
    "SurrogateVault",
    "SurrogateKey",
    "SurrogateEntry",
    "SurrogateSource",
    "VaultConsistencyReport",
    "VaultRotationResult",
    "InMemorySurrogateStore",
    "JsonFileSurrogateStore",
    "ENCRYPTION_SCHEME",
    "IndicNameNormalizer",
    "canonical_indic_name_key",
    "indic_names_match",
    "detect_name_script",
]
