"""Top-level interface for the OpenMed library."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Union

from .__about__ import __version__
from .core import ModelLoader, OpenMedConfig, load_model
from .core.anonymizer import (
    LANG_TO_LOCALE,
    Anonymizer,
    AnonymizerConfig,
    register_clinical_provider,
    register_label_generator,
)
from .core.audit import AuditReport, AuditSignature, AuditSpan, DetectorInfo
from .core.custom_recognizer import CustomRecognizer
from .core.document_stream import (
    DocumentStreamDeidentifier,
    DocumentStreamResult,
    deidentify_document_stream,
)
from .core.explain import ExplainReport, explain
from .core.labels import CANONICAL_LABELS, normalize_label
from .core.model_registry import (
    get_all_models,
    get_default_pii_model,
    get_model_info,
    get_model_suggestions,
    get_models_by_category,
    get_pii_models_by_language,
    list_model_categories,
)
from .core.model_search import ModelQuery, ModelSearchResult, search_models
from .core.offline import network_blocked_if_offline
from .core.pii import (
    DeidentificationResult,
    PIIEntity,
    deidentify,
    extract_pii,
    reidentify,
)
from .core.pii_entity_merger import (
    PII_PATTERNS,
    PIIPattern,
    calculate_dominant_label,
    find_semantic_units,
    merge_entities_with_semantic_units,
)
from .core.pii_i18n import (
    DEFAULT_PII_MODELS,
    LANGUAGE_PII_PATTERNS,
    SUPPORTED_LANGUAGES,
    get_patterns_for_language,
)
from .core.redaction_preview import redaction_preview, render_redaction_preview
from .core.result_cache import (
    get_result_cache,
    make_cache_key,
)
from .core.results import AnalyzeResult
from .core.streaming import (
    StreamingBufferError,
    StreamingDeidentificationEvent,
    StreamingDeidentifier,
    deidentify_stream,
)
from .core.surrogate_vault import (
    ENCRYPTION_SCHEME,
    InMemorySurrogateStore,
    JsonFileSurrogateStore,
    SurrogateEntry,
    SurrogateKey,
    SurrogateSource,
    SurrogateVault,
    VaultConsistencyReport,
    VaultRotationResult,
)
from .mlx.lm import (
    OpenMedMLXLanguageModel,
    OpenMedPagedKVCache,
    PagedKVCacheConfig,
    PagedKVCachePlan,
    PagedKVCacheStats,
    TokenRange,
    generate_text,
)
from .processing import (
    BatchItem,
    BatchItemResult,
    BatchProcessor,
    BatchProgress,
    BatchResult,
    DatasetRedactionResult,
    DatasetRedactionSummary,
    OutputFormatter,
    TextProcessor,
    TokenizationHelper,
    format_predictions,
    postprocess_text,
    preprocess_text,
    process_batch,
    redact_dataset,
)
from .processing import sentences as sentence_utils
from .processing.advanced_ner import (
    AdvancedNERProcessor,
    StreamingReplayResult,
    StreamingTokenClassifier,
    create_advanced_processor,
    replay_token_classifier,
    stream_token_classifier,
)
from .processing.outputs import PredictionResult
from .utils import (
    Profiler,
    ProfileReport,
    Timer,
    disable_profiling,
    enable_profiling,
    get_logger,
    get_profile_report,
    profile,
    setup_logging,
    timed,
    validate_input,
    validate_model_name,
)
from .utils.validation import (
    sanitize_filename,
    validate_batch_size,
    validate_confidence_threshold,
    validate_output_format,
)

_PLACEHOLDER_SEGMENT_PATTERN = re.compile(r"(?:_{3,}|placeholder|^\W+$)", re.IGNORECASE)

logger = logging.getLogger(__name__)


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

    loader = ModelLoader(config)
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

    loader = loader or ModelLoader(config)
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

    loader = loader or ModelLoader(config)
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

    raw_segments: List[sentence_utils.SentenceSpan] = []
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

            sentence_index: Optional[int] = None
            for idx in chunk.get("segment_indices", []):
                seg_meta = processed_segments[idx]
                seg_start = seg_meta["start"]
                seg_end = seg_meta["end"]
                adj_start = adjusted.get("start")
                adj_end = adjusted.get("end")
                if isinstance(adj_start, int) and seg_start <= adj_start < seg_end:
                    sentence_index = idx
                    break
                if (
                    sentence_index is None
                    and isinstance(adj_end, int)
                    and seg_start < adj_end <= seg_end
                ):
                    sentence_index = idx
                    break

            if sentence_index is None and chunk.get("segment_indices"):
                sentence_index = chunk["segment_indices"][0]

            if sentence_index is not None:
                seg_meta = processed_segments[sentence_index]
                if seg_meta.get("suppress_predictions"):
                    continue
                span_metadata = dict(adjusted.get("metadata") or {})
                span_metadata.setdefault("sentence_index", sentence_index)
                span_metadata.setdefault("sentence_text", seg_meta["text"])
                span_metadata.setdefault("sentence_start", seg_meta["start"])
                span_metadata.setdefault("sentence_end", seg_meta["end"])
                adjusted["metadata"] = span_metadata
            elif sentence_detection:
                span_metadata = dict(adjusted.get("metadata") or {})
                span_metadata.setdefault("sentence_index", -1)
                span_metadata.setdefault("sentence_text", "")
                span_metadata.setdefault("sentence_start", chunk["start"])
                span_metadata.setdefault("sentence_end", chunk["end"])
                adjusted["metadata"] = span_metadata

            adj_start = adjusted.get("start")
            adj_end = adjusted.get("end")

            if not (
                isinstance(adj_start, int)
                and isinstance(adj_end, int)
                and adj_end > adj_start
            ):
                continue

            span_slice = validated_text[adj_start:adj_end]
            stripped = span_slice.strip()
            if not stripped:
                continue
            if _PLACEHOLDER_SEGMENT_PATTERN.search(stripped):
                continue

            flattened_predictions.append(adjusted)

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
    if fmt_output == "dict" and isinstance(result, PredictionResult):
        final_result = AnalyzeResult.from_prediction_result(result)
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
    "TextProcessor",
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
    "Timer",
    "enable_profiling",
    "disable_profiling",
    "get_profile_report",
    "profile",
    "timed",
    # PII detection and de-identification
    "extract_pii",
    "deidentify",
    "reidentify",
    "PIIEntity",
    "DeidentificationResult",
    "CustomRecognizer",
    "StreamingBufferError",
    "StreamingDeidentificationEvent",
    "StreamingDeidentifier",
    "deidentify_stream",
    "DocumentStreamDeidentifier",
    "DocumentStreamResult",
    "deidentify_document_stream",
    "replay_token_classifier",
    "stream_token_classifier",
    "redaction_preview",
    "render_redaction_preview",
    # PII entity merging utilities
    "merge_entities_with_semantic_units",
    "find_semantic_units",
    "calculate_dominant_label",
    "PII_PATTERNS",
    "PIIPattern",
    # Multilingual PII support
    "SUPPORTED_LANGUAGES",
    "DEFAULT_PII_MODELS",
    "LANGUAGE_PII_PATTERNS",
    "get_patterns_for_language",
    # Canonical label taxonomy
    "CANONICAL_LABELS",
    "normalize_label",
    # Anonymization engine
    "Anonymizer",
    "AnonymizerConfig",
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
]
