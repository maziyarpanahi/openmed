"""Top-level interface for the OpenMed library."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Union, List

from .core import ModelLoader, load_model, OpenMedConfig
from .core.model_registry import (
    get_model_info,
    get_models_by_category,
    get_all_models,
    list_model_categories,
    get_model_suggestions,
)
from .processing import (
    TextProcessor,
    preprocess_text,
    postprocess_text,
    TokenizationHelper,
    OutputFormatter,
    format_predictions,
)
from .processing.advanced_ner import AdvancedNERProcessor, create_advanced_processor
from .processing.outputs import PredictionResult
from .utils import setup_logging, get_logger, validate_input, validate_model_name
from .utils.validation import (
    validate_confidence_threshold,
    validate_output_format,
    validate_batch_size,
    sanitize_filename,
)

__version__ = "0.1.1"


def list_models(
    *,
    include_registry: bool = True,
    config: Optional[OpenMedConfig] = None,
) -> List[str]:
    """Return available OpenMed model identifiers.

    Args:
        include_registry: Include entries from the bundled registry in addition to
            results fetched from Hugging Face.
        config: Optional custom configuration for model discovery.
    """

    loader = ModelLoader(config)
    return loader.list_available_models(include_registry=include_registry)


def analyze_text(
    text: str,
    model_name: str = "disease_detection_superclinical",
    *,
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
    **pipeline_kwargs: Any,
) -> Union[PredictionResult, str, List[Dict[str, Any]]]:
    """Run a token-classification model on ``text`` and format the predictions.

    Args:
        text: Clinical or biomedical text to analyse.
        model_name: Registry key or fully-qualified Hugging Face model id.
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
        **pipeline_kwargs: Additional arguments passed to
            :meth:`openmed.core.models.ModelLoader.create_pipeline`.

    Returns:
        Prediction result in the requested ``output_format``.
    """

    validated_text = validate_input(text)
    validated_model = validate_model_name(model_name)

    loader = loader or ModelLoader(config)

    pipeline_args: Dict[str, Any] = {
        "task": "token-classification",
        "aggregation_strategy": aggregation_strategy,
        "use_fast_tokenizer": use_fast_tokenizer,
    }
    pipeline_args.update(pipeline_kwargs)

    ner_pipeline = loader.create_pipeline(validated_model, **pipeline_args)

    start_time = time.time()
    predictions = ner_pipeline(validated_text)
    processing_time = time.time() - start_time

    fmt_kwargs: Dict[str, Any] = {
        "include_confidence": include_confidence,
        "group_entities": group_entities,
        "metadata": metadata,
        "processing_time": processing_time,
    }

    if confidence_threshold is not None:
        fmt_kwargs["confidence_threshold"] = validate_confidence_threshold(
            confidence_threshold
        )

    if formatter_kwargs:
        fmt_kwargs.update(formatter_kwargs)

    fmt_output = validate_output_format(output_format)

    return format_predictions(
        predictions,
        validated_text,
        model_name=validated_model,
        output_format=fmt_output,
        **fmt_kwargs,
    )


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
    "AdvancedNERProcessor",
    "create_advanced_processor",
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
    "list_models",
    "analyze_text",
]
