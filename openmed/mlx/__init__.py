"""MLX inference backend for OpenMed.

Provides hardware-accelerated NER/PII inference on Apple Silicon
via Apple's MLX framework.

Install with: ``pip install openmed[mlx]``
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from openmed.core.capabilities import is_backend_available as _is_backend_available
from openmed.core.capabilities import require_backend as _require_backend
from openmed.mlx.inference import (
    GLiClassMLXPipeline,
    GLiNERMLXPipeline,
    GLiNERRelexMLXPipeline,
    MLXTokenClassificationPipeline,
    PrivacyFilterMLXPipeline,
    create_mlx_language_model,
    create_mlx_pipeline,
)
from openmed.mlx.lm import (
    DEFAULT_SPECULATIVE_TOKENS,
    LANEFORMER_DRAFT_MLX_MODEL,
    LANEFORMER_MLX_MODEL,
    LANEFORMER_SOURCE_MODEL,
    MAPLE_MLX_MODEL,
    MAPLE_MLX_REVISION,
    MAPLE_SOURCE_MODEL,
    OpenMedMLXLanguageModel,
    OpenMedPagedKVCache,
    PagedKVCacheConfig,
    PagedKVCachePlan,
    PagedKVCacheStats,
    SpeculativeDecodeMetrics,
    SpeculativeDecodeResult,
    TokenRange,
    generate_text,
    resolve_mlx_draft_language_model,
    resolve_mlx_language_model,
    tokenizers_are_aligned,
)
from openmed.mlx.maple import (
    MAPLE_MEDICAL_DISCLAIMER,
    MapleClinicalAssistant,
    MapleRelation,
    MapleResponseError,
    MapleSpan,
    MapleTask,
    MapleTaskResult,
    build_maple_task_messages,
    parse_maple_task_response,
    redact_maple_spans,
    visible_maple_response,
)
from openmed.mlx.vlm import (
    DEFAULT_NORTH_MICRO_VISION_MODEL,
    CohereCompassProcessor,
    OpenMedMLXVisionLanguageModel,
    OpenMedVisionLanguageArtifactError,
    OpenMedVisionLanguageError,
    VisionLanguageGeneration,
    generate_vision_text,
    resolve_mlx_vision_language_model,
    smart_resize,
)

_MAPLE_EXPORT_NAMES = frozenset(
    {
        "MAPLE_EXPORT_BITS",
        "MAPLE_EXPORT_GROUP_SIZE",
        "MAPLE_EXPORT_MANIFEST",
        "MAPLE_SOURCE_REVISION",
        "MapleMLXExportPlan",
        "MapleMLXVariant",
        "export_maple_mlx_variants",
        "plan_maple_mlx_variants",
    }
)


def is_mlx_available() -> bool:
    """Return True when the ``mlx`` extra is importable, without importing it."""

    return _is_backend_available("mlx")


def ensure_mlx_available() -> None:
    """Raise an actionable error when the ``mlx`` extra is not installed."""

    _require_backend("mlx", feature="MLX inference")


def __getattr__(name: str) -> Any:
    """Load the standalone Maple exporter lazily for warning-free CLI use."""

    if name in _MAPLE_EXPORT_NAMES:
        module = import_module("openmed.mlx.maple_export")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ensure_mlx_available",
    "is_mlx_available",
    "DEFAULT_SPECULATIVE_TOKENS",
    "LANEFORMER_DRAFT_MLX_MODEL",
    "LANEFORMER_MLX_MODEL",
    "LANEFORMER_SOURCE_MODEL",
    "MAPLE_MEDICAL_DISCLAIMER",
    "MAPLE_EXPORT_BITS",
    "MAPLE_EXPORT_GROUP_SIZE",
    "MAPLE_EXPORT_MANIFEST",
    "MAPLE_MLX_MODEL",
    "MAPLE_MLX_REVISION",
    "MAPLE_SOURCE_MODEL",
    "MAPLE_SOURCE_REVISION",
    "MLXTokenClassificationPipeline",
    "GLiNERMLXPipeline",
    "GLiClassMLXPipeline",
    "GLiNERRelexMLXPipeline",
    "OpenMedMLXLanguageModel",
    "MapleClinicalAssistant",
    "MapleMLXExportPlan",
    "MapleMLXVariant",
    "MapleRelation",
    "MapleResponseError",
    "MapleSpan",
    "MapleTask",
    "MapleTaskResult",
    "OpenMedPagedKVCache",
    "PagedKVCacheConfig",
    "PagedKVCachePlan",
    "PagedKVCacheStats",
    "PrivacyFilterMLXPipeline",
    "SpeculativeDecodeMetrics",
    "SpeculativeDecodeResult",
    "TokenRange",
    "create_mlx_language_model",
    "create_mlx_pipeline",
    "build_maple_task_messages",
    "export_maple_mlx_variants",
    "generate_text",
    "parse_maple_task_response",
    "plan_maple_mlx_variants",
    "redact_maple_spans",
    "resolve_mlx_draft_language_model",
    "resolve_mlx_language_model",
    "tokenizers_are_aligned",
    "visible_maple_response",
    "CohereCompassProcessor",
    "DEFAULT_NORTH_MICRO_VISION_MODEL",
    "OpenMedVisionLanguageArtifactError",
    "OpenMedVisionLanguageError",
    "OpenMedMLXVisionLanguageModel",
    "VisionLanguageGeneration",
    "generate_vision_text",
    "resolve_mlx_vision_language_model",
    "smart_resize",
]
