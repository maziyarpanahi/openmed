"""MLX inference backend for OpenMed.

Provides hardware-accelerated NER/PII inference on Apple Silicon
via Apple's MLX framework.

Install with: ``pip install openmed[mlx]``
"""

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


def is_mlx_available() -> bool:
    """Return True when the ``mlx`` extra is importable, without importing it."""

    return _is_backend_available("mlx")


def ensure_mlx_available() -> None:
    """Raise an actionable error when the ``mlx`` extra is not installed."""

    _require_backend("mlx", feature="MLX inference")


__all__ = [
    "ensure_mlx_available",
    "is_mlx_available",
    "DEFAULT_SPECULATIVE_TOKENS",
    "LANEFORMER_DRAFT_MLX_MODEL",
    "LANEFORMER_MLX_MODEL",
    "LANEFORMER_SOURCE_MODEL",
    "MLXTokenClassificationPipeline",
    "GLiNERMLXPipeline",
    "GLiClassMLXPipeline",
    "GLiNERRelexMLXPipeline",
    "OpenMedMLXLanguageModel",
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
    "generate_text",
    "resolve_mlx_draft_language_model",
    "resolve_mlx_language_model",
    "tokenizers_are_aligned",
]
