"""MLX inference backend for OpenMed.

Provides hardware-accelerated NER/PII inference on Apple Silicon
via Apple's MLX framework.

Install with: ``pip install openmed[mlx]``
"""

from openmed.mlx.inference import (
    GLiClassMLXPipeline,
    GLiNERMLXPipeline,
    GLiNERRelexMLXPipeline,
    MLXTokenClassificationPipeline,
    PrivacyFilterMLXPipeline,
    create_mlx_pipeline,
)
from openmed.mlx.lm import (
    LANEFORMER_MLX_MODEL,
    LANEFORMER_SOURCE_MODEL,
    OpenMedMLXLanguageModel,
    OpenMedPagedKVCache,
    PagedKVCacheConfig,
    PagedKVCachePlan,
    PagedKVCacheStats,
    TokenRange,
    generate_text,
    resolve_mlx_language_model,
)

__all__ = [
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
    "TokenRange",
    "create_mlx_pipeline",
    "generate_text",
    "resolve_mlx_language_model",
]
