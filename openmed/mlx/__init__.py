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
    create_mlx_language_model,
    create_mlx_pipeline,
)
from openmed.mlx.lm import (
    DEFAULT_SPECULATIVE_TOKENS,
    LANEFORMER_DRAFT_MLX_MODEL,
    LANEFORMER_MLX_MODEL,
    LANEFORMER_SOURCE_MODEL,
    OpenMedMLXLanguageModel,
    SpeculativeDecodeMetrics,
    SpeculativeDecodeResult,
    generate_text,
    resolve_mlx_draft_language_model,
    resolve_mlx_language_model,
    tokenizers_are_aligned,
)

__all__ = [
    "DEFAULT_SPECULATIVE_TOKENS",
    "LANEFORMER_DRAFT_MLX_MODEL",
    "LANEFORMER_MLX_MODEL",
    "LANEFORMER_SOURCE_MODEL",
    "MLXTokenClassificationPipeline",
    "GLiNERMLXPipeline",
    "GLiClassMLXPipeline",
    "GLiNERRelexMLXPipeline",
    "OpenMedMLXLanguageModel",
    "PrivacyFilterMLXPipeline",
    "SpeculativeDecodeMetrics",
    "SpeculativeDecodeResult",
    "create_mlx_language_model",
    "create_mlx_pipeline",
    "generate_text",
    "resolve_mlx_draft_language_model",
    "resolve_mlx_language_model",
    "tokenizers_are_aligned",
]
