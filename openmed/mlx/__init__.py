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
    create_mlx_pipeline,
)

__all__ = [
    "MLXTokenClassificationPipeline",
    "GLiNERMLXPipeline",
    "GLiClassMLXPipeline",
    "GLiNERRelexMLXPipeline",
    "create_mlx_pipeline",
]
