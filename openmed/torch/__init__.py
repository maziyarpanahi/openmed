"""PyTorch / HuggingFace Transformers backends for OpenMed.

This subpackage wraps token-classification models that run on PyTorch
(CPU or CUDA) so they slot into the same OpenMed pipeline surface as
the MLX path. Use this when the host machine is not Apple Silicon, or
when MLX is unavailable.
"""

from .privacy_filter import PrivacyFilterTorchPipeline

__all__ = ["PrivacyFilterTorchPipeline"]
