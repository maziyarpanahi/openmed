"""PyTorch / HuggingFace Transformers backends for OpenMed.

This subpackage wraps token-classification models that run on PyTorch
(CPU or CUDA) so they slot into the same OpenMed pipeline surface as
the MLX path. Use this when the host machine is not Apple Silicon, or
when MLX is unavailable.
"""

from .calibration import load_awq_calibration_texts
from .privacy_filter import PrivacyFilterTorchPipeline
from .quantize_awq import AwqQuantizationResult, quantize_awq

__all__ = [
    "AwqQuantizationResult",
    "PrivacyFilterTorchPipeline",
    "load_awq_calibration_texts",
    "quantize_awq",
]
