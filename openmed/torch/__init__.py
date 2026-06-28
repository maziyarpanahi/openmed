"""PyTorch / HuggingFace Transformers backends for OpenMed.

This subpackage wraps token-classification models that run on PyTorch
(MPS, CUDA, or CPU) so they slot into the same OpenMed pipeline surface
as the MLX path. Use this when MLX is unavailable, or when you
intentionally choose the Hugging Face/PyTorch backend.
"""

from .attention import select_attn_implementation
from .calibration import load_awq_calibration_texts
from .device import apply_mps_tuning, resolve_torch_device
from .privacy_filter import PrivacyFilterTorchPipeline
from .quantize_awq import AwqQuantizationResult, quantize_awq

__all__ = [
    "AwqQuantizationResult",
    "PrivacyFilterTorchPipeline",
    "apply_mps_tuning",
    "load_awq_calibration_texts",
    "quantize_awq",
    "resolve_torch_device",
    "select_attn_implementation",
]
