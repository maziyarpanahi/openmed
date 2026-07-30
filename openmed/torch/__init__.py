"""PyTorch / HuggingFace Transformers backends for OpenMed.

This subpackage wraps token-classification models that run on PyTorch
(MPS, CUDA, or CPU) so they slot into the same OpenMed pipeline surface
as the MLX path. Use this when MLX is unavailable, or when you
intentionally choose the Hugging Face/PyTorch backend.
"""

from .attention import select_attn_implementation
from .awq_grounding import (
    AwqGroundingQuantizationResult,
    GroundingAwqRejected,
    GroundingRecallGate,
    HuggingFaceGroundingEmbedder,
    certify_grounding_recall,
    load_awq_grounding_embedder,
    quantize_awq_grounding,
)
from .calibration import load_awq_calibration_texts, load_quantization_calibration_texts
from .device import apply_mps_tuning, resolve_torch_device
from .privacy_filter import PrivacyFilterTorchPipeline
from .quantize_awq import AwqQuantizationResult, quantize_awq
from .quantize_gptq import GptqQuantizationResult, quantize_gptq

__all__ = [
    "AwqQuantizationResult",
    "AwqGroundingQuantizationResult",
    "GptqQuantizationResult",
    "GroundingAwqRejected",
    "GroundingRecallGate",
    "HuggingFaceGroundingEmbedder",
    "PrivacyFilterTorchPipeline",
    "apply_mps_tuning",
    "certify_grounding_recall",
    "load_awq_calibration_texts",
    "load_awq_grounding_embedder",
    "load_quantization_calibration_texts",
    "quantize_awq",
    "quantize_awq_grounding",
    "quantize_gptq",
    "resolve_torch_device",
    "select_attn_implementation",
]
