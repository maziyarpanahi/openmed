"""Backend-agnostic decoding utilities (Viterbi, BIOES span construction).

These utilities are extracted from the MLX privacy-filter pipeline so they
can be reused by the PyTorch wrapper as well. They depend only on the
standard library — no torch, no mlx.
"""

from .spans import refine_privacy_filter_span, trim_span_whitespace
from .viterbi import (
    TokenLabelInfo,
    VITERBI_BIAS_KEYS,
    build_label_info,
    labels_to_token_spans,
    viterbi_decode,
    zero_viterbi_biases,
)

__all__ = [
    "TokenLabelInfo",
    "VITERBI_BIAS_KEYS",
    "build_label_info",
    "labels_to_token_spans",
    "refine_privacy_filter_span",
    "trim_span_whitespace",
    "viterbi_decode",
    "zero_viterbi_biases",
]
