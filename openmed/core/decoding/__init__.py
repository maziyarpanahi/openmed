"""Backend-agnostic decoding utilities.

These utilities are extracted from the MLX privacy-filter pipeline so they
can be reused by the PyTorch wrapper as well. They depend only on the
standard library: no torch, no mlx.
"""

from .graph import (
    EdgeCardinality,
    EdgeDecisionTrace,
    GraphExplainReport,
    SpanEdge,
    SpanGraph,
    SpanGraphConstraints,
    SpanNode,
    decode_span_graph,
    edge_f1,
)
from .spans import refine_privacy_filter_span, trim_span_whitespace
from .viterbi import (
    VITERBI_BIAS_KEYS,
    TokenLabelInfo,
    build_label_info,
    labels_to_token_spans,
    viterbi_decode,
    zero_viterbi_biases,
)

__all__ = [
    "EdgeCardinality",
    "EdgeDecisionTrace",
    "GraphExplainReport",
    "SpanEdge",
    "SpanGraph",
    "SpanGraphConstraints",
    "SpanNode",
    "TokenLabelInfo",
    "VITERBI_BIAS_KEYS",
    "build_label_info",
    "decode_span_graph",
    "edge_f1",
    "labels_to_token_spans",
    "refine_privacy_filter_span",
    "trim_span_whitespace",
    "viterbi_decode",
    "zero_viterbi_biases",
]
