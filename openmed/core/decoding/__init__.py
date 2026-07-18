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
from .spans import (
    TokenClassificationSpan,
    TokenClassificationStreamEvent,
    coerce_token_classification_spans,
    reconcile_stream_spans,
    refine_privacy_filter_span,
    stable_span_id,
    stable_span_key,
    trim_span_whitespace,
)
from .viterbi import (
    VITERBI_BIAS_KEYS,
    IncrementalViterbiState,
    TokenLabelInfo,
    build_label_info,
    labels_to_token_spans,
    resolve_viterbi_biases,
    viterbi_decode,
    viterbi_decode_incremental,
    zero_viterbi_biases,
)

__all__ = [
    "EdgeCardinality",
    "EdgeDecisionTrace",
    "GraphExplainReport",
    "IncrementalViterbiState",
    "SpanEdge",
    "SpanGraph",
    "SpanGraphConstraints",
    "SpanNode",
    "TokenClassificationSpan",
    "TokenClassificationStreamEvent",
    "TokenLabelInfo",
    "VITERBI_BIAS_KEYS",
    "build_label_info",
    "coerce_token_classification_spans",
    "decode_span_graph",
    "edge_f1",
    "labels_to_token_spans",
    "reconcile_stream_spans",
    "refine_privacy_filter_span",
    "resolve_viterbi_biases",
    "stable_span_key",
    "stable_span_id",
    "trim_span_whitespace",
    "viterbi_decode",
    "viterbi_decode_incremental",
    "zero_viterbi_biases",
]
