"""Synthetic annotation task generation and BRAT/CoNLL interchange."""

from openmed.eval.annotation.brat_io import (
    format_brat,
    parse_brat,
    read_brat,
    write_brat,
)
from openmed.eval.annotation.conll_io import (
    format_conll,
    parse_conll,
    read_conll,
    write_conll,
)
from openmed.eval.annotation.toolkit import (
    AnnotationIssue,
    AnnotationTask,
    AnnotationValidationError,
    Prelabeler,
    SpanProposal,
    generate_annotation_task,
    generate_synthetic_annotation_task,
    span_from_offsets,
    validate_spans,
)

__all__ = [
    "AnnotationIssue",
    "AnnotationTask",
    "AnnotationValidationError",
    "Prelabeler",
    "SpanProposal",
    "format_brat",
    "format_conll",
    "generate_annotation_task",
    "generate_synthetic_annotation_task",
    "parse_brat",
    "parse_conll",
    "read_brat",
    "read_conll",
    "span_from_offsets",
    "validate_spans",
    "write_brat",
    "write_conll",
]
