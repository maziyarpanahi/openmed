"""Zero-shot NER utilities for OpenMed.

Currently provides model indexing helpers that scan a directory of models
and persist a structured metadata index for downstream tooling.
"""

from __future__ import annotations

from .adapter import TokenAnnotation, TokenClassificationResult, to_token_classification
from .exceptions import MissingDependencyError
from .families import (
    INDIC_ENCODER_SPECS,
    EncoderOutput,
    IndicEncoderLoadResult,
    IndicEncoderSpec,
    ModelFamily,
    SupportsEncoding,
    ensure_gliner2_available,
    ensure_gliner_available,
    is_gliner2_available,
    is_gliner_available,
    is_indic_encoder_available,
    load_indic_encoder,
)
from .indexing import (
    BUILTIN_MODEL_RECORDS,
    GLINER_BIOMED_MODEL_ID,
    GLINER_BIOMED_RECORD,
    ModelIndex,
    ModelRecord,
    build_index,
    discover_models,
    load_index,
    write_index,
)
from .infer import Entity, NerRequest, NerResponse, infer, infer_biomedical
from .labels import (
    available_domains,
    get_default_labels,
    load_default_label_map,
    reload_default_label_map,
)

__all__ = [
    "ModelRecord",
    "ModelIndex",
    "GLINER_BIOMED_MODEL_ID",
    "GLINER_BIOMED_RECORD",
    "BUILTIN_MODEL_RECORDS",
    "build_index",
    "discover_models",
    "write_index",
    "load_index",
    "ModelFamily",
    "EncoderOutput",
    "SupportsEncoding",
    "INDIC_ENCODER_SPECS",
    "IndicEncoderSpec",
    "IndicEncoderLoadResult",
    "is_indic_encoder_available",
    "load_indic_encoder",
    "MissingDependencyError",
    "ensure_gliner_available",
    "is_gliner_available",
    "ensure_gliner2_available",
    "is_gliner2_available",
    "load_default_label_map",
    "reload_default_label_map",
    "get_default_labels",
    "available_domains",
    "NerRequest",
    "NerResponse",
    "Entity",
    "infer",
    "infer_biomedical",
    "TokenAnnotation",
    "TokenClassificationResult",
    "to_token_classification",
]
