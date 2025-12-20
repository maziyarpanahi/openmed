"""Zero-shot NER utilities for OpenMed.

Currently provides model indexing helpers that scan a directory of models
and persist a structured metadata index for downstream tooling.
"""

from __future__ import annotations

from .indexing import (
    ModelRecord,
    ModelIndex,
    build_index,
    discover_models,
    write_index,
    load_index,
)
from .exceptions import MissingDependencyError
from .families import (
    ModelFamily,
    ensure_gliner_available,
    is_gliner_available,
    ensure_gliner2_available,
    is_gliner2_available,
)
from .labels import (
    available_domains,
    get_default_labels,
    load_default_label_map,
    reload_default_label_map,
)
from .infer import Entity, NerRequest, NerResponse, infer
from .adapter import TokenAnnotation, TokenClassificationResult, to_token_classification

__all__ = [
    "ModelRecord",
    "ModelIndex",
    "build_index",
    "discover_models",
    "write_index",
    "load_index",
    "ModelFamily",
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
    "TokenAnnotation",
    "TokenClassificationResult",
    "to_token_classification",
]
