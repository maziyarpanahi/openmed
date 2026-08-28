"""Training-data builders for public clinical corpora."""

from .drugprot_relations import (
    DRUGPROT_RELATION_LABELS,
    DRUGPROT_TO_RELATION_LABEL,
    DrugProtRelationExample,
    build_drugprot_relation_examples,
    load_drugprot_relation_examples,
    map_drugprot_relation_type,
)

__all__ = [
    "DRUGPROT_RELATION_LABELS",
    "DRUGPROT_TO_RELATION_LABEL",
    "DrugProtRelationExample",
    "build_drugprot_relation_examples",
    "load_drugprot_relation_examples",
    "map_drugprot_relation_type",
]
