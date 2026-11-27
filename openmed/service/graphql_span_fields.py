"""Canonical ``OpenMedSpan`` GraphQL fields, generated from the span schema.

Generated from ``openmed/core/schemas/json/span.schema.json`` by
``scripts/codegen/span_graphql.py``. Do not edit by hand: change the schema,
then run ``python scripts/codegen/span_graphql.py``.

Each entry is ``(property, GraphQL type)`` in the schema's canonical order.
Scalars are nullable so a partial span or entity mapping still resolves; arrays
and objects are non-null.
"""

from __future__ import annotations

#: Canonical span properties and their GraphQL types, in canonical order.
CANONICAL_SPAN_FIELDS: tuple[tuple[str, str], ...] = (
    ("schema_version", "Int"),
    ("doc_id", "String"),
    ("start", "Int"),
    ("end", "Int"),
    ("text_hash", "String"),
    ("entity_type", "String"),
    ("canonical_label", "String"),
    ("policy_label", "String"),
    ("regulatory_tags", "[String!]!"),
    ("score", "Float"),
    ("detector", "String"),
    ("evidence", "JSON!"),
    ("action", "String"),
    ("replacement", "String"),
    ("reversible_id", "String"),
    ("section", "String"),
    ("metadata", "JSON!"),
)

__all__ = ["CANONICAL_SPAN_FIELDS"]
