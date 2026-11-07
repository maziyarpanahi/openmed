"""Draft 2020-12 JSON Schema for content-free OMOP rollback manifests."""

from __future__ import annotations

import json
from typing import Any

from .omop.mutation_batch import MAX_BATCH_MUTATIONS, MutationOperation
from .omop_rollback_manifest import OMOP_ROLLBACK_MANIFEST_SCHEMA, RollbackStrategy

OMOP_ROLLBACK_MANIFEST_JSON_SCHEMA_ID = (
    "https://openmed.ai/schemas/interop/omop-rollback-manifest-v1.schema.json"
)

# These are the standard OMOP tables for which OpenMed's staged mutation API
# owns primary-key and reference metadata. Custom mutation tables remain a
# local adapter concern and are intentionally excluded from the interoperable
# rollback-manifest schema.
OMOP_ROLLBACK_MANIFEST_TABLES = (
    "concept",
    "condition_occurrence",
    "drug_exposure",
    "measurement",
    "note",
    "note_nlp",
    "observation",
    "person",
    "procedure_occurrence",
    "source_to_concept_map",
    "visit_occurrence",
)

_JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_DIGEST_PATTERN = "^sha256:[0-9a-f]{64}$"


def export_omop_rollback_manifest_schema() -> dict[str, Any]:
    """Return the strict Draft 2020-12 rollback-manifest JSON Schema.

    The returned mapping is newly built on every call, contains only local
    ``$ref`` targets, and derives operation and strategy values from the
    manifest's runtime enums.
    """

    operations = [operation.value for operation in MutationOperation]
    strategies = [strategy.value for strategy in RollbackStrategy]
    operation_properties = {
        operation: {"$ref": "#/$defs/positiveCount"} for operation in operations
    }

    return {
        "$schema": _JSON_SCHEMA_DIALECT,
        "$id": OMOP_ROLLBACK_MANIFEST_JSON_SCHEMA_ID,
        "title": "OpenMed OMOP rollback manifest",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "batch_digest",
            "entries",
            "manifest_digest",
            "mutation_count",
            "operation_counts",
            "schema",
            "tables",
            "vocabulary_snapshot_digest",
        ],
        "properties": {
            "batch_digest": {"$ref": "#/$defs/digest"},
            "entries": {
                "type": "array",
                "items": {"$ref": "#/$defs/entry"},
                "minItems": 1,
                "maxItems": MAX_BATCH_MUTATIONS,
                "uniqueItems": True,
            },
            "manifest_digest": {"$ref": "#/$defs/digest"},
            "mutation_count": {"$ref": "#/$defs/positiveCount"},
            "operation_counts": {"$ref": "#/$defs/operationCounts"},
            "schema": {"const": OMOP_ROLLBACK_MANIFEST_SCHEMA},
            "tables": {
                "type": "array",
                "items": {"$ref": "#/$defs/tableSummary"},
                "minItems": 1,
                "maxItems": len(OMOP_ROLLBACK_MANIFEST_TABLES),
                "uniqueItems": True,
            },
            "vocabulary_snapshot_digest": {
                "$ref": "#/$defs/vocabularySnapshotReference"
            },
        },
        "$defs": {
            "digest": {
                "type": "string",
                "pattern": _DIGEST_PATTERN,
            },
            "entry": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "mutation_digest",
                    "mutation_ordinal",
                    "operation",
                    "rollback_artifact_digest",
                    "rollback_ordinal",
                    "strategy",
                    "table",
                ],
                "properties": {
                    "mutation_digest": {"$ref": "#/$defs/digest"},
                    "mutation_ordinal": {"$ref": "#/$defs/ordinal"},
                    "operation": {"enum": operations},
                    "rollback_artifact_digest": {"$ref": "#/$defs/digest"},
                    "rollback_ordinal": {"$ref": "#/$defs/ordinal"},
                    "strategy": {"enum": strategies},
                    "table": {"$ref": "#/$defs/table"},
                },
                "allOf": [
                    {
                        "if": {
                            "properties": {"operation": {"const": "insert"}},
                            "required": ["operation"],
                        },
                        "then": {
                            "properties": {"strategy": {"const": "delete_inserted_row"}}
                        },
                    },
                    {
                        "if": {
                            "properties": {"operation": {"const": "update"}},
                            "required": ["operation"],
                        },
                        "then": {
                            "properties": {
                                "strategy": {"const": "restore_before_image"}
                            }
                        },
                    },
                    {
                        "if": {
                            "properties": {"operation": {"const": "tombstone"}},
                            "required": ["operation"],
                        },
                        "then": {
                            "properties": {
                                "strategy": {"const": "reinsert_tombstoned_row"}
                            }
                        },
                    },
                ],
            },
            "operationCounts": {
                "type": "object",
                "additionalProperties": False,
                "minProperties": 1,
                "maxProperties": len(operations),
                "properties": operation_properties,
            },
            "ordinal": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_BATCH_MUTATIONS - 1,
            },
            "positiveCount": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_BATCH_MUTATIONS,
            },
            "table": {
                "type": "string",
                "enum": list(OMOP_ROLLBACK_MANIFEST_TABLES),
            },
            "tableSummary": {
                "type": "object",
                "additionalProperties": False,
                "required": ["mutation_count", "operation_counts", "table"],
                "properties": {
                    "mutation_count": {"$ref": "#/$defs/positiveCount"},
                    "operation_counts": {"$ref": "#/$defs/operationCounts"},
                    "table": {"$ref": "#/$defs/table"},
                },
            },
            "vocabularySnapshotReference": {
                "type": "string",
                "pattern": _DIGEST_PATTERN,
                "description": (
                    "SHA-256 binding to the caller-supplied vocabulary snapshot."
                ),
            },
        },
    }


def export_omop_rollback_manifest_schema_json() -> str:
    """Return a byte-stable canonical JSON rendering of the schema."""

    return json.dumps(
        export_omop_rollback_manifest_schema(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "OMOP_ROLLBACK_MANIFEST_JSON_SCHEMA_ID",
    "OMOP_ROLLBACK_MANIFEST_TABLES",
    "export_omop_rollback_manifest_schema",
    "export_omop_rollback_manifest_schema_json",
]
