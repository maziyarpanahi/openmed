"""Self-contained Draft 2020-12 schema for rendered capability matrix records."""

from __future__ import annotations

import json
from typing import Any, Final

from .capability_matrix import (
    CAPABILITY_CODE_PATTERN,
    CAPABILITY_VERSION_PATTERN,
    COMPATIBILITY_REASON_OPTIONS,
    MAX_CAPABILITY_MATRIX_ENTRIES,
)

CAPABILITY_MATRIX_SCHEMA_DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"
CAPABILITY_MATRIX_SCHEMA_ID: Final = "urn:openmed:multimodal:capability-matrix:1"


def build_capability_matrix_schema() -> dict[str, Any]:
    """Return a fresh, closed schema for the JSON renderer's array of cells.

    Outcome/reason combinations, identifier grammars and record bounds share
    the Python contract's constants. References resolve inside this document.
    JSON Schema checks structural records and identical duplicates; uniqueness
    of provider/profile pairs still requires the Python matrix renderer.

    Returns:
        A mutable, JSON-compatible schema independent of all prior exports.
    """
    reasons = sorted(
        {
            reason
            for options in COMPATIBILITY_REASON_OPTIONS.values()
            for option in options
            for reason in option
        }
    )
    return {
        "$schema": CAPABILITY_MATRIX_SCHEMA_DIALECT,
        "$id": CAPABILITY_MATRIX_SCHEMA_ID,
        "title": "OpenMed provider capability compatibility matrix",
        "type": "array",
        "maxItems": MAX_CAPABILITY_MATRIX_ENTRIES,
        "uniqueItems": True,
        "items": {"$ref": "#/$defs/cell"},
        "$defs": {
            "opaque_id": {
                "type": "string",
                # A negative lookahead is a strict end anchor in ECMA-262 and
                # Python alike; '$' alone would admit a trailing newline.
                "pattern": rf"^(?:{CAPABILITY_CODE_PATTERN})(?![\s\S])",
            },
            "capability_version": {
                "type": "string",
                "pattern": rf"^(?:{CAPABILITY_VERSION_PATTERN})(?![\s\S])",
            },
            "cell": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "provider_id",
                    "capability_version",
                    "profile_id",
                    "outcome",
                    "reasons",
                ],
                "properties": {
                    "provider_id": {"$ref": "#/$defs/opaque_id"},
                    "capability_version": {"$ref": "#/$defs/capability_version"},
                    "profile_id": {"$ref": "#/$defs/opaque_id"},
                    "outcome": {
                        "type": "string",
                        "enum": sorted(COMPATIBILITY_REASON_OPTIONS),
                    },
                    "reasons": {
                        "type": "array",
                        "maxItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": reasons},
                    },
                },
                "oneOf": [
                    {
                        "properties": {
                            "outcome": {"const": outcome},
                            "reasons": {
                                "enum": [
                                    list(option)
                                    for option in COMPATIBILITY_REASON_OPTIONS[outcome]
                                ]
                            },
                        }
                    }
                    for outcome in sorted(COMPATIBILITY_REASON_OPTIONS)
                ],
            },
        },
    }


def render_capability_matrix_schema() -> str:
    """Return byte-stable compact JSON without contacting a schema registry."""
    return (
        json.dumps(
            build_capability_matrix_schema(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


__all__ = [
    "CAPABILITY_MATRIX_SCHEMA_DIALECT",
    "CAPABILITY_MATRIX_SCHEMA_ID",
    "build_capability_matrix_schema",
    "render_capability_matrix_schema",
]
