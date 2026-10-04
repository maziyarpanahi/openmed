"""Draft 2020-12 JSON Schemas for differential-privacy budget policies.

The committed epsilon-policy config validated by
``openmed.risk.budget.load_epsilon_policies`` is a versioned JSON document.
This module exports that contract as deterministic, self-contained JSON Schemas
so documentation, CI gates, and downstream tooling can validate budget policy
documents without importing the runtime. Every ``$ref`` target is
fragment-local and no validator implementation is imported here.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Final, get_args

from .budget import CURRENT_EPSILON_POLICY_SCHEMA_VERSION, CompositionRule

PRIVACY_BUDGET_POLICY_JSON_SCHEMA_ID = (
    "https://openmed.ai/schemas/risk/privacy-budget-policy-v1.schema.json"
)
EPSILON_POLICY_JSON_SCHEMA_ID = (
    "https://openmed.ai/schemas/risk/epsilon-policy-v1.schema.json"
)

_JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

# Largest finite IEEE-754 double. Committed policy numbers must be finite:
# ``NaN`` already fails ``exclusiveMinimum`` and a positive infinity exceeds this
# ceiling, so a non-finite policy fails closed in schema-only validation exactly
# as ``openmed.risk.budget`` rejects it at load time.
MAX_FINITE_POLICY_NUMBER: Final = sys.float_info.max


def _policy_definitions() -> dict[str, Any]:
    """Return the shared, fragment-local definitions for both exported schemas."""

    return {
        "epsilonBound": {
            "description": (
                "Positive, finite epsilon ceiling; the ceiling only rejects "
                "non-finite values."
            ),
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": MAX_FINITE_POLICY_NUMBER,
        },
        "deltaBound": {
            "description": (
                "Failure probability inside the open interval (0, 1); a "
                "permissive default of one or more would publish a budget bound "
                "that can never be exhausted."
            ),
            "type": "number",
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 1,
        },
        "deltaPrime": {
            "description": (
                "Advanced-composition slack; the basic rule normalises it to "
                "zero and the advanced rule requires a positive value below the "
                "declared delta bound."
            ),
            "type": "number",
            "minimum": 0,
            "maximum": MAX_FINITE_POLICY_NUMBER,
        },
        "compositionRule": {
            "description": (
                "Closed composition rule; it selects the accountant branch, so "
                "an unrecognised rule fails closed instead of silently widening "
                "the accumulated spend."
            ),
            "enum": list(get_args(CompositionRule)),
        },
        "policy": {
            "type": "object",
            "additionalProperties": False,
            "required": ["max_epsilon", "max_delta"],
            "properties": {
                "max_epsilon": {"$ref": "#/$defs/epsilonBound"},
                "max_delta": {"$ref": "#/$defs/deltaBound"},
                "composition": {"$ref": "#/$defs/compositionRule"},
                "delta_prime": {"$ref": "#/$defs/deltaPrime"},
            },
            "allOf": [
                {
                    "if": {
                        "required": ["composition"],
                        "properties": {"composition": {"const": "advanced"}},
                    },
                    "then": {
                        "required": ["delta_prime"],
                        "properties": {"delta_prime": {"exclusiveMinimum": 0}},
                    },
                }
            ],
        },
        "policyRecord": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "scope",
                "max_epsilon",
                "max_delta",
                "composition",
                "delta_prime",
            ],
            "properties": {
                "scope": {"type": "string", "minLength": 1},
                "max_epsilon": {"$ref": "#/$defs/epsilonBound"},
                "max_delta": {"$ref": "#/$defs/deltaBound"},
                "composition": {"$ref": "#/$defs/compositionRule"},
                "delta_prime": {"$ref": "#/$defs/deltaPrime"},
            },
            "allOf": [
                {
                    "if": {"properties": {"composition": {"const": "advanced"}}},
                    "then": {"properties": {"delta_prime": {"exclusiveMinimum": 0}}},
                }
            ],
        },
    }


def export_privacy_budget_policy_schema() -> dict[str, Any]:
    """Return the strict Draft 2020-12 schema for a policy config document.

    The returned mapping is newly built on every call, is closed to unknown
    fields, and requires the versioned ``schema_version`` plus a non-empty
    ``policies`` map keyed by release scope.
    """

    return {
        "$schema": _JSON_SCHEMA_DIALECT,
        "$id": PRIVACY_BUDGET_POLICY_JSON_SCHEMA_ID,
        "title": "OpenMed differential-privacy budget policy config",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "policies"],
        "properties": {
            "schema_version": {"const": CURRENT_EPSILON_POLICY_SCHEMA_VERSION},
            "policies": {
                "type": "object",
                "minProperties": 1,
                "propertyNames": {"type": "string", "minLength": 1},
                "additionalProperties": {"$ref": "#/$defs/policy"},
            },
        },
        "$defs": _policy_definitions(),
    }


def export_epsilon_policy_schema() -> dict[str, Any]:
    """Return the strict Draft 2020-12 schema for one resolved epsilon policy."""

    definitions = _policy_definitions()
    return {
        "$schema": _JSON_SCHEMA_DIALECT,
        "$id": EPSILON_POLICY_JSON_SCHEMA_ID,
        "title": "OpenMed epsilon policy record",
        **definitions["policyRecord"],
        "$defs": definitions,
    }


def export_privacy_budget_policy_schema_json() -> str:
    """Return a byte-stable canonical JSON rendering of the config schema."""

    return json.dumps(
        export_privacy_budget_policy_schema(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def export_epsilon_policy_schema_json() -> str:
    """Return a byte-stable canonical JSON rendering of the policy schema."""

    return json.dumps(
        export_epsilon_policy_schema(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "EPSILON_POLICY_JSON_SCHEMA_ID",
    "MAX_FINITE_POLICY_NUMBER",
    "PRIVACY_BUDGET_POLICY_JSON_SCHEMA_ID",
    "export_epsilon_policy_schema",
    "export_epsilon_policy_schema_json",
    "export_privacy_budget_policy_schema",
    "export_privacy_budget_policy_schema_json",
]
