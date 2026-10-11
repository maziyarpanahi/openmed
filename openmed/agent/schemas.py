"""Deterministic, offline JSON Schemas for public agent governance records.

The catalog describes serialized outcome, correlation, timing, and run-summary
records. It imports only their standard-library-backed source modules; schema
validation remains the responsibility of the consuming adapter.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Final

from .correlation import (
    ACTION_ID_PREFIX,
    CORRELATION_SCHEMA_VERSION,
    CORRELATION_TOKEN_BYTES,
    RUN_ID_PREFIX,
)
from .outcomes import OUTCOME_SCHEMA_VERSION, OutcomeClass, allowed_reason_codes
from .run_summary import (
    _MAX_DURATION_SECONDS,
    _MAX_EVENTS,
    _MAX_SUMMARY_DIGESTS,
    _MAX_TOOL_CALLS,
    _MAX_WORKFLOWS,
    RUN_SUMMARY_SCHEMA_VERSION,
)

AGENT_SCHEMA_DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"
_TIMING_SCHEMA_ID: Final = "urn:openmed:agent:timing:v1"
_END = r"(?![\s\S])"  # Unlike $, this cannot match before a final newline.
_SUMMARY_IDENTIFIER_PATTERN = (
    r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?" + _END
)
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}" + _END
_TIMING_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}" + _END


def _header(schema_id: str) -> dict[str, Any]:
    return {"$schema": AGENT_SCHEMA_DIALECT, "$id": schema_id}


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _outcome_schema() -> dict[str, Any]:
    classes = sorted(item.value for item in OutcomeClass)
    reasons = {name: sorted(allowed_reason_codes(name)) for name in classes}
    definitions = {
        f"outcome_{name}": _object(
            {
                "schema_version": {"const": OUTCOME_SCHEMA_VERSION},
                "outcome_class": {"const": name},
                "reason_code": {"enum": reasons[name]},
            },
            ["schema_version", "outcome_class", "reason_code"],
        )
        for name in classes
    }
    return {
        **_header("urn:" + OUTCOME_SCHEMA_VERSION.replace(".", ":")),
        "$defs": definitions,
        **_object(
            {
                "schema_version": {"const": OUTCOME_SCHEMA_VERSION},
                "outcome_class": {"enum": classes},
                "reason_code": {
                    "enum": sorted(
                        {code for codes in reasons.values() for code in codes}
                    )
                },
            },
            ["schema_version", "outcome_class", "reason_code"],
        ),
        "oneOf": [{"$ref": f"#/$defs/outcome_{name}"} for name in classes],
    }


def _correlation_schema() -> dict[str, Any]:
    digits = CORRELATION_TOKEN_BYTES * 2

    def identifier(prefix: str) -> dict[str, Any]:
        size = len(prefix) + digits
        return {
            "type": "string",
            "minLength": size,
            "maxLength": size,
            "pattern": rf"^{prefix}[0-9a-f]{{{digits}}}" + _END,
        }

    return {
        **_header("urn:" + CORRELATION_SCHEMA_VERSION.replace(".", ":")),
        "$defs": {
            "correlation_run_id": identifier(RUN_ID_PREFIX),
            "correlation_action_id": identifier(ACTION_ID_PREFIX),
        },
        **_object(
            {
                "schema_version": {"const": CORRELATION_SCHEMA_VERSION},
                "run_id": {"$ref": "#/$defs/correlation_run_id"},
                "action_id": {"$ref": "#/$defs/correlation_action_id"},
                "parent_action_id": {
                    "anyOf": [
                        {"type": "null"},
                        {"$ref": "#/$defs/correlation_action_id"},
                    ]
                },
            },
            ["schema_version", "run_id", "action_id", "parent_action_id"],
        ),
    }


def _timing_schema() -> dict[str, Any]:
    def nanoseconds() -> dict[str, Any]:
        return {"type": "integer", "minimum": 0}

    def interval() -> dict[str, Any]:
        return {
            "start_ns": nanoseconds(),
            "end_ns": nanoseconds(),
            "duration_ns": nanoseconds(),
        }

    identifier = {
        "type": "string",
        "minLength": 1,
        "maxLength": 128,
        "pattern": _TIMING_IDENTIFIER_PATTERN,
    }
    return {
        **_header(_TIMING_SCHEMA_ID),
        "$defs": {
            "timing_identifier": identifier,
            "timing_run": _object(
                {
                    **interval(),
                    "correlation_id": {"$ref": "#/$defs/timing_identifier"},
                },
                ["start_ns", "end_ns", "duration_ns"],
            ),
            "timing_action": _object(
                {
                    "action_id": {"$ref": "#/$defs/timing_identifier"},
                    **interval(),
                    "parent_action_id": {"$ref": "#/$defs/timing_identifier"},
                    "correlation_id": {"$ref": "#/$defs/timing_identifier"},
                },
                ["action_id", "start_ns", "end_ns", "duration_ns"],
            ),
        },
        **_object(
            {
                "run": {"$ref": "#/$defs/timing_run"},
                "actions": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/timing_action"},
                },
            },
            ["run", "actions"],
        ),
    }


def _run_summary_schema() -> dict[str, Any]:
    outcomes = sorted(item.value for item in OutcomeClass)
    return {
        **_header("urn:" + RUN_SUMMARY_SCHEMA_VERSION.replace(".", ":")),
        "$defs": {
            "summary_identifier": {
                "type": "string",
                "minLength": 1,
                "maxLength": 128,
                "pattern": _SUMMARY_IDENTIFIER_PATTERN,
            },
            "summary_digest": {
                "type": "string",
                "minLength": 71,
                "maxLength": 71,
                "pattern": _DIGEST_PATTERN,
            },
            "summary_outcome_counts": _object(
                {
                    name: {"type": "integer", "minimum": 0, "maximum": _MAX_EVENTS}
                    for name in outcomes
                },
                outcomes,
            ),
        },
        **_object(
            {
                "schema_version": {"const": RUN_SUMMARY_SCHEMA_VERSION},
                "workflow_ids": {
                    "type": "array",
                    "maxItems": _MAX_WORKFLOWS,
                    "uniqueItems": True,
                    "items": {"$ref": "#/$defs/summary_identifier"},
                },
                "outcome_counts": {"$ref": "#/$defs/summary_outcome_counts"},
                "tool_call_count": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": _MAX_TOOL_CALLS,
                },
                "duration_seconds": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": _MAX_DURATION_SECONDS,
                },
                "artifact_digests": {
                    "type": "array",
                    "maxItems": _MAX_SUMMARY_DIGESTS,
                    "uniqueItems": True,
                    "items": {"$ref": "#/$defs/summary_digest"},
                },
            },
            [
                "schema_version",
                "workflow_ids",
                "outcome_counts",
                "tool_call_count",
                "duration_seconds",
                "artifact_digests",
            ],
        ),
    }


_BUILDERS: Final[dict[str, Callable[[], dict[str, Any]]]] = {
    "outcome": _outcome_schema,
    "correlation": _correlation_schema,
    "timing": _timing_schema,
    "run_summary": _run_summary_schema,
}


def list_agent_schema_names() -> tuple[str, ...]:
    """Return the four stable catalog names in deterministic order."""
    return tuple(sorted(_BUILDERS))


def build_agent_schema(name: str) -> dict[str, Any]:
    """Return a fresh, self-contained Draft 2020-12 schema.

    Args:
        name: A name returned by :func:`list_agent_schema_names`.

    Raises:
        ValueError: If the name is unknown; submitted text is never echoed.
    """
    if type(name) is not str or name not in _BUILDERS:
        raise ValueError("schema: unknown_name")
    return _BUILDERS[name]()


def build_agent_schema_catalog() -> dict[str, dict[str, Any]]:
    """Return independent schema mappings indexed by catalog name."""
    return {name: build_agent_schema(name) for name in list_agent_schema_names()}


def render_agent_schema(name: str) -> str:
    """Render a catalog entry as byte-stable compact ASCII JSON."""
    return json.dumps(
        build_agent_schema(name),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "AGENT_SCHEMA_DIALECT",
    "build_agent_schema",
    "build_agent_schema_catalog",
    "list_agent_schema_names",
    "render_agent_schema",
]
