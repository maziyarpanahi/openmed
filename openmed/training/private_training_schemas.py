"""Draft 2020-12 schema export for private-training metadata.

The exported schemas are self-contained and use only fragment-local references.
They describe the public federated training JSON projections (lifecycle,
schedule, round status, update metadata, and aggregate metric envelopes) without
importing a validator, reading model contents, or contacting a schema registry.
Cross-field arithmetic remains enforced by the Python validators.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from types import MappingProxyType
from typing import Any, Final, Mapping

from openmed.training.federated_metrics import (
    FEDERATED_METRIC_SCHEMA_VERSION,
    FederatedMetricKind,
    FederatedParticipantCountBand,
    FederatedPrivacyMechanism,
    FederatedUncertaintyMethod,
)
from openmed.training.federated_round import (
    FEDERATED_ROUND_SCHEMA_VERSION,
    FederatedRoundState,
)
from openmed.training.federated_schedule import (
    FEDERATED_SCHEDULE_SCHEMA_VERSION,
    FEDERATED_SCHEDULE_V2_SCHEMA_VERSION,
    MAX_FEDERATED_CLOCK_SKEW_SECONDS,
    MAX_FEDERATED_PHASE_DURATION_SECONDS,
)
from openmed.training.federated_status import (
    _ABORT_REASON_CODES,
    _HOLD_REASON_CODES,
    FEDERATED_ROUND_STATUS_SCHEMA_VERSION,
    FederatedCompletionBand,
    FederatedQuorumStatus,
    FederatedRoundReasonCode,
)
from openmed.training.federated_update_metadata import (
    _DTYPES,
    _MAX_DIMENSION,
    _MAX_ELEMENTS,
    _MAX_PARAMETERS,
    _MAX_RANK,
    _PARAMETER_FIELDS,
    _UPDATE_FIELDS,
    FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
)

PRIVATE_TRAINING_SCHEMA_DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"

_DIGEST_PATTERN: Final = r"^sha256:[0-9a-f]{64}(?![\s\S])"
_CANONICAL_UTC_TIMESTAMP: Final = (
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{6})?Z(?![\s\S])"
)
_METRIC_ID_PATTERN: Final = r"^[a-z][a-z0-9_.-]{0,63}(?![\s\S])"
_MECHANISM_VERSION_PATTERN: Final = r"^v[1-9][0-9]{0,3}(?![\s\S])"
_PARAMETER_NAME_PATTERN: Final = (
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+))*(?![\s\S])"
)

_SCHEDULE_BOUNDARY_FIELDS: Final = frozenset(
    {
        "aggregation_starts_at",
        "enrollment_starts_at",
        "evaluation_starts_at",
        "finishes_at",
        "update_submission_starts_at",
    }
)
_SCHEDULE_MAXIMUM_FIELDS: Final = frozenset(
    {"aggregation", "enrollment", "evaluation", "update_submission"}
)


def build_federated_round_lifecycle_schema() -> dict[str, Any]:
    """Build the JSON Schema for federated round lifecycle payloads.

    Returns:
        A new JSON-compatible Draft 2020-12 schema mapping. Mutating the
        returned mapping cannot affect a later export.
    """

    return {
        "$schema": PRIVATE_TRAINING_SCHEMA_DIALECT,
        "$id": f"urn:openmed:training:federated_round_lifecycle:{FEDERATED_ROUND_SCHEMA_VERSION}",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "state"],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": FEDERATED_ROUND_SCHEMA_VERSION,
            },
            "state": {
                "type": "string",
                "enum": [item.value for item in FederatedRoundState],
            },
        },
    }


def build_federated_round_schedule_schema() -> dict[str, Any]:
    """Build the JSON Schema for federated round schedule payloads."""

    return {
        "$schema": PRIVATE_TRAINING_SCHEMA_DIALECT,
        "$id": f"urn:openmed:training:federated_round_schedule:{FEDERATED_SCHEDULE_SCHEMA_VERSION}",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "boundaries",
            "maximum_duration_seconds",
            "schema_version",
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": FEDERATED_SCHEDULE_SCHEMA_VERSION,
            },
            "boundaries": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_SCHEDULE_BOUNDARY_FIELDS),
                "properties": {
                    field: {
                        "type": "string",
                        "pattern": _CANONICAL_UTC_TIMESTAMP,
                        "format": "date-time",
                    }
                    for field in sorted(_SCHEDULE_BOUNDARY_FIELDS)
                },
            },
            "maximum_duration_seconds": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_SCHEDULE_MAXIMUM_FIELDS),
                "properties": {
                    field: {
                        "anyOf": [
                            {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": MAX_FEDERATED_PHASE_DURATION_SECONDS,
                            },
                            {"type": "null"},
                        ]
                    }
                    for field in sorted(_SCHEDULE_MAXIMUM_FIELDS)
                },
            },
        },
    }


def build_federated_round_schedule_v2_schema() -> dict[str, Any]:
    """Build the JSON Schema for skew-aware federated schedule payloads.

    Returns:
        A new JSON-compatible Draft 2020-12 schema mapping. It extends the v1
        schedule shape with the required ``max_clock_skew_seconds`` bound.
    """

    schema = build_federated_round_schedule_schema()
    schema["$id"] = (
        "urn:openmed:training:federated_round_schedule:"
        f"{FEDERATED_SCHEDULE_V2_SCHEMA_VERSION}"
    )
    schema["required"] = sorted([*schema["required"], "max_clock_skew_seconds"])
    schema["properties"]["schema_version"]["const"] = (
        FEDERATED_SCHEDULE_V2_SCHEMA_VERSION
    )
    schema["properties"]["max_clock_skew_seconds"] = {
        "type": "integer",
        "minimum": 0,
        "maximum": MAX_FEDERATED_CLOCK_SKEW_SECONDS,
    }
    return schema


def build_federated_round_status_schema() -> dict[str, Any]:
    """Build the JSON Schema for federated round status payloads."""

    return {
        "$schema": PRIVATE_TRAINING_SCHEMA_DIALECT,
        "$id": f"urn:openmed:training:federated_round_status:{FEDERATED_ROUND_STATUS_SCHEMA_VERSION}",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "aggregate_digest_refs",
            "completed_participant_count",
            "completion_band",
            "minimum_group_size",
            "participant_count",
            "quorum_status",
            "reason_code",
            "schema_version",
            "state",
        ],
        "allOf": [
            {
                "if": {
                    "properties": {"state": {"const": FederatedRoundState.HELD.value}}
                },
                "then": {
                    "properties": {
                        "reason_code": {
                            "enum": sorted(item.value for item in _HOLD_REASON_CODES)
                        }
                    }
                },
            },
            {
                "if": {
                    "properties": {
                        "state": {"const": FederatedRoundState.ABORTED.value}
                    }
                },
                "then": {
                    "properties": {
                        "reason_code": {
                            "enum": sorted(item.value for item in _ABORT_REASON_CODES)
                        }
                    }
                },
            },
            {
                "if": {
                    "properties": {
                        "state": {
                            "not": {
                                "enum": [
                                    FederatedRoundState.HELD.value,
                                    FederatedRoundState.ABORTED.value,
                                ]
                            }
                        }
                    }
                },
                "then": {"properties": {"reason_code": {"type": "null"}}},
            },
            {
                "if": {"properties": {"participant_count": {"type": "null"}}},
                "then": {
                    "properties": {"completed_participant_count": {"type": "null"}}
                },
            },
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": FEDERATED_ROUND_STATUS_SCHEMA_VERSION,
            },
            "state": {
                "type": "string",
                "enum": [item.value for item in FederatedRoundState],
            },
            "quorum_status": {
                "type": "string",
                "enum": [item.value for item in FederatedQuorumStatus],
            },
            "participant_count": {
                "anyOf": [
                    {"type": "integer", "minimum": 2},
                    {"type": "null"},
                ]
            },
            "completed_participant_count": {
                "anyOf": [
                    {"type": "integer", "minimum": 2},
                    {"type": "null"},
                ]
            },
            "minimum_group_size": {
                "type": "integer",
                "minimum": 2,
            },
            "completion_band": {
                "type": "string",
                "enum": [item.value for item in FederatedCompletionBand],
            },
            "reason_code": {
                "anyOf": [
                    {
                        "type": "string",
                        "enum": [item.value for item in FederatedRoundReasonCode],
                    },
                    {"type": "null"},
                ]
            },
            "aggregate_digest_refs": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "pattern": _DIGEST_PATTERN},
            },
        },
    }


def build_federated_update_metadata_schema() -> dict[str, Any]:
    """Build the JSON Schema for federated update metadata payloads."""

    return {
        "$schema": PRIVATE_TRAINING_SCHEMA_DIALECT,
        "$id": f"urn:openmed:training:federated_update_metadata:{FEDERATED_UPDATE_METADATA_SCHEMA_VERSION}",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_UPDATE_FIELDS),
        "properties": {
            "schema_version": {
                "type": "string",
                "const": FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
            },
            "model_digest": {
                "type": "string",
                "pattern": _DIGEST_PATTERN,
            },
            "adapter_format": {"const": "dense"},
            "parameters": {
                "type": "array",
                "minItems": 1,
                "maxItems": _MAX_PARAMETERS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(_PARAMETER_FIELDS),
                    "properties": {
                        "name": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 256,
                            "pattern": _PARAMETER_NAME_PATTERN,
                        },
                        "shape": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": _MAX_RANK,
                            "items": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": _MAX_DIMENSION,
                            },
                        },
                        "dtype": {
                            "type": "string",
                            "enum": sorted(_DTYPES),
                        },
                    },
                },
            },
            "total_elements": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_ELEMENTS,
            },
            "update_digest": {
                "type": "string",
                "pattern": _DIGEST_PATTERN,
            },
            "clipped": {"type": "boolean"},
        },
    }


def build_federated_aggregate_metric_schema() -> dict[str, Any]:
    """Build the JSON Schema for federated aggregate metric envelopes."""

    return {
        "$schema": PRIVATE_TRAINING_SCHEMA_DIALECT,
        "$id": f"urn:openmed:training:federated_aggregate_metric:{FEDERATED_METRIC_SCHEMA_VERSION}",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "aggregate_value",
            "clipping_lower_bound",
            "clipping_upper_bound",
            "confidence_level",
            "metric_id",
            "metric_kind",
            "minimum_group_size",
            "participant_count_band",
            "privacy_mechanism",
            "privacy_mechanism_version",
            "schema_version",
            "uncertainty_lower_bound",
            "uncertainty_method",
            "uncertainty_upper_bound",
        ],
        "$defs": {
            "number": {"type": "number"},
            "nullableNumber": {"type": ["number", "null"]},
            "confidence": {
                "type": ["number", "null"],
                "exclusiveMinimum": 0,
                "exclusiveMaximum": 1,
            },
        },
        "allOf": [
            {
                "if": {
                    "properties": {
                        "participant_count_band": {
                            "const": FederatedParticipantCountBand.SUPPRESSED.value
                        }
                    }
                },
                "then": {
                    "properties": {
                        "aggregate_value": {"type": "null"},
                        "uncertainty_method": {
                            "const": FederatedUncertaintyMethod.SUPPRESSED.value
                        },
                    }
                },
                "else": {
                    "properties": {
                        "aggregate_value": {"type": "number"},
                        "uncertainty_method": {
                            "enum": [
                                FederatedUncertaintyMethod.NONE.value,
                                FederatedUncertaintyMethod.CONFIDENCE_INTERVAL.value,
                            ]
                        },
                    }
                },
            },
            {
                "if": {
                    "properties": {
                        "uncertainty_method": {
                            "enum": [
                                FederatedUncertaintyMethod.NONE.value,
                                FederatedUncertaintyMethod.SUPPRESSED.value,
                            ]
                        }
                    }
                },
                "then": {
                    "properties": {
                        field: {"type": "null"}
                        for field in (
                            "uncertainty_lower_bound",
                            "uncertainty_upper_bound",
                            "confidence_level",
                        )
                    }
                },
                "else": {
                    "properties": {
                        field: {"type": "number"}
                        for field in (
                            "uncertainty_lower_bound",
                            "uncertainty_upper_bound",
                            "confidence_level",
                        )
                    }
                },
            },
            {
                "if": {
                    "properties": {
                        "metric_kind": {"const": FederatedMetricKind.COUNT.value}
                    }
                },
                "then": {
                    "properties": {
                        "aggregate_value": {"type": ["integer", "null"], "minimum": 0},
                        "clipping_lower_bound": {"type": "integer", "minimum": 0},
                        "clipping_upper_bound": {"type": "integer", "minimum": 0},
                    }
                },
            },
            {
                "if": {
                    "properties": {
                        "metric_kind": {"const": FederatedMetricKind.RATE.value}
                    }
                },
                "then": {
                    "properties": {
                        field: {"minimum": 0, "maximum": 1}
                        for field in (
                            "aggregate_value",
                            "clipping_lower_bound",
                            "clipping_upper_bound",
                            "uncertainty_lower_bound",
                            "uncertainty_upper_bound",
                        )
                    }
                },
            },
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": FEDERATED_METRIC_SCHEMA_VERSION,
            },
            "metric_id": {
                "type": "string",
                "pattern": _METRIC_ID_PATTERN,
            },
            "metric_kind": {
                "type": "string",
                "enum": [item.value for item in FederatedMetricKind],
            },
            "aggregate_value": {"$ref": "#/$defs/nullableNumber"},
            "clipping_lower_bound": {"$ref": "#/$defs/number"},
            "clipping_upper_bound": {"$ref": "#/$defs/number"},
            "privacy_mechanism": {
                "type": "string",
                "enum": [item.value for item in FederatedPrivacyMechanism],
            },
            "privacy_mechanism_version": {
                "type": "string",
                "pattern": _MECHANISM_VERSION_PATTERN,
            },
            "minimum_group_size": {
                "type": "integer",
                "minimum": 2,
            },
            "participant_count_band": {
                "type": "string",
                "enum": [item.value for item in FederatedParticipantCountBand],
            },
            "uncertainty_method": {
                "type": "string",
                "enum": [item.value for item in FederatedUncertaintyMethod],
            },
            "uncertainty_lower_bound": {"$ref": "#/$defs/nullableNumber"},
            "uncertainty_upper_bound": {"$ref": "#/$defs/nullableNumber"},
            "confidence_level": {"$ref": "#/$defs/confidence"},
        },
    }


PRIVATE_TRAINING_SCHEMA_BUILDERS: Final[Mapping[str, Callable[[], dict[str, Any]]]] = (
    MappingProxyType(
        {
            "federated_round_lifecycle": build_federated_round_lifecycle_schema,
            "federated_round_schedule": build_federated_round_schedule_schema,
            "federated_round_schedule_v2": build_federated_round_schedule_v2_schema,
            "federated_round_status": build_federated_round_status_schema,
            "federated_update_metadata": build_federated_update_metadata_schema,
            "federated_aggregate_metric": build_federated_aggregate_metric_schema,
        }
    )
)


class PrivateTrainingSchemaError(ValueError):
    """Raised when an unknown private-training schema is requested."""


def build_private_training_schemas() -> dict[str, Any]:
    """Build every private-training schema as a fresh catalog mapping.

    Returns:
        A new mapping from stable catalog name to schema. Mutating one entry
        cannot affect a later export.
    """

    return {
        name: builder() for name, builder in PRIVATE_TRAINING_SCHEMA_BUILDERS.items()
    }


def render_private_training_schemas() -> str:
    """Render the whole catalog as byte-stable compact JSON."""

    return json.dumps(
        build_private_training_schemas(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def build_schema(name: str) -> dict[str, Any]:
    """Build one cataloged schema by its stable name.

    Args:
        name: Stable catalog key, for example ``federated_round_lifecycle``.

    Returns:
        A fresh JSON-compatible Draft 2020-12 schema mapping.

    Raises:
        PrivateTrainingSchemaError: If the name is not part of the catalog.
    """

    if type(name) is not str or name not in PRIVATE_TRAINING_SCHEMA_BUILDERS:
        raise PrivateTrainingSchemaError("unknown private-training schema")
    return PRIVATE_TRAINING_SCHEMA_BUILDERS[name]()


def render_schema(name: str) -> str:
    """Render one cataloged schema as byte-stable compact JSON."""

    return json.dumps(
        build_schema(name),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "PRIVATE_TRAINING_SCHEMA_BUILDERS",
    "PRIVATE_TRAINING_SCHEMA_DIALECT",
    "PrivateTrainingSchemaError",
    "build_federated_aggregate_metric_schema",
    "build_federated_round_lifecycle_schema",
    "build_federated_round_schedule_schema",
    "build_federated_round_schedule_v2_schema",
    "build_federated_round_status_schema",
    "build_federated_update_metadata_schema",
    "build_private_training_schemas",
    "build_schema",
    "render_private_training_schemas",
    "render_schema",
]
