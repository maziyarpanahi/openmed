"""Deterministic JSON Schema exports for multimodal preflight artifacts.

The multimodal boundary exchanges a small set of metadata-only artifacts:
privacy-safe asset manifests, canonically ordered asset batches, abstention
records, processing summaries, and provider-result envelopes. Each artifact
already enforces a strict contract in Python; this module exports that contract
as a self-contained Draft 2020-12 JSON Schema so build systems, fixtures, and
partners can validate payloads offline.

Every exported document is built fresh on each call, declares its definitions
under ``$defs``, references them only with fragment-local ``$ref`` values, and
closes each declared object. Field names, closed enumerations, digest and
identifier formats, schema versions, and bounded numeric ranges mirror the
modules that own them. Cross-record invariants that JSON Schema cannot express
-- unique asset identifiers, canonical asset order, aggregate agreement with the
manifests, and integer-only JSON tokens -- stay enforced by the Python loaders.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, Final

from .abstention import ABSTENTION_SCHEMA_VERSION, AbstentionReason, AbstentionStage
from .asset_batch import BATCH_VERSION, MAX_BATCH_ASSETS
from .asset_manifest import (
    MANIFEST_VERSION,
    MAX_MANIFEST_BYTE_SIZE,
    MAX_MANIFEST_COUNT,
    MAX_MANIFEST_DURATION_SECONDS,
)
from .processing_summary import PROCESSING_SUMMARY_SCHEMA_VERSION, ProcessingOutcome
from .provider_result import (
    PROVIDER_RESULT_SCHEMA_VERSION,
    ProviderAbstentionCode,
    ProviderResultOutcome,
)

__all__ = [
    "ABSTENTION_SCHEMA_ID",
    "BATCH_SCHEMA_ID",
    "MANIFEST_SCHEMA_ID",
    "MULTIMODAL_SCHEMA_DIALECT",
    "MULTIMODAL_SCHEMA_NAMES",
    "PROCESSING_SUMMARY_SCHEMA_ID",
    "PROVIDER_RESULT_SCHEMA_ID",
    "build_multimodal_schemas",
    "export_multimodal_schema",
    "export_multimodal_schema_json",
    "export_multimodal_schemas_json",
]

MULTIMODAL_SCHEMA_DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"

MANIFEST_SCHEMA_ID: Final = (
    "https://openmed.ai/schemas/multimodal/asset-manifest-v1.schema.json"
)
BATCH_SCHEMA_ID: Final = (
    "https://openmed.ai/schemas/multimodal/asset-batch-v1.schema.json"
)
ABSTENTION_SCHEMA_ID: Final = (
    "https://openmed.ai/schemas/multimodal/abstention-record-v1.schema.json"
)
PROCESSING_SUMMARY_SCHEMA_ID: Final = (
    "https://openmed.ai/schemas/multimodal/processing-summary-v1.schema.json"
)
PROVIDER_RESULT_SCHEMA_ID: Final = (
    "https://openmed.ai/schemas/multimodal/provider-result-v1.schema.json"
)

MULTIMODAL_SCHEMA_NAMES: Final = (
    "asset_manifest",
    "asset_batch",
    "abstention_record",
    "processing_summary",
    "provider_result",
)

# Mirrors of the owning modules' formats, enumerations, and bounds.
# ``tests/unit/multimodal/test_schemas.py`` binds every value below to its
# source constant, so the exported schema cannot drift from the Python contract
# without a failing test.
_SUPPORTED_EXACT_MEDIA_TYPES: Final = frozenset(
    {"application/dicom", "application/dicom+json", "application/pdf"}
)
_SUPPORTED_MEDIA_PREFIXES: Final = ("audio/", "image/")
_OPAQUE_IDENTIFIER_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
_PATH_OR_URL_PATTERN: Final = r"://|(^|[A-Za-z]):[\\/]|[\\/]|~"
_MEDIA_TYPE_PATTERN: Final = r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$"
_SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"
_PROVIDER_IDENTIFIER_PATTERN: Final = r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$"
_PROVIDER_FORBIDDEN_PART_PATTERN: Final = (
    r"(^|[._-])(?:bearer|credential|mrn|password|patient|prompt|secret|token)"
    r"([._-]|$)"
)
_PROVIDER_MAX_COUNT: Final = (1 << 63) - 1
_PROVIDER_MAX_DURATION_MS: Final = 86_400_000.0
_PROVIDER_COUNT_FIELDS: Final = (
    "detection_count",
    "frame_count",
    "input_bytes",
    "input_items",
    "output_items",
    "page_count",
    "sample_count",
    "segment_count",
    "token_count",
)
_PROVIDER_REQUIRED_FIELDS: Final = (
    "duration_ms",
    "input_digest",
    "model_id",
    "outcome",
    "provider_id",
    "schema_version",
)
_MANIFEST_REQUIRED_FIELDS: Final = ("asset_id", "byte_size", "media_type", "sha256")
_SUMMARY_FIELDS: Final = (
    "schema_version",
    "total_assets",
    "total_bytes",
    "total_duration_seconds",
    "by_media_type",
    "outcome_counts",
    "abstention_counts",
    "asset_digests",
    "asset_count_with_output_digest",
)
_REASONS_BY_STAGE: Final = {
    "preflight": ("provider_unavailable", "resource_limit", "unsupported_media"),
    "decode": ("low_quality", "malformed_media", "resource_limit"),
    "inference": (
        "low_quality",
        "phi_uncertainty",
        "provider_unavailable",
        "resource_limit",
        "speaker_uncertainty",
        "temporal_instability",
    ),
    "post_process": (
        "low_quality",
        "phi_uncertainty",
        "resource_limit",
        "speaker_uncertainty",
        "temporal_instability",
    ),
}


def build_multimodal_schemas() -> dict[str, dict[str, Any]]:
    """Return a fresh Draft 2020-12 document for every exported artifact."""

    return {
        "asset_manifest": _build_asset_manifest_schema(),
        "asset_batch": _build_asset_batch_schema(),
        "abstention_record": _build_abstention_record_schema(),
        "processing_summary": _build_processing_summary_schema(),
        "provider_result": _build_provider_result_schema(),
    }


def export_multimodal_schema(name: str) -> dict[str, Any]:
    """Return a fresh schema document for one exported artifact name."""

    schemas = build_multimodal_schemas()
    try:
        return schemas[name]
    except KeyError:
        raise ValueError("unknown multimodal schema name") from None


def export_multimodal_schema_json(name: str) -> str:
    """Render one exported artifact schema as byte-stable compact JSON."""

    return _render(export_multimodal_schema(name))


def export_multimodal_schemas_json() -> str:
    """Render the whole schema catalog as byte-stable compact JSON."""

    return _render(build_multimodal_schemas())


def _render(document: Mapping[str, Any]) -> str:
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest_definition() -> dict[str, Any]:
    return {"type": "string", "pattern": _SHA256_PATTERN}


def _enum(values: Iterable[Any]) -> dict[str, Any]:
    return {"enum": [value.value for value in values]}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    """Allow an optional field to be absent or explicitly ``null``."""

    return {"anyOf": [schema, {"type": "null"}]}


def _document(
    *,
    schema_id: str,
    title: str,
    description: str,
    root: str,
    definitions: dict[str, Any],
) -> dict[str, Any]:
    return {
        "$schema": MULTIMODAL_SCHEMA_DIALECT,
        "$id": schema_id,
        "title": title,
        "description": description,
        "$ref": f"#/$defs/{root}",
        "$defs": definitions,
    }


def _shared_definitions() -> dict[str, Any]:
    """Return fresh definitions reused by more than one artifact schema."""

    return {
        "opaqueIdentifier": {
            "type": "string",
            "pattern": _OPAQUE_IDENTIFIER_PATTERN,
            "not": {"pattern": _PATH_OR_URL_PATTERN},
        },
        "sha256Digest": _digest_definition(),
        "mediaType": {
            "type": "string",
            "pattern": _MEDIA_TYPE_PATTERN,
            "anyOf": [
                {"enum": sorted(_SUPPORTED_EXACT_MEDIA_TYPES)},
                *(
                    {"pattern": f"^{re.escape(prefix)}"}
                    for prefix in _SUPPORTED_MEDIA_PREFIXES
                ),
            ],
        },
        "positiveByteSize": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_MANIFEST_BYTE_SIZE,
        },
        "positiveCount": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_MANIFEST_COUNT,
        },
        "positiveDuration": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": MAX_MANIFEST_DURATION_SECONDS,
        },
    }


def _stage_reason_constraints() -> list[dict[str, Any]]:
    """Constrain ``reason`` to the codes the owning module allows per stage."""

    constraints: list[dict[str, Any]] = []
    for stage, reasons in _REASONS_BY_STAGE.items():
        constraints.append(
            {
                "if": {
                    "properties": {"stage": {"const": stage}},
                    "required": ["stage"],
                },
                "then": {"properties": {"reason": {"enum": list(reasons)}}},
            }
        )
    return constraints


def _manifest_definitions() -> dict[str, Any]:
    definitions = _shared_definitions()
    definitions["manifest"] = {
        "type": "object",
        "additionalProperties": False,
        "required": list(_MANIFEST_REQUIRED_FIELDS),
        "properties": {
            "version": {"const": MANIFEST_VERSION},
            "asset_id": {"$ref": "#/$defs/opaqueIdentifier"},
            "media_type": {"$ref": "#/$defs/mediaType"},
            "sha256": {"$ref": "#/$defs/sha256Digest"},
            "byte_size": {"$ref": "#/$defs/positiveByteSize"},
            "pages": _nullable({"$ref": "#/$defs/positiveCount"}),
            "width": _nullable({"$ref": "#/$defs/positiveCount"}),
            "height": _nullable({"$ref": "#/$defs/positiveCount"}),
            "frames": _nullable({"$ref": "#/$defs/positiveCount"}),
            "duration_seconds": _nullable({"$ref": "#/$defs/positiveDuration"}),
        },
    }
    return definitions


def _build_asset_manifest_schema() -> dict[str, Any]:
    return _document(
        schema_id=MANIFEST_SCHEMA_ID,
        title="OpenMed multimodal asset manifest",
        description=(
            "Privacy-safe, versioned description of one multimodal input asset."
        ),
        root="manifest",
        definitions=_manifest_definitions(),
    )


def _build_asset_batch_schema() -> dict[str, Any]:
    definitions = _manifest_definitions()
    definitions["batch"] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["assets", "batch_id"],
        "properties": {
            "version": {"const": BATCH_VERSION},
            "batch_id": {"$ref": "#/$defs/opaqueIdentifier"},
            "assets": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_BATCH_ASSETS,
                "uniqueItems": True,
                "items": {"$ref": "#/$defs/manifest"},
            },
            "asset_count": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_BATCH_ASSETS,
            },
            "total_bytes": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_MANIFEST_BYTE_SIZE,
            },
            "total_pages": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_MANIFEST_COUNT,
            },
            "total_frames": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_MANIFEST_COUNT,
            },
            "total_duration_seconds": {
                "type": "number",
                "minimum": 0,
                "maximum": MAX_MANIFEST_DURATION_SECONDS,
            },
        },
    }
    return _document(
        schema_id=BATCH_SCHEMA_ID,
        title="OpenMed multimodal asset batch",
        description=(
            "Canonically ordered batch of privacy-safe asset manifests with "
            "derived aggregate totals."
        ),
        root="batch",
        definitions=definitions,
    )


def _build_abstention_record_schema() -> dict[str, Any]:
    definitions = {
        "abstentionRecord": {
            "type": "object",
            "additionalProperties": False,
            "required": ["reason", "schema_version", "stage"],
            "properties": {
                "schema_version": {"const": ABSTENTION_SCHEMA_VERSION},
                "stage": _enum(AbstentionStage),
                "reason": _enum(AbstentionReason),
            },
            "allOf": _stage_reason_constraints(),
        }
    }
    return _document(
        schema_id=ABSTENTION_SCHEMA_ID,
        title="OpenMed multimodal abstention record",
        description=(
            "Content-free record of the stage and reason at which multimodal "
            "processing stopped."
        ),
        root="abstentionRecord",
        definitions=definitions,
    )


def _build_processing_summary_schema() -> dict[str, Any]:
    definitions = _shared_definitions()
    definitions.update(
        {
            "mediaTypeTotals": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "count",
                    "media_type",
                    "total_bytes",
                    "total_frames",
                    "total_pages",
                ],
                "properties": {
                    "media_type": {"$ref": "#/$defs/mediaType"},
                    "count": {"type": "integer", "minimum": 1},
                    "total_bytes": {"type": "integer", "minimum": 0},
                    "total_pages": {"type": "integer", "minimum": 0},
                    "total_frames": {"type": "integer", "minimum": 0},
                },
            },
            "outcomeCount": {
                "type": "object",
                "additionalProperties": False,
                "required": ["count", "outcome"],
                "properties": {
                    "outcome": _enum(ProcessingOutcome),
                    "count": {"type": "integer", "minimum": 1},
                },
            },
            "abstentionCount": {
                "type": "object",
                "additionalProperties": False,
                "required": ["count", "reason", "stage"],
                "properties": {
                    "stage": _enum(AbstentionStage),
                    "reason": _enum(AbstentionReason),
                    "count": {"type": "integer", "minimum": 1},
                },
                "allOf": _stage_reason_constraints(),
            },
            "assetDigestEntry": {
                "type": "object",
                "additionalProperties": False,
                "required": ["asset_id", "input_sha256"],
                "properties": {
                    "asset_id": {"$ref": "#/$defs/opaqueIdentifier"},
                    "input_sha256": {"$ref": "#/$defs/sha256Digest"},
                    "output_sha256": _nullable({"$ref": "#/$defs/sha256Digest"}),
                },
            },
            "processingSummary": {
                "type": "object",
                "additionalProperties": False,
                "required": list(_SUMMARY_FIELDS),
                "properties": {
                    "schema_version": {"const": PROCESSING_SUMMARY_SCHEMA_VERSION},
                    "total_assets": {"type": "integer", "minimum": 0},
                    "total_bytes": {"type": "integer", "minimum": 0},
                    "total_duration_seconds": {"type": "number", "minimum": 0},
                    "by_media_type": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/mediaTypeTotals"},
                    },
                    "outcome_counts": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/outcomeCount"},
                    },
                    "abstention_counts": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/abstentionCount"},
                    },
                    "asset_digests": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/assetDigestEntry"},
                    },
                    "asset_count_with_output_digest": {
                        "type": "integer",
                        "minimum": 0,
                    },
                },
            },
        }
    )
    return _document(
        schema_id=PROCESSING_SUMMARY_SCHEMA_ID,
        title="OpenMed multimodal processing summary",
        description=(
            "Deterministic aggregate completion artifact for one multimodal run."
        ),
        root="processingSummary",
        definitions=definitions,
    )


def _provider_outcome_constraints() -> list[dict[str, Any]]:
    """Bind ``outcome`` to the optional fields the envelope requires."""

    other_outcomes = [
        outcome.value
        for outcome in ProviderResultOutcome
        if outcome is not ProviderResultOutcome.SUCCESS
    ]
    non_abstentions = [
        outcome.value
        for outcome in ProviderResultOutcome
        if outcome is not ProviderResultOutcome.ABSTENTION
    ]
    return [
        {
            "if": {
                "properties": {
                    "outcome": {"const": ProviderResultOutcome.SUCCESS.value}
                },
                "required": ["outcome"],
            },
            "then": {
                "properties": {
                    "output_digest": {"$ref": "#/$defs/sha256Digest"},
                },
                "required": ["output_digest"],
            },
        },
        {
            "if": {
                "properties": {"outcome": {"enum": other_outcomes}},
                "required": ["outcome"],
            },
            "then": {"properties": {"output_digest": {"type": "null"}}},
        },
        {
            "if": {
                "properties": {
                    "outcome": {"const": ProviderResultOutcome.ABSTENTION.value}
                },
                "required": ["outcome"],
            },
            "then": {
                "properties": {
                    "abstention_code": {"$ref": "#/$defs/providerAbstentionCode"},
                },
                "required": ["abstention_code"],
            },
        },
        {
            "if": {
                "properties": {"outcome": {"enum": non_abstentions}},
                "required": ["outcome"],
            },
            "then": {"properties": {"abstention_code": {"type": "null"}}},
        },
    ]


def _build_provider_result_schema() -> dict[str, Any]:
    definitions = {
        "sha256Digest": _digest_definition(),
        "providerIdentifier": {
            "type": "string",
            "pattern": _PROVIDER_IDENTIFIER_PATTERN,
            "not": {"pattern": _PROVIDER_FORBIDDEN_PART_PATTERN},
        },
        "providerAbstentionCode": _enum(ProviderAbstentionCode),
        "providerResultEnvelope": {
            "type": "object",
            "additionalProperties": False,
            "required": list(_PROVIDER_REQUIRED_FIELDS),
            "properties": {
                "schema_version": {"const": PROVIDER_RESULT_SCHEMA_VERSION},
                "provider_id": {"$ref": "#/$defs/providerIdentifier"},
                "model_id": {"$ref": "#/$defs/providerIdentifier"},
                "input_digest": {"$ref": "#/$defs/sha256Digest"},
                "output_digest": _nullable({"$ref": "#/$defs/sha256Digest"}),
                "outcome": _enum(ProviderResultOutcome),
                "abstention_code": _nullable(
                    {"$ref": "#/$defs/providerAbstentionCode"}
                ),
                "duration_ms": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": _PROVIDER_MAX_DURATION_MS,
                },
                "count_metadata": {
                    "type": "object",
                    "propertyNames": {"enum": sorted(_PROVIDER_COUNT_FIELDS)},
                    "additionalProperties": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": _PROVIDER_MAX_COUNT,
                    },
                },
            },
            "allOf": _provider_outcome_constraints(),
        },
    }
    return _document(
        schema_id=PROVIDER_RESULT_SCHEMA_ID,
        title="OpenMed multimodal provider result",
        description=(
            "Content-free envelope describing the terminal outcome of one "
            "multimodal provider call."
        ),
        root="providerResultEnvelope",
        definitions=definitions,
    )
