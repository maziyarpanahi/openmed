"""Draft 2020-12 schema export for content-free agent exchange records.

The exported schemas are deterministic, self-contained, and safe to generate
offline. They describe the public JSON projections of approval tokens and
receipts, reviewer handoff packets, capability grant manifests, and recovery
checkpoints and decisions without importing a validator or contacting a schema
registry.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Final

from openmed.agent.approvals.tokens import (
    APPROVAL_RECEIPT_SCHEMA_VERSION,
    APPROVAL_TOKEN_SCHEMA_VERSION,
)
from openmed.agent.artifact_reference import (
    ARTIFACT_REFERENCE_VERSION,
    MAX_ARTIFACT_BYTE_SIZE,
    ArtifactKind,
)
from openmed.agent.permissions.grants import CAPABILITY_GRANT_SCHEMA_VERSION
from openmed.agent.reviewer_handoff import (
    MAX_HANDOFF_EVIDENCE_REFERENCES,
    REVIEWER_HANDOFF_SCHEMA_VERSION,
    RequestedDecision,
    allowed_handoff_reason_codes,
)
from openmed.agent.workflows.recovery import (
    RECOVERY_CHECKPOINT_SCHEMA_VERSION,
    RECOVERY_EVIDENCE_SCHEMA_VERSION,
    CompensationLimit,
    EffectKind,
    EffectState,
    RecoveryDisposition,
    RecoveryPhase,
    RecoveryReason,
)

EXCHANGE_RECORD_SCHEMA_DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"

_MAX_NAMESPACE_LENGTH: Final = 253
_MAX_GOVERNANCE_IDENTIFIER_LENGTH: Final = 512
_MAX_TIMESTAMP: Final = (1 << 63) - 1
_ACTION_ID_LENGTH: Final = 36
_ARTIFACT_ID_LENGTH: Final = 36
_DIGEST_LENGTH: Final = 71
_IDEMPOTENCY_KEY_LENGTH: Final = 69
_NONCE_LENGTH: Final = 38
_RUN_ID_LENGTH: Final = 36
_SHA256_HEX_LENGTH: Final = 64
_SIGNATURE_LENGTH: Final = 76

# JSON Schema ``pattern`` is unanchored, so every exported pattern keeps an
# explicit start anchor and a true end-of-string guard. A trailing newline that
# ``$`` would tolerate therefore fails, matching the runtime ``fullmatch``
# checks.
_END = r"(?![\s\S])"
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_NAMESPACE = rf"{_LABEL}(?:\.{_LABEL})+"
_LOCAL_NAME = r"[a-z][a-z0-9-]{0,63}"
_NUMBER = r"(?:0|[1-9][0-9]*)"
_VERSION = rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}"
_NAMESPACE_GUARD = rf"(?=[^/]{{1,{_MAX_NAMESPACE_LENGTH}}}/)"


def _kind_pattern(kind: str) -> str:
    return (
        rf"^{kind}:{_NAMESPACE_GUARD}{_NAMESPACE}/{_LOCAL_NAME}"
        rf"(?:@{_VERSION})?{_END}"
    )


_DIGEST_PATTERN = rf"^sha256:[0-9a-f]{{64}}{_END}"
_SIGNATURE_PATTERN = rf"^(?:hmac-sha256:[0-9a-f]{{64}})?{_END}"
_NONCE_PATTERN = rf"^nonce_[0-9a-f]{{32}}{_END}"
_RUN_ID_PATTERN = rf"^run_[0-9a-f]{{32}}{_END}"
_ACTION_ID_PATTERN = rf"^act_[0-9a-f]{{32}}{_END}"
_IDEMPOTENCY_KEY_PATTERN = rf"^idem_[0-9a-f]{{64}}{_END}"
_KEY_ID_PATTERN = rf"^[a-z][a-z0-9._-]{{0,127}}{_END}"
_ARTIFACT_ID_PATTERN = rf"^art_[0-9a-f]{{32}}{_END}"
_ARTIFACT_SCHEMA_ID_PATTERN = rf"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+\.v[1-9][0-9]*{_END}"
_SHA256_HEX_PATTERN = rf"^[0-9a-f]{{64}}{_END}"
_TIMESTAMP_PATTERN = (
    rf"^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}T[0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}}Z{_END}"
)
_ROLE_PATTERN = _kind_pattern("role")
_WORKFLOW_ID_PATTERN = _kind_pattern("workflow")
_TOOL_ID_PATTERN = _kind_pattern("tool")
_RESOURCE_PATTERN = _kind_pattern("resource")
_CAPABILITY_ACTION_PATTERN = _kind_pattern("action")
_POLICY_PROFILE_PATTERN = _kind_pattern("policy")


class ExchangeRecordSchemaError(ValueError):
    """Raised when an exchange record schema name is unknown or not a string."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _bounded_string(
    pattern: str,
    *,
    min_length: int | None = None,
    max_length: int | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "pattern": pattern}
    if min_length is not None:
        schema["minLength"] = min_length
    if max_length is not None:
        schema["maxLength"] = max_length
    return schema


def _digest() -> dict[str, Any]:
    return _bounded_string(
        _DIGEST_PATTERN, min_length=_DIGEST_LENGTH, max_length=_DIGEST_LENGTH
    )


def _action_id() -> dict[str, Any]:
    return _bounded_string(
        _ACTION_ID_PATTERN, min_length=_ACTION_ID_LENGTH, max_length=_ACTION_ID_LENGTH
    )


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [{"type": "null"}, schema]}


def _timestamp() -> dict[str, Any]:
    return {"type": "integer", "minimum": 0, "maximum": _MAX_TIMESTAMP}


def _enum(enum_type: type[Any]) -> dict[str, Any]:
    return {"type": "string", "enum": [item.value for item in enum_type]}


def _closed_object(
    properties: Mapping[str, Any],
    *,
    required: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    names = tuple(required) if required is not None else tuple(properties)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(names),
        "properties": dict(properties),
    }


def _on_field(field_name: str, value: Any) -> dict[str, Any]:
    return {"properties": {field_name: {"const": value}}, "required": [field_name]}


def _if_then(
    condition: Mapping[str, Any], consequence: Mapping[str, Any]
) -> dict[str, Any]:
    return {"if": dict(condition), "then": dict(consequence)}


def _approval_token_schema() -> dict[str, Any]:
    return {
        "$schema": EXCHANGE_RECORD_SCHEMA_DIALECT,
        **_closed_object(
            {
                "schema_version": {
                    "type": "string",
                    "const": APPROVAL_TOKEN_SCHEMA_VERSION,
                },
                "action_digest": _digest(),
                "reviewer_role": _bounded_string(
                    _ROLE_PATTERN, max_length=_MAX_GOVERNANCE_IDENTIFIER_LENGTH
                ),
                "expires_at": _timestamp(),
                "nonce": _bounded_string(
                    _NONCE_PATTERN, min_length=_NONCE_LENGTH, max_length=_NONCE_LENGTH
                ),
                "signature": _bounded_string(
                    _SIGNATURE_PATTERN, max_length=_SIGNATURE_LENGTH
                ),
            }
        ),
    }


def _approval_receipt_schema() -> dict[str, Any]:
    return {
        "$schema": EXCHANGE_RECORD_SCHEMA_DIALECT,
        **_closed_object(
            {
                "schema_version": {
                    "type": "string",
                    "const": APPROVAL_RECEIPT_SCHEMA_VERSION,
                },
                "action_digest": _digest(),
                "reviewer_role": _bounded_string(
                    _ROLE_PATTERN, max_length=_MAX_GOVERNANCE_IDENTIFIER_LENGTH
                ),
                "token_digest": _digest(),
                "consumed_at": _timestamp(),
                "expires_at": _timestamp(),
            }
        ),
    }


def _artifact_reference_schema() -> dict[str, Any]:
    return _closed_object(
        {
            "version": {"type": "integer", "const": ARTIFACT_REFERENCE_VERSION},
            "artifact_id": _bounded_string(
                _ARTIFACT_ID_PATTERN,
                min_length=_ARTIFACT_ID_LENGTH,
                max_length=_ARTIFACT_ID_LENGTH,
            ),
            "kind": _enum(ArtifactKind),
            "schema_id": _bounded_string(_ARTIFACT_SCHEMA_ID_PATTERN),
            "sha256": _bounded_string(
                _SHA256_HEX_PATTERN,
                min_length=_SHA256_HEX_LENGTH,
                max_length=_SHA256_HEX_LENGTH,
            ),
            "byte_size": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_ARTIFACT_BYTE_SIZE,
            },
        },
        required=(
            "artifact_id",
            "kind",
            "schema_id",
            "sha256",
            "byte_size",
        ),
    )


def _reviewer_handoff_schema() -> dict[str, Any]:
    return {
        "$schema": EXCHANGE_RECORD_SCHEMA_DIALECT,
        "$defs": {"artifact_reference": _artifact_reference_schema()},
        **_closed_object(
            {
                "schema_version": {
                    "type": "string",
                    "const": REVIEWER_HANDOFF_SCHEMA_VERSION,
                },
                "run_id": _bounded_string(
                    _RUN_ID_PATTERN,
                    min_length=_RUN_ID_LENGTH,
                    max_length=_RUN_ID_LENGTH,
                ),
                "workflow_id": _bounded_string(
                    _WORKFLOW_ID_PATTERN, max_length=_MAX_GOVERNANCE_IDENTIFIER_LENGTH
                ),
                "reason_code": {
                    "type": "string",
                    "enum": sorted(allowed_handoff_reason_codes()),
                },
                "requested_decision": _enum(RequestedDecision),
                "evidence_references": {
                    "type": "array",
                    "maxItems": MAX_HANDOFF_EVIDENCE_REFERENCES,
                    "uniqueItems": True,
                    "items": {"$ref": "#/$defs/artifact_reference"},
                },
                "issued_at": _bounded_string(_TIMESTAMP_PATTERN),
                "expires_at": _bounded_string(_TIMESTAMP_PATTERN),
            }
        ),
    }


def _capability_grant_constraint_schema() -> dict[str, Any]:
    return _closed_object(
        {
            "tool": _bounded_string(
                _TOOL_ID_PATTERN, max_length=_MAX_GOVERNANCE_IDENTIFIER_LENGTH
            ),
            "resource": _bounded_string(
                _RESOURCE_PATTERN, max_length=_MAX_GOVERNANCE_IDENTIFIER_LENGTH
            ),
            "action": _bounded_string(
                _CAPABILITY_ACTION_PATTERN,
                max_length=_MAX_GOVERNANCE_IDENTIFIER_LENGTH,
            ),
            "policy_profile": _bounded_string(
                _POLICY_PROFILE_PATTERN, max_length=_MAX_GOVERNANCE_IDENTIFIER_LENGTH
            ),
        }
    )


def _capability_grant_schema() -> dict[str, Any]:
    return {
        "$schema": EXCHANGE_RECORD_SCHEMA_DIALECT,
        "$defs": {"capability_grant_constraint": _capability_grant_constraint_schema()},
        **_closed_object(
            {
                "schema_version": {
                    "type": "string",
                    "const": CAPABILITY_GRANT_SCHEMA_VERSION,
                },
                "constraints": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"$ref": "#/$defs/capability_grant_constraint"},
                },
                "expires_at": _timestamp(),
                "key_id": _bounded_string(_KEY_ID_PATTERN, max_length=128),
                "signature": _bounded_string(
                    _SIGNATURE_PATTERN, max_length=_SIGNATURE_LENGTH
                ),
            }
        ),
    }


def _effect_record_schema() -> dict[str, Any]:
    return {
        **_closed_object(
            {
                "ordinal": _timestamp(),
                "action_id": _action_id(),
                "tool_id": _bounded_string(
                    _TOOL_ID_PATTERN, max_length=_MAX_GOVERNANCE_IDENTIFIER_LENGTH
                ),
                "kind": _enum(EffectKind),
                "operation_digest": _digest(),
                "idempotency_key": _bounded_string(
                    _IDEMPOTENCY_KEY_PATTERN,
                    min_length=_IDEMPOTENCY_KEY_LENGTH,
                    max_length=_IDEMPOTENCY_KEY_LENGTH,
                ),
                "approval_required": {"type": "boolean"},
                "compensation_limit": _enum(CompensationLimit),
                "state": _enum(EffectState),
                "commit_evidence_digest": _nullable(_digest()),
            }
        ),
        "allOf": [
            _if_then(
                _on_field("state", EffectState.PENDING.value),
                {"properties": {"commit_evidence_digest": {"type": "null"}}},
            ),
            _if_then(
                _on_field("state", EffectState.COMMITTED.value),
                {"properties": {"commit_evidence_digest": _digest()}},
            ),
        ],
    }


def _recovery_checkpoint_schema() -> dict[str, Any]:
    approval_fields = (
        "approval_action_digest",
        "approval_receipt_digest",
        "approval_expires_at",
    )
    properties: dict[str, Any] = {
        "schema_version": {
            "type": "string",
            "const": RECOVERY_CHECKPOINT_SCHEMA_VERSION,
        },
        "workflow_id": _bounded_string(
            _WORKFLOW_ID_PATTERN, max_length=_MAX_GOVERNANCE_IDENTIFIER_LENGTH
        ),
        "run_id": _bounded_string(
            _RUN_ID_PATTERN, min_length=_RUN_ID_LENGTH, max_length=_RUN_ID_LENGTH
        ),
        "sequence": _timestamp(),
        "phase": _enum(RecoveryPhase),
        "plan_digest": _digest(),
        "approval_action_digest": _nullable(_digest()),
        "approval_receipt_digest": _nullable(_digest()),
        "approval_expires_at": _nullable(_timestamp()),
        "previous_checkpoint_digest": _nullable(_digest()),
        "recovery_evidence_digest": _nullable(_digest()),
        "effects": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/effect_record"},
        },
        "checkpoint_digest": _digest(),
    }
    return {
        "$schema": EXCHANGE_RECORD_SCHEMA_DIALECT,
        "$defs": {"effect_record": _effect_record_schema()},
        **_closed_object(properties),
        "allOf": [
            {
                "anyOf": [
                    {
                        "properties": {
                            field_name: {"type": "null"}
                            for field_name in approval_fields
                        }
                    },
                    {
                        "properties": {
                            "approval_action_digest": _digest(),
                            "approval_receipt_digest": _digest(),
                            "approval_expires_at": _timestamp(),
                        }
                    },
                ]
            },
            _if_then(
                _on_field("phase", RecoveryPhase.APPROVAL_RECORDED.value),
                {"properties": {"approval_receipt_digest": _digest()}},
            ),
            _if_then(
                {
                    "properties": {
                        "phase": {
                            "enum": [
                                RecoveryPhase.DISPATCHING.value,
                                RecoveryPhase.COMPLETED.value,
                            ]
                        }
                    },
                    "required": ["phase"],
                },
                _if_then(
                    {
                        "properties": {
                            "effects": {
                                "contains": {
                                    "properties": {
                                        "approval_required": {"const": True}
                                    },
                                    "required": ["approval_required"],
                                }
                            }
                        },
                        "required": ["effects"],
                    },
                    {"properties": {"approval_receipt_digest": _digest()}},
                ),
            ),
            _if_then(
                _on_field("phase", RecoveryPhase.COMPLETED.value),
                {
                    "properties": {
                        "effects": {
                            "items": {
                                "allOf": [
                                    {"$ref": "#/$defs/effect_record"},
                                    {
                                        "properties": {
                                            "state": {
                                                "const": EffectState.COMMITTED.value
                                            }
                                        },
                                        "required": ["state"],
                                    },
                                ]
                            }
                        }
                    }
                },
            ),
            _if_then(
                _on_field("sequence", 0),
                {"properties": {"previous_checkpoint_digest": {"type": "null"}}},
            ),
            _if_then(
                {
                    "properties": {"sequence": {"type": "integer", "minimum": 1}},
                    "required": ["sequence"],
                },
                {"properties": {"previous_checkpoint_digest": _digest()}},
            ),
        ],
    }


def _recovery_decision_schema() -> dict[str, Any]:
    committed_effect = _closed_object(
        {
            "action_id": _action_id(),
            "commit_evidence_digest": _digest(),
        }
    )
    return {
        "$schema": EXCHANGE_RECORD_SCHEMA_DIALECT,
        **_closed_object(
            {
                "schema_version": {
                    "type": "string",
                    "const": RECOVERY_EVIDENCE_SCHEMA_VERSION,
                },
                "disposition": _enum(RecoveryDisposition),
                "reason": _enum(RecoveryReason),
                "source_checkpoint_digest": _digest(),
                "retry_effect_ids": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": _action_id(),
                },
                "retry_idempotency_keys": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": _bounded_string(
                        _IDEMPOTENCY_KEY_PATTERN,
                        min_length=_IDEMPOTENCY_KEY_LENGTH,
                        max_length=_IDEMPOTENCY_KEY_LENGTH,
                    ),
                },
                "committed_effects": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": committed_effect,
                },
                "compensation_effect_ids": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": _action_id(),
                },
                "evidence_digest": _digest(),
            }
        ),
        "allOf": [
            _if_then(
                _on_field("disposition", RecoveryDisposition.RESUME.value),
                {
                    "properties": {
                        "reason": {"const": RecoveryReason.SAFE_TO_RESUME.value},
                        "retry_effect_ids": {"minItems": 1},
                        "compensation_effect_ids": {"maxItems": 0},
                    }
                },
            ),
            _if_then(
                _on_field("disposition", RecoveryDisposition.COMPLETE.value),
                {
                    "properties": {
                        "reason": {
                            "enum": [
                                RecoveryReason.EFFECTS_RECONCILED.value,
                                RecoveryReason.ALREADY_COMPLETE.value,
                            ]
                        },
                        "retry_effect_ids": {"maxItems": 0},
                        "compensation_effect_ids": {"maxItems": 0},
                    }
                },
            ),
            _if_then(
                _on_field("disposition", RecoveryDisposition.REVIEW_REQUIRED.value),
                {
                    "properties": {
                        "reason": {
                            "enum": [
                                RecoveryReason.WORKFLOW_ABORTED.value,
                                RecoveryReason.AMBIGUOUS_EFFECT.value,
                                RecoveryReason.EFFECT_MISMATCH.value,
                                RecoveryReason.APPROVAL_MISSING.value,
                                RecoveryReason.APPROVAL_EXPIRED.value,
                            ]
                        },
                        "retry_effect_ids": {"maxItems": 0},
                    }
                },
            ),
        ],
    }


_EXCHANGE_RECORD_BUILDERS: Final[Mapping[str, Callable[[], dict[str, Any]]]] = (
    MappingProxyType(
        {
            "approval_receipt": _approval_receipt_schema,
            "approval_token": _approval_token_schema,
            "capability_grant": _capability_grant_schema,
            "recovery_checkpoint": _recovery_checkpoint_schema,
            "recovery_decision": _recovery_decision_schema,
            "reviewer_handoff": _reviewer_handoff_schema,
        }
    )
)


def list_exchange_record_schema_names() -> tuple[str, ...]:
    """Return the sorted names of every exported exchange record schema.

    Returns:
        A tuple of stable schema names accepted by
        :func:`build_exchange_record_schema`.
    """

    return tuple(sorted(_EXCHANGE_RECORD_BUILDERS))


def build_exchange_record_schema(name: str) -> dict[str, Any]:
    """Build one self-contained JSON Schema for an exchange record.

    Args:
        name: One of the names returned by
            :func:`list_exchange_record_schema_names`.

    Returns:
        A new JSON-compatible Draft 2020-12 schema mapping. Mutating the
        returned mapping cannot affect a later export.

    Raises:
        ExchangeRecordSchemaError: If ``name`` is not an exported schema name.
            The message never echoes the submitted value.
    """

    if type(name) is not str or name not in _EXCHANGE_RECORD_BUILDERS:
        raise ExchangeRecordSchemaError("unknown exchange record schema")
    return _EXCHANGE_RECORD_BUILDERS[name]()


def build_exchange_record_schema_catalog() -> dict[str, dict[str, Any]]:
    """Build every exported exchange record schema keyed by stable name.

    Returns:
        A new mapping from schema name to a fresh Draft 2020-12 schema.
    """

    return {
        name: _EXCHANGE_RECORD_BUILDERS[name]()
        for name in list_exchange_record_schema_names()
    }


def render_exchange_record_schema(name: str) -> str:
    """Render one exchange record schema as byte-stable compact JSON.

    Args:
        name: One of the names returned by
            :func:`list_exchange_record_schema_names`.

    Returns:
        Canonical JSON text with sorted keys, ASCII escaping, and fixed
        separators.

    Raises:
        ExchangeRecordSchemaError: If ``name`` is not an exported schema name.
    """

    return json.dumps(
        build_exchange_record_schema(name),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def render_exchange_record_schema_catalog() -> str:
    """Render every exchange record schema as byte-stable compact JSON.

    Returns:
        Canonical JSON text for the catalog returned by
        :func:`build_exchange_record_schema_catalog`.
    """

    return json.dumps(
        build_exchange_record_schema_catalog(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "EXCHANGE_RECORD_SCHEMA_DIALECT",
    "ExchangeRecordSchemaError",
    "build_exchange_record_schema",
    "build_exchange_record_schema_catalog",
    "list_exchange_record_schema_names",
    "render_exchange_record_schema",
    "render_exchange_record_schema_catalog",
]
