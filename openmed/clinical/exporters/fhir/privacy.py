"""Leakage-safe handling for Indian health identifiers in FHIR resources."""

from __future__ import annotations

import re
from typing import Any, Mapping

from openmed.core.pii_i18n import (
    validate_abha_address,
    validate_abha_number,
    validate_indian_ration_card,
    validate_upi_id,
)

INDIA_HEALTH_ID_REDACTION = "[REDACTED-INDIA-HEALTH-ID]"

_IDENTIFIER_KEYS = frozenset({"identifier"})
_PATIENT_ID_KEYS = frozenset({"patientid", "patientidentifier"})
_RATION_SYSTEM_CUES = ("ration", "pds", "public-distribution")
_EXPLICIT_RATION_CARD = re.compile(
    r"[A-Za-z]{1,3}[\s/-]\d{8,12}(?:[\s/-][A-Za-z0-9]{1,4})?"
)


def is_india_health_identifier(value: str, *, system: str = "") -> bool:
    """Return whether ``value`` is an Indian health-adjacent identifier."""

    candidate = value.strip()
    if not candidate:
        return False
    if (
        validate_abha_number(candidate)
        or validate_abha_address(candidate)
        or validate_upi_id(candidate)
    ):
        return True
    if not validate_indian_ration_card(candidate):
        return False
    if _EXPLICIT_RATION_CARD.fullmatch(candidate):
        return True
    normalized_system = system.casefold()
    return any(cue in normalized_system for cue in _RATION_SYSTEM_CUES)


def sanitize_india_health_identifiers(resource: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copy a FHIR resource and redact India IDs in identifier fields.

    Only FHIR ``identifier`` containers and PatientID-style fields are in
    scope. Narrative and clinical ``value`` fields are preserved, while a
    matching identifier surface is replaced with a stable non-secret marker.
    """

    return _sanitize_node(resource)


def _sanitize_node(node: Any) -> Any:
    if isinstance(node, Mapping):
        result: dict[str, Any] = {}
        for key, value in node.items():
            normalized_key = _field_key(str(key))
            if normalized_key in _IDENTIFIER_KEYS:
                result[str(key)] = _sanitize_identifier_container(value)
            elif normalized_key in _PATIENT_ID_KEYS:
                result[str(key)] = _sanitize_patient_id(value)
            else:
                result[str(key)] = _sanitize_node(value)
        return result
    if isinstance(node, (list, tuple)):
        return [_sanitize_node(item) for item in node]
    return node


def _sanitize_identifier_container(value: Any) -> Any:
    if isinstance(value, Mapping):
        system = str(value.get("system") or "")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if _field_key(str(key)) == "value" and isinstance(item, str):
                result[str(key)] = (
                    INDIA_HEALTH_ID_REDACTION
                    if is_india_health_identifier(item, system=system)
                    else item
                )
            else:
                result[str(key)] = _sanitize_identifier_container(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize_identifier_container(item) for item in value]
    if isinstance(value, str) and is_india_health_identifier(value):
        return INDIA_HEALTH_ID_REDACTION
    return value


def _sanitize_patient_id(value: Any) -> Any:
    if isinstance(value, str):
        if is_india_health_identifier(value):
            return INDIA_HEALTH_ID_REDACTION
        return value
    return _sanitize_identifier_container(value)


def _field_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


__all__ = [
    "INDIA_HEALTH_ID_REDACTION",
    "is_india_health_identifier",
    "sanitize_india_health_identifiers",
]
