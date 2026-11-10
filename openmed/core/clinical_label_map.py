"""Explicit model-label projection for the clinical-preserving privacy profile.

Source labels deliberately kept as clinical/context attributes are listed
separately. An unknown model label must never silently become a kept OTHER span.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .labels import normalize_label

CLINICAL_LABEL_MAP_VERSION = "clinical-label-map-v1"
_PREFIX = re.compile(r"^[BIESUL][_-]", re.I)
_IDENTIFIER_ALIASES = {
    "bank_routing_number": "ACCOUNT_NUMBER",
    "biometric_identifier": "ID_NUM",
    "certificate_license_number": "ID_NUM",
    "company_name": "ORGANIZATION",
    "coordinate": "GPS_COORDINATES",
    "customer_id": "ID_NUM",
    "date_time": "DATE",
    "device_identifier": "ID_NUM",
    "employee_id": "ID_NUM",
    "employment_status": "OCCUPATION",
    "fax_number": "PHONE",
    "health_plan_beneficiary_number": "ID_NUM",
    "http_cookie": "ID_NUM",
    "ipv4": "IP_ADDRESS",
    "ipv6": "IP_ADDRESS",
    "race_ethnicity": "ETHNICITY",
    "swift_bic": "BIC",
    "tax_id": "ID_NUM",
    "unique_id": "ID_NUM",
    "vehicle_identifier": "VIN",
}
# The clinical-preserve profile retains these context attributes. OTHER here
# is an explicit, versioned projection, never an unknown-label fallback.
_CONTEXT_ATTRIBUTES = frozenset(
    {
        "blood_type",
        "education_level",
        "language",
        "political_view",
        "religious_belief",
        "sexuality",
    }
)


def clinical_label(label: str) -> str:
    """Resolve a reviewed source/canonical label or reject an unknown label."""
    raw = _PREFIX.sub("", label).strip()
    if raw == "O":
        return "O"
    key = raw.lower()
    if key in _IDENTIFIER_ALIASES:
        return _IDENTIFIER_ALIASES[key]
    if key in _CONTEXT_ATTRIBUTES:
        return "OTHER"
    normalized = normalize_label(raw)
    if normalized == "OTHER":
        raise ValueError("model label has no reviewed clinical-policy mapping")
    return normalized


def clinical_label_map(labels: Iterable[str]) -> dict[str, str]:
    """Validate a complete classifier head before serving any model requests."""
    mapping = {label: clinical_label(label) for label in labels}
    if not mapping or "O" not in mapping.values():
        raise ValueError("classifier label map requires an outside label")
    return mapping


__all__ = ["CLINICAL_LABEL_MAP_VERSION", "clinical_label", "clinical_label_map"]
