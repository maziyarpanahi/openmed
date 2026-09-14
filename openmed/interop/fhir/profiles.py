"""Machine-readable FHIR profile matrix used by the local validator."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .versions import SUPPORTED_RESOURCE_TYPES

__all__ = [
    "PROFILE_MATRIX_PATH",
    "PROFILE_MATRIX",
    "SUPPORTED_PROFILE_MATRIX",
    "get_profile",
    "profile_matrix",
    "validate_profile_matrix",
]

PROFILE_MATRIX_PATH = Path(__file__).with_name("profile_matrix.json")


def profile_matrix() -> dict[str, Any]:
    """Load and validate a fresh copy of the supported-profile matrix."""

    try:
        payload = json.loads(PROFILE_MATRIX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to load FHIR profile matrix: {exc}") from exc
    validate_profile_matrix(payload)
    return copy.deepcopy(payload)


def validate_profile_matrix(payload: Any) -> None:
    """Raise ``ValueError`` when the checked-in profile matrix drifts."""

    if not isinstance(payload, dict):
        raise ValueError("FHIR profile matrix must be a JSON object")
    if payload.get("schema_version") != "1.0":
        raise ValueError("FHIR profile matrix schema_version must be '1.0'")
    resource_types = payload.get("supported_resource_types")
    if not isinstance(resource_types, list):
        raise ValueError("FHIR profile matrix resource types must be an array")
    if set(resource_types) != set(SUPPORTED_RESOURCE_TYPES):
        raise ValueError("FHIR profile matrix resource types do not match the adapter")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("FHIR profile matrix must contain profiles")
    ids: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("FHIR profile matrix entries must be objects")
        profile_id = profile.get("id")
        if not isinstance(profile_id, str) or not profile_id or profile_id in ids:
            raise ValueError("FHIR profile matrix profile ids must be unique")
        ids.add(profile_id)
        if profile.get("fhir_release") not in {"R4", "R5"}:
            raise ValueError(f"unsupported FHIR release in matrix entry {profile_id}")
        if not isinstance(profile.get("fhir_version"), str):
            raise ValueError(f"missing exact FHIR version in matrix entry {profile_id}")
        if not isinstance(profile.get("package"), str) or "#" not in profile["package"]:
            raise ValueError(f"missing package version in matrix entry {profile_id}")
        if not isinstance(profile.get("support"), str) or not profile["support"]:
            raise ValueError(f"missing support statement in matrix entry {profile_id}")


SUPPORTED_PROFILE_MATRIX = profile_matrix()
PROFILE_MATRIX = SUPPORTED_PROFILE_MATRIX


def get_profile(profile_id: str) -> dict[str, Any]:
    """Return one profile entry by stable matrix id."""

    normalized = str(profile_id or "").strip().lower()
    for profile in SUPPORTED_PROFILE_MATRIX["profiles"]:
        if profile["id"].lower() == normalized:
            return copy.deepcopy(profile)
    known = ", ".join(profile["id"] for profile in SUPPORTED_PROFILE_MATRIX["profiles"])
    raise KeyError(f"unknown FHIR profile {profile_id!r}; expected one of {known}")
