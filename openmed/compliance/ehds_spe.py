"""Offline technical check of a declared secure processing environment profile."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_BOOLEAN_CONTROLS = (
    "outbound_egress_denied",
    "authentication_enabled",
    "audit_recording_enabled",
    "encryption_at_rest",
    "access_logging_enabled",
    "minimum_necessary_configured",
)
_EXPORT_FORMATS = {"anonymised", "aggregate"}
_NOTICE = "Technical check of declared controls only. This report provides neither a legal determination nor a data permit."


def check_ehds_spe(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Check seven deployment declarations and return value-free results.

    Boolean controls require an explicit JSON boolean. The export control
    accepts an explicit list of permitted output classes. An empty list means
    all exports are disabled. Absent declarations remain unassessed.
    """
    if not isinstance(profile, Mapping):
        raise ValueError("invalid_deployment_profile")
    controls = {}
    for code in (*_BOOLEAN_CONTROLS, "export_classes"):
        if code not in profile:
            status, reason = "not_assessed", "control_undeclared"
        elif code == "export_classes":
            values = profile[code]
            if not isinstance(values, list) or not all(
                type(value) is str for value in values
            ):
                status, reason = "fail", "invalid_declaration"
            elif all(value in _EXPORT_FORMATS for value in values):
                status, reason = "pass", "exports_restricted"
            else:
                status, reason = "fail", "export_class_not_permitted"
        elif type(profile[code]) is not bool:
            status, reason = "fail", "invalid_declaration"
        elif profile[code]:
            status, reason = "pass", "control_enabled"
        else:
            status, reason = "fail", "control_disabled"
        controls[code] = {"status": status, "reason": reason}
    statuses = {control["status"] for control in controls.values()}
    overall = (
        "fail"
        if "fail" in statuses
        else "not_assessed"
        if "not_assessed" in statuses
        else "pass"
    )
    return {
        "schema_version": "1.0.0",
        "status": overall,
        "controls": controls,
        "notice": _NOTICE,
    }


def load_deployment_profile(path: Path) -> dict:
    """Load a local declaration, rejecting ambiguous duplicate JSON keys."""

    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("invalid_deployment_profile")
        return result

    try:
        profile = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
        if not isinstance(profile, dict):
            raise ValueError
    except (OSError, ValueError, RecursionError):
        raise ValueError("invalid_deployment_profile") from None
    return profile
