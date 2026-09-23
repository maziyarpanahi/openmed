#!/usr/bin/env python3
"""Compare synthetic workflow needs with declared SMART v2 scopes offline."""

from __future__ import annotations

import json
from typing import Any

from openmed.interop.smart_scope_audit import audit_smart_scopes

_WORKFLOW_CASES = (
    {
        "workflow_id": "patient-read-summary",
        "description": "Patient context read-only summary workflow.",
        "required_scopes": (
            "patient/SyntheticCondition.r",
            "patient/SyntheticObservation.r",
        ),
        "declared_scopes": (
            "patient/SyntheticCondition.r",
            "patient/SyntheticObservation.r",
        ),
    },
    {
        "workflow_id": "patient-read-missing",
        "description": "Patient context workflow with one missing read scope.",
        "required_scopes": (
            "patient/SyntheticCondition.r",
            "patient/SyntheticObservation.r",
        ),
        "declared_scopes": ("patient/SyntheticObservation.r",),
    },
    {
        "workflow_id": "user-write-overbroad",
        "description": "User context write workflow with search over-claimed.",
        "required_scopes": (
            "user/SyntheticCarePlan.cu",
            "user/SyntheticCondition.r",
        ),
        "declared_scopes": (
            "user/SyntheticCarePlan.cus",
            "user/SyntheticCondition.rs",
        ),
    },
    {
        "workflow_id": "system-export-mixed",
        "description": "System context read workflow with missing and excessive scopes.",
        "required_scopes": (
            "system/SyntheticEncounter.r",
            "system/SyntheticObservation.r",
        ),
        "declared_scopes": (
            "system/SyntheticEncounter.rs",
            "system/SyntheticMedication.r",
        ),
    },
)


def build_report() -> dict[str, Any]:
    """Return deterministic offline SMART scope audit examples."""

    audits = [
        audit_smart_scopes(
            workflow_id=str(case["workflow_id"]),
            required_scopes=case["required_scopes"],
            declared_scopes=case["declared_scopes"],
        ).to_dict()
        | {"description": case["description"]}
        for case in _WORKFLOW_CASES
    ]
    return {
        "schema_version": 1,
        "source": "synthetic-offline-example",
        "privacy": {
            "offline": True,
            "contains_oauth_tokens": False,
            "contains_endpoints": False,
            "contains_launch_context": False,
            "contains_patient_data": False,
        },
        "audits": audits,
    }


def main() -> dict[str, Any]:
    """Print the synthetic SMART scope audit report as JSON."""

    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
