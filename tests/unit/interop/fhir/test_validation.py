"""Offline validation tests use only committed synthetic fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.interop.fhir import validate_document

FIXTURE_ROOT = Path(__file__).parents[3] / "fixtures" / "fhir"


def _load(name: str) -> dict:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def test_synthetic_ips_fixture_validates_locally():
    outcome = validate_document(_load("synthetic_ips_r4.json"), profile="ips")

    assert outcome["issue"] == [
        {
            "severity": "information",
            "code": "informational",
            "diagnostics": "No issues detected.",
        }
    ]


def test_synthetic_clinical_document_fixture_validates_locally():
    outcome = validate_document(
        _load("synthetic_clinical_document_r4.json"),
    )

    assert outcome["issue"] == [
        {
            "severity": "information",
            "code": "informational",
            "diagnostics": "No issues detected.",
        }
    ]
