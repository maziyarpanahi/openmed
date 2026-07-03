"""Tests for the committed REST service OpenAPI artifact."""

from __future__ import annotations

import json

import openmed
from scripts.export_openapi import DEFAULT_OUTPUT_PATH, render_openapi_spec

EXPECTED_PATHS = {
    "/health",
    "/models/loaded",
    "/models/unload",
    "/analyze",
    "/fhir/smart-backend/ingestions",
    "/fhir/smart-backend/ingestions/{job_id}",
    "/fhir/smart-backend/ingestions/{job_id}/summary",
    "/pii/extract",
    "/pii/extract/stream",
    "/pii/deidentify",
    "/jobs",
    "/jobs/{job_id}",
}


def test_committed_openapi_spec_matches_exporter() -> None:
    assert DEFAULT_OUTPUT_PATH.exists(), (
        "docs/api/openapi.json is missing. Re-run "
        ".venv/bin/python scripts/export_openapi.py."
    )

    actual = DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8")
    expected = render_openapi_spec().decode("utf-8")

    assert actual == expected, (
        "docs/api/openapi.json is stale. Re-run "
        ".venv/bin/python scripts/export_openapi.py."
    )


def test_committed_openapi_spec_lists_current_service_paths() -> None:
    spec = json.loads(DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))

    assert spec["info"]["version"] == openmed.__version__
    assert EXPECTED_PATHS.issubset(set(spec["paths"]))
