"""Focused drift checks for the checked-in FHIR profile matrix."""

from __future__ import annotations

import json

from openmed.interop.fhir import (
    PROFILE_MATRIX_PATH,
    SUPPORTED_PROFILE_MATRIX,
    profile_matrix,
    validate_profile_matrix,
)


def test_profile_matrix_names_exact_core_and_implementation_guide_versions():
    matrix = profile_matrix()
    by_id = {profile["id"]: profile for profile in matrix["profiles"]}

    assert by_id["fhir-r4-core"]["fhir_version"] == "4.0.1"
    assert by_id["fhir-r5-core"]["fhir_version"] == "5.0.0"
    assert by_id["ips"]["package"] == "hl7.fhir.uv.ips#2.0.1"
    assert by_id["ipa"]["package"] == "hl7.fhir.uv.ipa#1.1.0"
    assert by_id["clinical-document"]["package"] == (
        "hl7.fhir.uv.fhir-clinical-document#1.1.0"
    )


def test_loaded_matrix_matches_checked_in_json_without_drift():
    on_disk = json.loads(PROFILE_MATRIX_PATH.read_text(encoding="utf-8"))

    validate_profile_matrix(on_disk)
    assert on_disk == SUPPORTED_PROFILE_MATRIX
