"""Focused offline integration proof for the OpenMed V2.2 exchange surface."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from openmed.eval.v22_conformance import (
    assert_zero_critical_leakage,
    canonical_json,
    run_v22_negative_checks,
    run_v22_reference_flow,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "v22"


def _deny_network(*args: object, **kwargs: object) -> None:
    del args, kwargs
    raise AssertionError("V2.2 conformance fixtures must not open network sockets")


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "create_connection", _deny_network)
    monkeypatch.setattr(socket.socket, "connect", _deny_network)


@pytest.mark.integration
def test_reference_exchange_flow_is_deterministic_offline_and_phi_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _block_network(monkeypatch)
    expected = json.loads(
        (FIXTURES / "expected_reference_hashes.json").read_text(encoding="utf-8")
    )
    form_fixture = json.loads(
        (FIXTURES / "reference_form.json").read_text(encoding="utf-8")
    )
    direct_identifiers = form_fixture["synthetic_direct_identifiers"]

    first = run_v22_reference_flow(FIXTURES, tmp_path / "first")
    second = run_v22_reference_flow(FIXTURES, tmp_path / "second")

    assert first.hashes == second.hashes == expected
    assert first.grounding_audit["code"] == "E11.9"
    assert first.grounding_audit["system_uri"] == ("http://hl7.org/fhir/sid/icd-10-cm")
    assert first.grounding_audit["end"] - first.grounding_audit["start"] == len(
        "type 2 diabetes"
    )
    assert first.evidence["fhir"]["reference_integrity"]["valid"] is True
    assert first.evidence["fhir"]["resource_counts"] == {
        "Composition": 1,
        "Condition": 1,
        "Patient": 1,
    }
    assert first.evidence["omop"]["summary"]["resource_counts"] == {
        "Condition": 1,
        "Patient": 1,
    }
    assert first.evidence["omop"]["summary"]["mapped_codes"] == 1
    assert first.evidence["critical_leakage"] == {
        "maximum_allowed": 0,
        "observed": 0,
        "surfaces": [
            "form review",
            "grounding audit",
            "FHIR serialization",
            "OMOP report",
            "evidence artifact",
            "temporary paths",
        ],
    }

    condition = next(
        entry["resource"]
        for entry in first.fhir_bundle["entry"]
        if entry["resource"]["resourceType"] == "Condition"
    )
    patient_entry = next(
        entry
        for entry in first.fhir_bundle["entry"]
        if entry["resource"]["resourceType"] == "Patient"
    )
    assert condition["code"]["coding"][0]["code"] == "E11.9"
    assert condition["subject"]["reference"] == patient_entry["fullUrl"]

    generated_contents = {
        name: path.read_text(encoding="utf-8")
        for name, path in first.artifact_paths.items()
    }
    for name, path in first.artifact_paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert path.read_text(encoding="utf-8") == canonical_json(payload) + "\n"
        assert name in generated_contents
    leakage_counts = assert_zero_critical_leakage(
        direct_identifiers,
        {
            "review": first.review_artifact,
            "grounding": first.grounding_audit,
            "fhir": first.fhir_bundle,
            "omop": first.omop_report,
            "evidence": first.evidence,
            "files": generated_contents,
            "temporary_paths": [str(path) for path in first.artifact_paths.values()],
            "logs": caplog.text,
        },
    )
    assert set(leakage_counts.values()) == {0}


@pytest.mark.integration
def test_negative_conformance_fixtures_fail_at_safe_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _block_network(monkeypatch)
    results = run_v22_negative_checks(FIXTURES)
    rendered = [result.to_dict() for result in results]

    assert {
        result["case_id"]: (result["category"], result["boundary"])
        for result in rendered
    } == {
        "fhir-r5-unsupported-field": (
            "unsupported_cross_version_field",
            "Observation.unsupportedCrossVersionField",
        ),
        "bulk-data-resume-scope-mismatch": (
            "incompatible_checkpoint",
            "endpoint_scope",
        ),
        "structured-privacy-membership-threshold": (
            "membership_inference_threshold",
            "structured_privacy_policy",
        ),
        "negative-unapproved-state-change": (
            "unapproved_state_change",
            "state_change_policy",
        ),
    }
    assert all(result["status"] == "expected_failure" for result in rendered)
    assert all(result["error_sha256"].startswith("sha256:") for result in rendered)

    sensitive_values: list[str] = []
    for name in (
        "fhir_r5_negative.json",
        "bulk_resume_negative.json",
        "structured_privacy_negative.json",
        "mcp_authorization_negative.json",
    ):
        fixture = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        sensitive_values.extend(fixture["sensitive_values"])
    counts = assert_zero_critical_leakage(
        sensitive_values,
        {"safe_results": rendered, "logs": caplog.text},
    )
    assert set(counts.values()) == {0}
