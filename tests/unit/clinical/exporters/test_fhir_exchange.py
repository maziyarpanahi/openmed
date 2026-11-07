"""Synthetic end-to-end tests for the FHIR clinical exchange workbench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from openmed.clinical.exporters.fhir import (
    FHIRExchangeWorkbench,
    build_clinical_document,
    build_ipa_patient_access,
    build_ips_patient_summary,
    deidentify_fhir,
    validate_exchange,
)

FIXTURE_ROOT = Path(__file__).parents[3] / "fixtures" / "fhir"


@dataclass
class _FakeResult:
    deidentified_text: str


def _fake_deidentify(
    text: str, *, method: str = "mask", policy: str = "hipaa_safe_harbor"
) -> _FakeResult:
    del method, policy
    redacted = text
    for source, replacement in (
        ("Avery Example", "[NAME]"),
        ("Avery", "[FIRST]"),
        ("Example", "[LAST]"),
    ):
        redacted = redacted.replace(source, replacement)
    return _FakeResult(redacted)


def _load(name: str) -> dict:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _resources(bundle: dict) -> list[dict]:
    return [entry["resource"] for entry in bundle["entry"]]


def test_r4_ips_import_deidentify_export_has_no_source_identifiers_and_keeps_codes():
    source = _load("synthetic_ips_r4.json")
    deidentified = deidentify_fhir(
        source,
        deidentifier=_fake_deidentify,
        document_id="ips-test",
    )
    serialized = json.dumps(deidentified, sort_keys=True)

    assert "Avery Example" not in serialized
    assert "SYNTH-MRN-001" not in serialized
    assert "ips-bundle-original" not in serialized
    assert "patient-original" not in serialized
    assert "urn:uuid:synthetic-patient" not in serialized
    assert "44054006" in serialized
    assert "4548-4" in serialized
    assert deidentified["entry"][2]["resource"]["subject"]["reference"].startswith(
        "Patient/openmed-"
    )
    assert deidentified["entry"][7]["resource"]["target"][0]["reference"].startswith(
        "Condition/openmed-"
    )
    assert "[NAME]" in serialized


def test_document_and_patient_access_builders_are_profile_checked():
    ips = build_ips_patient_summary(_load("synthetic_ips_r4.json"))
    clinical = build_clinical_document(_load("synthetic_clinical_document_r4.json"))
    ipa = build_ipa_patient_access(_load("synthetic_ips_r4.json"))

    assert ips["type"] == "document"
    assert ips["entry"][0]["resource"]["resourceType"] == "Composition"
    assert (
        validate_exchange(ips, profile="ips")["issue"][0]["severity"] == "information"
    )

    assert clinical["type"] == "document"
    assert clinical["entry"][0]["resource"]["resourceType"] == "Composition"
    assert (
        validate_exchange(clinical, profile="clinical-document")["issue"][0]["severity"]
        == "information"
    )

    assert ipa["type"] == "searchset"
    assert ipa["total"] == len(ipa["entry"])
    assert (
        validate_exchange(ipa, profile="ipa")["issue"][0]["severity"] == "information"
    )


def test_workbench_keeps_explicit_source_and_target_releases():
    source = _load("synthetic_ips_r4.json")
    workbench = FHIRExchangeWorkbench("R4", target_version="R5", document_id="wb-test")

    imported = workbench.import_bundle(source, validate_input=True)
    converted = workbench.convert(imported)

    assert converted["resourceType"] == "Bundle"
    assert converted["entry"][0]["resource"]["resourceType"] == "Composition"
    assert (
        converted["entry"][4]["resource"]["medication"]["concept"]["coding"][0]["code"]
        == "860975"
    )
