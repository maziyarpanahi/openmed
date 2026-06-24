"""Tests for the FHIR ``$de-identify`` operation logic.

All cases run fully offline: the privacy pipeline is replaced by a deterministic
``fake_deidentify`` that maps a fixed set of synthetic PHI tokens to redaction
placeholders, so no model download or network access is required.
"""

from __future__ import annotations

from dataclasses import dataclass

from openmed.interop.fhir_operations import (
    de_identify,
    de_identify_bundle,
    de_identify_resource,
)

# Longest-first so multi-word names redact before their parts.
_PHI_REPLACEMENTS = [
    ("John Doe", "[NAME]"),
    ("123 Main Street", "[ADDRESS]"),
    ("555-0100", "[PHONE]"),
    ("Springfield", "[CITY]"),
    ("Doe", "[LAST]"),
    ("John", "[FIRST]"),
]


@dataclass
class _FakeResult:
    deidentified_text: str


def fake_deidentify(text, *, method="replace", policy="hipaa_safe_harbor"):
    """Deterministic offline stand-in for ``openmed.core.pii.deidentify``."""

    redacted = text
    for needle, replacement in _PHI_REPLACEMENTS:
        redacted = redacted.replace(needle, replacement)
    return _FakeResult(deidentified_text=redacted)


def _patient() -> dict:
    return {
        "resourceType": "Patient",
        "id": "pat-1",
        "text": {
            "status": "generated",
            "div": '<div xmlns="http://www.w3.org/1999/xhtml">'
            "<p>Patient <b>John Doe</b> of Springfield</p></div>",
        },
        "identifier": [{"system": "http://hospital.example/mrn", "value": "MRN-123"}],
        "name": [
            {"use": "official", "text": "John Doe", "family": "Doe", "given": ["John"]}
        ],
        "gender": "male",
        "birthDate": "1980-01-15",
        "telecom": [{"system": "phone", "value": "555-0100", "use": "home"}],
        "address": [{"line": ["123 Main Street"], "city": "Springfield"}],
    }


def _observation() -> dict:
    return {
        "resourceType": "Observation",
        "id": "obs-1",
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "29463-7",
                    "display": "Body Weight",
                }
            ],
            "text": "Body Weight for John Doe",
        },
        "valueQuantity": {
            "value": 72.5,
            "unit": "kg",
            "system": "http://unitsofmeasure.org",
            "code": "kg",
        },
        "note": [{"text": "Recorded by John Doe at 123 Main Street."}],
    }


def test_resource_deidentifies_free_text_and_narrative():
    result = de_identify_resource(_patient(), deidentifier=fake_deidentify)

    assert result["name"][0]["family"] == "[LAST]"
    assert result["name"][0]["given"] == ["[FIRST]"]
    assert result["name"][0]["text"] == "[NAME]"
    assert result["address"][0]["line"] == ["[ADDRESS]"]
    assert result["address"][0]["city"] == "[CITY]"
    assert result["telecom"][0]["value"] == "[PHONE]"
    assert "John Doe" not in result["text"]["div"]
    assert "[NAME]" in result["text"]["div"]
    # Narrative markup is preserved.
    assert result["text"]["div"].startswith("<div")
    assert "<b>" in result["text"]["div"]


def test_resource_leaves_coded_and_structural_elements_untouched():
    result = de_identify_resource(_patient(), deidentifier=fake_deidentify)

    assert result["gender"] == "male"
    assert result["birthDate"] == "1980-01-15"
    assert result["name"][0]["use"] == "official"
    assert result["telecom"][0]["system"] == "phone"
    assert result["identifier"][0]["system"] == "http://hospital.example/mrn"
    assert result["identifier"][0]["value"] == "MRN-123"
    assert result["text"]["status"] == "generated"


def test_observation_coded_values_preserved_but_free_text_redacted():
    result = de_identify_resource(_observation(), deidentifier=fake_deidentify)

    assert result["code"]["coding"][0]["code"] == "29463-7"
    assert result["code"]["coding"][0]["display"] == "Body Weight"
    assert result["code"]["text"] == "Body Weight for John Doe"
    assert result["valueQuantity"]["value"] == 72.5
    assert result["valueQuantity"]["unit"] == "kg"
    assert result["note"][0]["text"] == "Recorded by [NAME] at [ADDRESS]."


def test_input_resource_is_not_mutated():
    original = _patient()
    de_identify_resource(original, deidentifier=fake_deidentify)

    assert original["name"][0]["family"] == "Doe"
    assert "John Doe" in original["text"]["div"]


def test_resource_without_phi_returns_unchanged_copy():
    resource = {
        "resourceType": "Observation",
        "id": "obs-2",
        "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": "1234-5"}]},
    }
    result = de_identify_resource(resource, deidentifier=fake_deidentify)

    assert result == resource
    assert result is not resource


def test_bundle_preserves_structure_and_references():
    bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "fullUrl": "urn:uuid:patient-1",
                "resource": _patient(),
                "request": {"method": "POST", "url": "Patient"},
            },
            {
                "fullUrl": "urn:uuid:obs-1",
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-1",
                    "status": "final",
                    "subject": {
                        "reference": "Patient/pat-1",
                        "display": "John Doe",
                    },
                    "note": [{"text": "Seen by John Doe."}],
                },
            },
        ],
    }

    result = de_identify_bundle(bundle, deidentifier=fake_deidentify)

    assert result["type"] == "transaction"
    assert len(result["entry"]) == 2
    assert result["entry"][0]["fullUrl"] == "urn:uuid:patient-1"
    assert result["entry"][0]["request"] == {"method": "POST", "url": "Patient"}
    obs = result["entry"][1]["resource"]
    assert obs["subject"]["reference"] == "Patient/pat-1"
    assert obs["subject"]["display"] == "[NAME]"
    assert obs["note"][0]["text"] == "Seen by [NAME]."


def test_parameters_envelope_round_trips_policy_and_method():
    parameters = {
        "resourceType": "Parameters",
        "parameter": [
            {"name": "resource", "resource": _patient()},
            {"name": "policy", "valueString": "hipaa_safe_harbor"},
            {"name": "method", "valueCode": "replace"},
        ],
    }

    output = de_identify(parameters, deidentifier=fake_deidentify)

    assert output["resourceType"] == "Parameters"
    by_name = {p["name"]: p for p in output["parameter"]}
    assert by_name["policy"]["valueString"] == "hipaa_safe_harbor"
    assert by_name["method"]["valueCode"] == "replace"
    assert by_name["resource"]["resource"]["name"][0]["family"] == "[LAST]"

    outcome = by_name["outcome"]["resource"]
    assert outcome["resourceType"] == "OperationOutcome"
    expressions = {
        expr for issue in outcome["issue"] for expr in issue.get("expression", [])
    }
    assert "Patient.name[0].family" in expressions
    assert "Patient.text.div" in expressions
    assert all(issue["severity"] == "information" for issue in outcome["issue"])


def test_parameters_defaults_when_policy_and_method_absent():
    parameters = {
        "resourceType": "Parameters",
        "parameter": [{"name": "resource", "resource": _patient()}],
    }

    output = de_identify(parameters, deidentifier=fake_deidentify)

    by_name = {p["name"]: p for p in output["parameter"]}
    assert by_name["policy"]["valueString"] == "hipaa_safe_harbor"
    assert by_name["method"]["valueCode"] == "replace"


def test_parameters_bundle_input_reports_entry_paths():
    parameters = {
        "resourceType": "Parameters",
        "parameter": [
            {
                "name": "bundle",
                "resource": {
                    "resourceType": "Bundle",
                    "type": "collection",
                    "entry": [{"resource": _patient()}],
                },
            }
        ],
    }

    output = de_identify(parameters, deidentifier=fake_deidentify)

    by_name = {p["name"]: p for p in output["parameter"]}
    assert by_name["bundle"]["resource"]["type"] == "collection"
    outcome = by_name["outcome"]["resource"]
    expressions = {
        expr for issue in outcome["issue"] for expr in issue.get("expression", [])
    }
    assert "Bundle.entry[0].resource.name[0].family" in expressions


def test_parameters_with_no_changes_yields_all_ok_outcome():
    parameters = {
        "resourceType": "Parameters",
        "parameter": [
            {
                "name": "resource",
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-3",
                    "status": "final",
                    "code": {"coding": [{"code": "1234-5"}]},
                },
            }
        ],
    }

    output = de_identify(parameters, deidentifier=fake_deidentify)

    by_name = {p["name"]: p for p in output["parameter"]}
    outcome = by_name["outcome"]["resource"]
    assert outcome["issue"] == [
        {
            "severity": "information",
            "code": "informational",
            "diagnostics": "No issues detected.",
        }
    ]
