"""Field-level schema-policy tests with synthetic offline resources."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from openmed.core.date_shift import stable_offset_for
from openmed.structured.schema_policy import (
    apply_omop_file,
    apply_schema_policy,
    lint_schema_policy,
    list_schema_policies,
    load_schema_policy,
    validate_schema_policy,
)
from openmed.structured.table_io import read_table, write_table

FIXTURES = Path(__file__).parents[2] / "fixtures" / "structured"
DATE_SECRET = b"synthetic-schema-policy-test-key"


@dataclass(frozen=True)
class _FakeResult:
    deidentified_text: str


def _fake_deidentify(text, *, method="replace", policy="hipaa_safe_harbor"):
    assert method == "replace"
    assert policy in {"hipaa_safe_harbor", "research_limited_dataset"}
    redacted = text.replace("Avery Example", "[NAME]")
    redacted = redacted.replace("555-0100", "[PHONE]")
    return _FakeResult(redacted)


def _load_bundle() -> dict:
    return json.loads(
        (FIXTURES / "fhir_patient_observation_bundle.json").read_text(encoding="utf-8")
    )


def _date_offset(before: str, after: str) -> int:
    return (date.fromisoformat(after[:10]) - date.fromisoformat(before[:10])).days


def test_bundled_schema_policies_are_valid_and_reviewable():
    assert list_schema_policies() == (
        "fhir_hipaa_safe_harbor",
        "fhir_research_limited_dataset",
        "omop_hipaa_safe_harbor",
        "omop_research_limited_dataset",
    )

    for name in list_schema_policies():
        policy = load_schema_policy(name)
        assert policy.name == name
        assert policy.rules
        assert validate_schema_policy(policy) == ()


def test_apply_schema_policy_to_fhir_bundle_dispatches_every_action():
    bundle = _load_bundle()

    transformed = apply_schema_policy(
        bundle,
        "fhir_hipaa_safe_harbor",
        date_shift_secret=DATE_SECRET,
        deidentifier=_fake_deidentify,
    )

    assert transformed is not bundle
    patient = transformed["entry"][0]["resource"]
    observation = transformed["entry"][1]["resource"]

    assert "name" not in patient
    assert "telecom" not in patient
    assert patient["identifier"] == [{"system": "https://example.invalid/mrn"}]
    assert patient["address"] == [
        {
            "state": "MA",
            "postalCode": "021",
            "country": "US",
        }
    ]
    assert observation["subject"] == {"reference": "Patient/synthetic-patient-1"}
    assert observation["code"]["text"] == "Result discussed with [NAME]"
    assert observation["note"][0]["text"] == "[NAME] called [PHONE]."

    patient_offset = _date_offset("1980-01-15", patient["birthDate"])
    observation_offset = _date_offset(
        "2024-03-20T09:30:00Z",
        observation["effectiveDateTime"],
    )
    assert patient_offset == observation_offset
    assert patient_offset == stable_offset_for(
        "synthetic-patient-1",
        max_days=365,
        secret=DATE_SECRET,
    )

    serialized = json.dumps(transformed, sort_keys=True)
    for direct_identifier in (
        "Avery Example",
        "555-0100",
        "SYN-MRN-001",
        "100 Test Avenue",
        "Exampleton",
    ):
        assert direct_identifier not in serialized
    assert lint_schema_policy(bundle, "fhir_hipaa_safe_harbor") == ()
    assert bundle["entry"][0]["resource"]["name"][0]["text"] == "Avery Example"


def test_apply_schema_policy_to_linked_omop_rows_is_subject_consistent():
    tables = {
        "person": read_table(FIXTURES / "omop_person.csv"),
        "visit_occurrence": read_table(FIXTURES / "omop_visit_occurrence.csv"),
    }

    transformed = apply_schema_policy(
        tables,
        "omop_hipaa_safe_harbor",
        date_shift_secret=DATE_SECRET,
        deidentifier=_fake_deidentify,
    )

    person = transformed["person"][0]
    visit = transformed["visit_occurrence"][0]
    assert person["person_id"] == "1001"
    assert visit["person_id"] == "1001"
    assert "month_of_birth" not in person
    assert "day_of_birth" not in person
    assert "location_id" not in person
    assert person["person_source_value"] == "Synthetic [NAME]"
    assert visit["visit_source_value"] == "Visit for Synthetic [NAME]"

    birth_offset = _date_offset("1980-01-15", person["birth_datetime"])
    visit_date_offset = _date_offset("2024-03-20", visit["visit_start_date"])
    visit_datetime_offset = _date_offset(
        "2024-03-20T09:30:00",
        visit["visit_start_datetime"],
    )
    assert birth_offset == visit_date_offset == visit_datetime_offset
    assert birth_offset == stable_offset_for("1001", max_days=365, secret=DATE_SECRET)

    serialized = json.dumps(transformed, sort_keys=True)
    assert "Avery Example" not in serialized
    assert lint_schema_policy(tables, "omop_hipaa_safe_harbor") == ()


def test_uncovered_identifier_fails_closed_and_lint_reports_uncovered_fields():
    patient = {
        "resourceType": "Patient",
        "id": "synthetic-patient-2",
        "national_id": "synthetic-national-id",
        "customClinicalField": "non-identifying value",
    }

    findings = lint_schema_policy(patient, "fhir_hipaa_safe_harbor")
    assert [(finding.code, finding.path, finding.severity) for finding in findings] == [
        ("uncovered-field", "Patient.customClinicalField", "warning"),
        ("uncovered-identifier", "Patient.national_id", "error"),
    ]

    transformed = apply_schema_policy(patient, "fhir_hipaa_safe_harbor")
    assert transformed == {
        "resourceType": "Patient",
        "id": "synthetic-patient-2",
        "customClinicalField": "non-identifying value",
    }


def test_validate_policy_reports_unknown_and_uncovered_schema_paths():
    policy = {
        "schema_version": 1,
        "name": "synthetic_validation_policy",
        "schema": "fhir",
        "base_policy": "hipaa_safe_harbor",
        "fields": {"Patient.typoField": "keep"},
    }

    findings = validate_schema_policy(
        policy,
        {"Patient": ("resourceType", "id")},
    )

    assert {finding.code for finding in findings} == {
        "unknown-policy-field",
        "uncovered-schema-field",
    }
    assert any(
        finding.path == "Patient.id" and finding.severity == "error"
        for finding in findings
    )


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_apply_omop_file_supports_csv_and_parquet(tmp_path, suffix):
    rows = read_table(FIXTURES / "omop_visit_occurrence.csv")
    source = tmp_path / f"visit_occurrence{suffix}"
    destination = tmp_path / f"deidentified{suffix}"
    write_table(source, rows)

    result = apply_omop_file(
        source,
        destination,
        "omop_hipaa_safe_harbor",
        date_shift_secret=DATE_SECRET,
        deidentifier=_fake_deidentify,
    )

    assert result == destination
    output = read_table(destination)
    assert output[0]["visit_source_value"] == "Visit for Synthetic [NAME]"
    assert output[0]["visit_start_date"] != "2024-03-20"
