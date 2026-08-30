from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from openmed.interop.fhir_omop import (
    FhirOmopMappingError,
    build_conformance_report,
    inspect_fhir_from_omop,
    load_fhir_bundle,
    write_fhir_omop_sqlite,
)
from openmed.interop.omop import validate_omop_tables

FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "interop"
BUNDLE_PATH = FIXTURE_DIR / "fhir_omop_bundle.json"
VOCABULARY_PATH = FIXTURE_DIR / "fhir_omop_vocabulary.json"
NEGATIVE_PATH = FIXTURE_DIR / "fhir_omop_fabricated_concept.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle() -> dict[str, Any]:
    return _read_json(BUNDLE_PATH)


def _vocabulary() -> dict[str, Any]:
    return _read_json(VOCABULARY_PATH)


def _mapping(bundle: dict[str, Any]) -> list[tuple[str, str]]:
    values = []
    for entry in bundle["entry"]:
        resource = entry["resource"]
        resource_type = resource["resourceType"]
        path = {
            "Condition": "code",
            "Observation": "code",
            "MedicationRequest": "medicationCodeableConcept",
            "Procedure": "code",
        }.get(resource_type)
        if path is None:
            continue
        for coding in resource[path]["coding"]:
            values.append((coding["system"], coding["code"]))
    return values


def test_load_is_deterministic_and_keeps_standard_source_concepts_distinct() -> None:
    bundle = _bundle()
    vocabulary = _vocabulary()
    first = load_fhir_bundle(
        bundle,
        vocabulary=vocabulary,
        vocabulary_snapshot="synthetic-v1",
        source_system="synthetic-fhir-source",
        source_version="2026-r4",
    )
    second = load_fhir_bundle(
        bundle,
        vocabulary=vocabulary,
        vocabulary_snapshot="synthetic-v1",
        source_system="synthetic-fhir-source",
        source_version="2026-r4",
    )

    assert first.row_counts == second.row_counts
    assert first.keys == second.keys
    assert first.summary.resource_counts == {
        "Condition": 1,
        "Encounter": 1,
        "MedicationRequest": 1,
        "Observation": 1,
        "Patient": 1,
        "Procedure": 1,
    }
    assert validate_omop_tables(first.omop) == ()
    condition = first.table("condition_occurrence")[0]
    assert condition["condition_concept_id"] == 201826
    assert condition["condition_source_concept_id"] == 9001001
    assert first.table("source_to_concept_map")[0]["vocabulary_version"] == (
        "synthetic-v1"
    )
    assert first.provenance[0]["source_system"] == "synthetic-fhir-source"
    assert first.provenance[0]["source_version"] == "2026-r4"

    concept_ids = {row["concept_id"] for row in first.table("concept")}
    for table in first.omop.tables.values():
        for row in table:
            for column, value in row.items():
                if value is not None and (
                    column.endswith("_concept_id")
                    or column in {"source_concept_id", "target_concept_id"}
                ):
                    assert value in concept_ids


def test_unmapped_code_is_concept_zero_with_a_reason_and_hash_only_traceability() -> (
    None
):
    loaded = load_fhir_bundle(
        _bundle(),
        vocabulary=_vocabulary(),
        vocabulary_snapshot="synthetic-v1",
    )
    procedure_code = next(
        row for row in loaded.codes if row["resource_type"] == "Procedure"
    )
    assert procedure_code["target_concept_id"] == 0
    assert procedure_code["unmapped_reason"] == "no_user_supplied_mapping"
    procedure_map = next(
        row
        for row in loaded.table("source_to_concept_map")
        if row["source_code"] == "80146002"
    )
    assert procedure_map["invalid_reason"] == "no_user_supplied_mapping"
    assert procedure_map["target_concept_id"] == 0

    safe_payload = json.dumps(loaded.to_dict(), sort_keys=True)
    for identifier in (
        "patient-synthetic-001",
        "encounter-synthetic-001",
        "condition-synthetic-001",
        "observation-synthetic-001",
        "medication-synthetic-001",
        "procedure-synthetic-001",
    ):
        assert identifier not in safe_payload
    assert all(len(row["source_id_hash"]) == 64 for row in loaded.resources)
    assert all(row["element_path"] for row in loaded.provenance)
    assert all(
        row["vocabulary_snapshot"] == "synthetic-v1" for row in loaded.provenance
    )


def test_reverse_inspection_reproduces_codings_and_reports_defined_loss() -> None:
    loaded = load_fhir_bundle(
        _bundle(),
        vocabulary=_vocabulary(),
        vocabulary_snapshot="synthetic-v1",
    )
    reverse = inspect_fhir_from_omop(loaded)
    resources = {resource["resourceType"]: resource for resource in reverse.resources}

    assert resources["Condition"]["code"] == _bundle()["entry"][2]["resource"]["code"]
    assert resources["Observation"]["code"] == _bundle()["entry"][3]["resource"]["code"]
    assert resources["Observation"]["valueQuantity"]["value"] == 8.2
    assert resources["Encounter"]["class"] == _bundle()["entry"][1]["resource"]["class"]
    assert (
        resources["Encounter"]["period"] == _bundle()["entry"][1]["resource"]["period"]
    )
    assert resources["Encounter"]["subject"]["reference"].startswith("Patient/")
    assert "name" not in resources["Patient"]
    assert all(
        identifier not in json.dumps(reverse.to_dict(), sort_keys=True)
        for identifier in ("patient-synthetic-001", "condition-synthetic-001")
    )
    loss_paths = {finding.element_path for finding in reverse.information_loss}
    assert "Patient.name" in loss_paths
    assert "Condition.id" in loss_paths
    assert "Observation.subject.reference" in loss_paths


def test_fabricated_concept_fixture_fails_and_report_is_negative() -> None:
    negative = _read_json(NEGATIVE_PATH)

    with pytest.raises(FhirOmopMappingError):
        load_fhir_bundle(negative, vocabulary=_vocabulary())

    report = build_conformance_report(negative, vocabulary=_vocabulary())
    assert not report.passed
    assert report.errors == ("fabricated_concept_identifier",)


def test_sqlite_writer_is_idempotent_for_omop_and_sidecar_rows(tmp_path: Path) -> None:
    loaded = load_fhir_bundle(
        _bundle(),
        vocabulary=_vocabulary(),
        vocabulary_snapshot="synthetic-v1",
    )
    con = write_fhir_omop_sqlite(loaded, tmp_path / "fhir-omop.sqlite")
    write_fhir_omop_sqlite(loaded, con)

    assert isinstance(con, sqlite3.Connection)
    for table, count in loaded.row_counts.items():
        assert con.execute(f"SELECT count(*) FROM {table}").fetchone() == (count,)
    assert con.execute("SELECT count(*) FROM fhir_resource").fetchone() == (
        len(loaded.resources),
    )
    assert con.execute("SELECT count(*) FROM fhir_provenance").fetchone() == (
        len(loaded.provenance),
    )
    assert con.execute(
        "SELECT count(*) FROM fhir_provenance WHERE source_resource_hash LIKE '%patient-synthetic%'"
    ).fetchone() == (0,)


def test_conformance_report_covers_reload_foreign_keys_traceability_and_reverse() -> (
    None
):
    report = build_conformance_report(
        _bundle(),
        vocabulary=_vocabulary(),
        vocabulary_snapshot="synthetic-v1",
    )

    assert report.passed
    assert {check["name"] for check in report.checks} == {
        "deterministic_reload_keys",
        "concept_foreign_keys_resolve",
        "source_traceability",
        "explicit_unmapped_reasons",
        "reverse_supported_coded_content",
    }
    assert "Patient.name" in report.unsupported_elements


def test_missing_vocabulary_never_infers_a_concept_id() -> None:
    loaded = load_fhir_bundle(_bundle())

    assert loaded.summary.mapped_codes == 0
    assert all(row["target_concept_id"] == 0 for row in loaded.codes)
    assert all(row["unmapped_reason"] for row in loaded.codes)
    assert _mapping(_bundle())
