from __future__ import annotations

import copy
import json

import pytest

from openmed.interop.omop import OmopCdmTables, OmopLoadSummary, load_grounded_notes
from openmed.interop.omop_cohort_check import (
    OmopCohortExportValidationError,
    OmopCohortValidationReport,
    OmopCohortViolation,
    assert_valid_omop_cohort_export,
    check_omop_cohort_export,
    omop_row_fingerprint,
    validate_omop_cohort_export,
)

_SOURCE_NOTE_HASH = "a" * 64


def _synthetic_export() -> dict[str, list[dict[str, object]]]:
    return {
        "concept": [
            {
                "concept_id": 0,
                "vocabulary_id": "UNMAPPED",
                "standard_concept": "",
            },
            {
                "concept_id": 10,
                "vocabulary_id": "SYNTHETIC",
                "standard_concept": "S",
            },
            {
                "concept_id": 20,
                "vocabulary_id": "SYNTHETIC",
                "standard_concept": "",
            },
        ],
        "person": [{"person_id": 1, "person_source_value": "synthetic-person"}],
        "visit_occurrence": [{"visit_occurrence_id": 2, "person_id": 1}],
        "note": [
            {
                "note_id": 3,
                "person_id": 1,
                "visit_occurrence_id": 2,
                "source_note_hash": _SOURCE_NOTE_HASH,
                "note_text": "synthetic note corpus value",
            }
        ],
        "note_nlp": [
            {
                "note_nlp_id": 4,
                "note_id": 3,
                "note_nlp_event_id": 5,
            }
        ],
        "condition_occurrence": [
            {
                "condition_occurrence_id": 5,
                "person_id": 1,
                "condition_concept_id": 10,
                "condition_source_concept_id": 20,
                "visit_occurrence_id": 2,
                "note_id": 3,
                "note_nlp_id": 4,
                "source_note_hash": _SOURCE_NOTE_HASH,
            }
        ],
        "source_to_concept_map": [
            {
                "source_to_concept_map_id": 6,
                "source_code": "SYN-CODE",
                "source_concept_id": 20,
                "source_vocabulary_id": "SYNTHETIC",
                "target_concept_id": 10,
                "target_vocabulary_id": "SYNTHETIC",
                "note_nlp_id": 4,
                "source_note_hash": _SOURCE_NOTE_HASH,
            }
        ],
    }


def test_validates_relationship_vocabulary_and_provenance_invariants() -> None:
    export = _synthetic_export()

    report = validate_omop_cohort_export(export)

    assert report.is_valid
    assert report.violation_count == 0
    assert report.to_dict()["by_table"] == {}
    assert report.to_dict()["by_reason"] == {}
    assert report.to_dict()["row_counts"]["condition_occurrence"] == 1


def test_reports_deterministic_counts_and_fingerprints_without_source_values() -> None:
    export = _synthetic_export()
    broken = copy.deepcopy(export)
    broken["condition_occurrence"][0]["visit_occurrence_id"] = 999
    broken["condition_occurrence"][0]["source_note_hash"] = "b" * 64
    broken["source_to_concept_map"][0]["target_vocabulary_id"] = "OTHER"
    broken["note_nlp"][0]["note_nlp_event_id"] = 999

    report = validate_omop_cohort_export(broken)
    repeated = validate_omop_cohort_export(copy.deepcopy(broken))
    serialized = json.dumps(report.to_dict(), sort_keys=True)

    assert report.is_valid is False
    assert report.violation_count >= 4
    assert report.to_dict() == repeated.to_dict()
    assert "synthetic-note corpus value" not in serialized
    assert "synthetic-person" not in serialized
    assert all(
        fingerprint.startswith("sha256:")
        for violation in report.violations
        for fingerprint in violation.row_fingerprints
    )
    assert (
        "source_note_hash" in report.by_reason
        or "provenance_mismatch" in report.by_reason
    )


def test_checker_accepts_loader_tables_and_aliases() -> None:
    export = _synthetic_export()
    tables = OmopCdmTables(
        tables={name: tuple(rows) for name, rows in export.items()},
        summary=OmopLoadSummary(
            row_counts={name: len(rows) for name, rows in export.items()},
            rejection_counts={},
        ),
    )

    assert (
        check_omop_cohort_export(tables).to_dict()
        == validate_omop_cohort_export(tables.to_dict()).to_dict()
    )
    assert assert_valid_omop_cohort_export(tables).is_valid


def test_checker_accepts_synthetic_loader_output() -> None:
    note_text = "Synthetic finding."
    start = note_text.index("finding")
    tables = load_grounded_notes(
        [
            {
                "document_id": "synthetic-document",
                "person_id": "synthetic-person",
                "visit_id": "synthetic-visit",
                "note_text": note_text,
                "entities": [
                    {
                        "text": "finding",
                        "start": start,
                        "end": start + len("finding"),
                        "domain_id": "Condition",
                        "concept_id": 10,
                        "code": "SYN-1",
                        "vocabulary_id": "SYNTHETIC",
                        "concept_name": "Synthetic finding",
                    }
                ],
            }
        ]
    )

    assert validate_omop_cohort_export(tables).is_valid


def test_row_fingerprints_are_canonical_and_validation_errors_are_phi_free() -> None:
    assert omop_row_fingerprint("person", {"b": 2, "a": 1}) == omop_row_fingerprint(
        "person", {"a": 1, "b": 2}
    )

    broken = _synthetic_export()
    broken["condition_occurrence"][0]["note_id"] = None
    with pytest.raises(OmopCohortExportValidationError) as exc_info:
        assert_valid_omop_cohort_export(broken)

    assert "synthetic" not in str(exc_info.value)
    assert exc_info.value.report.is_valid is False


def test_missing_and_lossy_primary_keys_are_rejected() -> None:
    export = {
        "person": [
            {"person_id": None},
            {"person_id": 1.5},
            {"person_id": 2**63},
        ]
    }

    report = validate_omop_cohort_export(export)

    assert report.by_reason == {
        "invalid_primary_key": 2,
        "missing_primary_key": 1,
    }


def test_duplicate_primary_keys_are_all_flagged_and_not_referenceable() -> None:
    export = {
        "person": [
            {"person_id": 1, "person_source_value": "synthetic-first"},
            {"person_id": 1, "person_source_value": "synthetic-second"},
        ],
        "visit_occurrence": [{"visit_occurrence_id": 2, "person_id": 1}],
    }

    report = validate_omop_cohort_export(export)

    assert report.by_reason == {
        "duplicate_primary_key": 2,
        "missing_reference": 1,
    }
    assert "synthetic-first" not in json.dumps(report.to_dict())
    assert "synthetic-second" not in json.dumps(report.to_dict())


def test_hostile_and_cyclic_rows_fail_without_echoing_source_values() -> None:
    sentinel = "RAW-SYNTHETIC-PATIENT"

    class HostileValue:
        def __str__(self) -> str:
            raise ValueError(sentinel)

    cyclic: dict[str, object] = {}
    cyclic["nested"] = cyclic
    nested: object = sentinel
    for _ in range(65):
        nested = [nested]

    for row in ({"source_value": HostileValue()}, cyclic, {"nested": nested}):
        with pytest.raises(ValueError) as exc_info:
            validate_omop_cohort_export({"person": [row]})
        assert sentinel not in str(exc_info.value)
        assert len(str(exc_info.value)) < 100


def test_hostile_row_iterators_fail_without_echoing_source_values() -> None:
    sentinel = "RAW-SYNTHETIC-PATIENT"

    class HostileRows:
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise ValueError(sentinel)

    with pytest.raises(ValueError) as exc_info:
        validate_omop_cohort_export({"person": HostileRows()})

    assert sentinel not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_fingerprints_preserve_type_distinct_mapping_keys() -> None:
    colliding_keys = {1: "first", "1": "second"}
    string_key_only = {"1": "second"}

    assert omop_row_fingerprint(
        "person", {"nested": colliding_keys}
    ) != omop_row_fingerprint("person", {"nested": string_key_only})


def test_provenance_hashes_must_be_sha256_values() -> None:
    export = _synthetic_export()
    export["note"][0]["source_note_hash"] = "synthetic-not-a-hash"

    report = validate_omop_cohort_export(export)

    assert report.by_reason["invalid_provenance"] >= 1
    assert "synthetic-not-a-hash" not in json.dumps(report.to_dict())


def test_domain_rows_require_target_and_source_concept_references() -> None:
    export = _synthetic_export()
    del export["condition_occurrence"][0]["condition_concept_id"]
    del export["condition_occurrence"][0]["condition_source_concept_id"]

    report = validate_omop_cohort_export(export)

    violations = {(item.column, item.reason): item.count for item in report.violations}
    assert violations[("condition_concept_id", "missing_reference")] == 1
    assert violations[("condition_source_concept_id", "missing_reference")] == 1


def test_note_nlp_event_link_must_resolve_to_one_domain_row() -> None:
    export = _synthetic_export()
    export["drug_exposure"] = [
        {
            "drug_exposure_id": 5,
            "person_id": 1,
            "drug_concept_id": 10,
            "drug_source_concept_id": 20,
            "visit_occurrence_id": 2,
            "note_id": 3,
            "note_nlp_id": 4,
            "source_note_hash": _SOURCE_NOTE_HASH,
        }
    ]

    report = validate_omop_cohort_export(export)

    assert report.by_reason["ambiguous_event"] == 1


def test_public_report_types_reject_untrusted_diagnostic_labels() -> None:
    fingerprint = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match="table is unsupported") as exc_info:
        OmopCohortViolation(
            table="raw-synthetic-patient",
            column=None,
            reason="missing_primary_key",
            count=1,
            row_fingerprints=(fingerprint,),
        )
    assert "raw-synthetic-patient" not in str(exc_info.value)

    with pytest.raises(ValueError, match="unsupported table"):
        OmopCohortValidationReport(
            row_counts={"raw-synthetic-patient": 1},
            violations=(),
        )
