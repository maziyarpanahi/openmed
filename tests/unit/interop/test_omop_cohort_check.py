from __future__ import annotations

import copy
import json

import pytest

from openmed.interop.omop import OmopCdmTables, OmopLoadSummary, load_grounded_notes
from openmed.interop.omop_cohort_check import (
    OmopCohortExportValidationError,
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
