"""Quality-floor integration tests for downstream ETL entry points."""

from __future__ import annotations

import pytest

from openmed.interop.cdm_etl import notes_to_cdm
from openmed.interop.omop import load_grounded_notes
from openmed.structured.quality import QualityGateError


def _good_record() -> dict[str, object]:
    return {
        "note_id": "synthetic-note",
        "person_id": "synthetic-person",
        "note_text": "x",
        "entities": [
            {
                "label": "condition",
                "text": "x",
                "start": 0,
                "end": 1,
                "concept_id": 123,
            }
        ],
    }


def _bad_record() -> dict[str, object]:
    return {
        "note_id": "synthetic-note",
        "person_id": "synthetic-person",
        "entities": [],
    }


def test_omop_loader_blocks_batches_below_quality_floor() -> None:
    with pytest.raises(QualityGateError) as error:
        load_grounded_notes(
            [_bad_record()],
            completeness_floor=0.5,
            required_fields=("condition",),
        )

    assert error.value.report["gate"]["passed"] is False
    assert error.value.report["completeness"]["note_count"] == 1


def test_cdm_etl_applies_the_same_quality_floor_and_allows_valid_batch() -> None:
    tables = notes_to_cdm(
        [
            {
                "document_id": "synthetic-document",
                "patient_id": "synthetic-patient",
                "entities": _good_record()["entities"],
            }
        ],
        completeness_floor=0.5,
        required_fields=("condition",),
    )
    assert tables.summary.row_counts["condition_occurrence"] == 1

    with pytest.raises(QualityGateError):
        notes_to_cdm(
            [
                {
                    "document_id": "synthetic-document",
                    "patient_id": "synthetic-patient",
                    "entities": [],
                }
            ],
            completeness_floor=0.5,
            required_fields=("condition",),
        )
