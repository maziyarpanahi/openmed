from __future__ import annotations

import json
from pathlib import Path

from openmed.interop import adapter_spec, available_adapters
from openmed.interop.athena import load_athena_vocab
from openmed.interop.cdm_etl import (
    UNMAPPED_CONCEPT_ID,
    ClinicalEntity,
    ClinicalNote,
    notes_to_cdm,
)

FIXTURES = Path(__file__).with_name("fixtures")


def _synthetic_vocabulary():
    return {
        "LOCAL": {
            "C-1": {
                "concept_id": 101,
                "concept_name": "Synthetic condition",
                "domain_id": "Condition",
                "vocabulary_id": "LOCAL",
                "concept_code": "C-1",
                "aliases": ["condition alpha"],
            },
            "D-1": {
                "concept_id": 202,
                "concept_name": "Synthetic medicine",
                "domain_id": "Drug",
                "vocabulary_id": "LOCAL",
                "concept_code": "D-1",
                "aliases": ["medicine beta"],
            },
            "M-1": {
                "concept_id": 303,
                "concept_name": "Synthetic lab",
                "domain_id": "Measurement",
                "vocabulary_id": "LOCAL",
                "concept_code": "M-1",
                "aliases": ["glucose"],
            },
        },
        "_meta": {"user_supplied": True},
    }


def test_notes_to_cdm_is_deterministic_for_identical_input() -> None:
    notes = [
        ClinicalNote(
            document_id="doc-1",
            patient_id="patient-1",
            visit_id="visit-1",
            document_version="v1",
            entities=[
                ClinicalEntity(
                    label="condition",
                    text="condition alpha",
                    source_entity_id="e-condition",
                    start=10,
                    end=25,
                ),
                ClinicalEntity(
                    label="medication",
                    text="medicine beta",
                    source_entity_id="e-drug",
                    start=40,
                    end=53,
                ),
                ClinicalEntity(
                    label="lab",
                    text="glucose",
                    source_entity_id="e-measurement",
                    start=70,
                    end=77,
                ),
            ],
        )
    ]

    first = notes_to_cdm(notes, vocabulary_index=_synthetic_vocabulary())
    second = notes_to_cdm(notes, vocabulary_index=_synthetic_vocabulary())

    assert first.to_dict() == second.to_dict()
    assert first.condition_occurrence[0].condition_concept_id == 101
    assert first.drug_exposure[0].drug_concept_id == 202
    assert first.measurement[0].measurement_concept_id == 303
    assert first.summary.row_counts == {
        "person": 1,
        "visit_occurrence": 1,
        "condition_occurrence": 1,
        "drug_exposure": 1,
        "measurement": 1,
    }


def test_notes_to_cdm_resolves_aliases_from_loaded_athena_vocab() -> None:
    vocabulary = load_athena_vocab(FIXTURES)
    tables = notes_to_cdm(
        [
            {
                "document_id": "doc-2",
                "patient_id": "patient-2",
                "entities": [
                    {
                        "label": "diagnosis",
                        "text": "Alpha condition",
                        "source_entity_id": "dx-1",
                    },
                    {
                        "label": "drug",
                        "text": "Synthetic medicine beta",
                        "source_entity_id": "rx-1",
                    },
                ],
            }
        ],
        vocabulary_index=vocabulary,
    )

    assert tables.condition_occurrence[0].condition_concept_id == 1001
    assert tables.condition_occurrence[0].source_vocabulary_id == "TESTVOCAB"
    assert tables.drug_exposure[0].drug_concept_id == 2001
    assert tables.drug_exposure[0].source_vocabulary_id == "RXTEST"


def test_notes_to_cdm_uses_unmapped_placeholder_without_vocabulary() -> None:
    tables = notes_to_cdm(
        [
            ClinicalNote(
                document_id="doc-3",
                patient_id="patient-3",
                entities=[
                    ClinicalEntity(label="condition", text="condition alpha"),
                    ClinicalEntity(label="medication", text="medicine beta"),
                    ClinicalEntity(label="measurement", text="glucose"),
                ],
            )
        ]
    )

    assert tables.condition_occurrence[0].condition_concept_id == UNMAPPED_CONCEPT_ID
    assert tables.drug_exposure[0].drug_concept_id == UNMAPPED_CONCEPT_ID
    assert tables.measurement[0].measurement_concept_id == UNMAPPED_CONCEPT_ID
    assert tables.summary.concept_counts == {
        "condition_occurrence": {"mapped": 0, "unmapped": 1},
        "drug_exposure": {"mapped": 0, "unmapped": 1},
        "measurement": {"mapped": 0, "unmapped": 1},
    }


def test_notes_to_cdm_routes_supported_domains_and_skips_unknown_entities() -> None:
    tables = notes_to_cdm(
        [
            {
                "document_id": "doc-4",
                "patient_id": "patient-4",
                "entities": [
                    {"entity_group": "problem", "word": "condition alpha"},
                    {"entity_group": "rx", "word": "medicine beta"},
                    {"entity_group": "vital_sign", "word": "blood pressure"},
                    {"entity_group": "anatomy", "word": "left arm"},
                ],
            }
        ],
        vocabulary_index=_synthetic_vocabulary(),
    )

    assert len(tables.condition_occurrence) == 1
    assert len(tables.drug_exposure) == 1
    assert len(tables.measurement) == 1
    assert tables.summary.row_counts["condition_occurrence"] == 1
    assert tables.summary.row_counts["drug_exposure"] == 1
    assert tables.summary.row_counts["measurement"] == 1


def test_cdm_etl_summary_excludes_identifiers_and_text() -> None:
    tables = notes_to_cdm(
        [
            ClinicalNote(
                document_id="sensitive-doc-id",
                patient_id="sensitive-patient-id",
                entities=[
                    ClinicalEntity(
                        label="condition",
                        text="sensitive condition text",
                        source_entity_id="sensitive-entity-id",
                    )
                ],
            )
        ],
        vocabulary_index=_synthetic_vocabulary(),
    )

    summary_payload = json.dumps(tables.summary.to_dict(), sort_keys=True)

    assert "sensitive-doc-id" not in summary_payload
    assert "sensitive-patient-id" not in summary_payload
    assert "sensitive-entity-id" not in summary_payload
    assert "sensitive condition text" not in summary_payload


def test_cdm_etl_is_available_through_interop_registry() -> None:
    assert "cdm_etl" in available_adapters()
    assert adapter_spec("cdm-etl").module == "openmed.interop.cdm_etl"
