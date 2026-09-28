"""Golden Journey coverage for correction-aware OMOP 5.4 projection."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    EvidenceLocator,
    canonical_digest,
    sha256_digest,
)
from openmed.interop.omop import (
    OmopConceptMapping,
    OmopFactProjectionInput,
    OmopVocabularySnapshot,
    project_clinical_facts_to_omop,
    validate_omop_fact_projection,
)
from openmed.structured.store import SQLiteJourneyStore

pytestmark = pytest.mark.integration

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "interop"
    / "omop"
    / "fact_projection.json"
)


def _fixture_inputs() -> tuple[
    dict[str, object],
    OmopVocabularySnapshot,
    tuple[OmopFactProjectionInput, ...],
]:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    inputs = []
    for raw in fixture["facts"]:
        record = dict(raw)
        mapping = OmopConceptMapping.from_dict(record.pop("mapping"))
        inputs.append(
            OmopFactProjectionInput(
                fact=ClinicalFact.from_dict(record),
                source_key=fixture["source_key"],
                source_revision=fixture["source_revision"],
                mapping=mapping,
                dataset_split=fixture["dataset_split"],
            )
        )
    return (
        fixture,
        OmopVocabularySnapshot.from_dict(fixture["vocabulary_snapshot"]),
        tuple(inputs),
    )


def test_current_journey_facts_project_and_correct_without_duplicate_rows(
    tmp_path: Path,
) -> None:
    fixture, snapshot, inputs = _fixture_inputs()
    store = SQLiteJourneyStore(tmp_path / "journey-omop.sqlite3")
    artifact = ClinicalArtifact(
        artifact_id="artifact_aaaaaaaaaaaaaaaa",
        artifact_type="clinical_note",
        media_type="text/plain",
        content_hash=sha256_digest("synthetic-omop-golden-source"),
        byte_size=len("synthetic-omop-golden-source"),
        source_id=fixture["source_key"],
        recorded_at=fixture["occurred_at"],
        subject_id=inputs[0].fact.subject_id,
        encounter_id=inputs[0].fact.encounter_id,
    )
    with store.transaction(committed_at=fixture["occurred_at"]) as transaction:
        assert transaction.put_artifact(artifact).ok
        for index, item in enumerate(inputs):
            assert transaction.put_evidence(
                EvidenceLocator(
                    locator_id=item.fact.evidence_ids[0],
                    artifact_id=artifact.artifact_id,
                    location_type="text_span",
                    location={"start": index * 10, "end": index * 10 + 5},
                )
            ).ok
            assert transaction.put_fact(item.fact).ok
    current = store.list_facts(inputs[0].fact.subject_id)
    assert current.ok and current.value is not None
    inputs_by_id = {item.fact.fact_id: item for item in inputs}
    stored_inputs = tuple(
        replace(inputs_by_id[fact.fact_id], fact=fact) for fact in current.value
    )

    first = project_clinical_facts_to_omop(
        stored_inputs,
        vocabulary_snapshot=snapshot,
        etl_version=fixture["etl_version"],
        occurred_at=fixture["occurred_at"],
    )
    assert first.ok and first.value is not None
    assert first.value.summary.current_fact_count == 5
    assert not validate_omop_fact_projection(first.value)

    source = inputs[0]
    corrected_fact = replace(
        source.fact,
        fact_id="fact_9999999999999999",
        value={"code": "synthetic-condition-amended", "system": "synthetic"},
        status="corrected",
        derivation_hash=canonical_digest({"kind": "synthetic-correction"}),
        parent_fact_ids=(source.fact.fact_id,),
    )
    assert store.put_fact(
        corrected_fact,
        committed_at="2026-09-21T11:00:00Z",
    ).ok
    correction = replace(
        source,
        fact=corrected_fact,
        source_revision="sha256:9999999999999999999999999999999999999999999999999999999999999999",
        mapping=replace(
            source.mapping,
            source_code="synthetic-condition-amended",
            source_concept_id=1901,
            standard_concept_id=2901,
            standard_code="condition-amended-standard",
        ),
    )
    updated = project_clinical_facts_to_omop(
        (correction,),
        vocabulary_snapshot=snapshot,
        etl_version=fixture["etl_version"],
        occurred_at="2026-09-21T11:00:00Z",
        previous=first.value,
    )

    assert updated.ok and updated.value is not None
    assert updated.value.current_fact_ids == (corrected_fact.fact_id,)
    assert updated.value.summary.row_counts["condition_occurrence"] == 1
    assert updated.value.summary.row_counts["source_to_concept_map"] == 1
    assert len(updated.value.etl_runs) == 2
    assert not validate_omop_fact_projection(updated.value)
    store.close()
