"""Tests for deterministic ClinicalFact projection into OMOP 5.4 rows."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from jsonschema import Draft202012Validator

from openmed.clinical.exporters.omop import (
    CONDITION_OCCURRENCE_COLUMNS,
    DRUG_EXPOSURE_COLUMNS,
    MEASUREMENT_COLUMNS,
    PROCEDURE_OCCURRENCE_COLUMNS,
    VISIT_OCCURRENCE_COLUMNS,
)
from openmed.clinical.journey_contracts import ClinicalFact, canonical_digest
from openmed.interop.omop.fact_projection import (
    OMOP_FACT_TABLES,
    OmopConceptMapping,
    OmopFactProjection,
    OmopFactProjectionInput,
    OmopVocabularySnapshot,
    assess_omop_fact_round_trip,
    load_omop_fact_projection_schema,
    project_clinical_facts_to_omop,
    validate_omop_fact_projection,
)
from openmed.structured.store import StoreState

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "interop"
    / "omop"
    / "fact_projection.json"
)


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _inputs() -> tuple[OmopFactProjectionInput, ...]:
    data = _fixture()
    result = []
    for raw in data["facts"]:  # type: ignore[index]
        record = dict(raw)
        mapping = OmopConceptMapping.from_dict(record.pop("mapping"))
        result.append(
            OmopFactProjectionInput(
                fact=ClinicalFact.from_dict(record),
                source_key=str(data["source_key"]),
                source_revision=str(data["source_revision"]),
                mapping=mapping,
                dataset_split=str(data["dataset_split"]),
            )
        )
    return tuple(result)


def _snapshot(**changes: object) -> OmopVocabularySnapshot:
    data = dict(_fixture()["vocabulary_snapshot"])  # type: ignore[arg-type]
    data.update(changes)
    return OmopVocabularySnapshot.from_dict(data)


def _project(
    inputs: tuple[OmopFactProjectionInput, ...] | None = None,
    *,
    previous: OmopFactProjection | None = None,
    occurred_at: str | None = None,
):
    data = _fixture()
    return project_clinical_facts_to_omop(
        inputs or _inputs(),
        vocabulary_snapshot=_snapshot(),
        etl_version=str(data["etl_version"]),
        occurred_at=occurred_at or str(data["occurred_at"]),
        previous=previous,
    )


def test_projects_all_supported_domains_with_referential_integrity() -> None:
    result = _project()

    assert result.ok and result.value is not None
    projection = result.value
    assert projection.summary.row_counts == {
        "person": 1,
        "visit_occurrence": 1,
        "note": 1,
        "condition_occurrence": 1,
        "drug_exposure": 1,
        "procedure_occurrence": 1,
        "measurement": 1,
        "observation": 1,
        "source_to_concept_map": 5,
    }
    assert projection.summary.current_fact_count == 5
    assert projection.summary.mapping_counts["mapped"] == 5
    assert projection.vocabulary_snapshot.digest == _snapshot().digest
    assert projection.etl_runs[0].etl_version == "3.0.0"
    assert all(
        item.source_evidence_key.startswith("sha256:") for item in projection.provenance
    )
    assert all(
        item.mapping.outcome_id.startswith("sha256:")
        for item in projection.mapping_outcomes
    )
    assert set(projection.table("visit_occurrence")[0]) == set(VISIT_OCCURRENCE_COLUMNS)
    assert set(projection.table("condition_occurrence")[0]) == set(
        CONDITION_OCCURRENCE_COLUMNS
    )
    assert set(projection.table("drug_exposure")[0]) == set(DRUG_EXPOSURE_COLUMNS)
    assert set(projection.table("procedure_occurrence")[0]) == set(
        PROCEDURE_OCCURRENCE_COLUMNS
    )
    assert set(projection.table("measurement")[0]) == set(MEASUREMENT_COLUMNS)
    assert not validate_omop_fact_projection(projection)
    assert assess_omop_fact_round_trip(_inputs(), projection).lossless


def test_projection_is_order_independent_and_schema_valid() -> None:
    inputs = _inputs()
    forward = _project(inputs)
    reverse = _project(tuple(reversed(inputs)))

    assert forward.ok and reverse.ok
    assert forward.value is not None and reverse.value is not None
    assert forward.value.to_json() == reverse.value.to_json()
    restored = OmopFactProjection.from_json(forward.value.to_json())
    assert restored == forward.value
    schema = load_omop_fact_projection_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(forward.value.to_dict())


def test_condition_cohort_membership_is_reproducible() -> None:
    first = _inputs()[0]
    second = replace(
        first,
        fact=replace(
            first.fact,
            fact_id="fact_4444444444444444",
            subject_id="subject_bbbbbbbbbbbbbbbb",
            encounter_id="encounter_bbbbbbbbbbbbbbbb",
            evidence_ids=("evidence_4444444444444444",),
            derivation_hash=canonical_digest({"subject": "synthetic-b"}),
        ),
        source_key="source_bbbbbbbbbbbbbbbb",
        source_revision="sha256:4444444444444444444444444444444444444444444444444444444444444444",
    )
    forward = _project((first, second))
    reverse = _project((second, first))

    assert forward.ok and reverse.ok
    assert forward.value is not None and reverse.value is not None

    def cohort(projection: OmopFactProjection) -> tuple[int, ...]:
        return tuple(
            sorted(
                int(row["person_id"])
                for row in projection.table("condition_occurrence")
                if row["condition_concept_id"] == 2101
            )
        )

    assert len(cohort(forward.value)) == 2
    assert cohort(forward.value) == cohort(reverse.value)
    assert forward.value.to_json() == reverse.value.to_json()


def test_unmapped_concept_zero_is_visible_with_an_explicit_reason() -> None:
    inputs = list(_inputs())
    first = inputs[0]
    inputs[0] = replace(
        first,
        mapping=OmopConceptMapping.unmapped(
            source_system="synthetic",
            source_code="synthetic-condition",
            source_concept_id=1101,
            snapshot_digest=_snapshot().digest,
            reason_code="standard_concept_not_found",
        ),
    )

    result = _project(tuple(inputs))

    assert result.state is StoreState.PARTIAL
    assert result.code == "projection_mapping_review_required"
    assert result.value is not None
    row = result.value.table("condition_occurrence")[0]
    assert row["condition_concept_id"] == 0
    outcome = next(
        item
        for item in result.value.mapping_outcomes
        if item.fact_id == first.fact.fact_id
    )
    assert outcome.mapping.reason_code == "standard_concept_not_found"
    assert outcome.mapping.requires_review
    source_map = next(
        item
        for item in result.value.table("source_to_concept_map")
        if item["source_code"] == "synthetic-condition"
    )
    assert source_map["target_concept_id"] == 0
    assert source_map["invalid_reason"] is None
    assert source_map["valid_start_date"] == "1970-01-01"
    assert source_map["valid_end_date"] == "2099-12-31"


def test_replay_is_idempotent_and_correction_replaces_current_source() -> None:
    first = _project()
    assert first.ok and first.value is not None

    replay = _project(previous=first.value)
    assert replay.ok and replay.value == first.value

    original = _inputs()[0]
    corrected_fact = replace(
        original.fact,
        fact_id="fact_ffffffffffffffff",
        value={"code": "synthetic-condition-corrected", "system": "synthetic"},
        derivation_hash=canonical_digest({"correction": "synthetic"}),
        parent_fact_ids=(original.fact.fact_id,),
    )
    corrected_mapping = replace(
        original.mapping,
        source_code="synthetic-condition-corrected",
        source_concept_id=1201,
        standard_concept_id=2201,
        standard_code="condition-corrected-standard",
    )
    correction = replace(
        original,
        fact=corrected_fact,
        mapping=corrected_mapping,
        source_revision="sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    )
    result = _project(
        (correction,),
        previous=first.value,
        occurred_at="2026-09-21T11:00:00Z",
    )

    assert result.ok and result.value is not None
    projection = result.value
    assert projection.current_fact_ids == (corrected_fact.fact_id,)
    assert len(projection.etl_runs) == 2
    assert projection.etl_runs[-1].superseded_fact_ids == tuple(
        sorted(item.fact.fact_id for item in _inputs())
    )
    assert projection.provenance[0].correction_of == (original.fact.fact_id,)
    assert projection.summary.row_counts["source_to_concept_map"] == 1
    assert not validate_omop_fact_projection(projection)


def test_correction_recomputes_visit_dates_from_current_facts() -> None:
    first = _project()
    assert first.ok and first.value is not None
    source = _inputs()[0]
    correction = replace(
        source,
        fact=replace(source.fact, effective_time={"start": "2026-03-01"}),
        source_revision=canonical_digest({"revision": "corrected-date"}),
    )
    result = _project(
        (correction,), previous=first.value, occurred_at="2026-09-22T11:00:00Z"
    )
    assert result.ok and result.value is not None
    visit = result.value.table("visit_occurrence")[0]
    assert visit["visit_start_date"] == "2026-03-01"
    assert visit["visit_end_date"] == "2026-03-01"
    assert len(result.value.etl_runs) == 2


def test_incremental_visit_dates_include_the_entire_incoming_batch() -> None:
    first_source = replace(
        _inputs()[0],
        source_key="source_bbbbbbbbbbbbbbbb",
        fact=replace(_inputs()[0].fact, effective_time={"start": "2026-01-03"}),
    )
    first = _project((first_source,))
    assert first.ok and first.value is not None
    result = _project(
        _inputs()[1:], previous=first.value, occurred_at="2026-09-22T11:00:00Z"
    )
    assert result.ok and result.value is not None
    visit = result.value.table("visit_occurrence")[0]
    assert visit["visit_start_date"] == "2026-01-02"
    assert visit["visit_end_date"] == "2026-01-05"


def test_typed_outcomes_cover_unsupported_split_snapshot_and_license_gates() -> None:
    source = _inputs()[0]
    unsupported = replace(
        source,
        fact=replace(source.fact, fact_id="fact_1111111111111111", fact_type="allergy"),
    )
    result = _project((unsupported,))
    assert result.state is StoreState.UNSUPPORTED
    assert result.code == "projection_has_no_supported_facts"

    missing_date = replace(
        source,
        fact=replace(
            source.fact,
            fact_id="fact_3333333333333333",
            effective_time={},
        ),
    )
    result = _project((missing_date,))
    assert result.state is StoreState.UNSUPPORTED
    assert result.code == "projection_has_no_supported_facts"

    mixed = _project((source, unsupported))
    assert mixed.state is StoreState.PARTIAL
    assert mixed.code == "projection_information_loss"
    assert mixed.value is not None
    round_trip = assess_omop_fact_round_trip((source, unsupported), mixed.value)
    assert not round_trip.lossless
    assert round_trip.loss_fact_ids == (unsupported.fact.fact_id,)
    assert not round_trip.missing_fact_ids

    first = _project((source,))
    assert first.ok and first.value is not None
    split_conflict = replace(
        _inputs()[1],
        source_key="source_bbbbbbbbbbbbbbbb",
        dataset_split="train",
    )
    result = _project((split_conflict,), previous=first.value)
    assert result.state is StoreState.CONFLICT
    assert result.code == "projection_input_conflict"

    mismatch = replace(
        source,
        mapping=replace(
            source.mapping,
            snapshot_digest="sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
        ),
    )
    result = _project((mismatch,))
    assert result.state is StoreState.CONFLICT

    data = _fixture()
    denied = project_clinical_facts_to_omop(
        (source,),
        vocabulary_snapshot=_snapshot(
            license="restricted-vocabulary",
            usage_lane="user_supplied",
            bundled=True,
        ),
        etl_version=str(data["etl_version"]),
        occurred_at=str(data["occurred_at"]),
    )
    assert denied.state is StoreState.DENIED
    allowed = project_clinical_facts_to_omop(
        (source,),
        vocabulary_snapshot=_snapshot(
            license="restricted-vocabulary",
            usage_lane="user_supplied",
            bundled=False,
        ),
        etl_version=str(data["etl_version"]),
        occurred_at=str(data["occurred_at"]),
    )
    assert allowed.ok


def test_projection_custody_does_not_retain_raw_fact_text() -> None:
    canary = "SYNTHETIC-PHI-CANARY"
    source = _inputs()[-1]
    fact = replace(
        source.fact,
        fact_id="fact_2222222222222222",
        value=canary,
        derivation_hash=canonical_digest({"canary": canary}),
    )
    result = _project((replace(source, fact=fact),))

    assert result.state is StoreState.PARTIAL
    assert result.code == "projection_information_loss"
    assert result.value is not None
    assert canary not in result.value.to_json()
    assert result.value.losses[0].reason_code == "free_text_value_omitted"
    assert set(result.value.tables) == set(OMOP_FACT_TABLES)
