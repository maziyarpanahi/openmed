"""Golden saved-cohort to governed-dataset journey."""

from __future__ import annotations

from pathlib import Path

from openmed.structured.cohort import (
    CohortMembership,
    CohortSourceSnapshot,
    CriterionMembership,
    MembershipEvidence,
    MembershipState,
    PhenotypeDefinition,
    build_cohort_execution,
    save_cohort_definition,
)
from openmed.structured.datasets import (
    DatasetAnnotation,
    DatasetBuildSpec,
    DatasetExportFormat,
    DatasetLicenseConstraint,
    DatasetRecord,
    DatasetSelection,
    RedistributionPolicy,
    build_dataset_snapshot,
)
from openmed.structured.store import StoreState

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "cohort"
    / "phenotypes"
    / "diabetes_on_metformin.json"
)


def _criterion(
    criterion_id: str,
    state: MembershipState,
    suffix: str,
) -> CriterionMembership:
    return CriterionMembership(
        criterion_id=criterion_id,
        state=state,
        evidence=(
            ()
            if state is MembershipState.UNKNOWN
            else (
                MembershipEvidence(
                    evidence_id=f"evidence_{suffix * 16}",
                    fact_id=f"fact_{suffix * 16}",
                    time_window_id=f"window_{suffix * 16}",
                ),
            )
        ),
        reason_codes=("evidence_missing",) if state is MembershipState.UNKNOWN else (),
    )


def test_saved_cohort_selects_only_resolved_members_into_governed_snapshot() -> None:
    definition = save_cohort_definition(PhenotypeDefinition.load(FIXTURE))
    eligible = CohortMembership(
        patient_key="patient_eligible00000001",
        state=MembershipState.MET,
        criteria=(
            _criterion("has-diabetes", MembershipState.MET, "d"),
            _criterion("has-metformin", MembershipState.MET, "m"),
        ),
    )
    unresolved = CohortMembership(
        patient_key="patient_unresolved000001",
        state=MembershipState.UNKNOWN,
        criteria=(
            _criterion("has-diabetes", MembershipState.MET, "e"),
            _criterion("has-metformin", MembershipState.UNKNOWN, "u"),
        ),
    )
    execution_result = build_cohort_execution(
        definition,
        source_snapshot=CohortSourceSnapshot(
            snapshot_id="snapshot_cohortdataset001",
            digest="sha256:" + "1" * 64,
            schema_version="journey-snapshot-v1",
            license_tags=("synthetic",),
        ),
        vocabulary_digest="sha256:" + "2" * 64,
        policy_digest="sha256:" + "3" * 64,
        evaluator_version="cohort-evaluator-1.0",
        memberships=(eligible, unresolved),
    )
    assert execution_result.value is not None
    selection = DatasetSelection.from_cohort_execution(execution_result.value)

    record = DatasetRecord(
        record_id="record_eligible00000001",
        patient_key=eligible.patient_key,
        split="train",
        source_fact_ids=("fact_dddddddddddddddd", "fact_mmmmmmmmmmmmmmmm"),
        evidence_ids=("evidence_dddddddddddddddd", "evidence_mmmmmmmmmmmmmmmm"),
        labels=("Condition", "Medication"),
        annotations=(
            DatasetAnnotation(
                annotation_id="annotation_eligible00000001",
                label="Condition",
                start=4,
                end=13,
                source_digest="sha256:" + "4" * 64,
                evidence_id="evidence_dddddddddddddddd",
            ),
        ),
        values={"clinical_code": "synthetic-condition"},
    )
    spec = DatasetBuildSpec(
        dataset_id="dataset_cohortdataset001",
        created_at="2026-01-02T03:04:05Z",
        selection=selection,
        query_digest="sha256:" + "5" * 64,
        policy_digest="sha256:" + "6" * 64,
        schema_digest="sha256:" + "7" * 64,
        vocabulary_digest="sha256:" + "2" * 64,
        component_versions={"dataset_builder": "1.0.0"},
        model_versions={"clinical_encoder": "synthetic-1.0"},
        licenses=(
            DatasetLicenseConstraint(
                source_id="synthetic_fixture",
                license_id="Apache-2.0",
                terms_digest="sha256:" + "8" * 64,
                redistribution=RedistributionPolicy.PERMITTED,
            ),
        ),
        formats=(
            DatasetExportFormat.JSONL,
            DatasetExportFormat.ANNOTATION_JSONL,
        ),
    )
    built = build_dataset_snapshot(spec, (record,))

    assert built.state is StoreState.SUCCESS
    assert built.value is not None
    assert selection.eligible_patient_keys == (eligible.patient_key,)
    assert selection.review_excluded_count == 1
    assert built.value.manifest.snapshot.record_count == 1
    assert "synthetic-condition" not in built.value.files["records.jsonl"].decode()
    assert built.value.manifest.manifest_digest == (
        "sha256:03bd9f5834cf1fc609cb44b0d34aa3ab25c6efeb7076a7ee6fd22bf2d050749b"
    )

    blocked_record = DatasetRecord(
        record_id="record_unresolved000001",
        patient_key=unresolved.patient_key,
        split="train",
        source_fact_ids=("fact_eeeeeeeeeeeeeeee",),
    )
    blocked = build_dataset_snapshot(spec, (record, blocked_record))
    assert blocked.state is StoreState.CONFLICT
