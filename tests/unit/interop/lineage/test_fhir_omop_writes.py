from __future__ import annotations

from typing import Any

import pytest

from openmed.interop.lineage import (
    FhirElementReference,
    FhirOmopLineageError,
    FhirOmopLineageLink,
    FhirOmopWriteLineage,
    VocabularyLineageEvidence,
)
from openmed.interop.omop import (
    CommitStatus,
    OmopMutation,
    OmopMutationBatch,
    OmopRowKey,
    VocabularyMappingProvenance,
)

RESOURCE_A = "sha256:" + "a" * 64
RESOURCE_B = "sha256:" + "b" * 64
SNAPSHOT_DIGEST = "sha256:" + "c" * 64
RECEIPT_DIGEST = "sha256:" + "d" * 64


def _source(resource_digest: str, path: str) -> FhirElementReference:
    return FhirElementReference(resource_digest, path)


def _batch() -> OmopMutationBatch:
    return OmopMutationBatch(
        (
            OmopMutation.insert("person", {"person_id": 1101}),
            OmopMutation.insert(
                "measurement",
                {
                    "measurement_id": 2202,
                    "person_id": 1101,
                    "measurement_concept_id": 3303,
                },
            ),
        )
    )


def _evidence() -> VocabularyLineageEvidence:
    mapping = VocabularyMappingProvenance(
        target_concept_id=3303,
        target_vocabulary_id="LOINC",
        vocabulary_version="synthetic-release",
    )
    return VocabularyLineageEvidence.from_mapping(
        mapping,
        snapshot_digest=SNAPSHOT_DIGEST,
    )


def _complete_lineage(
    batch: OmopMutationBatch,
) -> tuple[FhirOmopWriteLineage, tuple[FhirElementReference, ...]]:
    sources = (
        _source(RESOURCE_A, "Patient.id"),
        _source(RESOURCE_B, "Observation.code"),
        _source(RESOURCE_B, "Observation.valueQuantity.value"),
    )
    lineage = FhirOmopWriteLineage(
        (
            FhirOmopLineageLink(
                sources,
                "fhir.measurement.v1",
                tuple(mutation.row_digest for mutation in batch.mutations),
                vocabulary_evidence=(_evidence(),),
            ),
        )
    )
    return lineage, sources


def test_many_to_many_crosswalk_is_deterministic_and_value_free() -> None:
    batch = _batch()
    lineage, sources = _complete_lineage(batch)

    first = lineage.verify(batch, required_sources=reversed(sources))
    second = lineage.verify(batch, required_sources=sources)

    assert first == second
    assert first.is_approvable
    assert first.required_source_count == 3
    assert first.covered_source_count == 3
    assert first.target_row_count == 2
    assert first.covered_target_count == 2
    assert first.vocabulary_evidence_count == 1
    rendered = first.to_json()
    assert RESOURCE_A not in rendered
    assert RESOURCE_B not in rendered
    assert "1101" not in rendered
    assert "2202" not in rendered
    assert "3303" not in rendered


def test_bridge_resource_hash_is_normalized_to_digest_form() -> None:
    source = FhirElementReference("f" * 64, "Observation.code")

    assert source.resource_digest == "sha256:" + "f" * 64


def test_verification_reports_missing_source_and_target_coverage() -> None:
    batch = _batch()
    required = (
        _source(RESOURCE_A, "Patient.id"),
        _source(RESOURCE_B, "Observation.code"),
    )
    lineage = FhirOmopWriteLineage(
        (
            FhirOmopLineageLink(
                (required[0],),
                "fhir.person.v1",
                (batch.mutations[0].row_digest,),
            ),
        )
    )

    report = lineage.verify(batch, required_sources=required)

    assert not report.is_approvable
    assert [issue.code for issue in report.issues] == [
        "missing_source_lineage",
        "missing_target_lineage",
    ]
    assert report.issues[0].element_path == "Observation.code"


def test_concept_write_requires_vocabulary_evidence() -> None:
    batch = _batch()
    source = _source(RESOURCE_B, "Observation.code")
    lineage = FhirOmopWriteLineage(
        (
            FhirOmopLineageLink(
                (source,),
                "fhir.measurement.v1",
                tuple(mutation.row_digest for mutation in batch.mutations),
            ),
        )
    )

    report = lineage.verify(batch, required_sources=(source,))

    assert [issue.code for issue in report.issues] == ["missing_vocabulary_evidence"]
    assert report.issues[0].target_row_digest == batch.mutations[1].row_digest


def test_lossy_transform_is_identified_and_blocks_approval() -> None:
    batch = _batch()
    source = _source(RESOURCE_B, "Observation.valueString")
    lineage = FhirOmopWriteLineage(
        (
            FhirOmopLineageLink(
                (source,),
                "fhir.drop_unsupported.v1",
                (),
                loss_reason="unsupported_element",
            ),
            FhirOmopLineageLink(
                (_source(RESOURCE_A, "Patient.id"),),
                "fhir.person.v1",
                (batch.mutations[0].row_digest,),
            ),
            FhirOmopLineageLink(
                (_source(RESOURCE_B, "Observation.code"),),
                "fhir.measurement.v1",
                (batch.mutations[1].row_digest,),
                vocabulary_evidence=(_evidence(),),
            ),
        )
    )
    preview = batch.preview(
        existing_rows=(OmopRowKey("concept", {"concept_id": 3303}),)
    )

    with pytest.raises(
        FhirOmopLineageError,
        match="lineage_verification_failed",
    ) as captured:
        lineage.bind_approval(
            batch,
            preview,
            required_sources=(source,),
            approved_preview_digest=preview.preview_digest,
            approval_receipt_digest=RECEIPT_DIGEST,
        )

    report = captured.value.report
    assert report is not None
    assert report.lossy_link_count == 1
    assert any(issue.code == "lossy_transformation" for issue in report.issues)
    assert "unsupported_element" in report.to_json()


def test_unknown_target_digest_fails_closed() -> None:
    batch = _batch()
    source = _source(RESOURCE_A, "Patient.id")
    lineage = FhirOmopWriteLineage(
        (
            FhirOmopLineageLink(
                (source,),
                "fhir.person.v1",
                ("sha256:" + "e" * 64,),
            ),
        )
    )

    report = lineage.verify(batch, required_sources=(source,))

    assert [issue.code for issue in report.issues] == [
        "missing_target_lineage",
        "missing_target_lineage",
        "unknown_target_row",
    ]


class _RecordingCommitter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def commit_batch(self, mutations: Any, **metadata: Any) -> None:
        self.calls.append({"mutations": mutations, **metadata})


def test_passing_lineage_binds_and_commits_exact_batch() -> None:
    batch = _batch()
    lineage, sources = _complete_lineage(batch)
    preview = batch.preview(
        existing_rows=(OmopRowKey("concept", {"concept_id": 3303}),)
    )
    committer = _RecordingCommitter()

    approval = lineage.bind_approval(
        batch,
        preview,
        required_sources=sources,
        approved_preview_digest=preview.preview_digest,
        approval_receipt_digest=RECEIPT_DIGEST,
    )
    result = approval.commit(batch, committer)

    assert result.status is CommitStatus.COMMITTED
    assert result.mutation_count == 2
    assert approval.crosswalk_digest == lineage.crosswalk_digest
    assert len(committer.calls) == 1


def test_link_without_target_must_declare_loss() -> None:
    with pytest.raises(FhirOmopLineageError, match="target_required"):
        FhirOmopLineageLink(
            (_source(RESOURCE_A, "Patient.id"),),
            "fhir.drop.v1",
            (),
        )
