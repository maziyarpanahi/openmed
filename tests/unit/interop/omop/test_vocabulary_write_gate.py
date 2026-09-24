from __future__ import annotations

from typing import Any

import pytest

from openmed.interop.omop import VocabularyWriteGate as PublicVocabularyWriteGate
from openmed.interop.omop.mutation_batch import (
    CommitStatus,
    OmopApprovalBinding,
    OmopMutation,
    OmopMutationBatch,
)
from openmed.interop.omop.vocab_router import SourceToConceptMapping
from openmed.interop.omop.vocabulary_write_gate import (
    VocabularyCompatibility,
    VocabularyConcept,
    VocabularyMappingProvenance,
    VocabularySnapshot,
    VocabularyWriteGate,
    VocabularyWriteGateError,
)

RECEIPT_DIGEST = "sha256:" + "a" * 64


def _snapshot() -> VocabularySnapshot:
    return VocabularySnapshot(
        {"SYNTHETIC": "release-current"},
        (
            VocabularyConcept(101, "SYNTHETIC", standard_concept="S"),
            VocabularyConcept(
                202,
                "SYNTHETIC",
                standard_concept="S",
                invalid_reason="D",
            ),
            VocabularyConcept(303, "SYNTHETIC"),
        ),
    )


def _mapping(
    concept_id: int,
    *,
    version: str | None = "release-current",
    vocabulary_id: str = "SYNTHETIC",
) -> VocabularyMappingProvenance:
    return VocabularyMappingProvenance(
        target_concept_id=concept_id,
        target_vocabulary_id=vocabulary_id,
        vocabulary_version=version,
    )


def test_gate_is_exported_from_omop_package() -> None:
    assert PublicVocabularyWriteGate is VocabularyWriteGate


def test_gate_classifies_compatible_remap_required_and_retired() -> None:
    gate = VocabularyWriteGate(_snapshot())
    mappings = (
        _mapping(101),
        _mapping(101, version="release-stale"),
        _mapping(202),
    )

    first = gate.evaluate(mappings)
    second = gate.evaluate(mappings)

    assert first == second
    assert not first.is_compatible
    assert [decision.compatibility for decision in first.decisions] == [
        VocabularyCompatibility.COMPATIBLE,
        VocabularyCompatibility.REMAP_REQUIRED,
        VocabularyCompatibility.RETIRED,
    ]
    assert [decision.reason for decision in first.decisions] == [
        "snapshot_match",
        "vocabulary_version_changed",
        "concept_retired",
    ]
    assert first.classification_counts == (
        ("compatible", 1),
        ("remap_required", 1),
        ("retired", 1),
    )
    assert [item.ordinal for item in first.remapping_queue] == [1, 2]


def test_report_and_reprs_are_value_free() -> None:
    snapshot = _snapshot()
    mapping = _mapping(101, version="synthetic-private-version")
    report = VocabularyWriteGate(snapshot).evaluate((mapping,))

    rendered = report.to_json()
    assert "concept_id" not in rendered
    for raw_value in ("release-current", "synthetic-private-version"):
        assert raw_value not in rendered
        assert raw_value not in repr(snapshot)
        assert raw_value not in repr(mapping)
    assert report.decisions[0].source_version_digest is not None
    assert report.decisions[0].target_version_digest is not None


def test_router_mapping_extraction_ignores_source_values() -> None:
    private_code = "synthetic-private-source-code"
    private_description = "synthetic-private-description"
    routed = SourceToConceptMapping(
        source_code=private_code,
        source_vocabulary_id="LOCAL",
        source_concept_id=909,
        source_code_description=private_description,
        target_concept_id=101,
        target_vocabulary_id="SYNTHETIC",
        target_domain_id="Condition",
        domain="Condition",
        cdm_table="condition_occurrence",
        standard_concept="S",
        vocabulary_version="release-current",
        mapping_status="mapped",
    )

    provenance = VocabularyMappingProvenance.from_mapping(routed)
    report = VocabularyWriteGate(_snapshot()).evaluate((provenance,))

    assert report.is_compatible
    assert private_code not in repr(provenance)
    assert private_description not in report.to_json()


@pytest.mark.parametrize(
    ("mapping", "reason"),
    [
        (_mapping(0), "unmapped_concept"),
        (_mapping(404), "concept_not_found"),
        (_mapping(303), "concept_not_standard"),
        (_mapping(101, version=None), "mapping_version_missing"),
        (_mapping(101, vocabulary_id="OTHER"), "concept_vocabulary_changed"),
    ],
)
def test_unsafe_mapping_states_fail_closed(
    mapping: VocabularyMappingProvenance,
    reason: str,
) -> None:
    report = VocabularyWriteGate(_snapshot()).evaluate((mapping,))

    assert not report.is_compatible
    assert report.decisions[0].reason == reason
    assert report.remapping_queue[0].reason == reason


def test_missing_target_vocabulary_version_requires_remapping() -> None:
    snapshot = VocabularySnapshot(
        {},
        (VocabularyConcept(101, "SYNTHETIC", standard_concept="S"),),
    )

    report = VocabularyWriteGate(snapshot).evaluate((_mapping(101),))

    assert report.decisions[0].reason == "target_version_missing"


def test_assert_writable_exposes_only_safe_report_on_error() -> None:
    private_version = "synthetic-private-version"
    gate = VocabularyWriteGate(_snapshot())

    with pytest.raises(
        VocabularyWriteGateError,
        match="^incompatible_vocabulary_snapshot$",
    ) as captured:
        gate.assert_writable((_mapping(101, version=private_version),))

    assert captured.value.report is not None
    assert private_version not in str(captured.value)
    assert private_version not in captured.value.report.to_json()


class _RecordingCommitter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def commit_batch(
        self,
        mutations: tuple[OmopMutation, ...],
        *,
        batch_digest: str,
        approval: OmopApprovalBinding,
    ) -> None:
        self.calls.append(
            {
                "approval": approval,
                "batch_digest": batch_digest,
                "mutations": mutations,
            }
        )


def _approved_batch() -> tuple[OmopMutationBatch, OmopApprovalBinding]:
    batch = OmopMutationBatch((OmopMutation.insert("person", {"person_id": 1}),))
    preview = batch.preview()
    approval = batch.bind_approval(
        preview,
        approved_preview_digest=preview.preview_digest,
        approval_receipt_digest=RECEIPT_DIGEST,
    )
    return batch, approval


def test_commit_blocks_before_calling_committer_on_material_drift() -> None:
    batch, approval = _approved_batch()
    committer = _RecordingCommitter()

    with pytest.raises(VocabularyWriteGateError, match="incompatible"):
        VocabularyWriteGate(_snapshot()).commit(
            batch,
            committer,
            approval=approval,
            mappings=(_mapping(101, version="release-stale"),),
        )

    assert committer.calls == []


def test_commit_allows_exact_snapshot_match() -> None:
    batch, approval = _approved_batch()
    committer = _RecordingCommitter()

    result = VocabularyWriteGate(_snapshot()).commit(
        batch,
        committer,
        approval=approval,
        mappings=(_mapping(101),),
    )

    assert result.status is CommitStatus.COMMITTED
    assert len(committer.calls) == 1


def test_mapping_order_is_bound_into_report_digest() -> None:
    gate = VocabularyWriteGate(_snapshot())
    first = _mapping(101)
    second = _mapping(202)

    forward = gate.evaluate((first, second))
    reverse = gate.evaluate((second, first))

    assert forward.mappings_digest != reverse.mappings_digest
    assert forward.report_digest != reverse.report_digest


def test_invalid_collections_and_duplicate_concepts_are_rejected_safely() -> None:
    with pytest.raises(VocabularyWriteGateError, match="duplicate_concept"):
        VocabularySnapshot(
            {"SYNTHETIC": "release-current"},
            (
                VocabularyConcept(101, "SYNTHETIC"),
                VocabularyConcept(101, "SYNTHETIC"),
            ),
        )

    gate = VocabularyWriteGate(_snapshot())
    with pytest.raises(VocabularyWriteGateError, match="empty_mapping_collection"):
        gate.evaluate(())
    with pytest.raises(VocabularyWriteGateError, match="invalid_mapping_collection"):
        gate.evaluate({})
