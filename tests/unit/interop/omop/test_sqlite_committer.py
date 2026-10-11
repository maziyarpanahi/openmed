"""Controlled public metadata and bounded proposal validation."""

from __future__ import annotations

from itertools import repeat

import pytest

from openmed.interop.omop import (
    OmopMutation,
    OmopMutationBatch,
    VocabularyConcept,
    VocabularyMappingProvenance,
    VocabularySnapshot,
)
from openmed.interop.omop.sqlite_committer import (
    OmopDatabaseError,
    OmopDatabasePreview,
    OmopDatabaseResult,
    OmopDatabaseStatus,
    SQLiteOmopBatchCommitter,
)
from openmed.interop.omop_rollback_manifest import (
    OmopRollbackInstruction,
    RollbackStrategy,
    build_omop_rollback_manifest,
)

SAFE = "sha256:" + "a" * 64
CANARY = "SYNTHETIC-PRIVATE-METADATA-CANARY"


@pytest.mark.parametrize("bad", [CANARY, True, [], None])
def test_result_refuses_non_digest_private_metadata(bad):
    with pytest.raises(OmopDatabaseError, match="invalid_input") as exc:
        OmopDatabaseResult(
            OmopDatabaseStatus.COMMITTED, None, bad, SAFE, SAFE, SAFE, 1, 1
        )
    assert CANARY not in str(exc.value)


@pytest.mark.parametrize("count", [0, True, -1, 10001])
def test_result_requires_bounded_non_boolean_counts(count):
    with pytest.raises(OmopDatabaseError, match="invalid_input"):
        OmopDatabaseResult(
            OmopDatabaseStatus.COMMITTED, None, SAFE, SAFE, SAFE, SAFE, count, count
        )


def test_unknown_outcome_never_claims_an_applied_row_count():
    with pytest.raises(OmopDatabaseError, match="invalid_input"):
        OmopDatabaseResult(
            OmopDatabaseStatus.UNKNOWN, "commit_unknown", SAFE, SAFE, SAFE, SAFE, 1, 0
        )


def test_unsupported_custom_table_does_not_open_any_connection():
    proposal = OmopMutationBatch(
        (
            OmopMutation.insert(
                "private_custom_table", {"opaque_id": CANARY}, key={"opaque_id": CANARY}
            ),
        )
    )
    preview = OmopDatabasePreview(proposal.preview(), SAFE, SAFE, SAFE, SAFE, SAFE)
    with pytest.raises(OmopDatabaseError, match="unsupported_batch"):
        SQLiteOmopBatchCommitter(
            proposal,
            preview,
            vocabulary_snapshot=None,
            vocabulary_mappings=(),
            rollback_manifest=None,
            connection_factory=lambda: pytest.fail("Connection opened"),
            authorize=lambda *args: True,
            admit=lambda *args: True,
            rollback_ready=lambda *args: True,
        )


def test_mapping_collection_is_bounded_before_opening_database():
    proposal = OmopMutationBatch(
        (
            OmopMutation.insert(
                "person", {"person_id": 1, "person_source_value": "synthetic"}
            ),
        )
    )
    snapshot = VocabularySnapshot(
        {"SYNTHETIC": "release"},
        (VocabularyConcept(1, "SYNTHETIC", standard_concept="S"),),
    )
    manifest = build_omop_rollback_manifest(
        proposal,
        snapshot,
        (OmopRollbackInstruction(0, RollbackStrategy.DELETE_INSERTED_ROW, SAFE),),
    )
    packet = OmopDatabasePreview(
        proposal.preview(),
        SAFE,
        SAFE,
        snapshot.snapshot_digest,
        SAFE,
        manifest.manifest_digest,
    )
    with pytest.raises(OmopDatabaseError, match="invalid_input"):
        SQLiteOmopBatchCommitter(
            proposal,
            packet,
            vocabulary_snapshot=snapshot,
            vocabulary_mappings=repeat(
                VocabularyMappingProvenance(1, "SYNTHETIC", "release")
            ),
            rollback_manifest=manifest,
            connection_factory=lambda: pytest.fail("Connection opened"),
            authorize=lambda *args: True,
            admit=lambda *args: True,
            rollback_ready=lambda *args: True,
        )
