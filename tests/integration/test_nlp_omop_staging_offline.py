"""Synthetic offline loader, lineage, approval and protected-custody composition."""

from __future__ import annotations

import hashlib
import json
import socket

import pytest

from openmed.interop.omop import (
    CommitStatus,
    OmopMutationError,
    VocabularyConcept,
    VocabularyMappingProvenance,
    VocabularySnapshot,
    cdm_loader,
    load_grounded_notes,
    stage_nlp_omop_tables,
)
from openmed.interop.omop_rollback_manifest import OmopRollbackInstruction

pytestmark = pytest.mark.integration


def proposal(pipeline):
    text = "Synthetic private condition."
    loaded = load_grounded_notes(
        [
            {
                "note_id": "SYNTHETIC-PRIVATE-NOTE",
                "person_id": "SYNTHETIC-PRIVATE-PERSON",
                "note_text": text,
                "entities": [
                    {
                        "text": "private",
                        "start": 10,
                        "end": 17,
                        "domain_id": "Condition",
                        "concept_id": 701,
                        "vocabulary_id": "SYNTHETIC",
                        "code": "SYNTHETIC-701",
                    }
                ],
            }
        ],
        vocabulary_version="SYNTHETIC-RELEASE",
    )
    return stage_nlp_omop_tables(
        loaded,
        pipeline_digest=pipeline,
        vocabulary_snapshot=VocabularySnapshot(
            {"SYNTHETIC": "SYNTHETIC-RELEASE"},
            (VocabularyConcept(701, "SYNTHETIC", standard_concept="S"),),
        ),
        vocabulary_mappings=(
            VocabularyMappingProvenance(701, "SYNTHETIC", "SYNTHETIC-RELEASE"),
        ),
    )


def test_preview_is_offline_and_only_explicit_commit_reaches_injected_adapter(
    monkeypatch, caplog
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unrequested network or direct-writer effect")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for name in ("write_omop_sqlite", "write_omop_duckdb", "write_omop_parquet"):
        monkeypatch.setattr(cdm_loader, name, forbidden)

    stage = proposal("sha256:" + "a" * 64)
    # A synthetic in-memory custody adapter independently hashes the exact
    # required payload bytes. This test makes no durable-storage claim.
    custody = {}
    instructions = []
    for required in stage.rollback_requirements:
        material = stage.rollback_material(required.mutation_ordinal)
        data = json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        custody[digest] = data
        instructions.append(
            OmopRollbackInstruction(
                required.mutation_ordinal, required.strategy, digest
            )
        )
    manifest = stage.build_rollback_manifest(instructions)
    packet = stage.preview(manifest)
    assert packet.is_approvable
    assert all(i.rollback_artifact_digest in custody for i in instructions)

    class SyntheticAtomicCommitter:
        def __init__(self):
            self.calls = []

        def commit_batch(self, mutations, *, batch_digest, approval):
            assert approval.batch_digest == batch_digest
            self.calls.append(tuple(mutations))

    committer = SyntheticAtomicCommitter()
    assert not committer.calls
    # Fixed receipt is a synthetic test fixture, not a human authorization.
    approval = stage.bind_approval(
        packet,
        manifest,
        approved_preview_digest=packet.preview_digest,
        approval_receipt_digest="sha256:" + "b" * 64,
    )
    changed = proposal("sha256:" + "c" * 64)
    with pytest.raises(OmopMutationError, match="batch_changed"):
        changed.batch.commit(committer, approval=approval)
    assert not committer.calls
    result = stage.batch.commit(committer, approval=approval)
    assert result.status is CommitStatus.COMMITTED
    assert committer.calls == [stage.batch.mutations]
    safe = (
        json.dumps(packet.to_dict())
        + json.dumps(manifest.to_dict())
        + json.dumps(stage.lineage.to_dict())
        + json.dumps(result.to_dict())
        + caplog.text
    )
    assert "SYNTHETIC-PRIVATE" not in safe
    assert "Synthetic private condition" not in safe
