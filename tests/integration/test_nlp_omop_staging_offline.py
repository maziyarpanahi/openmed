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


def test_staging_and_sqlite_need_independent_outer_review_before_real_effects(
    tmp_path, caplog
):
    import sqlite3

    from openmed.interop.omop import (
        OmopDatabaseStatus,
        SQLiteOmopBatchCommitter,
        create_omop_schema,
        initialize_omop_commit_metadata,
        preview_omop_database,
    )

    stage = proposal("sha256:" + "d" * 64)
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
        ).encode("utf-8")
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        custody[digest] = data
        instructions.append(
            OmopRollbackInstruction(
                required.mutation_ordinal,
                required.strategy,
                digest,
            )
        )
    manifest = stage.build_rollback_manifest(instructions)
    staging_packet = stage.preview(manifest)
    staging_receipt = "sha256:" + "e" * 64
    stage_approval = stage.bind_approval(
        staging_packet,
        manifest,
        approved_preview_digest=staging_packet.preview_digest,
        approval_receipt_digest=staging_receipt,
    )
    path = tmp_path / "synthetic-composition.db"
    with sqlite3.connect(path) as con:
        create_omop_schema(con)
        con.commit()
        initialize_omop_commit_metadata(con, stage.vocabulary_snapshot)
        database_packet = preview_omop_database(
            con,
            stage.batch,
            vocabulary_snapshot=stage.vocabulary_snapshot,
            vocabulary_mappings=stage.vocabulary_mappings,
            rollback_manifest=manifest,
        )
    assert database_packet.preview_digest != staging_packet.preview_digest
    # Host custody/approval are synthetic doubles, not storage qualification
    # or a human authorization. A stage-only receipt must not authorize SQL.
    independently_approved = {staging_receipt: {staging_packet.preview_digest}}
    opened = []

    def factory():
        opened.append(True)
        return sqlite3.connect(path)

    def authorize(binding, packet):
        return {
            staging_packet.preview_digest,
            packet.preview_digest,
        } <= independently_approved.get(binding.approval_receipt_digest, set())

    adapter = SQLiteOmopBatchCommitter(
        stage.batch,
        database_packet,
        vocabulary_snapshot=stage.vocabulary_snapshot,
        vocabulary_mappings=stage.vocabulary_mappings,
        rollback_manifest=manifest,
        connection_factory=factory,
        authorize=authorize,
        admit=lambda digest: digest == database_packet.preview_digest,
        rollback_ready=lambda value: (
            value == manifest
            and all(
                instruction.rollback_artifact_digest in custody
                for instruction in manifest.entries
            )
        ),
    )
    refused = adapter.submit(stage_approval)
    assert refused.status in (OmopDatabaseStatus.DENIED, OmopDatabaseStatus.CONFLICT)
    assert not opened
    database_receipt = "sha256:" + "f" * 64
    independently_approved[database_receipt] = {
        staging_packet.preview_digest,
        database_packet.preview_digest,
    }
    approval = stage.batch.bind_approval(
        database_packet.batch_preview,
        approved_preview_digest=database_packet.batch_preview.preview_digest,
        approval_receipt_digest=database_receipt,
    )
    first = adapter.submit(approval)
    assert first.status is OmopDatabaseStatus.COMMITTED
    assert adapter.submit(approval) == first
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT count(*) FROM note_nlp").fetchone()[0] == 1
        assert (
            con.execute("SELECT count(*) FROM condition_occurrence").fetchone()[0] == 1
        )
        rows = con.execute(
            "SELECT result_json FROM _openmed_omop_commit_receipts"
        ).fetchall()
    assert len(rows) == 1
    public = json.dumps(first.to_dict()) + str(rows) + caplog.text
    assert "SYNTHETIC-PRIVATE" not in public
    assert "Synthetic private condition" not in public
