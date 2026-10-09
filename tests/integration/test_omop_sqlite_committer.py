"""Real local SQLite transactions with exclusively synthetic clinical fixtures."""

from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from openmed.interop.omop import (
    OmopDatabaseError,
    OmopDatabaseStatus,
    OmopMutation,
    OmopMutationBatch,
    SQLiteOmopBatchCommitter,
    VocabularyConcept,
    VocabularyMappingProvenance,
    VocabularySnapshot,
    create_omop_schema,
    initialize_omop_commit_metadata,
    load_grounded_notes,
    preview_omop_database,
)
from openmed.interop.omop_rollback_manifest import (
    OmopRollbackInstruction,
    RollbackStrategy,
    build_omop_rollback_manifest,
)

pytestmark = pytest.mark.integration
CANARY = "SYNTHETIC-PRIVATE-ROW-CANARY"
RECEIPT = "sha256:" + "a" * 64


def vocab():
    return VocabularySnapshot(
        {"SYNTHETIC": "release-1"},
        (VocabularyConcept(101, "SYNTHETIC", standard_concept="S"),),
    )


def mappings():
    return (VocabularyMappingProvenance(101, "SYNTHETIC", "release-1"),)


def batch():
    text = f"Synthetic {CANARY}."
    tables = load_grounded_notes(
        [
            {
                "note_id": "synthetic-note",
                "person_id": "synthetic-person",
                "note_text": text,
                "entities": [
                    {
                        "text": CANARY,
                        "start": 10,
                        "end": 10 + len(CANARY),
                        "domain_id": "Condition",
                        "concept_id": 101,
                        "vocabulary_id": "SYNTHETIC",
                        "code": "C-101",
                        "standard_concept": "S",
                    }
                ],
            }
        ],
        vocabulary_version="release-1",
    )
    return OmopMutationBatch(
        OmopMutation.insert(name, row)
        for name, rows in tables.tables.items()
        for row in rows
    )


def rollback(proposal):
    strategies = {
        "insert": RollbackStrategy.DELETE_INSERTED_ROW,
        "update": RollbackStrategy.RESTORE_BEFORE_IMAGE,
        "tombstone": RollbackStrategy.REINSERT_TOMBSTONED_ROW,
    }
    return build_omop_rollback_manifest(
        proposal,
        vocab(),
        tuple(
            OmopRollbackInstruction(
                i, strategies[m.operation.value], "sha256:" + f"{i + 1:064x}"
            )
            for i, m in enumerate(proposal.mutations)
        ),
    )


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "synthetic.db"
    con = sqlite3.connect(path)
    create_omop_schema(con)
    con.commit()
    initialize_omop_commit_metadata(con, vocab())
    con.close()
    return path


def prepared(
    path,
    *,
    proposal=None,
    factory=None,
    authorize=None,
    admit=None,
    ready=None,
    receipt=RECEIPT,
):
    proposal = proposal or batch()
    manifest = rollback(proposal)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys=ON")
    packet = preview_omop_database(
        con,
        proposal,
        vocabulary_snapshot=vocab(),
        vocabulary_mappings=mappings(),
        rollback_manifest=manifest,
    )
    con.close()
    approval = proposal.bind_approval(
        packet.batch_preview,
        approved_preview_digest=packet.batch_preview.preview_digest,
        approval_receipt_digest=receipt,
    )
    authorize = authorize or (
        lambda receipt, preview: (
            receipt == approval and preview.preview_digest == packet.preview_digest
        )
    )
    adapter = SQLiteOmopBatchCommitter(
        proposal,
        packet,
        vocabulary_snapshot=vocab(),
        vocabulary_mappings=mappings(),
        rollback_manifest=manifest,
        connection_factory=factory or (lambda: sqlite3.connect(path)),
        authorize=authorize,
        admit=admit or (lambda digest: digest == packet.preview_digest),
        rollback_ready=ready or (lambda material: material == manifest),
    )
    return adapter, approval, packet


def row_counts(path):
    with sqlite3.connect(path) as con:
        return tuple(
            con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
            for table in (
                "concept",
                "person",
                "visit_occurrence",
                "note",
                "note_nlp",
                "condition_occurrence",
                "source_to_concept_map",
                "_openmed_omop_commit_receipts",
            )
        )


def test_real_atomic_commit_and_authorized_duplicate_return_identical_results(
    database, caplog
):
    adapter, approval, packet = prepared(database)
    assert row_counts(database) == (0,) * 8
    first = adapter.submit(approval)
    assert first.status is OmopDatabaseStatus.COMMITTED
    second = adapter.submit(approval)
    assert second == first
    assert row_counts(database) == (2, 1, 1, 1, 1, 1, 1, 1)
    with sqlite3.connect(database) as con:
        receipt = con.execute(
            "SELECT result_json FROM _openmed_omop_commit_receipts"
        ).fetchone()[0]
        assert json.loads(receipt) == first.to_dict()
        assert CANARY in con.execute("SELECT note_text FROM note").fetchone()[0]
    assert (
        CANARY
        not in receipt
        + json.dumps(packet.to_dict())
        + json.dumps(first.to_dict())
        + repr(packet)
        + caplog.text
    )


def test_stale_inner_approval_refuses_before_opening_database(database):
    opened = []

    def factory():
        opened.append(True)
        return sqlite3.connect(database)

    adapter, approval, _ = prepared(database, factory=factory)
    result = adapter.submit(replace(approval, preview_digest="sha256:" + "b" * 64))
    assert (
        result.status is OmopDatabaseStatus.CONFLICT
        and result.code == "invalid_approval"
    )
    assert not opened and row_counts(database) == (0,) * 8


@pytest.mark.parametrize("guard", ["authorize", "admit", "ready"])
@pytest.mark.parametrize("ack", [False, 1, "true"])
def test_guards_require_exact_true_and_leave_no_rows(database, guard, ack):
    adapter, approval, _ = prepared(database, **{guard: lambda *args: ack})
    result = adapter.submit(approval)
    assert result.status is OmopDatabaseStatus.DENIED
    assert row_counts(database) == (0,) * 8


def test_emergency_stop_during_batch_rolls_back_earlier_sql(database):
    calls = []

    def admission(*args):
        calls.append(True)
        return len(calls) < 6

    adapter, approval, _ = prepared(database, admit=admission)
    result = adapter.submit(approval)
    assert result.code == "admission_denied" and len(calls) == 6
    assert row_counts(database) == (0,) * 8


def test_driver_failure_mid_transaction_never_leaves_a_partial_batch(database, caplog):
    class Broken(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql.startswith('INSERT INTO "note_nlp"'):
                raise RuntimeError(CANARY)
            return super().execute(sql, parameters)

    adapter, approval, _ = prepared(
        database, factory=lambda: sqlite3.connect(database, factory=Broken)
    )
    result = adapter.submit(approval)
    assert result.status is OmopDatabaseStatus.FAILED
    assert row_counts(database) == (0,) * 8
    assert CANARY not in json.dumps(result.to_dict()) + caplog.text


@pytest.mark.parametrize("ack", ["lost", "not_committed"])
def test_commit_acknowledgement_uncertainty_is_explicit_and_recovery_is_read_only(
    database, ack
):
    class Uncertain(sqlite3.Connection):
        def commit(self):
            if ack == "lost":
                super().commit()
                raise RuntimeError(CANARY)
            return None

    adapter, approval, _ = prepared(
        database, factory=lambda: sqlite3.connect(database, factory=Uncertain)
    )
    result = adapter.submit(approval)
    assert result.status is OmopDatabaseStatus.UNKNOWN and result.applied_count is None
    assert result.code == "commit_unknown"
    before = row_counts(database)
    recovered = adapter.recover(approval)
    assert row_counts(database) == before
    assert recovered.status is (
        OmopDatabaseStatus.COMMITTED if ack == "lost" else OmopDatabaseStatus.UNKNOWN
    )


def test_existing_protocol_preserves_unknown_in_last_result(database):
    class Uncertain(sqlite3.Connection):
        def commit(self):
            super().commit()
            raise RuntimeError(CANARY)

    adapter, approval, _ = prepared(
        database, factory=lambda: sqlite3.connect(database, factory=Uncertain)
    )
    result = adapter.batch.commit(adapter, approval=approval)
    assert result.error_code == "committer_error"
    assert adapter.last_result.status is OmopDatabaseStatus.UNKNOWN


def test_changed_private_rows_with_unchanged_keys_invalidate_review(database):
    first, approval, _ = prepared(database)
    assert first.submit(approval).status is OmopDatabaseStatus.COMMITTED
    with sqlite3.connect(database) as con:
        existing = con.execute(
            "SELECT condition_occurrence_id FROM condition_occurrence"
        ).fetchone()[0]
    proposal = OmopMutationBatch(
        (
            OmopMutation.update(
                "condition_occurrence",
                {"condition_occurrence_id": existing},
                {"condition_source_value": "synthetic-revision"},
            ),
        )
    )
    adapter, approval, packet = prepared(database, proposal=proposal)
    with sqlite3.connect(database) as con:
        con.execute("DELETE FROM _openmed_omop_commit_receipts")
        con.execute("UPDATE note SET note_text=?", (f"Synthetic {CANARY} altered.",))
    before = row_counts(database)
    result = adapter.submit(approval)
    assert (
        result.status is OmopDatabaseStatus.CONFLICT and result.code == "state_changed"
    )
    assert row_counts(database) == before


def test_vocabulary_metadata_drift_refuses_without_mutations(database):
    adapter, approval, _ = prepared(database)
    with sqlite3.connect(database) as con:
        con.execute(
            "UPDATE _openmed_omop_commit_vocabulary SET snapshot_digest=?",
            ("sha256:" + "f" * 64,),
        )
    result = adapter.submit(approval)
    assert result.code == "vocabulary_changed"
    assert row_counts(database) == (0,) * 8


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TRIGGER unsafe AFTER INSERT ON person BEGIN SELECT 1; END",
        "CREATE TABLE other (id INTEGER PRIMARY KEY, parent BIGINT REFERENCES person(person_id) ON DELETE CASCADE)",
        "ALTER TABLE note ADD COLUMN private_extra TEXT",
    ],
)
def test_unreviewed_schema_effects_are_refused(database, sql):
    adapter, approval, _ = prepared(database)
    with sqlite3.connect(database) as con:
        con.execute(sql)
    result = adapter.submit(approval)
    assert (
        result.status is OmopDatabaseStatus.CONFLICT
        and result.code == "unsupported_schema"
    )
    assert row_counts(database) == (0,) * 8


def test_tampered_idempotency_record_is_not_reported_as_committed(database):
    adapter, approval, _ = prepared(database)
    assert adapter.submit(approval).status is OmopDatabaseStatus.COMMITTED
    with sqlite3.connect(database) as con:
        con.execute("UPDATE _openmed_omop_commit_receipts SET result_json=?", (CANARY,))
    result = adapter.submit(approval)
    assert (
        result.status is OmopDatabaseStatus.CONFLICT
        and result.code == "receipt_conflict"
    )
    assert CANARY not in json.dumps(result.to_dict())


def test_metadata_initialization_does_not_rollback_a_callers_transaction(database):
    con = sqlite3.connect(database)
    con.execute("CREATE TABLE unrelated (value TEXT)")
    con.execute("INSERT INTO unrelated VALUES ('synthetic')")
    with pytest.raises(OmopDatabaseError, match="invalid_input"):
        initialize_omop_commit_metadata(con, vocab())
    assert con.in_transaction
    assert con.execute("SELECT count(*) FROM unrelated").fetchone() == (1,)
    con.rollback()
    con.close()


def test_update_and_explicit_tombstone_insert_replacement_commit_atomically(database):
    initial, approval, _ = prepared(database)
    assert initial.submit(approval).status is OmopDatabaseStatus.COMMITTED
    incoming = []
    for mutation in initial.batch.mutations:
        if mutation.table in {"concept", "person", "visit_occurrence"}:
            continue
        values = mutation.values
        if mutation.table == "note":
            values["note_text"] += " Synthetic revision."
        incoming.append(OmopMutation.insert(mutation.table, values))
    removals = tuple(
        OmopMutation.tombstone(m.table, m.key.values) for m in reversed(incoming)
    )
    replacement = OmopMutationBatch((*removals, *incoming))
    adapter, approval, _ = prepared(
        database, proposal=replacement, receipt="sha256:" + "b" * 64
    )
    result = adapter.submit(approval)
    assert result.status is OmopDatabaseStatus.COMMITTED and result.applied_count == 8
    with sqlite3.connect(database) as con:
        assert (
            con.execute("SELECT note_text FROM note")
            .fetchone()[0]
            .endswith("Synthetic revision.")
        )
        key = con.execute(
            "SELECT condition_occurrence_id FROM condition_occurrence"
        ).fetchone()[0]
    update = OmopMutationBatch(
        (
            OmopMutation.update(
                "condition_occurrence",
                {"condition_occurrence_id": key},
                {"condition_source_value": "Synthetic normalized finding"},
            ),
        )
    )
    adapter, approval, _ = prepared(
        database, proposal=update, receipt="sha256:" + "c" * 64
    )
    result = adapter.submit(approval)
    assert result.status is OmopDatabaseStatus.COMMITTED and result.applied_count == 1
    assert row_counts(database) == (2, 1, 1, 1, 1, 1, 1, 3)


def test_rollback_acknowledgement_failure_remains_unknown(database):
    class Uncertain(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql.startswith('INSERT INTO "note_nlp"'):
                raise RuntimeError(CANARY)
            return super().execute(sql, parameters)

        def rollback(self):
            raise RuntimeError(CANARY)

    adapter, approval, _ = prepared(
        database, factory=lambda: sqlite3.connect(database, factory=Uncertain)
    )
    result = adapter.submit(approval)
    assert (
        result.status is OmopDatabaseStatus.UNKNOWN
        and result.code == "rollback_unknown"
    )
    assert result.applied_count is None
    assert row_counts(database) == (0,) * 8


def test_trace_callback_does_not_receive_patient_sql_parameters(database):
    trace = []

    def factory():
        con = sqlite3.connect(database)
        con.set_trace_callback(trace.append)
        return con

    adapter, approval, _ = prepared(database, factory=factory)
    assert adapter.submit(approval).status is OmopDatabaseStatus.COMMITTED
    assert not trace


@pytest.mark.parametrize("kind", ["concept", "provenance"])
def test_actual_target_catalog_and_mapping_versions_must_match_declared_snapshot(
    database, kind
):
    original = batch()
    altered = []
    for mutation in original.mutations:
        values = mutation.values
        if (
            kind == "concept"
            and mutation.table == "concept"
            and values["concept_id"] == 101
        ):
            values["vocabulary_id"] = "OTHER"
        elif kind == "provenance" and mutation.table == "source_to_concept_map":
            values["vocabulary_version"] = "OLD"
        altered.append(OmopMutation.insert(mutation.table, values))
    with pytest.raises(OmopDatabaseError, match="incompatible_vocabulary"):
        prepared(database, proposal=OmopMutationBatch(altered))
    assert row_counts(database) == (0,) * 8


def test_incompatible_retired_vocabulary_leaves_database_unchanged(database):
    retired = VocabularySnapshot(
        {"SYNTHETIC": "release-1"},
        (
            VocabularyConcept(
                101, "SYNTHETIC", standard_concept="S", invalid_reason="D"
            ),
        ),
    )
    proposal = batch()
    manifest = build_omop_rollback_manifest(
        proposal,
        retired,
        tuple(
            OmopRollbackInstruction(
                i, RollbackStrategy.DELETE_INSERTED_ROW, "sha256:" + f"{i + 1:064x}"
            )
            for i in range(len(proposal.mutations))
        ),
    )
    con = sqlite3.connect(database)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute(
        "UPDATE _openmed_omop_commit_vocabulary SET snapshot_digest=?",
        (retired.snapshot_digest,),
    )
    con.commit()
    with pytest.raises(OmopDatabaseError, match="incompatible_vocabulary"):
        preview_omop_database(
            con,
            proposal,
            vocabulary_snapshot=retired,
            vocabulary_mappings=mappings(),
            rollback_manifest=manifest,
        )
    con.close()
    assert row_counts(database) == (0,) * 8


def test_two_concurrent_authorized_duplicates_commit_only_once(database):
    barrier = threading.Barrier(2)

    def factory():
        barrier.wait(timeout=5)
        return sqlite3.connect(database, timeout=5)

    first, approval, _ = prepared(database, factory=factory)
    second, other, _ = prepared(database, factory=factory)
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [
            workers.submit(first.submit, approval),
            workers.submit(second.submit, other),
        ]
        results = [future.result(timeout=10) for future in futures]
    assert results[0] == results[1]
    assert results[0].status is OmopDatabaseStatus.COMMITTED
    assert row_counts(database) == (2, 1, 1, 1, 1, 1, 1, 1)


def test_expired_receipt_at_final_guard_rolls_back_the_batch(database):
    calls = []
    proposal = batch()

    def authorize(*args):
        calls.append(True)
        return len(calls) <= len(proposal.mutations) + 2

    adapter, approval, _ = prepared(database, authorize=authorize)
    result = adapter.submit(approval)
    assert (
        result.status is OmopDatabaseStatus.DENIED
        and result.code == "authorization_denied"
    )
    assert len(calls) == len(proposal.mutations) + 3
    assert row_counts(database) == (0,) * 8


def test_recovery_refuses_an_unauthorized_receipt(database):
    allowed = [True]
    adapter, approval, _ = prepared(database, authorize=lambda *args: allowed[0])
    assert adapter.submit(approval).status is OmopDatabaseStatus.COMMITTED
    allowed[0] = False
    result = adapter.recover(approval)
    assert (
        result.status is OmopDatabaseStatus.DENIED
        and result.code == "authorization_denied"
    )
    assert row_counts(database) == (2, 1, 1, 1, 1, 1, 1, 1)
