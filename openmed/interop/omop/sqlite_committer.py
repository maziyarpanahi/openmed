"""Explicit local SQLite commit adapter for the loader-owned OMOP subset.

Clinical rows stay in the caller's protected database or private memory. Review,
receipt and result objects contain only digests, counts and closed codes.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from itertools import islice
from typing import Any

from ..omop_rollback_manifest import (
    OmopRollbackManifest,
    OmopRollbackManifestError,
    validate_omop_rollback_manifest,
)
from .cdm_loader import (
    _PRIMARY_KEYS,
    _SCHEMA_COLUMNS,
    _TABLE_ORDER,
    create_omop_schema,
    validate_omop_database,
)
from .mutation_batch import (
    MAX_BATCH_MUTATIONS,
    MutationOperation,
    OmopApprovalBinding,
    OmopMutation,
    OmopMutationBatch,
    OmopMutationPreview,
    OmopRowKey,
)
from .vocabulary_write_gate import (
    VocabularyMappingProvenance,
    VocabularySnapshot,
    VocabularyWriteGate,
    VocabularyWriteGateError,
)

__all__ = [
    "OmopDatabaseError",
    "OmopDatabaseStatus",
    "OmopDatabasePreview",
    "OmopDatabaseResult",
    "initialize_omop_commit_metadata",
    "preview_omop_database",
    "SQLiteOmopBatchCommitter",
]
_RECEIPTS = "_openmed_omop_commit_receipts"
_VOCABULARY = "_openmed_omop_commit_vocabulary"
_DDL = (
    f"CREATE TABLE IF NOT EXISTS {_RECEIPTS} (receipt_digest TEXT PRIMARY KEY, result_json TEXT NOT NULL)",
    f"CREATE TABLE IF NOT EXISTS {_VOCABULARY} (singleton INTEGER PRIMARY KEY CHECK(singleton=1), snapshot_digest TEXT NOT NULL)",
)
_CODES = frozenset(
    {
        "invalid_input",
        "unsupported_batch",
        "unsupported_schema",
        "invalid_snapshot",
        "invalid_rollback",
        "invalid_approval",
        "state_changed",
        "vocabulary_changed",
        "incompatible_vocabulary",
        "reference_check_failed",
        "authorization_denied",
        "admission_denied",
        "rollback_unavailable",
        "transaction_conflict",
        "transaction_failed",
        "rollback_unknown",
        "commit_unknown",
        "receipt_not_found",
        "receipt_conflict",
        "database_unavailable",
        "snapshot_too_large",
    }
)
_TARGETS = {
    "note_nlp": "note_nlp_concept_id",
    "condition_occurrence": "condition_concept_id",
    "drug_exposure": "drug_concept_id",
    "measurement": "measurement_concept_id",
    "procedure_occurrence": "procedure_concept_id",
    "observation": "observation_concept_id",
    "source_to_concept_map": "target_concept_id",
}


class OmopDatabaseError(ValueError):
    """A closed database-adapter failure with no SQL or row-value details."""

    def __init__(self, code: str) -> None:
        self.code = code if type(code) is str and code in _CODES else "invalid_input"
        super().__init__(self.code)


class OmopDatabaseStatus(str, Enum):
    """Explicit durable, refused, failed and uncertain submission outcomes."""

    COMMITTED = "committed"
    CONFLICT = "conflict"
    DENIED = "denied"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode()).hexdigest()


def _require_digest(value: Any) -> None:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(c not in "0123456789abcdef" for c in value[7:])
    ):
        raise OmopDatabaseError("invalid_input")


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _descriptor(con: sqlite3.Connection, table: str) -> tuple[Any, ...]:
    sql = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return (
        tuple(tuple(row) for row in con.execute(f"PRAGMA table_info({_quote(table)})")),
        tuple(
            sorted(
                tuple(row)
                for row in con.execute(f"PRAGMA foreign_key_list({_quote(table)})")
            )
        ),
        " ".join(sql[0].split()).casefold() if sql else None,
    )


@lru_cache(maxsize=1)
def _expected_schema() -> tuple[tuple[str, tuple[Any, ...]], ...]:
    con = sqlite3.connect(":memory:")
    try:
        create_omop_schema(con)
        for ddl in _DDL:
            con.execute(ddl)
        return tuple(
            (name, _descriptor(con, name))
            for name in (*_TABLE_ORDER, _RECEIPTS, _VOCABULARY)
        )
    finally:
        con.close()


def _check_schema(con: sqlite3.Connection, *, metadata: bool = True) -> str:
    expected = _expected_schema()
    for table, descriptor in expected:
        if not metadata and table in {_RECEIPTS, _VOCABULARY}:
            continue
        if _descriptor(con, table) != descriptor:
            raise OmopDatabaseError("unsupported_schema")
    # Refuse triggers and extra inbound FK effects rather than hiding writes
    # that were never in the reviewed mutation batch.
    objects = con.execute(
        "SELECT type,name,tbl_name FROM sqlite_master WHERE type IN ('table','trigger','view')"
    ).fetchmany(257)
    if len(objects) > 256:
        raise OmopDatabaseError("unsupported_schema")
    owned = frozenset((*_TABLE_ORDER, _RECEIPTS, _VOCABULARY))
    for kind, name, table in objects:
        if kind == "trigger" and table in owned:
            raise OmopDatabaseError("unsupported_schema")
        if kind == "table" and name not in owned and not name.startswith("sqlite_"):
            if any(
                row[2] in owned
                for row in con.execute(f"PRAGMA foreign_key_list({_quote(name)})")
            ):
                raise OmopDatabaseError("unsupported_schema")
    return _digest(expected)


def _private_snapshot(
    con: sqlite3.Connection,
) -> tuple[str, tuple[OmopRowKey, ...], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    keys = []
    by_key = {}
    for table in _TABLE_ORDER:
        columns = _SCHEMA_COLUMNS[table]
        selected = con.execute(
            f"SELECT {','.join(_quote(c) for c in columns)} FROM {_quote(table)} ORDER BY {_quote(_PRIMARY_KEYS[table])}"
        ).fetchmany(MAX_BATCH_MUTATIONS + 1 - len(rows))
        for values in selected:
            row = dict(zip(columns, values))
            keys.append(
                OmopRowKey(table, {_PRIMARY_KEYS[table]: row[_PRIMARY_KEYS[table]]})
            )
            by_key[keys[-1].digest] = row
            rows.append({"table": table, "row": row})
        if len(rows) > MAX_BATCH_MUTATIONS:
            raise OmopDatabaseError("snapshot_too_large")
    if validate_omop_database(con):
        raise OmopDatabaseError("invalid_snapshot")
    return _digest(rows), tuple(keys), by_key


def _bounded_mappings(
    values: Iterable[VocabularyMappingProvenance],
) -> tuple[VocabularyMappingProvenance, ...]:
    try:
        if isinstance(values, (str, bytes, dict)):
            raise OmopDatabaseError("invalid_input")
        result = tuple(islice(values, MAX_BATCH_MUTATIONS + 1))
        if len(result) > MAX_BATCH_MUTATIONS or any(
            type(m) is not VocabularyMappingProvenance for m in result
        ):
            raise OmopDatabaseError("invalid_input")
        return result
    except Exception:
        raise OmopDatabaseError("invalid_input") from None


def _validate_batch(batch: OmopMutationBatch) -> None:
    if type(batch) is not OmopMutationBatch:
        raise OmopDatabaseError("invalid_input")
    descriptors = dict(_expected_schema())
    for mutation in batch.mutations:
        table, values = mutation.table, mutation.values
        if table not in _TABLE_ORDER or mutation._explicit_references:
            raise OmopDatabaseError("unsupported_batch")
        if set(mutation.key.values) != {_PRIMARY_KEYS[table]} or any(
            name not in _SCHEMA_COLUMNS[table] for name in values
        ):
            raise OmopDatabaseError("unsupported_batch")
        key_value = mutation.key.values[_PRIMARY_KEYS[table]]
        if (
            type(key_value) is not int
            or not (0 if table == "concept" else 1) <= key_value < 2**63
        ):
            raise OmopDatabaseError("unsupported_batch")
        if mutation.operation is MutationOperation.INSERT and set(values) != set(
            _SCHEMA_COLUMNS[table]
        ):
            raise OmopDatabaseError("unsupported_batch")
        if (
            mutation.operation is MutationOperation.UPDATE
            and _PRIMARY_KEYS[table] in values
        ):
            raise OmopDatabaseError("unsupported_batch")
        types = {row[1]: row[2] for row in descriptors[table][0]}
        for name, value in {**mutation.key.values, **values}.items():
            if value is None:
                continue
            if types[name] == "BIGINT":
                if type(value) is not int or not -(2**63) <= value < 2**63:
                    raise OmopDatabaseError("unsupported_batch")
            elif type(value) is not str:
                raise OmopDatabaseError("unsupported_batch")


@dataclass(frozen=True, slots=True)
class OmopDatabasePreview:
    """Value-free review binding for one batch and exact private database state."""

    batch_preview: OmopMutationPreview = field(repr=False)
    state_digest: str
    schema_digest: str
    vocabulary_snapshot_digest: str
    vocabulary_report_digest: str
    rollback_manifest_digest: str

    def __post_init__(self) -> None:
        if type(self.batch_preview) is not OmopMutationPreview:
            raise OmopDatabaseError("invalid_input")
        for value in (
            self.state_digest,
            self.schema_digest,
            self.vocabulary_snapshot_digest,
            self.vocabulary_report_digest,
            self.rollback_manifest_digest,
        ):
            _require_digest(value)
        for value in (
            self.batch_preview.batch_digest,
            self.batch_preview.preview_digest,
            self.batch_preview.reference_snapshot_digest,
        ):
            _require_digest(value)
        if (
            type(self.batch_preview.mutations) is not tuple
            or not 1 <= len(self.batch_preview.mutations) <= MAX_BATCH_MUTATIONS
        ):
            raise OmopDatabaseError("invalid_input")

    def to_dict(self) -> dict[str, Any]:
        """Return hashes and counts, excluding table/field names and row values."""
        return {
            "schema": "openmed.interop.omop.database_preview.v1",
            "batch_digest": self.batch_preview.batch_digest,
            "batch_preview_digest": self.batch_preview.preview_digest,
            "reference_snapshot_digest": self.batch_preview.reference_snapshot_digest,
            "mutation_count": len(self.batch_preview.mutations),
            "state_digest": self.state_digest,
            "schema_digest": self.schema_digest,
            "vocabulary_snapshot_digest": self.vocabulary_snapshot_digest,
            "vocabulary_report_digest": self.vocabulary_report_digest,
            "rollback_manifest_digest": self.rollback_manifest_digest,
        }

    @property
    def preview_digest(self) -> str:
        """Return the exact digest the external approval verifier must authorize."""
        return _digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class OmopDatabaseResult:
    """Value-free transaction result; unknown never asserts an applied count."""

    status: OmopDatabaseStatus
    code: str | None
    batch_digest: str
    preview_digest: str
    approval_receipt_digest: str
    rollback_manifest_digest: str
    proposed_count: int
    applied_count: int | None

    def __post_init__(self) -> None:
        if type(self.status) is not OmopDatabaseStatus or (
            self.code is not None
            and (type(self.code) is not str or self.code not in _CODES)
        ):
            raise OmopDatabaseError("invalid_input")
        for value in (
            self.batch_digest,
            self.preview_digest,
            self.approval_receipt_digest,
            self.rollback_manifest_digest,
        ):
            _require_digest(value)
        if (
            type(self.proposed_count) is not int
            or not 1 <= self.proposed_count <= MAX_BATCH_MUTATIONS
        ):
            raise OmopDatabaseError("invalid_input")
        expected = (
            self.proposed_count
            if self.status is OmopDatabaseStatus.COMMITTED
            else None
            if self.status is OmopDatabaseStatus.UNKNOWN
            else 0
        )
        if (
            (self.applied_count is not None and type(self.applied_count) is not int)
            or self.applied_count != expected
            or (self.status is OmopDatabaseStatus.COMMITTED) != (self.code is None)
        ):
            raise OmopDatabaseError("invalid_input")

    def to_dict(self) -> dict[str, Any]:
        """Return the controlled outcome for a reviewer or recovery ledger."""
        return {
            "status": self.status.value,
            "code": self.code,
            "batch_digest": self.batch_digest,
            "preview_digest": self.preview_digest,
            "approval_receipt_digest": self.approval_receipt_digest,
            "rollback_manifest_digest": self.rollback_manifest_digest,
            "proposed_count": self.proposed_count,
            "applied_count": self.applied_count,
        }


def initialize_omop_commit_metadata(
    con: sqlite3.Connection, snapshot: VocabularySnapshot
) -> str:
    """Explicitly provision receipt and vocabulary metadata on a prepared database.

    Args:
        con: Caller-owned SQLite connection outside any transaction.
        snapshot: Operator-attested vocabulary state, kept immutable here.

    Returns:
        The registered snapshot digest. Existing different metadata is refused.
    """
    begun = commit_attempted = False
    try:
        if (
            not isinstance(con, sqlite3.Connection)
            or type(snapshot) is not VocabularySnapshot
            or con.in_transaction
        ):
            raise OmopDatabaseError("invalid_input")
        con.set_trace_callback(None)
        con.execute("PRAGMA foreign_keys=ON")
        _check_schema(con, metadata=False)
        con.execute("BEGIN IMMEDIATE")
        begun = True
        for ddl in _DDL:
            con.execute(ddl)
        current = con.execute(
            f"SELECT snapshot_digest FROM {_VOCABULARY} WHERE singleton=1"
        ).fetchone()
        if current is not None and current[0] != snapshot.snapshot_digest:
            raise OmopDatabaseError("vocabulary_changed")
        if current is None:
            con.execute(
                f"INSERT INTO {_VOCABULARY} VALUES (1,?)", (snapshot.snapshot_digest,)
            )
        _check_schema(con)
        commit_attempted = True
        con.commit()
        if con.in_transaction:
            raise OmopDatabaseError("commit_unknown")
        begun = False
        return snapshot.snapshot_digest
    except Exception as exc:
        if begun:
            try:
                con.rollback()
            except Exception:
                raise OmopDatabaseError("rollback_unknown") from None
        if commit_attempted:
            raise OmopDatabaseError("commit_unknown") from None
        if type(exc) is OmopDatabaseError:
            raise
        raise OmopDatabaseError("database_unavailable") from None


def preview_omop_database(
    con: sqlite3.Connection,
    batch: OmopMutationBatch,
    *,
    vocabulary_snapshot: VocabularySnapshot,
    vocabulary_mappings: Iterable[VocabularyMappingProvenance],
    rollback_manifest: OmopRollbackManifest,
) -> OmopDatabasePreview:
    """Read a bounded private target snapshot and construct a complete review.

    Args:
        con: Caller-owned prepared SQLite connection with foreign keys enabled.
        batch: Existing immutable mutation proposal for the loader-owned subset.
        vocabulary_snapshot: Exact operator-attested target vocabulary snapshot.
        vocabulary_mappings: Provenance for every inserted clinical target.
        rollback_manifest: Existing manifest with complete structural coverage.

    Returns:
        Value-free row-state, schema, vocabulary, rollback and batch bindings.
    """
    owned_read = False
    try:
        _validate_batch(batch)
        if (
            not isinstance(con, sqlite3.Connection)
            or type(vocabulary_snapshot) is not VocabularySnapshot
        ):
            raise OmopDatabaseError("invalid_input")
        con.set_trace_callback(None)
        if not con.in_transaction:
            con.execute("BEGIN")
            owned_read = True
        if con.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise OmopDatabaseError("unsupported_schema")
        schema = _check_schema(con)
        current = con.execute(
            f"SELECT snapshot_digest FROM {_VOCABULARY} WHERE singleton=1"
        ).fetchone()
        if current != (vocabulary_snapshot.snapshot_digest,):
            raise OmopDatabaseError("vocabulary_changed")
        validate_omop_rollback_manifest(rollback_manifest, batch, vocabulary_snapshot)
        mappings = _bounded_mappings(vocabulary_mappings)
        state, keys, previous = _private_snapshot(con)
        required = set()
        by_id = {m.target_concept_id: m for m in mappings}
        for mutation in batch.mutations:
            if (
                mutation.operation is MutationOperation.TOMBSTONE
                or mutation.table not in _TARGETS
            ):
                continue
            proposed = {**previous.get(mutation.key.digest, {}), **mutation.values}
            target = proposed.get(_TARGETS[mutation.table])
            if type(target) is not int or target < 0:
                raise OmopDatabaseError("incompatible_vocabulary")
            required.add(target)
            if mutation.table == "source_to_concept_map" and target in by_id:
                if (
                    proposed.get("target_vocabulary_id"),
                    proposed.get("vocabulary_version"),
                ) != (
                    by_id[target].target_vocabulary_id,
                    by_id[target].vocabulary_version,
                ):
                    raise OmopDatabaseError("incompatible_vocabulary")
        if (
            len(mappings) > MAX_BATCH_MUTATIONS
            or any(type(m) is not VocabularyMappingProvenance for m in mappings)
            or len({m.target_concept_id for m in mappings}) != len(mappings)
            or required != {m.target_concept_id for m in mappings}
        ):
            raise OmopDatabaseError("incompatible_vocabulary")
        report = VocabularyWriteGate(vocabulary_snapshot).assert_writable(mappings)
        concepts = {key: row for key, row in previous.items() if "concept_id" in row}
        for mutation in batch.mutations:
            if mutation.table == "concept":
                if mutation.operation is MutationOperation.TOMBSTONE:
                    concepts.pop(mutation.key.digest, None)
                else:
                    concepts[mutation.key.digest] = {
                        **concepts.get(mutation.key.digest, {}),
                        **mutation.values,
                    }
        for mapping in mappings:
            actual = concepts.get(
                OmopRowKey("concept", {"concept_id": mapping.target_concept_id}).digest
            )
            declared = vocabulary_snapshot.concept(mapping.target_concept_id)
            if (
                actual is None
                or declared is None
                or actual.get("vocabulary_id") != declared.vocabulary_id
                or (actual.get("standard_concept") or None) != declared.standard_concept
            ):
                raise OmopDatabaseError("incompatible_vocabulary")
        preview = batch.preview(existing_rows=keys)
        if not preview.is_valid:
            raise OmopDatabaseError("reference_check_failed")
        return OmopDatabasePreview(
            preview,
            state,
            schema,
            vocabulary_snapshot.snapshot_digest,
            report.report_digest,
            rollback_manifest.manifest_digest,
        )
    except OmopDatabaseError:
        raise
    except VocabularyWriteGateError:
        raise OmopDatabaseError("incompatible_vocabulary") from None
    except OmopRollbackManifestError:
        raise OmopDatabaseError("invalid_rollback") from None
    except Exception:
        raise OmopDatabaseError("invalid_input") from None
    finally:
        if owned_read:
            try:
                con.rollback()
            except Exception:
                raise OmopDatabaseError("database_unavailable") from None


class SQLiteOmopBatchCommitter:
    """Explicit transactional committer behind the existing batch protocol.

    Args:
        batch: Exact immutable batch that was previewed.
        preview: Independently reviewed database preview including row-state hash.
        vocabulary_snapshot: Exact operator-attested target vocabulary snapshot.
        vocabulary_mappings: Existing target provenance for the write gate.
        rollback_manifest: Complete manifest for protected rollback custody.
        connection_factory: Explicit trusted factory; each returned connection is
            owned by this adapter and closed after submission or recovery.
        authorize: Required verifier of receipt authenticity, reviewer authority,
            expiry and binding to BOTH the outer preview and inner batch preview.
        admit: Required default-off workflow admission check, rechecked per effect.
        rollback_ready: Required protected-custody readiness verifier.
    """

    def __init__(
        self,
        batch: OmopMutationBatch,
        preview: OmopDatabasePreview,
        *,
        vocabulary_snapshot: VocabularySnapshot,
        vocabulary_mappings: Iterable[VocabularyMappingProvenance],
        rollback_manifest: OmopRollbackManifest,
        connection_factory: Callable[[], sqlite3.Connection],
        authorize: Callable[[OmopApprovalBinding, OmopDatabasePreview], bool],
        admit: Callable[[str], bool],
        rollback_ready: Callable[[OmopRollbackManifest], bool],
    ) -> None:
        _validate_batch(batch)
        if (
            type(preview) is not OmopDatabasePreview
            or preview.batch_preview.batch_digest != batch.batch_digest
            or not all(
                callable(f)
                for f in (connection_factory, authorize, admit, rollback_ready)
            )
        ):
            raise OmopDatabaseError("invalid_input")
        self.batch, self.preview = batch, preview
        if (
            type(vocabulary_snapshot) is not VocabularySnapshot
            or type(rollback_manifest) is not OmopRollbackManifest
        ):
            raise OmopDatabaseError("invalid_input")
        validate_omop_rollback_manifest(rollback_manifest, batch, vocabulary_snapshot)
        if (
            preview.vocabulary_snapshot_digest != vocabulary_snapshot.snapshot_digest
            or preview.rollback_manifest_digest != rollback_manifest.manifest_digest
        ):
            raise OmopDatabaseError("invalid_input")
        self._snapshot, self._mappings, self._manifest = (
            vocabulary_snapshot,
            _bounded_mappings(vocabulary_mappings),
            rollback_manifest,
        )
        self._factory, self._authorize, self._admit, self._rollback_ready = (
            connection_factory,
            authorize,
            admit,
            rollback_ready,
        )
        self.last_result: OmopDatabaseResult | None = None

    def _result(
        self,
        approval: OmopApprovalBinding,
        status: OmopDatabaseStatus,
        code: str | None,
    ) -> OmopDatabaseResult:
        return OmopDatabaseResult(
            status,
            code,
            self.batch.batch_digest,
            self.preview.preview_digest,
            approval.approval_receipt_digest,
            self._manifest.manifest_digest,
            len(self.batch.mutations),
            len(self.batch.mutations)
            if status is OmopDatabaseStatus.COMMITTED
            else None
            if status is OmopDatabaseStatus.UNKNOWN
            else 0,
        )

    def _guards(self, approval: OmopApprovalBinding) -> None:
        if self._authorize(approval, self.preview) is not True:
            raise OmopDatabaseError("authorization_denied")
        if self._admit(self.preview.preview_digest) is not True:
            raise OmopDatabaseError("admission_denied")
        if self._rollback_ready(self._manifest) is not True:
            raise OmopDatabaseError("rollback_unavailable")

    def _approval(self, approval: OmopApprovalBinding) -> None:
        if type(approval) is not OmopApprovalBinding or (
            approval.batch_digest,
            approval.preview_digest,
            approval.reference_snapshot_digest,
        ) != (
            self.batch.batch_digest,
            self.preview.batch_preview.preview_digest,
            self.preview.batch_preview.reference_snapshot_digest,
        ):
            raise OmopDatabaseError("invalid_approval")

    def _receipt(
        self, con: sqlite3.Connection, approval: OmopApprovalBinding
    ) -> OmopDatabaseResult | None:
        row = con.execute(
            f"SELECT result_json FROM {_RECEIPTS} WHERE receipt_digest=?",
            (approval.approval_receipt_digest,),
        ).fetchone()
        if row is None:
            return None
        expected = self._result(approval, OmopDatabaseStatus.COMMITTED, None)
        if row != (_json(expected.to_dict()),):
            raise OmopDatabaseError("receipt_conflict")
        return expected

    def submit(self, approval: OmopApprovalBinding) -> OmopDatabaseResult:
        """Explicitly submit one authorized batch, preserving unknown outcomes."""
        if type(approval) is not OmopApprovalBinding:
            raise OmopDatabaseError("invalid_approval")
        con = None
        begun = commit_attempted = False
        try:
            self._approval(approval)
            self._guards(approval)
            con = self._factory()
            if not isinstance(con, sqlite3.Connection) or con.in_transaction:
                raise OmopDatabaseError("transaction_conflict")
            con.set_trace_callback(None)
            con.execute("PRAGMA foreign_keys=ON")
            con.execute("BEGIN IMMEDIATE")
            begun = True
            _check_schema(con)
            previous = self._receipt(con, approval)
            if previous is not None:
                con.rollback()
                begun = False
                self.last_result = previous
                return previous
            fresh = preview_omop_database(
                con,
                self.batch,
                vocabulary_snapshot=self._snapshot,
                vocabulary_mappings=self._mappings,
                rollback_manifest=self._manifest,
            )
            if fresh != self.preview:
                raise OmopDatabaseError("state_changed")
            self._guards(approval)
            for mutation in self.batch.mutations:
                self._guards(approval)
                values, key = mutation.values, mutation.key.values
                table = _quote(mutation.table)
                if mutation.operation is MutationOperation.INSERT:
                    columns = tuple(values)
                    cursor = con.execute(
                        f"INSERT INTO {table} ({','.join(_quote(c) for c in columns)}) VALUES ({','.join('?' for _ in columns)})",
                        tuple(values[c] for c in columns),
                    )
                else:
                    where = " AND ".join(f"{_quote(c)}=?" for c in key)
                    if mutation.operation is MutationOperation.UPDATE:
                        cursor = con.execute(
                            f"UPDATE {table} SET {','.join(_quote(c) + '=?' for c in values)} WHERE {where}",
                            (*values.values(), *key.values()),
                        )
                    else:
                        cursor = con.execute(
                            f"DELETE FROM {table} WHERE {where}", tuple(key.values())
                        )
                if cursor.rowcount != 1:
                    raise OmopDatabaseError("transaction_conflict")
            _private_snapshot(con)
            if con.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise OmopDatabaseError("invalid_snapshot")
            self._guards(approval)
            result = self._result(approval, OmopDatabaseStatus.COMMITTED, None)
            con.execute(
                f"INSERT INTO {_RECEIPTS} VALUES (?,?)",
                (approval.approval_receipt_digest, _json(result.to_dict())),
            )
            commit_attempted = True
            con.commit()
            if con.in_transaction:
                raise OmopDatabaseError("commit_unknown")
            begun = False
            self.last_result = result
            return result
        except Exception as exc:
            code = exc.code if type(exc) is OmopDatabaseError else "transaction_failed"
            status = (
                OmopDatabaseStatus.DENIED
                if code
                in {"authorization_denied", "admission_denied", "rollback_unavailable"}
                else OmopDatabaseStatus.CONFLICT
                if code
                in {
                    "invalid_approval",
                    "state_changed",
                    "vocabulary_changed",
                    "incompatible_vocabulary",
                    "reference_check_failed",
                    "unsupported_schema",
                    "invalid_snapshot",
                    "receipt_conflict",
                    "transaction_conflict",
                }
                or isinstance(exc, sqlite3.IntegrityError)
                else OmopDatabaseStatus.FAILED
            )
            if commit_attempted:
                status, code = OmopDatabaseStatus.UNKNOWN, "commit_unknown"
            if begun and con is not None:
                try:
                    con.rollback()
                except Exception:
                    status, code = OmopDatabaseStatus.UNKNOWN, "rollback_unknown"
            self.last_result = self._result(approval, status, code)
            return self.last_result
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass

    def recover(self, approval: OmopApprovalBinding) -> OmopDatabaseResult:
        """Read the exact durable receipt without replaying or issuing any writes."""
        self._approval(approval)
        con = None
        try:
            if self._authorize(approval, self.preview) is not True:
                raise OmopDatabaseError("authorization_denied")
            con = self._factory()
            con.set_trace_callback(None)
            con.execute("PRAGMA query_only=ON")
            _check_schema(con)
            result = self._receipt(con, approval)
            if result is None:
                return self._result(
                    approval, OmopDatabaseStatus.UNKNOWN, "receipt_not_found"
                )
            return result
        except Exception as exc:
            if type(exc) is OmopDatabaseError and exc.code == "authorization_denied":
                return self._result(
                    approval, OmopDatabaseStatus.DENIED, "authorization_denied"
                )
            return self._result(
                approval, OmopDatabaseStatus.UNKNOWN, "database_unavailable"
            )
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass

    def commit_batch(
        self,
        mutations: tuple[OmopMutation, ...],
        *,
        batch_digest: str,
        approval: OmopApprovalBinding,
    ) -> None:
        """Implement the shipped protocol; use submit/last_result for richer states."""
        if mutations != self.batch.mutations or batch_digest != self.batch.batch_digest:
            raise OmopDatabaseError("invalid_approval")
        result = self.submit(approval)
        if result.status is not OmopDatabaseStatus.COMMITTED:
            raise OmopDatabaseError(result.code or "transaction_failed")
