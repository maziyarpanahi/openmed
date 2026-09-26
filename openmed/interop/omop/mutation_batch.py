"""Deterministic, reviewable batches of proposed OMOP mutations.

The batch keeps row values in memory for the eventual local committer, while
all preview and result types expose only table/field metadata, counts, stable
digests, and closed reason codes. Approval-token issuance and generic
side-effect rendering deliberately remain outside this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

MUTATION_BATCH_SCHEMA = "openmed.interop.omop.mutation_batch.v1"
MAX_BATCH_MUTATIONS = 10_000

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]{0,62}")

_PRIMARY_KEYS: Mapping[str, str] = {
    "concept": "concept_id",
    "person": "person_id",
    "visit_occurrence": "visit_occurrence_id",
    "note": "note_id",
    "note_nlp": "note_nlp_id",
    "condition_occurrence": "condition_occurrence_id",
    "drug_exposure": "drug_exposure_id",
    "measurement": "measurement_id",
    "procedure_occurrence": "procedure_occurrence_id",
    "observation": "observation_id",
    "source_to_concept_map": "source_to_concept_map_id",
}

_FOREIGN_KEYS: Mapping[str, Mapping[str, tuple[str, str]]] = {
    "visit_occurrence": {
        "person_id": ("person", "person_id"),
        "visit_concept_id": ("concept", "concept_id"),
        "visit_source_concept_id": ("concept", "concept_id"),
    },
    "note": {
        "person_id": ("person", "person_id"),
        "visit_occurrence_id": (
            "visit_occurrence",
            "visit_occurrence_id",
        ),
        "note_type_concept_id": ("concept", "concept_id"),
        "note_class_concept_id": ("concept", "concept_id"),
        "encoding_concept_id": ("concept", "concept_id"),
        "language_concept_id": ("concept", "concept_id"),
    },
    "note_nlp": {
        "note_id": ("note", "note_id"),
        "section_concept_id": ("concept", "concept_id"),
        "note_nlp_concept_id": ("concept", "concept_id"),
        "note_nlp_source_concept_id": ("concept", "concept_id"),
        "note_nlp_event_field_concept_id": ("concept", "concept_id"),
    },
    "condition_occurrence": {
        "person_id": ("person", "person_id"),
        "condition_concept_id": ("concept", "concept_id"),
        "condition_type_concept_id": ("concept", "concept_id"),
        "condition_source_concept_id": ("concept", "concept_id"),
        "visit_occurrence_id": (
            "visit_occurrence",
            "visit_occurrence_id",
        ),
        "note_id": ("note", "note_id"),
        "note_nlp_id": ("note_nlp", "note_nlp_id"),
    },
    "drug_exposure": {
        "person_id": ("person", "person_id"),
        "drug_concept_id": ("concept", "concept_id"),
        "drug_type_concept_id": ("concept", "concept_id"),
        "drug_source_concept_id": ("concept", "concept_id"),
        "visit_occurrence_id": (
            "visit_occurrence",
            "visit_occurrence_id",
        ),
        "note_id": ("note", "note_id"),
        "note_nlp_id": ("note_nlp", "note_nlp_id"),
    },
    "measurement": {
        "person_id": ("person", "person_id"),
        "measurement_concept_id": ("concept", "concept_id"),
        "measurement_type_concept_id": ("concept", "concept_id"),
        "measurement_source_concept_id": ("concept", "concept_id"),
        "visit_occurrence_id": (
            "visit_occurrence",
            "visit_occurrence_id",
        ),
        "note_id": ("note", "note_id"),
        "note_nlp_id": ("note_nlp", "note_nlp_id"),
    },
    "procedure_occurrence": {
        "person_id": ("person", "person_id"),
        "procedure_concept_id": ("concept", "concept_id"),
        "procedure_type_concept_id": ("concept", "concept_id"),
        "procedure_source_concept_id": ("concept", "concept_id"),
        "visit_occurrence_id": (
            "visit_occurrence",
            "visit_occurrence_id",
        ),
        "note_id": ("note", "note_id"),
        "note_nlp_id": ("note_nlp", "note_nlp_id"),
    },
    "observation": {
        "person_id": ("person", "person_id"),
        "observation_concept_id": ("concept", "concept_id"),
        "observation_type_concept_id": ("concept", "concept_id"),
        "observation_source_concept_id": ("concept", "concept_id"),
        "visit_occurrence_id": (
            "visit_occurrence",
            "visit_occurrence_id",
        ),
        "note_id": ("note", "note_id"),
        "note_nlp_id": ("note_nlp", "note_nlp_id"),
    },
    "source_to_concept_map": {
        "source_concept_id": ("concept", "concept_id"),
        "target_concept_id": ("concept", "concept_id"),
        "note_nlp_id": ("note_nlp", "note_nlp_id"),
    },
}

Scalar = str | int | float | bool | None
_Items = tuple[tuple[str, Scalar], ...]


class MutationOperation(str, Enum):
    """Closed set of staged row operations."""

    INSERT = "insert"
    UPDATE = "update"
    TOMBSTONE = "tombstone"


class CommitStatus(str, Enum):
    """Closed set of batch commit outcomes."""

    COMMITTED = "committed"
    FAILED = "failed"


class OmopMutationError(ValueError):
    """A value-free validation error for staged mutations."""

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True, repr=False, init=False)
class OmopRowKey:
    """An in-memory OMOP row identity whose representation hides values."""

    table: str
    _items: _Items = field(repr=False)

    def __init__(self, table: str, values: Mapping[str, Scalar]) -> None:
        object.__setattr__(self, "table", _validate_identifier(table, "table"))
        object.__setattr__(self, "_items", _normalize_mapping(values, "key"))

    @property
    def values(self) -> dict[str, Scalar]:
        """Return a copy of the row-key values for a local committer."""

        return dict(self._items)

    @property
    def digest(self) -> str:
        """Return the stable digest of this table-qualified row identity."""

        return _digest({"table": self.table, "key": dict(self._items)})

    def __repr__(self) -> str:
        return f"OmopRowKey(table={self.table!r}, digest={self.digest!r})"


@dataclass(frozen=True, slots=True, repr=False, init=False)
class OmopMutation:
    """One proposed OMOP insert, update, or tombstone.

    Use :meth:`insert`, :meth:`update`, or :meth:`tombstone`. Raw key and row
    values are intentionally absent from ``repr`` and all public summaries.
    """

    operation: MutationOperation
    table: str
    _key_items: _Items = field(repr=False)
    _value_items: _Items = field(repr=False)
    _explicit_references: tuple[OmopRowKey, ...] = field(repr=False)

    def __init__(
        self,
        operation: MutationOperation,
        table: str,
        key: Mapping[str, Scalar],
        values: Mapping[str, Scalar],
        references: Iterable[OmopRowKey] = (),
    ) -> None:
        if not isinstance(operation, MutationOperation):
            raise OmopMutationError("unknown_operation", "operation")
        normalized_table = _validate_identifier(table, "table")
        key_items = _normalize_mapping(key, "key")
        value_items = _normalize_mapping(values, "values", allow_empty=True)
        expected_primary_key = _PRIMARY_KEYS.get(normalized_table)
        if expected_primary_key is not None and tuple(
            name for name, _ in key_items
        ) != (expected_primary_key,):
            raise OmopMutationError("invalid_primary_key", "key")
        if operation is MutationOperation.INSERT and not value_items:
            raise OmopMutationError("empty_insert", "values")
        if operation is MutationOperation.UPDATE and not value_items:
            raise OmopMutationError("empty_update", "values")
        if operation is MutationOperation.TOMBSTONE and value_items:
            raise OmopMutationError("tombstone_has_values", "values")
        if operation is MutationOperation.INSERT:
            value_map = dict(value_items)
            if any(value_map.get(name, object()) != item for name, item in key_items):
                raise OmopMutationError("key_mismatch", "key")
        if (
            set(dict(key_items)) & set(dict(value_items))
            and operation is not MutationOperation.INSERT
        ):
            raise OmopMutationError("key_field_changed", "values")

        normalized_references = _normalize_references(references)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "table", normalized_table)
        object.__setattr__(self, "_key_items", key_items)
        object.__setattr__(self, "_value_items", value_items)
        object.__setattr__(self, "_explicit_references", normalized_references)

    @classmethod
    def insert(
        cls,
        table: str,
        row: Mapping[str, Scalar],
        *,
        key: Mapping[str, Scalar] | None = None,
        references: Iterable[OmopRowKey] = (),
    ) -> OmopMutation:
        """Stage an insert, deriving the key for supported OpenMed OMOP tables."""

        normalized_table = _validate_identifier(table, "table")
        row_items = _normalize_mapping(row, "row")
        row_values = dict(row_items)
        if key is None:
            primary_key = _PRIMARY_KEYS.get(normalized_table)
            if primary_key is None:
                raise OmopMutationError("key_required", "key")
            if primary_key not in row_values:
                raise OmopMutationError("missing_primary_key", "row")
            key = {primary_key: row_values[primary_key]}
        return cls(
            MutationOperation.INSERT,
            normalized_table,
            key,
            row_values,
            references,
        )

    @classmethod
    def update(
        cls,
        table: str,
        key: Mapping[str, Scalar],
        changes: Mapping[str, Scalar],
        *,
        references: Iterable[OmopRowKey] = (),
    ) -> OmopMutation:
        """Stage changes to an existing row."""

        return cls(MutationOperation.UPDATE, table, key, changes, references)

    @classmethod
    def tombstone(
        cls,
        table: str,
        key: Mapping[str, Scalar],
    ) -> OmopMutation:
        """Stage deletion of an existing row without exposing its identity."""

        return cls(MutationOperation.TOMBSTONE, table, key, {})

    @property
    def key(self) -> OmopRowKey:
        """Return the table-qualified row key."""

        return OmopRowKey(self.table, dict(self._key_items))

    @property
    def values(self) -> dict[str, Scalar]:
        """Return a copy of values for a trusted local committer."""

        return dict(self._value_items)

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return sorted field names without their values."""

        return tuple(name for name, _ in self._value_items)

    @property
    def references(self) -> tuple[tuple[str, OmopRowKey], ...]:
        """Return inferred and explicitly declared references."""

        references: list[tuple[str, OmopRowKey]] = []
        values = dict(self._value_items)
        for field_name, (target_table, target_field) in _FOREIGN_KEYS.get(
            self.table, {}
        ).items():
            if field_name not in values or values[field_name] is None:
                continue
            value = values[field_name]
            if target_table == "concept" and value == 0:
                continue
            references.append(
                (field_name, OmopRowKey(target_table, {target_field: value}))
            )
        references.extend(
            ("explicit_reference", value) for value in self._explicit_references
        )

        unique: dict[tuple[str, str], tuple[str, OmopRowKey]] = {}
        for field_name, reference in references:
            unique[(field_name, reference.digest)] = (field_name, reference)
        return tuple(unique[key] for key in sorted(unique))

    @property
    def row_digest(self) -> str:
        """Return a stable digest over the full proposed mutation."""

        return _digest(
            {
                "operation": self.operation.value,
                "table": self.table,
                "key": dict(self._key_items),
                "values": dict(self._value_items),
                "references": [
                    {"field": field_name, "key_digest": reference.digest}
                    for field_name, reference in self.references
                ],
            }
        )

    def __repr__(self) -> str:
        return (
            "OmopMutation("
            f"operation={self.operation.value!r}, table={self.table!r}, "
            f"row_digest={self.row_digest!r})"
        )


@dataclass(frozen=True, slots=True)
class OmopMutationSummary:
    """Value-free preview metadata for one ordered mutation."""

    ordinal: int
    operation: MutationOperation
    table: str
    field_names: tuple[str, ...]
    reference_count: int
    row_digest: str

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-safe summary fields."""

        return {
            "field_names": list(self.field_names),
            "operation": self.operation.value,
            "ordinal": self.ordinal,
            "reference_count": self.reference_count,
            "row_digest": self.row_digest,
            "table": self.table,
        }


@dataclass(frozen=True, slots=True)
class OmopReferenceIssue:
    """Value-free referential-check failure."""

    ordinal: int
    code: str
    table: str
    field_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-safe issue fields."""

        return {
            "code": self.code,
            "field_name": self.field_name,
            "ordinal": self.ordinal,
            "table": self.table,
        }


@dataclass(frozen=True, slots=True)
class OmopMutationPreview:
    """Reviewable, value-free view of an ordered mutation batch."""

    batch_digest: str
    preview_digest: str
    reference_snapshot_digest: str
    mutations: tuple[OmopMutationSummary, ...]
    issues: tuple[OmopReferenceIssue, ...]
    operation_counts: tuple[tuple[str, int], ...]
    schema: str = MUTATION_BATCH_SCHEMA

    @property
    def is_valid(self) -> bool:
        """Whether referential checks found no issues."""

        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic value-free audit summary."""

        return {
            "batch_digest": self.batch_digest,
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "mutation_count": len(self.mutations),
            "mutations": [mutation.to_dict() for mutation in self.mutations],
            "operation_counts": dict(self.operation_counts),
            "preview_digest": self.preview_digest,
            "reference_snapshot_digest": self.reference_snapshot_digest,
            "schema": self.schema,
        }

    def to_json(self) -> str:
        """Serialize the audit summary with stable key ordering."""

        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class OmopApprovalBinding:
    """Digest-only binding between an approved preview and a receipt."""

    batch_digest: str
    preview_digest: str
    reference_snapshot_digest: str
    approval_receipt_digest: str

    def __post_init__(self) -> None:
        _validate_digest(self.batch_digest, "batch_digest")
        _validate_digest(self.preview_digest, "preview_digest")
        _validate_digest(
            self.reference_snapshot_digest,
            "reference_snapshot_digest",
        )
        _validate_digest(self.approval_receipt_digest, "approval_receipt_digest")


@dataclass(frozen=True, slots=True)
class OmopCommitResult:
    """Explicit, value-free outcome from attempting one atomic batch commit."""

    status: CommitStatus
    batch_digest: str
    preview_digest: str
    approval_receipt_digest: str
    mutation_count: int
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-safe result fields."""

        return {
            "approval_receipt_digest": self.approval_receipt_digest,
            "batch_digest": self.batch_digest,
            "error_code": self.error_code,
            "mutation_count": self.mutation_count,
            "preview_digest": self.preview_digest,
            "status": self.status.value,
        }


class OmopBatchCommitter(Protocol):
    """Local adapter contract for atomically applying an ordered batch."""

    def commit_batch(
        self,
        mutations: tuple[OmopMutation, ...],
        *,
        batch_digest: str,
        approval: OmopApprovalBinding,
    ) -> None:
        """Commit all mutations atomically, or raise without a partial commit."""


@dataclass(frozen=True, slots=True, repr=False, init=False)
class OmopMutationBatch:
    """Immutable ordered collection of proposed OMOP row mutations."""

    mutations: tuple[OmopMutation, ...]

    def __init__(self, mutations: Iterable[OmopMutation]) -> None:
        if isinstance(mutations, (str, bytes, bytearray, Mapping)):
            raise OmopMutationError("invalid_mutation_collection")
        try:
            normalized = tuple(mutations)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise OmopMutationError("invalid_mutation_collection") from None
        if not normalized:
            raise OmopMutationError("empty_batch")
        if len(normalized) > MAX_BATCH_MUTATIONS:
            raise OmopMutationError("batch_too_large")
        if any(type(mutation) is not OmopMutation for mutation in normalized):
            raise OmopMutationError("invalid_mutation")
        object.__setattr__(self, "mutations", normalized)

    @property
    def batch_digest(self) -> str:
        """Return a stable digest that preserves mutation order."""

        return _digest(
            {
                "schema": MUTATION_BATCH_SCHEMA,
                "row_digests": [mutation.row_digest for mutation in self.mutations],
            }
        )

    def preview(
        self,
        *,
        existing_rows: Iterable[OmopRowKey] = (),
    ) -> OmopMutationPreview:
        """Build a deterministic preview and perform ordered reference checks.

        ``existing_rows`` should contain the local row identities needed by
        this batch. Their values never appear in the returned preview.
        """

        existing = _normalize_existing_rows(existing_rows)
        live_keys = {row.digest for row in existing}
        staged_edges: dict[str, dict[str, set[str]]] = {}
        issues: list[OmopReferenceIssue] = []
        summaries: list[OmopMutationSummary] = []

        for ordinal, mutation in enumerate(self.mutations):
            key_digest = mutation.key.digest
            if mutation.operation is MutationOperation.INSERT:
                if key_digest in live_keys:
                    issues.append(
                        OmopReferenceIssue(
                            ordinal=ordinal,
                            code="duplicate_insert",
                            table=mutation.table,
                        )
                    )
                live_keys.add(key_digest)
            elif key_digest not in live_keys:
                issues.append(
                    OmopReferenceIssue(
                        ordinal=ordinal,
                        code="missing_target",
                        table=mutation.table,
                    )
                )

            if mutation.operation is MutationOperation.TOMBSTONE:
                if any(
                    key_digest in targets
                    for edges in staged_edges.values()
                    for targets in edges.values()
                ):
                    issues.append(
                        OmopReferenceIssue(
                            ordinal=ordinal,
                            code="referenced_tombstone",
                            table=mutation.table,
                        )
                    )
                live_keys.discard(key_digest)
                staged_edges.pop(key_digest, None)
            else:
                if mutation.operation is MutationOperation.UPDATE:
                    edges = {
                        name: set(targets)
                        for name, targets in staged_edges.get(key_digest, {}).items()
                    }
                    for field_name in mutation.field_names:
                        edges.pop(field_name, None)
                    if mutation._explicit_references:
                        edges.pop("explicit_reference", None)
                else:
                    edges = {}
                for field_name, reference in mutation.references:
                    target_digest = reference.digest
                    edges.setdefault(field_name, set()).add(target_digest)
                    if target_digest not in live_keys:
                        issues.append(
                            OmopReferenceIssue(
                                ordinal=ordinal,
                                code="missing_reference",
                                table=mutation.table,
                                field_name=field_name,
                            )
                        )
                staged_edges[key_digest] = edges

            summaries.append(
                OmopMutationSummary(
                    ordinal=ordinal,
                    operation=mutation.operation,
                    table=mutation.table,
                    field_names=mutation.field_names,
                    reference_count=len(mutation.references),
                    row_digest=mutation.row_digest,
                )
            )

        snapshot_digest = _digest(sorted(row.digest for row in existing))
        counts = Counter(mutation.operation.value for mutation in self.mutations)
        operation_counts = tuple(sorted(counts.items()))
        preview_payload = _preview_payload(
            batch_digest=self.batch_digest,
            reference_snapshot_digest=snapshot_digest,
            mutations=tuple(summaries),
            issues=tuple(issues),
            operation_counts=operation_counts,
        )
        return OmopMutationPreview(
            batch_digest=self.batch_digest,
            preview_digest=_digest(preview_payload),
            reference_snapshot_digest=snapshot_digest,
            mutations=tuple(summaries),
            issues=tuple(issues),
            operation_counts=operation_counts,
        )

    def bind_approval(
        self,
        preview: OmopMutationPreview,
        *,
        approved_preview_digest: str,
        approval_receipt_digest: str,
    ) -> OmopApprovalBinding:
        """Bind a value-free approval receipt to this exact valid preview."""

        if type(preview) is not OmopMutationPreview:
            raise OmopMutationError("invalid_preview", "preview")
        if preview.batch_digest != self.batch_digest:
            raise OmopMutationError("batch_changed", "preview")
        if not preview.is_valid:
            raise OmopMutationError("reference_check_failed", "preview")
        try:
            expected_preview_digest = _digest(
                _preview_payload(
                    batch_digest=preview.batch_digest,
                    reference_snapshot_digest=preview.reference_snapshot_digest,
                    mutations=preview.mutations,
                    issues=preview.issues,
                    operation_counts=preview.operation_counts,
                )
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise OmopMutationError("invalid_preview", "preview") from None
        if preview.schema != MUTATION_BATCH_SCHEMA or (
            preview.preview_digest != expected_preview_digest
        ):
            raise OmopMutationError("invalid_preview", "preview")
        _validate_digest(approved_preview_digest, "approved_preview_digest")
        _validate_digest(approval_receipt_digest, "approval_receipt_digest")
        if approved_preview_digest != preview.preview_digest:
            raise OmopMutationError("preview_changed", "approved_preview_digest")
        return OmopApprovalBinding(
            batch_digest=self.batch_digest,
            preview_digest=preview.preview_digest,
            reference_snapshot_digest=preview.reference_snapshot_digest,
            approval_receipt_digest=approval_receipt_digest,
        )

    def commit(
        self,
        committer: OmopBatchCommitter,
        *,
        approval: OmopApprovalBinding,
    ) -> OmopCommitResult:
        """Ask a local committer to atomically apply the approved batch."""

        if type(approval) is not OmopApprovalBinding:
            raise OmopMutationError("invalid_approval", "approval")
        if approval.batch_digest != self.batch_digest:
            raise OmopMutationError("batch_changed", "approval")
        try:
            commit_batch = committer.commit_batch
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise OmopMutationError("invalid_committer", "committer") from None
        if not callable(commit_batch):
            raise OmopMutationError("invalid_committer", "committer")

        try:
            commit_batch(
                self.mutations,
                batch_digest=self.batch_digest,
                approval=approval,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return OmopCommitResult(
                status=CommitStatus.FAILED,
                batch_digest=self.batch_digest,
                preview_digest=approval.preview_digest,
                approval_receipt_digest=approval.approval_receipt_digest,
                mutation_count=0,
                error_code="committer_error",
            )
        return OmopCommitResult(
            status=CommitStatus.COMMITTED,
            batch_digest=self.batch_digest,
            preview_digest=approval.preview_digest,
            approval_receipt_digest=approval.approval_receipt_digest,
            mutation_count=len(self.mutations),
        )

    def __repr__(self) -> str:
        return (
            "OmopMutationBatch("
            f"mutation_count={len(self.mutations)}, "
            f"batch_digest={self.batch_digest!r})"
        )


def _normalize_mapping(
    value: Mapping[str, Scalar],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> _Items:
    if not isinstance(value, Mapping) or isinstance(value, (str, bytes, bytearray)):
        raise OmopMutationError("not_a_mapping", field_name)
    try:
        items = tuple(value.items())
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise OmopMutationError("unreadable_mapping", field_name) from None
    if not items and not allow_empty:
        raise OmopMutationError("empty_mapping", field_name)

    normalized: list[tuple[str, Scalar]] = []
    seen: set[str] = set()
    for name, item in items:
        normalized_name = _validate_identifier(name, field_name)
        if normalized_name in seen:
            raise OmopMutationError("duplicate_field", field_name)
        seen.add(normalized_name)
        normalized.append((normalized_name, _validate_scalar(item, field_name)))
    return tuple(sorted(normalized))


def _normalize_references(
    references: Iterable[OmopRowKey],
) -> tuple[OmopRowKey, ...]:
    if isinstance(references, (str, bytes, bytearray, Mapping)):
        raise OmopMutationError("invalid_references", "references")
    try:
        normalized = tuple(references)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise OmopMutationError("invalid_references", "references") from None
    if any(type(reference) is not OmopRowKey for reference in normalized):
        raise OmopMutationError("invalid_reference", "references")
    return normalized


def _normalize_existing_rows(
    rows: Iterable[OmopRowKey],
) -> tuple[OmopRowKey, ...]:
    if isinstance(rows, (str, bytes, bytearray, Mapping)):
        raise OmopMutationError("invalid_existing_rows", "existing_rows")
    try:
        normalized = tuple(rows)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise OmopMutationError("invalid_existing_rows", "existing_rows") from None
    if any(type(row) is not OmopRowKey for row in normalized):
        raise OmopMutationError("invalid_existing_row", "existing_rows")
    return normalized


def _validate_identifier(value: Any, field_name: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise OmopMutationError("invalid_identifier", field_name)
    return value


def _validate_scalar(value: Any, field_name: str) -> Scalar:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise OmopMutationError("invalid_scalar", field_name)


def _validate_digest(value: Any, field_name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise OmopMutationError("invalid_digest", field_name)
    return value


def _preview_payload(
    *,
    batch_digest: str,
    reference_snapshot_digest: str,
    mutations: tuple[OmopMutationSummary, ...],
    issues: tuple[OmopReferenceIssue, ...],
    operation_counts: tuple[tuple[str, int], ...],
) -> dict[str, Any]:
    return {
        "batch_digest": batch_digest,
        "is_valid": not issues,
        "issues": [issue.to_dict() for issue in issues],
        "mutation_count": len(mutations),
        "mutations": [mutation.to_dict() for mutation in mutations],
        "operation_counts": dict(operation_counts),
        "reference_snapshot_digest": reference_snapshot_digest,
        "schema": MUTATION_BATCH_SCHEMA,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: Any) -> str:
    payload = _canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


__all__ = [
    "MAX_BATCH_MUTATIONS",
    "MUTATION_BATCH_SCHEMA",
    "CommitStatus",
    "MutationOperation",
    "OmopApprovalBinding",
    "OmopBatchCommitter",
    "OmopCommitResult",
    "OmopMutation",
    "OmopMutationBatch",
    "OmopMutationError",
    "OmopMutationPreview",
    "OmopMutationSummary",
    "OmopReferenceIssue",
    "OmopRowKey",
]
