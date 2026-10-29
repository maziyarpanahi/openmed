"""Offline OMOP loader staging with digest-bound NLP span and rollback evidence.

Row values and rollback material remain private in memory. This module does not
write databases/files, issue approval receipts or execute clinical effects.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from itertools import islice
from typing import Any, cast

from ..omop.cdm_loader import (
    _PRIMARY_KEYS,
    _SCHEMA_COLUMNS,
    _TABLE_ORDER,
    OmopCdmTables,
    OmopLoadSummary,
    validate_omop_tables,
)
from ..omop.mutation_batch import (
    MAX_BATCH_MUTATIONS,
    MUTATION_BATCH_SCHEMA,
    MutationOperation,
    OmopApprovalBinding,
    OmopMutation,
    OmopMutationBatch,
    OmopMutationPreview,
    OmopMutationSummary,
    OmopReferenceIssue,
    OmopRowKey,
    Scalar,
)
from ..omop.vocabulary_write_gate import (
    VocabularyMappingProvenance,
    VocabularySnapshot,
    VocabularyWriteGate,
)
from ..omop_rollback_manifest import (
    OmopRollbackInstruction,
    OmopRollbackManifest,
    RollbackStrategy,
    build_omop_rollback_manifest,
    validate_omop_rollback_manifest,
)

__all__ = [
    "NLP_OMOP_STAGING_SCHEMA",
    "NlpOmopStagingError",
    "NlpOmopLineageRecord",
    "NlpOmopLineageReport",
    "NlpOmopWriteLineage",
    "NlpOmopStagedPreview",
    "NlpOmopStagedBatch",
    "stage_nlp_omop_tables",
]

NLP_OMOP_STAGING_SCHEMA = "openmed.interop.lineage.nlp_omop_staging.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_BARE_DIGEST = re.compile(r"[0-9a-f]{64}")
_DOMAIN_CONCEPT = {
    "condition_occurrence": "condition_concept_id",
    "drug_exposure": "drug_concept_id",
    "measurement": "measurement_concept_id",
    "procedure_occurrence": "procedure_concept_id",
    "observation": "observation_concept_id",
}
_LINEAGE_TABLES = frozenset({"note_nlp", "source_to_concept_map", *_DOMAIN_CONCEPT})
_PARENTS = frozenset({"concept", "person", "visit_occurrence"})
_CODES = frozenset(
    {
        "invalid_input",
        "invalid_digest",
        "invalid_integer",
        "invalid_lineage",
        "invalid_tables",
        "invalid_row",
        "invalid_row_key",
        "duplicate_row",
        "unknown_table",
        "unknown_column",
        "input_too_large",
        "summary_mismatch",
        "missing_source_note",
        "missing_source_span",
        "source_mismatch",
        "invalid_offsets",
        "missing_nlp_system",
        "missing_old_pipeline_evidence",
        "extra_old_pipeline_evidence",
        "replacement_snapshot_required",
        "parent_conflict",
        "empty_batch",
        "invalid_mapping",
        "invalid_rollback",
        "invalid_preview",
        "approval_mismatch",
        "preflight_failed",
    }
)


class NlpOmopStagingError(ValueError):
    """A closed, value-free staging or approval failure."""

    def __init__(self, code: str) -> None:
        self.code = code if type(code) is str and code in _CODES else "invalid_input"
        super().__init__(self.code)


def _json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _text_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_digest(value: Any) -> str:
    if type(value) is not str:
        raise NlpOmopStagingError("invalid_digest")
    if _BARE_DIGEST.fullmatch(value):
        return "sha256:" + value
    if _DIGEST.fullmatch(value) is None:
        raise NlpOmopStagingError("invalid_digest")
    return value


def _integer(value: Any, low: int = 0, high: int = 2**63 - 1) -> bool:
    return type(value) is int and low <= value <= high


def _bounded(
    values: Iterable[Any], limit: int = MAX_BATCH_MUTATIONS
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise NlpOmopStagingError("invalid_input")
    try:
        result = tuple(islice(values, limit + 1))
    except Exception:
        result = None
    if result is None:
        raise NlpOmopStagingError("invalid_input")
    if len(result) > limit:
        raise NlpOmopStagingError("input_too_large")
    return result


@dataclass(frozen=True, slots=True, order=True)
class NlpOmopLineageRecord:
    """Value-free origin of one NLP-derived mutation, including tombstones.

    Args:
        mutation_ordinal: Position of the exact proposed mutation.
        table: Closed loader table name for an NLP-derived row.
        row_digest: Digest of the complete operation, key and in-memory values.
        source_note_digest: Caller-declared source identity, validated as a digest.
        note_content_digest: Digest computed from the in-memory note text.
        start: Inclusive Python character offset into that text.
        end: Exclusive, non-empty span end.
        nlp_system_digest: Digest of the originating NLP system name.
        pipeline_digest: Caller-attested pipeline or model digest for this row.
    """

    mutation_ordinal: int
    table: str
    row_digest: str
    source_note_digest: str
    note_content_digest: str
    start: int
    end: int
    nlp_system_digest: str
    pipeline_digest: str

    def __post_init__(self) -> None:
        if (
            not _integer(self.mutation_ordinal, 0, MAX_BATCH_MUTATIONS - 1)
            or type(self.table) is not str
            or self.table not in _LINEAGE_TABLES
        ):
            raise NlpOmopStagingError("invalid_lineage")
        if not _integer(self.start, 0, 4_000_000) or not _integer(
            self.end, self.start + 1, 4_000_000
        ):
            raise NlpOmopStagingError("invalid_offsets")
        for name in (
            "row_digest",
            "source_note_digest",
            "note_content_digest",
            "nlp_system_digest",
            "pipeline_digest",
        ):
            object.__setattr__(self, name, _normalize_digest(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        """Return only closed table names, offsets, ordinals and digests."""
        return {
            "mutation_ordinal": self.mutation_ordinal,
            "table": self.table,
            "row_digest": self.row_digest,
            "source_note_digest": self.source_note_digest,
            "note_content_digest": self.note_content_digest,
            "start": self.start,
            "end": self.end,
            "nlp_system_digest": self.nlp_system_digest,
            "pipeline_digest": self.pipeline_digest,
        }


@dataclass(frozen=True, slots=True)
class NlpOmopLineageReport:
    """Counts and closed coverage findings for a proposed NLP-derived batch."""

    expected_count: int
    supplied_count: int
    missing_count: int
    extra_count: int
    duplicate_count: int
    mismatch_count: int
    lineage_digest: str

    def __post_init__(self) -> None:
        if any(
            not _integer(v, 0, MAX_BATCH_MUTATIONS)
            for v in (
                self.expected_count,
                self.supplied_count,
                self.missing_count,
                self.extra_count,
                self.duplicate_count,
                self.mismatch_count,
            )
        ):
            raise NlpOmopStagingError("invalid_lineage")
        object.__setattr__(
            self, "lineage_digest", _normalize_digest(self.lineage_digest)
        )

    @property
    def is_complete(self) -> bool:
        """Return whether every required row has exactly one matching record."""
        return not any(
            (
                self.missing_count,
                self.extra_count,
                self.duplicate_count,
                self.mismatch_count,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return bounded counts, a completeness flag and the crosswalk digest."""
        return {
            "expected_count": self.expected_count,
            "supplied_count": self.supplied_count,
            "missing_count": self.missing_count,
            "extra_count": self.extra_count,
            "duplicate_count": self.duplicate_count,
            "mismatch_count": self.mismatch_count,
            "is_complete": self.is_complete,
            "lineage_digest": self.lineage_digest,
        }


@dataclass(frozen=True, slots=True, repr=False, init=False)
class NlpOmopWriteLineage:
    """Immutable one-record-per-mutation NLP span crosswalk."""

    records: tuple[NlpOmopLineageRecord, ...]

    def __init__(self, records: Iterable[NlpOmopLineageRecord]) -> None:
        normalized = _bounded(records)
        if any(type(record) is not NlpOmopLineageRecord for record in normalized):
            raise NlpOmopStagingError("invalid_lineage")
        object.__setattr__(self, "records", tuple(sorted(normalized)))

    @property
    def lineage_digest(self) -> str:
        """Return the deterministic digest of every span origin and row binding."""
        return _digest(
            {
                "schema": NLP_OMOP_STAGING_SCHEMA,
                "records": [r.to_dict() for r in self.records],
            }
        )

    def verify(self, batch: OmopMutationBatch) -> NlpOmopLineageReport:
        """Verify exact ordinal/table/row-digest coverage without source values."""
        if type(batch) is not OmopMutationBatch:
            raise NlpOmopStagingError("invalid_input")
        expected = {
            i: m for i, m in enumerate(batch.mutations) if m.table in _LINEAGE_TABLES
        }
        seen = Counter(r.mutation_ordinal for r in self.records)
        return NlpOmopLineageReport(
            len(expected),
            len(self.records),
            len(set(expected) - set(seen)),
            len(set(seen) - set(expected)),
            sum(n - 1 for n in seen.values()),
            sum(
                r.mutation_ordinal in expected
                and (
                    r.table != expected[r.mutation_ordinal].table
                    or r.row_digest != expected[r.mutation_ordinal].row_digest
                )
                for r in self.records
            ),
            self.lineage_digest,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free crosswalk; it is still sensitive audit metadata."""
        return {
            "schema": NLP_OMOP_STAGING_SCHEMA,
            "lineage_digest": self.lineage_digest,
            "records": [r.to_dict() for r in self.records],
        }


@dataclass(frozen=True, slots=True)
class NlpOmopStagedPreview:
    """Complete review packet bound to rows, lineage, vocabulary and rollback."""

    batch_preview: OmopMutationPreview
    lineage_report: NlpOmopLineageReport
    vocabulary_snapshot_digest: str
    vocabulary_report_digest: str
    vocabulary_compatible: bool
    missing_mapping_count: int
    extra_mapping_count: int
    provenance_mismatch_count: int
    rollback_manifest_digest: str
    bindings_match: bool
    rejected_span_count: int

    def __post_init__(self) -> None:
        if (
            type(self.batch_preview) is not OmopMutationPreview
            or type(self.lineage_report) is not NlpOmopLineageReport
            or type(self.vocabulary_compatible) is not bool
            or type(self.bindings_match) is not bool
            or any(
                not _integer(v, 0, MAX_BATCH_MUTATIONS)
                for v in (
                    self.missing_mapping_count,
                    self.extra_mapping_count,
                    self.provenance_mismatch_count,
                    self.rejected_span_count,
                )
            )
        ):
            raise NlpOmopStagingError("invalid_preview")
        _validate_preview_metadata(self.batch_preview)
        for name in (
            "vocabulary_snapshot_digest",
            "vocabulary_report_digest",
            "rollback_manifest_digest",
        ):
            object.__setattr__(self, name, _normalize_digest(getattr(self, name)))

    @property
    def is_approvable(self) -> bool:
        """Return whether all current local preflights and evidence bindings pass."""
        return (
            self.batch_preview.is_valid
            and self.lineage_report.is_complete
            and self.vocabulary_compatible
            and self.bindings_match
            and not any(
                (
                    self.missing_mapping_count,
                    self.extra_mapping_count,
                    self.provenance_mismatch_count,
                )
            )
        )

    @property
    def preview_digest(self) -> str:
        """Return the digest that a separate human approval service must review."""
        return _digest(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": NLP_OMOP_STAGING_SCHEMA,
            "batch": self.batch_preview.to_dict(),
            "lineage": self.lineage_report.to_dict(),
            "vocabulary_snapshot_digest": self.vocabulary_snapshot_digest,
            "vocabulary_report_digest": self.vocabulary_report_digest,
            "vocabulary_compatible": self.vocabulary_compatible,
            "missing_mapping_count": self.missing_mapping_count,
            "extra_mapping_count": self.extra_mapping_count,
            "provenance_mismatch_count": self.provenance_mismatch_count,
            "rollback_manifest_digest": self.rollback_manifest_digest,
            "bindings_match": self.bindings_match,
            "rejected_span_count": self.rejected_span_count,
            "is_approvable": self.is_approvable,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free review packet with its canonical digest."""
        return {**self._payload(), "preview_digest": self.preview_digest}


def _validate_preview_metadata(preview: OmopMutationPreview) -> None:
    """Reject externally constructed preview metadata that could expose text."""
    try:
        if type(preview.schema) is not str or preview.schema != MUTATION_BATCH_SCHEMA:
            raise ValueError
        if (
            type(preview.mutations) is not tuple
            or type(preview.issues) is not tuple
            or type(preview.evidence_digests) is not tuple
            or len(preview.evidence_digests) > 16
        ):
            raise ValueError
        for digest in (
            preview.batch_digest,
            preview.preview_digest,
            preview.reference_snapshot_digest,
            *preview.evidence_digests,
        ):
            if type(digest) is not str or _DIGEST.fullmatch(digest) is None:
                raise ValueError
        if (
            not 1 <= len(preview.mutations) <= MAX_BATCH_MUTATIONS
            or len(preview.issues) > MAX_BATCH_MUTATIONS
        ):
            raise ValueError
        for ordinal, item in enumerate(preview.mutations):
            if (
                type(item) is not OmopMutationSummary
                or not _integer(item.ordinal)
                or item.ordinal != ordinal
                or type(item.table) is not str
                or item.table not in _TABLE_ORDER
                or type(item.operation) is not MutationOperation
                or item.operation
                not in (MutationOperation.INSERT, MutationOperation.TOMBSTONE)
            ):
                raise ValueError
            if (
                type(item.field_names) is not tuple
                or any(
                    type(name) is not str or name not in _SCHEMA_COLUMNS[item.table]
                    for name in item.field_names
                )
                or not _integer(item.reference_count, 0, 64)
            ):
                raise ValueError
            if (
                type(item.row_digest) is not str
                or _DIGEST.fullmatch(item.row_digest) is None
            ):
                raise ValueError
        counts = tuple(
            sorted(Counter(item.operation.value for item in preview.mutations).items())
        )
        if (
            type(preview.operation_counts) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or item[0] not in {"insert", "tombstone"}
                or not _integer(item[1], 1, MAX_BATCH_MUTATIONS)
                for item in preview.operation_counts
            )
            or preview.operation_counts != counts
        ):
            raise ValueError
        for issue in preview.issues:
            if (
                type(issue) is not OmopReferenceIssue
                or not _integer(issue.ordinal, 0, len(preview.mutations) - 1)
                or type(issue.table) is not str
                or issue.table not in _TABLE_ORDER
                or type(issue.code) is not str
                or issue.code
                not in {
                    "duplicate_insert",
                    "missing_target",
                    "missing_reference",
                    "referenced_tombstone",
                }
            ):
                raise ValueError
            if issue.field_name is not None and (
                type(issue.field_name) is not str
                or issue.field_name not in _SCHEMA_COLUMNS[issue.table]
            ):
                raise ValueError
    except Exception:
        invalid = True
    else:
        invalid = False
    if invalid:
        raise NlpOmopStagingError("invalid_preview")


@dataclass(frozen=True, slots=True, repr=False)
class NlpOmopStagedBatch:
    """Private staged rows with local-only review and approval binding helpers.

    Build through stage_nlp_omop_tables. No method executes mutations or creates
    an approval receipt. Rollback material must stay in protected local custody.
    """

    batch: OmopMutationBatch
    lineage: NlpOmopWriteLineage
    vocabulary_snapshot: VocabularySnapshot
    _mappings: tuple[VocabularyMappingProvenance, ...] = field(repr=False)
    _existing_keys: tuple[OmopRowKey, ...] = field(repr=False)
    _rollback_materials: tuple[str, ...] = field(repr=False)
    rejected_span_count: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.batch) is not OmopMutationBatch
            or type(self.lineage) is not NlpOmopWriteLineage
            or type(self.vocabulary_snapshot) is not VocabularySnapshot
            or type(self._mappings) is not tuple
            or any(type(m) is not VocabularyMappingProvenance for m in self._mappings)
            or type(self._existing_keys) is not tuple
            or any(type(k) is not OmopRowKey for k in self._existing_keys)
            or type(self._rollback_materials) is not tuple
            or len(self._rollback_materials) != len(self.batch.mutations)
            or any(type(v) is not str for v in self._rollback_materials)
            or not _integer(self.rejected_span_count, 0, MAX_BATCH_MUTATIONS)
        ):
            raise NlpOmopStagingError("invalid_input")

    @property
    def vocabulary_mappings(self) -> tuple[VocabularyMappingProvenance, ...]:
        """Return private terminology evidence for the existing write gate."""
        return self._mappings

    def rollback_material(self, mutation_ordinal: int) -> dict[str, Any]:
        """Return a fresh private payload for trusted, protected rollback custody.

        This contains row keys and before-images. Never log or put it in an audit
        report. No file is created and storage durability is not asserted here.
        """
        if not _integer(mutation_ordinal, 0, len(self.batch.mutations) - 1):
            raise NlpOmopStagingError("invalid_integer")
        try:
            return json.loads(self._rollback_materials[mutation_ordinal])
        except Exception:
            pass
        raise NlpOmopStagingError("invalid_rollback")

    @property
    def rollback_requirements(self) -> tuple[OmopRollbackInstruction, ...]:
        """Return required strategies and expected protected-payload digests."""
        return tuple(
            OmopRollbackInstruction(
                i,
                RollbackStrategy.REINSERT_TOMBSTONED_ROW
                if m.operation is MutationOperation.TOMBSTONE
                else RollbackStrategy.DELETE_INSERTED_ROW,
                _text_digest(material),
            )
            for i, (m, material) in enumerate(
                zip(self.batch.mutations, self._rollback_materials)
            )
        )

    @property
    def _rollback_digest(self) -> str:
        return _digest(
            [
                {
                    "ordinal": i.mutation_ordinal,
                    "strategy": i.strategy.value,
                    "payload_digest": i.rollback_artifact_digest,
                }
                for i in self.rollback_requirements
            ]
        )

    def build_rollback_manifest(
        self, instructions: Iterable[OmopRollbackInstruction]
    ) -> OmopRollbackManifest:
        """Validate caller custody digests and use the existing manifest builder."""
        try:
            manifest = build_omop_rollback_manifest(
                self.batch, self.vocabulary_snapshot, _bounded(instructions)
            )
            actual = {
                entry.mutation_ordinal: (entry.strategy, entry.rollback_artifact_digest)
                for entry in manifest.entries
            }
            required = {
                i.mutation_ordinal: (i.strategy, i.rollback_artifact_digest)
                for i in self.rollback_requirements
            }
            if actual != required:
                raise NlpOmopStagingError("invalid_rollback")
            return manifest
        except Exception:
            pass
        raise NlpOmopStagingError("invalid_rollback")

    def preview(self, rollback_manifest: OmopRollbackManifest) -> NlpOmopStagedPreview:
        """Preflight exact lineage, target vocabulary, references and rollback."""
        try:
            validate_omop_rollback_manifest(
                rollback_manifest, self.batch, self.vocabulary_snapshot
            )
            rebuilt = self.build_rollback_manifest(self.rollback_requirements)
            if rebuilt != rollback_manifest:
                raise NlpOmopStagingError("invalid_rollback")
        except Exception:
            invalid = True
        else:
            invalid = False
        if invalid:
            raise NlpOmopStagingError("invalid_rollback")
        # The existing gate rejects an empty mapping collection. Represent that
        # refusal in the preview rather than misclassifying it as rollback drift.
        gate = (
            VocabularyWriteGate(self.vocabulary_snapshot).evaluate(self._mappings)
            if self._mappings
            else None
        )
        required, mismatch = _mapping_requirements(self.batch.mutations, self._mappings)
        provided = {m.target_concept_id for m in self._mappings}
        bindings = tuple(
            sorted(
                {
                    self.lineage.lineage_digest,
                    self.vocabulary_snapshot.snapshot_digest,
                    _digest([m.mapping_digest for m in self._mappings]),
                    self._rollback_digest,
                }
            )
        )
        return NlpOmopStagedPreview(
            self.batch.preview(existing_rows=self._existing_keys),
            self.lineage.verify(self.batch),
            self.vocabulary_snapshot.snapshot_digest,
            gate.report_digest
            if gate is not None
            else _digest(
                {
                    "code": "empty_mapping_collection",
                    "snapshot_digest": self.vocabulary_snapshot.snapshot_digest,
                }
            ),
            gate.is_compatible if gate is not None else False,
            len(required - provided),
            len(provided - required),
            mismatch,
            rollback_manifest.manifest_digest,
            self.batch.evidence_digests == bindings,
            self.rejected_span_count,
        )

    def bind_approval(
        self,
        preview: NlpOmopStagedPreview,
        rollback_manifest: OmopRollbackManifest,
        *,
        approved_preview_digest: str,
        approval_receipt_digest: str,
    ) -> OmopApprovalBinding:
        """Bind a separately issued receipt only to the exact complete preview.

        The approval service owns reviewer authority, expiry and single-use
        enforcement. Supplying or copying a digest does not constitute approval.
        """
        if type(preview) is not NlpOmopStagedPreview:
            raise NlpOmopStagingError("invalid_preview")
        current = self.preview(rollback_manifest)
        if current != preview:
            raise NlpOmopStagingError("invalid_preview")
        if not current.is_approvable:
            raise NlpOmopStagingError("preflight_failed")
        if _normalize_digest(approved_preview_digest) != current.preview_digest:
            raise NlpOmopStagingError("approval_mismatch")
        try:
            return self.batch.bind_approval(
                current.batch_preview,
                approved_preview_digest=current.batch_preview.preview_digest,
                approval_receipt_digest=_normalize_digest(approval_receipt_digest),
            )
        except Exception:
            pass
        raise NlpOmopStagingError("invalid_preview")


def _copy_tables(value: OmopCdmTables) -> dict[str, tuple[dict[str, Scalar], ...]]:
    if (
        type(value) is not OmopCdmTables
        or type(value.summary) is not OmopLoadSummary
        or not isinstance(value.tables, Mapping)
    ):
        raise NlpOmopStagingError("invalid_tables")
    try:
        if any(
            type(name) is not str or name not in _TABLE_ORDER for name in value.tables
        ):
            raise NlpOmopStagingError("unknown_table")
        result: dict[str, tuple[dict[str, Scalar], ...]] = {}
        total = 0
        for table in _TABLE_ORDER:
            rows = _bounded(value.table(table))
            total += len(rows)
            if total > MAX_BATCH_MUTATIONS:
                raise NlpOmopStagingError("input_too_large")
            copied = []
            seen = set()
            for row in rows:
                if type(row) is not dict:
                    raise NlpOmopStagingError("invalid_row")
                if any(
                    type(column) is not str or column not in _SCHEMA_COLUMNS[table]
                    for column in row
                ):
                    raise NlpOmopStagingError("unknown_column")
                key = row.get(_PRIMARY_KEYS[table])
                if not _integer(key, 0 if table == "concept" else 1):
                    raise NlpOmopStagingError("invalid_row_key")
                if key in seen:
                    raise NlpOmopStagingError("duplicate_row")
                seen.add(key)
                copied.append(OmopMutation.insert(table, row).values)
            result[table] = tuple(
                sorted(copied, key=lambda r: cast(int, r[_PRIMARY_KEYS[table]]))
            )
        if not isinstance(value.summary.row_counts, Mapping) or any(
            type(name) is not str
            or name not in result
            or not _integer(count, 0, MAX_BATCH_MUTATIONS)
            or count != len(result[name])
            for name, count in value.summary.row_counts.items()
        ):
            raise NlpOmopStagingError("summary_mismatch")
        if type(value.summary.mode) is not str or value.summary.mode not in {
            "append",
            "replace_by_note",
        }:
            raise NlpOmopStagingError("invalid_input")
        # Reuse the loader's complete structural checks, including the inverse
        # NOTE_NLP -> domain-event link that ordinary batch FKs cannot express.
        # Detailed row findings stay private; only a closed failure escapes.
        if validate_omop_tables(replace(value, tables=result)):
            raise NlpOmopStagingError("invalid_row")
        mapping_rows = Counter(
            r["note_nlp_id"] for r in result["source_to_concept_map"]
        )
        if any(mapping_rows[r["note_nlp_id"]] != 1 for r in result["note_nlp"]):
            raise NlpOmopStagingError("missing_source_span")
        return result
    except NlpOmopStagingError:
        raise
    except Exception:
        raise NlpOmopStagingError("invalid_row") from None


def _key(table: str, row: Mapping[str, Scalar]) -> OmopRowKey:
    return OmopRowKey(table, {_PRIMARY_KEYS[table]: row[_PRIMARY_KEYS[table]]})


def _record(
    ordinal: int,
    mutation: OmopMutation,
    row: Mapping[str, Scalar],
    indexes: tuple[dict[Scalar, dict[str, Scalar]], dict[Scalar, dict[str, Scalar]]],
    pipeline: str,
) -> NlpOmopLineageRecord:
    nlps, notes = indexes
    nlp = row if mutation.table == "note_nlp" else nlps.get(row.get("note_nlp_id"))
    if nlp is None:
        raise NlpOmopStagingError("missing_source_span")
    note = notes.get(nlp.get("note_id"))
    if note is None:
        raise NlpOmopStagingError("missing_source_note")
    text, system = note.get("note_text"), nlp.get("nlp_system")
    if type(text) is not str or len(text) > 4_000_000:
        raise NlpOmopStagingError("invalid_row")
    if type(system) is not str or not system.strip() or len(system) > 512:
        raise NlpOmopStagingError("missing_nlp_system")
    start, end = nlp.get("offset"), nlp.get("offset_end")
    if (
        not isinstance(start, int)
        or not _integer(start, 0, len(text))
        or not isinstance(end, int)
        or not _integer(end, start + 1, len(text))
    ):
        raise NlpOmopStagingError("invalid_offsets")
    source = _normalize_digest(note.get("source_note_hash"))
    if mutation.table != "note_nlp":
        if _normalize_digest(row.get("source_note_hash")) != source:
            raise NlpOmopStagingError("source_mismatch")
        if mutation.table in _DOMAIN_CONCEPT and (
            row.get("note_id") != note.get("note_id")
            or row.get("person_id") != note.get("person_id")
            or row.get("visit_occurrence_id") != note.get("visit_occurrence_id")
            or row.get(_PRIMARY_KEYS[mutation.table]) != nlp.get("note_nlp_event_id")
            or row.get(_DOMAIN_CONCEPT[mutation.table])
            != nlp.get("note_nlp_concept_id")
        ):
            raise NlpOmopStagingError("source_mismatch")
        if mutation.table == "source_to_concept_map" and row.get(
            "target_concept_id"
        ) != nlp.get("note_nlp_concept_id"):
            raise NlpOmopStagingError("source_mismatch")
    return NlpOmopLineageRecord(
        ordinal,
        mutation.table,
        mutation.row_digest,
        source,
        _text_digest(text),
        start,
        end,
        _text_digest(system),
        pipeline,
    )


def _mapping_requirements(
    mutations: tuple[OmopMutation, ...],
    mappings: tuple[VocabularyMappingProvenance, ...],
) -> tuple[set[int], int]:
    required: set[int] = set()
    mismatch = 0
    by_id = {m.target_concept_id: m for m in mappings}
    for mutation in mutations:
        if mutation.operation is not MutationOperation.INSERT:
            continue
        values = mutation.values
        field_name = (
            "note_nlp_concept_id"
            if mutation.table == "note_nlp"
            else _DOMAIN_CONCEPT.get(mutation.table)
        )
        if field_name is not None:
            target = values.get(field_name)
            if not isinstance(target, int) or not _integer(target):
                mismatch += 1
            else:
                required.add(target)
        if mutation.table == "source_to_concept_map":
            target = values.get("target_concept_id")
            if not isinstance(target, int) or not _integer(target):
                mismatch += 1
                continue
            required.add(target)
            mapping = by_id.get(target)
            if mapping is not None and (
                mapping.target_vocabulary_id != values.get("target_vocabulary_id")
                or mapping.vocabulary_version != values.get("vocabulary_version")
            ):
                mismatch += 1
    return required, mismatch


def _stage_nlp_omop_tables(
    tables: OmopCdmTables,
    *,
    pipeline_digest: str,
    vocabulary_snapshot: VocabularySnapshot,
    vocabulary_mappings: Iterable[VocabularyMappingProvenance],
    existing_tables: OmopCdmTables | None = None,
    existing_pipeline_digests: Mapping[str, str] | None = None,
) -> NlpOmopStagedBatch:
    """Convert loader output into inserts and explicit, reviewed replacements.

    Args:
        tables: Caller-supplied in-memory loader output; no writer is invoked.
        pipeline_digest: Attested pipeline/model digest for incoming span rows.
        vocabulary_snapshot: Exact target snapshot for existing vocabulary gates.
        vocabulary_mappings: One target provenance record per referenced concept.
        existing_tables: Explicit complete local snapshot. Required for replacement;
            identical existing parent rows are reused without implicit updates.
        existing_pipeline_digests: Previous NOTE_NLP key digests mapped to their
            original pipeline/model digests; mandatory for removed old span rows.

    Returns:
        Immutable staged rows, value-free lineage and private rollback material.

    Raises:
        NlpOmopStagingError: A fixed failure code without source or target values.
    """
    pipeline = _normalize_digest(pipeline_digest)
    if type(vocabulary_snapshot) is not VocabularySnapshot:
        raise NlpOmopStagingError("invalid_input")
    incoming = _copy_tables(tables)
    if tables.summary.mode == "replace_by_note" and existing_tables is None:
        raise NlpOmopStagingError("replacement_snapshot_required")
    previous = (
        _copy_tables(existing_tables)
        if existing_tables is not None
        else {name: () for name in _TABLE_ORDER}
    )
    mappings = _bounded(vocabulary_mappings)
    if any(type(m) is not VocabularyMappingProvenance for m in mappings) or len(
        {m.target_concept_id for m in mappings}
    ) != len(mappings):
        raise NlpOmopStagingError("invalid_mapping")
    mappings = tuple(sorted(mappings, key=lambda m: m.mapping_digest))
    if existing_pipeline_digests is None:
        old_pipeline = {}
    elif (
        isinstance(existing_pipeline_digests, Mapping)
        and len(existing_pipeline_digests) <= MAX_BATCH_MUTATIONS
    ):
        old_pipeline = {
            _normalize_digest(k): _normalize_digest(v)
            for k, v in existing_pipeline_digests.items()
        }
    else:
        raise NlpOmopStagingError("invalid_input")
    if existing_pipeline_digests is not None and len(old_pipeline) != len(
        existing_pipeline_digests
    ):
        raise NlpOmopStagingError("invalid_lineage")
    mutations: list[OmopMutation] = []
    records: list[NlpOmopLineageRecord] = []
    materials: list[str] = []
    used_old = set()
    targets = {
        (r.get("person_id"), _normalize_digest(r.get("source_note_hash")))
        for r in incoming["note"]
    }
    old_notes = {
        r["note_id"]
        for r in previous["note"]
        if (r.get("person_id"), _normalize_digest(r.get("source_note_hash"))) in targets
    }
    old_nlps = {
        r["note_nlp_id"] for r in previous["note_nlp"] if r.get("note_id") in old_notes
    }
    existing_keys = tuple(
        _key(name, row) for name in _TABLE_ORDER for row in previous[name]
    )
    old_by_key = {
        _key(name, row).digest: row for name in _TABLE_ORDER for row in previous[name]
    }
    incoming_indexes = (
        {r["note_nlp_id"]: r for r in incoming["note_nlp"]},
        {r["note_id"]: r for r in incoming["note"]},
    )
    previous_indexes = (
        {r["note_nlp_id"]: r for r in previous["note_nlp"]},
        {r["note_id"]: r for r in previous["note"]},
    )

    def append(table: str, row: Mapping[str, Scalar], *, remove: bool = False) -> None:
        key = _key(table, row)
        mutation = (
            OmopMutation.tombstone(table, key.values)
            if remove
            else OmopMutation.insert(table, row)
        )
        ordinal = len(mutations)
        if ordinal >= MAX_BATCH_MUTATIONS:
            raise NlpOmopStagingError("input_too_large")
        if table in _LINEAGE_TABLES:
            source_indexes = previous_indexes if remove else incoming_indexes
            nlp_key = (
                key.digest
                if table == "note_nlp"
                else _key("note_nlp", {"note_nlp_id": row.get("note_nlp_id")}).digest
            )
            origin_pipeline = pipeline
            if remove:
                if nlp_key not in old_pipeline:
                    raise NlpOmopStagingError("missing_old_pipeline_evidence")
                used_old.add(nlp_key)
                origin_pipeline = old_pipeline[nlp_key]
            records.append(
                _record(ordinal, mutation, row, source_indexes, origin_pipeline)
            )
        mutations.append(mutation)
        materials.append(
            _json(
                {
                    "schema": "openmed.nlp_omop.rollback_material.v1",
                    "mutation_ordinal": ordinal,
                    "operation": mutation.operation.value,
                    "table": table,
                    "key": key.values,
                    "before_image": dict(row) if remove else None,
                }
            )
        )

    if tables.summary.mode == "replace_by_note":
        for table in reversed(_TABLE_ORDER):
            if table in _PARENTS:
                continue
            for row in previous[table]:
                selected = (
                    row["note_id"] in old_notes
                    if table == "note"
                    else row.get("note_nlp_id") in old_nlps
                    if table == "source_to_concept_map"
                    else row.get("note_id") in old_notes
                )
                if selected:
                    append(table, row, remove=True)
    if set(old_pipeline) != used_old:
        raise NlpOmopStagingError("extra_old_pipeline_evidence")
    for table in _TABLE_ORDER:
        for row in incoming[table]:
            existing = old_by_key.get(_key(table, row).digest)
            if table in _PARENTS and existing is not None:
                if dict(row) != existing:
                    raise NlpOmopStagingError("parent_conflict")
                continue
            append(table, row)
    if not mutations:
        raise NlpOmopStagingError("empty_batch")
    lineage = NlpOmopWriteLineage(records)
    unbound = OmopMutationBatch(mutations)
    temporary = NlpOmopStagedBatch(
        unbound, lineage, vocabulary_snapshot, mappings, existing_keys, tuple(materials)
    )
    batch = OmopMutationBatch(
        mutations,
        evidence_digests=(
            lineage.lineage_digest,
            vocabulary_snapshot.snapshot_digest,
            _digest([m.mapping_digest for m in mappings]),
            temporary._rollback_digest,
        ),
    )
    counts = tables.summary.rejection_counts
    if (
        not isinstance(counts, Mapping)
        or any(not _integer(v, 0, MAX_BATCH_MUTATIONS) for v in counts.values())
        or sum(counts.values()) > MAX_BATCH_MUTATIONS
    ):
        raise NlpOmopStagingError("invalid_input")
    return NlpOmopStagedBatch(
        batch,
        lineage,
        vocabulary_snapshot,
        mappings,
        existing_keys,
        tuple(materials),
        sum(counts.values()),
    )


def stage_nlp_omop_tables(
    tables: OmopCdmTables,
    *,
    pipeline_digest: str,
    vocabulary_snapshot: VocabularySnapshot,
    vocabulary_mappings: Iterable[VocabularyMappingProvenance],
    existing_tables: OmopCdmTables | None = None,
    existing_pipeline_digests: Mapping[str, str] | None = None,
) -> NlpOmopStagedBatch:
    """Stage loader rows and bind their lineage, vocabulary and rollback evidence.

    Args:
        tables: In-memory OmopCdmTables produced by the existing local loader.
        pipeline_digest: Caller-attested pipeline or model digest for new spans.
        vocabulary_snapshot: Exact caller-supplied target vocabulary snapshot.
        vocabulary_mappings: Target mapping provenance consumed by the existing gate.
        existing_tables: Complete local row snapshot, required for replace_by_note.
        existing_pipeline_digests: Original pipeline digests keyed by previous
            NOTE_NLP row-key digests. Removed spans never inherit the new pipeline.

    Returns:
        An immutable proposal with value-free lineage and private rollback material.

    Raises:
        NlpOmopStagingError: A closed code without adapter, note or row values.
    """
    try:
        return _stage_nlp_omop_tables(
            tables,
            pipeline_digest=pipeline_digest,
            vocabulary_snapshot=vocabulary_snapshot,
            vocabulary_mappings=vocabulary_mappings,
            existing_tables=existing_tables,
            existing_pipeline_digests=existing_pipeline_digests,
        )
    except NlpOmopStagingError as error:
        stored = vars(error).get("code") if type(error) is NlpOmopStagingError else None
        code = stored if type(stored) is str and stored in _CODES else "invalid_input"
    except Exception:
        code = "invalid_input"
    raise NlpOmopStagingError(code)
