"""Content-free rollback manifests for staged OMOP mutation batches.

The manifest binds each staged mutation to a protected local rollback artifact
without retaining row values, row keys, SQL, or connection details.  It checks
structural reversibility only; executing or proving a rollback remains the
responsibility of the local transactional adapter.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .omop.mutation_batch import MutationOperation, OmopMutationBatch
from .omop.vocabulary_write_gate import VocabularySnapshot

OMOP_ROLLBACK_MANIFEST_SCHEMA = "openmed.interop.omop.rollback_manifest.v1"

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]{0,62}")


class RollbackStrategy(str, Enum):
    """Closed set of structural rollback strategies."""

    DELETE_INSERTED_ROW = "delete_inserted_row"
    RESTORE_BEFORE_IMAGE = "restore_before_image"
    REINSERT_TOMBSTONED_ROW = "reinsert_tombstoned_row"


_EXPECTED_STRATEGIES = {
    MutationOperation.INSERT: RollbackStrategy.DELETE_INSERTED_ROW,
    MutationOperation.UPDATE: RollbackStrategy.RESTORE_BEFORE_IMAGE,
    MutationOperation.TOMBSTONE: RollbackStrategy.REINSERT_TOMBSTONED_ROW,
}


class OmopRollbackManifestError(ValueError):
    """A value-free rollback-manifest validation error."""

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True, repr=False)
class OmopRollbackInstruction:
    """Caller-supplied rollback coverage for one staged mutation.

    ``rollback_artifact_digest`` binds the instruction to a protected local
    artifact that contains any row identity or before-image data. The artifact
    itself must stay outside the manifest.
    """

    mutation_ordinal: int
    strategy: RollbackStrategy
    rollback_artifact_digest: str

    def __post_init__(self) -> None:
        _require_non_negative_integer(self.mutation_ordinal, "mutation_ordinal")
        if not isinstance(self.strategy, RollbackStrategy):
            raise OmopRollbackManifestError("unknown_strategy", "strategy")
        _require_digest(self.rollback_artifact_digest, "rollback_artifact_digest")

    def __repr__(self) -> str:
        return (
            "OmopRollbackInstruction("
            f"mutation_ordinal={self.mutation_ordinal}, "
            f"strategy={self.strategy.value!r}, "
            f"rollback_artifact_digest={self.rollback_artifact_digest!r})"
        )


@dataclass(frozen=True, slots=True)
class OmopRollbackTableSummary:
    """Value-free operation counts for one OMOP table."""

    table: str
    operation_counts: tuple[tuple[str, int], ...]
    mutation_count: int

    def __post_init__(self) -> None:
        _require_identifier(self.table, "table")
        _require_operation_counts(self.operation_counts, "operation_counts")
        _require_positive_integer(self.mutation_count, "mutation_count")
        if sum(count for _, count in self.operation_counts) != self.mutation_count:
            raise OmopRollbackManifestError(
                "operation_counts_mismatch", "operation_counts"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-safe table metadata."""

        return {
            "mutation_count": self.mutation_count,
            "operation_counts": dict(self.operation_counts),
            "table": self.table,
        }


@dataclass(frozen=True, slots=True)
class OmopRollbackEntry:
    """Value-free rollback metadata for one staged mutation."""

    rollback_ordinal: int
    mutation_ordinal: int
    table: str
    operation: MutationOperation
    strategy: RollbackStrategy
    mutation_digest: str
    rollback_artifact_digest: str

    def __post_init__(self) -> None:
        _require_non_negative_integer(self.rollback_ordinal, "rollback_ordinal")
        _require_non_negative_integer(self.mutation_ordinal, "mutation_ordinal")
        _require_identifier(self.table, "table")
        if not isinstance(self.operation, MutationOperation):
            raise OmopRollbackManifestError("unknown_operation", "operation")
        if not isinstance(self.strategy, RollbackStrategy):
            raise OmopRollbackManifestError("unknown_strategy", "strategy")
        _require_digest(self.mutation_digest, "mutation_digest")
        _require_digest(self.rollback_artifact_digest, "rollback_artifact_digest")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-safe entry metadata."""

        return {
            "mutation_digest": self.mutation_digest,
            "mutation_ordinal": self.mutation_ordinal,
            "operation": self.operation.value,
            "rollback_artifact_digest": self.rollback_artifact_digest,
            "rollback_ordinal": self.rollback_ordinal,
            "strategy": self.strategy.value,
            "table": self.table,
        }


@dataclass(frozen=True, slots=True, repr=False)
class OmopRollbackManifest:
    """Deterministic, content-free rollback metadata for one staged batch."""

    batch_digest: str
    vocabulary_snapshot_digest: str
    mutation_count: int
    operation_counts: tuple[tuple[str, int], ...]
    tables: tuple[OmopRollbackTableSummary, ...]
    entries: tuple[OmopRollbackEntry, ...]
    manifest_digest: str
    schema: str = OMOP_ROLLBACK_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != OMOP_ROLLBACK_MANIFEST_SCHEMA:
            raise OmopRollbackManifestError("unsupported_schema", "schema")
        _require_digest(self.batch_digest, "batch_digest")
        _require_digest(
            self.vocabulary_snapshot_digest,
            "vocabulary_snapshot_digest",
        )
        _require_digest(self.manifest_digest, "manifest_digest")
        _require_positive_integer(self.mutation_count, "mutation_count")
        _require_operation_counts(self.operation_counts, "operation_counts")
        if sum(count for _, count in self.operation_counts) != self.mutation_count:
            raise OmopRollbackManifestError(
                "operation_counts_mismatch", "operation_counts"
            )
        if not isinstance(self.tables, tuple) or any(
            type(item) is not OmopRollbackTableSummary for item in self.tables
        ):
            raise OmopRollbackManifestError("invalid_tables", "tables")
        if tuple(table.table for table in self.tables) != tuple(
            sorted(table.table for table in self.tables)
        ) or len({table.table for table in self.tables}) != len(self.tables):
            raise OmopRollbackManifestError("invalid_table_order", "tables")
        if sum(table.mutation_count for table in self.tables) != self.mutation_count:
            raise OmopRollbackManifestError("table_summary_mismatch", "tables")
        if not isinstance(self.entries, tuple) or any(
            type(item) is not OmopRollbackEntry for item in self.entries
        ):
            raise OmopRollbackManifestError("invalid_entries", "entries")
        if len(self.entries) != self.mutation_count:
            raise OmopRollbackManifestError("incomplete_coverage", "entries")

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic content-free manifest."""

        return {
            "batch_digest": self.batch_digest,
            "entries": [entry.to_dict() for entry in self.entries],
            "manifest_digest": self.manifest_digest,
            "mutation_count": self.mutation_count,
            "operation_counts": dict(self.operation_counts),
            "schema": self.schema,
            "tables": [table.to_dict() for table in self.tables],
            "vocabulary_snapshot_digest": self.vocabulary_snapshot_digest,
        }

    def to_json(self) -> str:
        """Serialize the manifest with stable key ordering."""

        return _canonical_json(self.to_dict())

    def validate(
        self,
        batch: OmopMutationBatch,
        vocabulary_snapshot: VocabularySnapshot,
    ) -> None:
        """Raise unless this manifest still matches the batch and snapshot."""

        validate_omop_rollback_manifest(self, batch, vocabulary_snapshot)

    def __repr__(self) -> str:
        return (
            "OmopRollbackManifest("
            f"mutation_count={self.mutation_count}, "
            f"manifest_digest={self.manifest_digest!r})"
        )


def build_omop_rollback_manifest(
    batch: OmopMutationBatch,
    vocabulary_snapshot: VocabularySnapshot,
    instructions: Iterable[OmopRollbackInstruction],
) -> OmopRollbackManifest:
    """Build a complete rollback manifest for a staged OMOP batch.

    Instructions may be supplied in any order. The resulting entries are
    always ordered for rollback: last staged mutation first.

    Args:
        batch: The exact staged mutation batch awaiting approval.
        vocabulary_snapshot: The local target vocabulary snapshot.
        instructions: One rollback instruction for every mutation ordinal.

    Returns:
        A deterministic manifest containing only metadata and digests.

    Raises:
        OmopRollbackManifestError: If coverage is incomplete, duplicated, or
            incompatible with a staged operation.
    """

    _require_batch(batch)
    _require_snapshot(vocabulary_snapshot)
    normalized = _normalize_instructions(instructions)
    by_ordinal: dict[int, OmopRollbackInstruction] = {}
    for instruction in normalized:
        if instruction.mutation_ordinal in by_ordinal:
            raise OmopRollbackManifestError(
                "duplicate_mutation_coverage", "instructions"
            )
        by_ordinal[instruction.mutation_ordinal] = instruction

    expected_ordinals = set(range(len(batch.mutations)))
    supplied_ordinals = set(by_ordinal)
    if supplied_ordinals - expected_ordinals:
        raise OmopRollbackManifestError("unknown_mutation", "instructions")
    if expected_ordinals - supplied_ordinals:
        raise OmopRollbackManifestError("incomplete_coverage", "instructions")

    entries: list[OmopRollbackEntry] = []
    for rollback_ordinal, mutation_ordinal in enumerate(
        reversed(range(len(batch.mutations)))
    ):
        mutation = batch.mutations[mutation_ordinal]
        instruction = by_ordinal[mutation_ordinal]
        expected_strategy = _EXPECTED_STRATEGIES[mutation.operation]
        if instruction.strategy is not expected_strategy:
            raise OmopRollbackManifestError(
                "strategy_operation_mismatch", "instructions"
            )
        entries.append(
            OmopRollbackEntry(
                rollback_ordinal=rollback_ordinal,
                mutation_ordinal=mutation_ordinal,
                table=mutation.table,
                operation=mutation.operation,
                strategy=instruction.strategy,
                mutation_digest=mutation.row_digest,
                rollback_artifact_digest=instruction.rollback_artifact_digest,
            )
        )

    operation_counts, tables = _summaries(batch)
    payload = _manifest_payload(
        batch_digest=batch.batch_digest,
        vocabulary_snapshot_digest=vocabulary_snapshot.snapshot_digest,
        mutation_count=len(batch.mutations),
        operation_counts=operation_counts,
        tables=tables,
        entries=tuple(entries),
    )
    return OmopRollbackManifest(
        batch_digest=batch.batch_digest,
        vocabulary_snapshot_digest=vocabulary_snapshot.snapshot_digest,
        mutation_count=len(batch.mutations),
        operation_counts=operation_counts,
        tables=tables,
        entries=tuple(entries),
        manifest_digest=_digest(payload),
    )


def validate_omop_rollback_manifest(
    manifest: OmopRollbackManifest,
    batch: OmopMutationBatch,
    vocabulary_snapshot: VocabularySnapshot,
) -> None:
    """Validate manifest integrity, coverage, order, and snapshot binding."""

    if type(manifest) is not OmopRollbackManifest:
        raise OmopRollbackManifestError("invalid_manifest", "manifest")
    _require_batch(batch)
    _require_snapshot(vocabulary_snapshot)
    if manifest.batch_digest != batch.batch_digest:
        raise OmopRollbackManifestError("batch_mismatch", "batch_digest")
    if manifest.vocabulary_snapshot_digest != vocabulary_snapshot.snapshot_digest:
        raise OmopRollbackManifestError(
            "vocabulary_snapshot_mismatch",
            "vocabulary_snapshot_digest",
        )

    operation_counts, tables = _summaries(batch)
    if manifest.mutation_count != len(batch.mutations):
        raise OmopRollbackManifestError("mutation_count_mismatch", "mutation_count")
    if manifest.operation_counts != operation_counts:
        raise OmopRollbackManifestError("operation_counts_mismatch", "operation_counts")
    if manifest.tables != tables:
        raise OmopRollbackManifestError("table_summary_mismatch", "tables")
    if len(manifest.entries) != len(batch.mutations):
        raise OmopRollbackManifestError("incomplete_coverage", "entries")

    seen: set[int] = set()
    for rollback_ordinal, entry in enumerate(manifest.entries):
        if entry.mutation_ordinal in seen:
            raise OmopRollbackManifestError("duplicate_mutation_coverage", "entries")
        seen.add(entry.mutation_ordinal)
        expected_mutation_ordinal = len(batch.mutations) - rollback_ordinal - 1
        if (
            entry.rollback_ordinal != rollback_ordinal
            or entry.mutation_ordinal != expected_mutation_ordinal
        ):
            raise OmopRollbackManifestError("invalid_rollback_order", "entries")
        mutation = batch.mutations[entry.mutation_ordinal]
        if (
            entry.table != mutation.table
            or entry.operation is not mutation.operation
            or entry.mutation_digest != mutation.row_digest
        ):
            raise OmopRollbackManifestError("mutation_metadata_mismatch", "entries")
        if entry.strategy is not _EXPECTED_STRATEGIES[mutation.operation]:
            raise OmopRollbackManifestError("strategy_operation_mismatch", "entries")
        _require_digest(entry.rollback_artifact_digest, "rollback_artifact_digest")

    payload = _manifest_payload(
        batch_digest=manifest.batch_digest,
        vocabulary_snapshot_digest=manifest.vocabulary_snapshot_digest,
        mutation_count=manifest.mutation_count,
        operation_counts=manifest.operation_counts,
        tables=manifest.tables,
        entries=manifest.entries,
    )
    if manifest.manifest_digest != _digest(payload):
        raise OmopRollbackManifestError("manifest_digest_mismatch", "manifest_digest")


def _normalize_instructions(
    instructions: Iterable[OmopRollbackInstruction],
) -> tuple[OmopRollbackInstruction, ...]:
    if isinstance(instructions, (str, bytes, bytearray, Mapping)):
        raise OmopRollbackManifestError(
            "invalid_instruction_collection", "instructions"
        )
    try:
        normalized = tuple(instructions)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise OmopRollbackManifestError(
            "invalid_instruction_collection", "instructions"
        ) from None
    if any(type(item) is not OmopRollbackInstruction for item in normalized):
        raise OmopRollbackManifestError("invalid_instruction", "instructions")
    return normalized


def _summaries(
    batch: OmopMutationBatch,
) -> tuple[
    tuple[tuple[str, int], ...],
    tuple[OmopRollbackTableSummary, ...],
]:
    total_counts = Counter(mutation.operation.value for mutation in batch.mutations)
    table_counts: dict[str, Counter[str]] = {}
    for mutation in batch.mutations:
        table_counts.setdefault(mutation.table, Counter())[
            mutation.operation.value
        ] += 1
    tables = tuple(
        OmopRollbackTableSummary(
            table=table,
            operation_counts=tuple(sorted(counts.items())),
            mutation_count=sum(counts.values()),
        )
        for table, counts in sorted(table_counts.items())
    )
    return tuple(sorted(total_counts.items())), tables


def _manifest_payload(
    *,
    batch_digest: str,
    vocabulary_snapshot_digest: str,
    mutation_count: int,
    operation_counts: tuple[tuple[str, int], ...],
    tables: tuple[OmopRollbackTableSummary, ...],
    entries: tuple[OmopRollbackEntry, ...],
) -> dict[str, Any]:
    return {
        "batch_digest": batch_digest,
        "entries": [entry.to_dict() for entry in entries],
        "mutation_count": mutation_count,
        "operation_counts": dict(operation_counts),
        "schema": OMOP_ROLLBACK_MANIFEST_SCHEMA,
        "tables": [table.to_dict() for table in tables],
        "vocabulary_snapshot_digest": vocabulary_snapshot_digest,
    }


def _require_batch(batch: OmopMutationBatch) -> None:
    if type(batch) is not OmopMutationBatch:
        raise OmopRollbackManifestError("invalid_batch", "batch")


def _require_snapshot(snapshot: VocabularySnapshot) -> None:
    if type(snapshot) is not VocabularySnapshot:
        raise OmopRollbackManifestError(
            "invalid_vocabulary_snapshot", "vocabulary_snapshot"
        )


def _require_non_negative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OmopRollbackManifestError("invalid_non_negative_integer", field_name)
    return value


def _require_positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OmopRollbackManifestError("invalid_positive_integer", field_name)
    return value


def _require_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise OmopRollbackManifestError("invalid_identifier", field_name)
    return value


def _require_operation_counts(
    value: Any,
    field_name: str,
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, tuple):
        raise OmopRollbackManifestError("invalid_operation_counts", field_name)
    allowed = {operation.value for operation in MutationOperation}
    if any(
        not isinstance(item, tuple)
        or len(item) != 2
        or item[0] not in allowed
        or isinstance(item[1], bool)
        or not isinstance(item[1], int)
        or item[1] <= 0
        for item in value
    ):
        raise OmopRollbackManifestError("invalid_operation_counts", field_name)
    names = tuple(name for name, _ in value)
    if names != tuple(sorted(names)) or len(set(names)) != len(names):
        raise OmopRollbackManifestError("invalid_operation_counts", field_name)
    return value


def _require_digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise OmopRollbackManifestError("invalid_digest", field_name)
    return value


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
    "OMOP_ROLLBACK_MANIFEST_SCHEMA",
    "OmopRollbackEntry",
    "OmopRollbackInstruction",
    "OmopRollbackManifest",
    "OmopRollbackManifestError",
    "OmopRollbackTableSummary",
    "RollbackStrategy",
    "build_omop_rollback_manifest",
    "validate_omop_rollback_manifest",
]
