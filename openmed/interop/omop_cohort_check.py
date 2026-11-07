"""Privacy-safe validation for OpenMed OMOP cohort exports.

The checker operates on in-memory OMOP rows and intentionally has no network,
database, or vocabulary-service dependency.  Validation output contains table
and column names, aggregate counts, failure reasons, and deterministic row
fingerprints only; it never copies source identifiers or note text into a
report or exception.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any, Final, TypeAlias

from .omop.cdm_loader import OmopCdmTables

TableRows: TypeAlias = Mapping[str, Iterable[Mapping[str, Any]]]

_MISSING: Final = object()
_INVALID: Final = object()
_MAX_CANONICAL_DEPTH: Final = 64
_MIN_BIGINT: Final = -(2**63)
_MAX_BIGINT: Final = 2**63 - 1
_FINGERPRINT_PATTERN: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SOURCE_HASH_PATTERN: Final = re.compile(r"[0-9a-f]{64}\Z")

_TABLE_ORDER: Final[tuple[str, ...]] = (
    "concept",
    "person",
    "visit_occurrence",
    "note",
    "note_nlp",
    "condition_occurrence",
    "drug_exposure",
    "measurement",
    "procedure_occurrence",
    "observation",
    "source_to_concept_map",
)

_PRIMARY_KEYS: Final[Mapping[str, str]] = {
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

_DOMAIN_TABLES: Final[Mapping[str, str]] = {
    "condition_occurrence": "condition_concept_id",
    "drug_exposure": "drug_concept_id",
    "measurement": "measurement_concept_id",
    "procedure_occurrence": "procedure_concept_id",
    "observation": "observation_concept_id",
}

_SOURCE_CONCEPT_COLUMNS: Final[Mapping[str, str]] = {
    "condition_occurrence": "condition_source_concept_id",
    "drug_exposure": "drug_source_concept_id",
    "measurement": "measurement_source_concept_id",
    "procedure_occurrence": "procedure_source_concept_id",
    "observation": "observation_source_concept_id",
}

_CONCEPT_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = {
    "visit_occurrence": ("visit_concept_id", "visit_source_concept_id"),
    "note": (
        "note_type_concept_id",
        "note_class_concept_id",
        "encoding_concept_id",
        "language_concept_id",
    ),
    "note_nlp": (
        "section_concept_id",
        "note_nlp_concept_id",
        "note_nlp_source_concept_id",
        "note_nlp_event_field_concept_id",
    ),
    "source_to_concept_map": ("source_concept_id", "target_concept_id"),
}

_ALLOWED_STANDARD_CONCEPTS: Final[frozenset[str]] = frozenset({"", "C", "N", "S"})
_VIOLATION_REASONS: Final[frozenset[str]] = frozenset(
    {
        "ambiguous_event",
        "conflicting_vocabulary_mapping",
        "duplicate_primary_key",
        "invalid_primary_key",
        "invalid_provenance",
        "invalid_reference",
        "invalid_standard_concept",
        "missing_primary_key",
        "missing_provenance",
        "missing_reference",
        "nonstandard_concept_reference",
        "nonstandard_target_concept",
        "person_visit_mismatch",
        "provenance_mismatch",
        "unreachable_event",
        "vocabulary_mismatch",
    }
)
_VIOLATION_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        *_PRIMARY_KEYS.values(),
        *_DOMAIN_TABLES.values(),
        *_SOURCE_CONCEPT_COLUMNS.values(),
        *(column for columns in _CONCEPT_COLUMNS.values() for column in columns),
        "note_id",
        "note_nlp_event_id",
        "note_nlp_id",
        "person_id",
        "source_code",
        "source_note_hash",
        "source_vocabulary_id",
        "standard_concept",
        "target_concept_id",
        "target_vocabulary_id",
        "visit_occurrence_id",
    }
)


class _CohortInputError(ValueError):
    """Internal marker for deliberately sanitized input errors."""


@dataclass(frozen=True)
class OmopCohortViolation:
    """One grouped, PHI-free cohort validation failure."""

    table: str
    column: str | None
    reason: str
    count: int
    row_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.table, str) or self.table not in _TABLE_ORDER:
            raise ValueError("violation table is unsupported")
        if self.column is not None and (
            not isinstance(self.column, str) or self.column not in _VIOLATION_COLUMNS
        ):
            raise ValueError("violation column is unsupported")
        if not isinstance(self.reason, str) or self.reason not in _VIOLATION_REASONS:
            raise ValueError("violation reason is unsupported")
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count <= 0
            or self.count > _MAX_BIGINT
        ):
            raise ValueError("violation count must be a positive bounded integer")
        try:
            fingerprints = tuple(sorted(set(self.row_fingerprints)))
        except MemoryError:
            raise
        except Exception:
            raise ValueError(
                "row_fingerprints must contain SHA-256 fingerprints"
            ) from None
        if not fingerprints or any(
            not isinstance(value, str) or not _FINGERPRINT_PATTERN.fullmatch(value)
            for value in fingerprints
        ):
            raise ValueError("row_fingerprints must contain SHA-256 fingerprints")
        if len(fingerprints) > self.count:
            raise ValueError("row_fingerprints cannot exceed the violation count")
        object.__setattr__(self, "row_fingerprints", fingerprints)

    @property
    def fingerprints(self) -> tuple[str, ...]:
        """Return deterministic fingerprints for affected rows."""

        return self.row_fingerprints

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible failure without source row values."""

        result: dict[str, Any] = {
            "table": self.table,
            "reason": self.reason,
            "count": self.count,
            "row_fingerprints": list(self.row_fingerprints),
        }
        if self.column is not None:
            result["column"] = self.column
        return result


@dataclass(frozen=True)
class OmopCohortValidationReport:
    """Aggregate diagnostics for one OMOP cohort export validation run."""

    row_counts: Mapping[str, int]
    violations: tuple[OmopCohortViolation, ...]

    def __post_init__(self) -> None:
        counts: dict[str, int] = {}
        try:
            items = tuple(self.row_counts.items())
        except MemoryError:
            raise
        except Exception:
            raise ValueError("row_counts must be a table count mapping") from None
        for table, count in items:
            if not isinstance(table, str) or table not in _TABLE_ORDER:
                raise ValueError("row_counts contains an unsupported table")
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                or count > _MAX_BIGINT
            ):
                raise ValueError(
                    "row_counts must contain bounded non-negative integers"
                )
            counts[table] = count
        object.__setattr__(
            self,
            "row_counts",
            MappingProxyType({name: counts.get(name, 0) for name in _TABLE_ORDER}),
        )
        try:
            violations = tuple(self.violations)
        except MemoryError:
            raise
        except Exception:
            raise ValueError("violations must be an iterable of violations") from None
        if not all(isinstance(item, OmopCohortViolation) for item in violations):
            raise TypeError("violations must contain OmopCohortViolation instances")
        object.__setattr__(self, "violations", violations)

    @property
    def violation_count(self) -> int:
        """Return the number of failed row-level invariants."""

        return sum(item.count for item in self.violations)

    @property
    def failure_count(self) -> int:
        """Alias for :attr:`violation_count` used by cohort checks."""

        return self.violation_count

    @property
    def failures(self) -> tuple[OmopCohortViolation, ...]:
        """Return grouped failures under the issue's terminology."""

        return self.violations

    @property
    def is_valid(self) -> bool:
        """Return whether all checked invariants passed."""

        return self.violation_count == 0

    @property
    def valid(self) -> bool:
        """Return :attr:`is_valid` as a compact compatibility alias."""

        return self.is_valid

    @property
    def by_table(self) -> Mapping[str, int]:
        """Return deterministic failure counts grouped by table."""

        counts: Counter[str] = Counter()
        for item in self.violations:
            counts[item.table] += item.count
        return dict(sorted(counts.items()))

    @property
    def by_reason(self) -> Mapping[str, int]:
        """Return deterministic failure counts grouped by reason."""

        counts: Counter[str] = Counter()
        for item in self.violations:
            counts[item.reason] += item.count
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        """Return counts, fingerprints, and no raw cohort values."""

        return {
            "count": self.violation_count,
            "row_counts": dict(self.row_counts),
            "by_table": dict(self.by_table),
            "by_reason": dict(self.by_reason),
            "violations": [item.to_dict() for item in self.violations],
        }


class OmopCohortExportValidationError(ValueError):
    """Raised by the assertion helper with only aggregate diagnostics."""

    def __init__(self, report: OmopCohortValidationReport) -> None:
        if not isinstance(report, OmopCohortValidationReport):
            raise TypeError("report must be an OmopCohortValidationReport")
        self.report = report
        super().__init__(
            "OMOP cohort export validation failed with "
            f"{report.violation_count} row-level failure(s)"
        )


class _FailureCollector:
    """Collect grouped fingerprints while retaining exact failure counts."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str | None, str], Counter[str]] = defaultdict(
            Counter
        )

    def add(
        self,
        table: str,
        row: Mapping[str, Any],
        reason: str,
        column: str | None = None,
    ) -> None:
        fingerprint = omop_row_fingerprint(table, row)
        self._items[(table, column, reason)][fingerprint] += 1

    def report(self, row_counts: Mapping[str, int]) -> OmopCohortValidationReport:
        table_order = {name: index for index, name in enumerate(_TABLE_ORDER)}
        ordered = sorted(
            self._items.items(),
            key=lambda item: (
                table_order.get(item[0][0], len(_TABLE_ORDER)),
                item[0][1] or "",
                item[0][2],
            ),
        )
        violations = tuple(
            OmopCohortViolation(
                table=key[0],
                column=key[1],
                reason=key[2],
                count=sum(fingerprints.values()),
                row_fingerprints=tuple(sorted(fingerprints)),
            )
            for key, fingerprints in ordered
        )
        return OmopCohortValidationReport(
            row_counts=dict(row_counts),
            violations=violations,
        )


def omop_row_fingerprint(table: str, row: Mapping[str, Any]) -> str:
    """Return a deterministic content fingerprint for one OMOP row.

    The table name is included so identical rows in different tables do not
    share a diagnostic fingerprint.  The input is serialized only to compute
    the digest; it is never returned or embedded in an exception.
    """

    try:
        if not isinstance(table, str) or table not in _TABLE_ORDER:
            raise ValueError("OMOP row table is unsupported")
        if not isinstance(row, Mapping):
            raise TypeError("OMOP row must be a mapping")
        payload = {
            "schema": "openmed.omop.cohort-check.v1",
            "table": table,
            "row": _canonicalize(row),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    except MemoryError:
        raise
    except Exception:
        raise ValueError("OMOP row cannot be fingerprinted safely") from None


def validate_omop_cohort_export(
    export: OmopCdmTables | TableRows | Mapping[str, Any],
) -> OmopCohortValidationReport:
    """Validate an in-memory OMOP cohort export without network access.

    ``export`` may be an :class:`~openmed.interop.omop.OmopCdmTables` instance,
    a mapping of OMOP table names to row iterables, or the mapping returned by
    ``OmopCdmTables.to_dict()``.  Missing tables are treated as empty, while
    rows in present tables are checked for key relationships, vocabulary
    consistency, and OpenMed provenance links.

    The returned report is deterministic for the same rows regardless of
    input mapping order.  Invalid rows do not appear in the report; only their
    aggregate counts and row fingerprints do.
    """

    try:
        return _validate_omop_cohort_export(export)
    except MemoryError:
        raise
    except _CohortInputError as exc:
        raise ValueError(str(exc)) from None
    except Exception:
        raise ValueError("cohort export could not be validated safely") from None


def _validate_omop_cohort_export(
    export: OmopCdmTables | TableRows | Mapping[str, Any],
) -> OmopCohortValidationReport:
    tables = _normalise_tables(export)
    row_counts = {name: len(tables[name]) for name in _TABLE_ORDER}
    collector = _FailureCollector()
    indexes = _build_primary_indexes(tables, collector)

    _validate_relationships(tables, indexes, collector)
    _validate_vocabulary(tables, indexes, collector)
    _validate_provenance(tables, indexes, collector)

    return collector.report(row_counts)


def check_omop_cohort_export(
    export: OmopCdmTables | TableRows | Mapping[str, Any],
) -> OmopCohortValidationReport:
    """Compatibility alias for :func:`validate_omop_cohort_export`."""

    return validate_omop_cohort_export(export)


def validate_cohort_export(
    export: OmopCdmTables | TableRows | Mapping[str, Any],
) -> OmopCohortValidationReport:
    """Short alias for :func:`validate_omop_cohort_export`."""

    return validate_omop_cohort_export(export)


def assert_valid_omop_cohort_export(
    export: OmopCdmTables | TableRows | Mapping[str, Any],
) -> OmopCohortValidationReport:
    """Validate an export and raise a PHI-free exception when it is invalid."""

    report = validate_omop_cohort_export(export)
    if not report.is_valid:
        raise OmopCohortExportValidationError(report)
    return report


def _normalise_tables(
    export: OmopCdmTables | TableRows | Mapping[str, Any],
) -> dict[str, tuple[dict[str, Any], ...]]:
    if isinstance(export, OmopCdmTables):
        source: Mapping[str, Any] = export.tables
    elif isinstance(export, Mapping):
        nested = export.get("tables")
        source = nested if isinstance(nested, Mapping) else export
    else:
        raise _CohortInputError("cohort export must be an OMOP table mapping")

    normalised: dict[str, tuple[dict[str, Any], ...]] = {}
    known_tables = set(_TABLE_ORDER)
    for raw_name, raw_rows in source.items():
        if not isinstance(raw_name, str):
            raise _CohortInputError("cohort export table names must be strings")
        name = raw_name.lower()
        if name not in known_tables:
            raise _CohortInputError("cohort export contains an unsupported table")
        if raw_rows is None:
            rows: Iterable[Any] = ()
        elif isinstance(raw_rows, (str, bytes, bytearray)):
            raise _CohortInputError("cohort export table rows must be mappings")
        else:
            try:
                rows = iter(raw_rows)
            except MemoryError:
                raise
            except Exception:
                raise _CohortInputError(
                    "cohort export table rows must be iterable"
                ) from None

        materialised: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise _CohortInputError("cohort export table rows must be mappings")
            copied = dict(row)
            if not all(isinstance(column, str) for column in copied):
                raise _CohortInputError("cohort export column names must be strings")
            materialised.append(copied)
        if name in normalised:
            raise _CohortInputError("cohort export contains duplicate table names")
        normalised[name] = tuple(materialised)

    return {name: normalised.get(name, ()) for name in _TABLE_ORDER}


def _build_primary_indexes(
    tables: Mapping[str, tuple[dict[str, Any], ...]],
    collector: _FailureCollector,
) -> dict[str, dict[int, Mapping[str, Any]]]:
    indexes: dict[str, dict[int, Mapping[str, Any]]] = {
        name: {} for name in _TABLE_ORDER
    }
    for table in _TABLE_ORDER:
        primary_key = _PRIMARY_KEYS[table]
        candidates: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in tables[table]:
            value = _identifier(row.get(primary_key, _MISSING))
            if value is _MISSING or value is None:
                collector.add(table, row, "missing_primary_key", primary_key)
            elif value is _INVALID or not isinstance(value, int):
                collector.add(table, row, "invalid_primary_key", primary_key)
            else:
                candidates[value].append(row)
        for value, rows in candidates.items():
            if len(rows) == 1:
                indexes[table][value] = rows[0]
                continue
            for duplicate_row in rows:
                collector.add(
                    table,
                    duplicate_row,
                    "duplicate_primary_key",
                    primary_key,
                )
    return indexes


def _validate_relationships(
    tables: Mapping[str, tuple[dict[str, Any], ...]],
    indexes: Mapping[str, Mapping[int, Mapping[str, Any]]],
    collector: _FailureCollector,
) -> None:
    required_foreign_keys: Mapping[str, tuple[tuple[str, str], ...]] = {
        "visit_occurrence": (("person_id", "person"),),
        "note": (("person_id", "person"),),
        "note_nlp": (("note_id", "note"),),
        "condition_occurrence": (("person_id", "person"),),
        "drug_exposure": (("person_id", "person"),),
        "measurement": (("person_id", "person"),),
        "procedure_occurrence": (("person_id", "person"),),
        "observation": (("person_id", "person"),),
    }

    for table, references in required_foreign_keys.items():
        for column, target in references:
            _check_reference(
                tables[table],
                indexes[target],
                table,
                column,
                collector,
                required=True,
            )

    optional_foreign_keys: Mapping[str, tuple[tuple[str, str], ...]] = {
        "visit_occurrence": (("visit_concept_id", "concept"),),
        "note": (("visit_occurrence_id", "visit_occurrence"),),
        "note_nlp": (),
        "condition_occurrence": (
            ("visit_occurrence_id", "visit_occurrence"),
            ("note_id", "note"),
            ("note_nlp_id", "note_nlp"),
        ),
        "drug_exposure": (
            ("visit_occurrence_id", "visit_occurrence"),
            ("note_id", "note"),
            ("note_nlp_id", "note_nlp"),
        ),
        "measurement": (
            ("visit_occurrence_id", "visit_occurrence"),
            ("note_id", "note"),
            ("note_nlp_id", "note_nlp"),
        ),
        "procedure_occurrence": (
            ("visit_occurrence_id", "visit_occurrence"),
            ("note_id", "note"),
            ("note_nlp_id", "note_nlp"),
        ),
        "observation": (
            ("visit_occurrence_id", "visit_occurrence"),
            ("note_id", "note"),
            ("note_nlp_id", "note_nlp"),
        ),
        "source_to_concept_map": (("note_nlp_id", "note_nlp"),),
    }
    for table, references in optional_foreign_keys.items():
        for column, target in references:
            if _field_enabled(tables[table], column):
                _check_reference(
                    tables[table],
                    indexes[target],
                    table,
                    column,
                    collector,
                    required=True,
                )

    for table, columns in _CONCEPT_COLUMNS.items():
        for column in columns:
            if _field_enabled(tables[table], column):
                _check_reference(
                    tables[table],
                    indexes["concept"],
                    table,
                    column,
                    collector,
                    required=True,
                )
    for table, column in _DOMAIN_TABLES.items():
        _check_reference(
            tables[table],
            indexes["concept"],
            table,
            column,
            collector,
            required=True,
        )

    for table, source_column in _SOURCE_CONCEPT_COLUMNS.items():
        _check_reference(
            tables[table],
            indexes["concept"],
            table,
            source_column,
            collector,
            required=True,
        )

    _validate_person_visit_consistency(tables, indexes, collector)


def _validate_person_visit_consistency(
    tables: Mapping[str, tuple[dict[str, Any], ...]],
    indexes: Mapping[str, Mapping[int, Mapping[str, Any]]],
    collector: _FailureCollector,
) -> None:
    for table in (
        "note",
        "condition_occurrence",
        "drug_exposure",
        "measurement",
        "procedure_occurrence",
        "observation",
    ):
        for row in tables[table]:
            visit_id = _identifier(row.get("visit_occurrence_id", _MISSING))
            person_id = _identifier(row.get("person_id", _MISSING))
            if not isinstance(visit_id, int) or not isinstance(person_id, int):
                continue
            visit = indexes["visit_occurrence"].get(visit_id)
            if visit is None:
                continue
            if _identifier(visit.get("person_id", _MISSING)) != person_id:
                collector.add(
                    table, row, "person_visit_mismatch", "visit_occurrence_id"
                )


def _validate_vocabulary(
    tables: Mapping[str, tuple[dict[str, Any], ...]],
    indexes: Mapping[str, Mapping[int, Mapping[str, Any]]],
    collector: _FailureCollector,
) -> None:
    concept_rows = tables["concept"]
    for row in concept_rows:
        standard = row.get("standard_concept", _MISSING)
        if (
            standard is not _MISSING
            and str(standard or "") not in _ALLOWED_STANDARD_CONCEPTS
        ):
            collector.add(
                "concept", row, "invalid_standard_concept", "standard_concept"
            )

    for table, concept_column in _DOMAIN_TABLES.items():
        for row in tables[table]:
            concept_id = _identifier(row.get(concept_column, _MISSING))
            if not isinstance(concept_id, int):
                continue
            concept = indexes["concept"].get(concept_id)
            if concept is None:
                continue
            standard = concept.get("standard_concept", _MISSING)
            if standard is not _MISSING and str(standard or "") not in {"", "S"}:
                collector.add(
                    table, row, "nonstandard_concept_reference", concept_column
                )

    for row in tables["source_to_concept_map"]:
        source_id = _identifier(row.get("source_concept_id", _MISSING))
        target_id = _identifier(row.get("target_concept_id", _MISSING))
        source = (
            indexes["concept"].get(source_id) if isinstance(source_id, int) else None
        )
        target = (
            indexes["concept"].get(target_id) if isinstance(target_id, int) else None
        )

        source_vocabulary = _text_value(row.get("source_vocabulary_id", _MISSING))
        if source is not None and source_vocabulary:
            concept_vocabulary = _text_value(source.get("vocabulary_id", _MISSING))
            if concept_vocabulary and source_vocabulary != concept_vocabulary:
                collector.add(
                    "source_to_concept_map",
                    row,
                    "vocabulary_mismatch",
                    "source_vocabulary_id",
                )

        target_vocabulary = _text_value(row.get("target_vocabulary_id", _MISSING))
        if target is not None and target_vocabulary:
            concept_vocabulary = _text_value(target.get("vocabulary_id", _MISSING))
            if concept_vocabulary and target_vocabulary != concept_vocabulary:
                collector.add(
                    "source_to_concept_map",
                    row,
                    "vocabulary_mismatch",
                    "target_vocabulary_id",
                )
            standard = target.get("standard_concept", _MISSING)
            if (
                target_id != 0
                and standard is not _MISSING
                and str(standard or "") not in {"", "S"}
            ):
                collector.add(
                    "source_to_concept_map",
                    row,
                    "nonstandard_target_concept",
                    "target_concept_id",
                )

    mappings: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in tables["source_to_concept_map"]:
        source_code = _text_value(row.get("source_code", _MISSING))
        source_vocabulary = _text_value(row.get("source_vocabulary_id", _MISSING))
        target_id = _identifier(row.get("target_concept_id", _MISSING))
        if source_code and source_vocabulary and isinstance(target_id, int):
            mappings[(source_vocabulary, source_code)].add(target_id)
    conflicting = {key for key, targets in mappings.items() if len(targets) > 1}
    for row in tables["source_to_concept_map"]:
        key = (
            _text_value(row.get("source_vocabulary_id", _MISSING)),
            _text_value(row.get("source_code", _MISSING)),
        )
        if key in conflicting:
            collector.add(
                "source_to_concept_map",
                row,
                "conflicting_vocabulary_mapping",
                "source_code",
            )


def _validate_provenance(
    tables: Mapping[str, tuple[dict[str, Any], ...]],
    indexes: Mapping[str, Mapping[int, Mapping[str, Any]]],
    collector: _FailureCollector,
) -> None:
    note_indexes = indexes["note"]
    note_nlp_indexes = indexes["note_nlp"]
    domain_tables = tuple(_DOMAIN_TABLES)

    for row in tables["note"]:
        if _field_enabled(tables["note"], "source_note_hash"):
            source_hash = _text_value(row.get("source_note_hash", _MISSING))
            if not _valid_source_hash(source_hash):
                collector.add("note", row, "invalid_provenance", "source_note_hash")

    for table in domain_tables:
        fields_enabled = {
            field
            for field in ("note_id", "note_nlp_id", "source_note_hash")
            if _field_enabled(tables[table], field)
        }
        for row in tables[table]:
            note_id = _identifier(row.get("note_id", _MISSING))
            note_nlp_id = _identifier(row.get("note_nlp_id", _MISSING))
            source_hash = _text_value(row.get("source_note_hash", _MISSING))

            for field in fields_enabled:
                if field not in row or _is_empty(row.get(field)):
                    collector.add(table, row, "missing_provenance", field)
            if source_hash and not _valid_source_hash(source_hash):
                collector.add(table, row, "invalid_provenance", "source_note_hash")

            note = note_indexes.get(note_id) if isinstance(note_id, int) else None
            note_nlp = (
                note_nlp_indexes.get(note_nlp_id)
                if isinstance(note_nlp_id, int)
                else None
            )
            if note is not None and source_hash:
                note_hash = _text_value(note.get("source_note_hash", _MISSING))
                if note_hash and note_hash != source_hash:
                    collector.add(table, row, "provenance_mismatch", "source_note_hash")
            if note is not None and note_nlp is not None:
                linked_note_id = _identifier(note_nlp.get("note_id", _MISSING))
                if linked_note_id != note_id:
                    collector.add(table, row, "provenance_mismatch", "note_nlp_id")

            primary_id = _identifier(row.get(_PRIMARY_KEYS[table], _MISSING))
            if note_nlp is not None and isinstance(primary_id, int):
                event_id = _identifier(note_nlp.get("note_nlp_event_id", _MISSING))
                if (
                    event_id not in (_MISSING, _INVALID, None)
                    and event_id != primary_id
                ):
                    collector.add(
                        table,
                        row,
                        "provenance_mismatch",
                        "note_nlp_event_id",
                    )

    event_rows: dict[int, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for table in domain_tables:
        for row in tables[table]:
            row_id = _identifier(row.get(_PRIMARY_KEYS[table], _MISSING))
            if isinstance(row_id, int):
                event_rows[row_id].append((table, row))

    for row in tables["note_nlp"]:
        event_id = _identifier(row.get("note_nlp_event_id", _MISSING))
        if not isinstance(event_id, int):
            continue
        linked = event_rows.get(event_id, ())
        if not linked:
            collector.add("note_nlp", row, "unreachable_event", "note_nlp_event_id")
            continue
        note_nlp_id = _identifier(row.get("note_nlp_id", _MISSING))
        matching_rows = [
            domain_row
            for _, domain_row in linked
            if _identifier(domain_row.get("note_nlp_id", _MISSING)) == note_nlp_id
        ]
        if not matching_rows:
            collector.add("note_nlp", row, "provenance_mismatch", "note_nlp_event_id")
        elif len(matching_rows) > 1:
            collector.add("note_nlp", row, "ambiguous_event", "note_nlp_event_id")

    for row in tables["source_to_concept_map"]:
        note_nlp_id = _identifier(row.get("note_nlp_id", _MISSING))
        source_hash = _text_value(row.get("source_note_hash", _MISSING))
        if source_hash and not _valid_source_hash(source_hash):
            collector.add(
                "source_to_concept_map",
                row,
                "invalid_provenance",
                "source_note_hash",
            )
            continue
        if not isinstance(note_nlp_id, int) or not source_hash:
            continue
        note_nlp = note_nlp_indexes.get(note_nlp_id)
        if note_nlp is None:
            continue
        note_id = _identifier(note_nlp.get("note_id", _MISSING))
        note = note_indexes.get(note_id) if isinstance(note_id, int) else None
        note_hash = _text_value(note.get("source_note_hash", _MISSING)) if note else ""
        if note_hash and note_hash != source_hash:
            collector.add(
                "source_to_concept_map",
                row,
                "provenance_mismatch",
                "source_note_hash",
            )


def _check_reference(
    rows: Iterable[Mapping[str, Any]],
    target: Mapping[int, Mapping[str, Any]],
    table: str,
    column: str,
    collector: _FailureCollector,
    *,
    required: bool,
) -> None:
    for row in rows:
        value = _identifier(row.get(column, _MISSING))
        if value is _MISSING:
            if required:
                collector.add(table, row, "missing_reference", column)
        elif value is _INVALID:
            collector.add(table, row, "invalid_reference", column)
        elif value is None:
            if required:
                collector.add(table, row, "missing_reference", column)
        elif value not in target:
            collector.add(table, row, "missing_reference", column)


def _field_enabled(rows: Iterable[Mapping[str, Any]], field: str) -> bool:
    return any(field in row for row in rows)


def _identifier(value: Any) -> int | object | None:
    if value is _MISSING:
        return _MISSING
    if value is None:
        return None
    if isinstance(value, bool):
        return _INVALID
    if isinstance(value, Integral):
        converted = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            converted = int(stripped)
        except ValueError:
            return _INVALID
    else:
        return _INVALID
    if converted < _MIN_BIGINT or converted > _MAX_BIGINT:
        return _INVALID
    return converted


def _text_value(value: Any) -> str:
    if value is _MISSING or value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _valid_source_hash(value: str) -> bool:
    return bool(_SOURCE_HASH_PATTERN.fullmatch(value))


def _is_empty(value: Any) -> bool:
    return (
        value is _MISSING
        or value is None
        or (isinstance(value, str) and not value.strip())
    )


def _canonicalize(
    value: Any,
    *,
    _active: set[int] | None = None,
    _depth: int = 0,
) -> Any:
    if _depth > _MAX_CANONICAL_DEPTH:
        raise ValueError("OMOP row nesting exceeds the supported depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"type": "float", "value": str(value)}
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        active = set() if _active is None else _active
        marker = id(value)
        if marker in active:
            raise ValueError("OMOP row contains a cyclic value")
        active.add(marker)
        try:
            if isinstance(value, Mapping):
                items = [
                    [
                        _canonicalize(key, _active=active, _depth=_depth + 1),
                        _canonicalize(item, _active=active, _depth=_depth + 1),
                    ]
                    for key, item in value.items()
                ]
                return {
                    "type": "mapping",
                    "items": sorted(items, key=_canonical_sort_key),
                }
            values = [
                _canonicalize(item, _active=active, _depth=_depth + 1) for item in value
            ]
            if isinstance(value, (set, frozenset)):
                values.sort(key=_canonical_sort_key)
            return values
        finally:
            active.remove(marker)
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": str(value),
    }


def _canonical_sort_key(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


OmopCohortCheckReport = OmopCohortValidationReport
OmopCohortExportReport = OmopCohortValidationReport

__all__ = [
    "OmopCohortCheckReport",
    "OmopCohortExportReport",
    "OmopCohortExportValidationError",
    "OmopCohortValidationReport",
    "OmopCohortViolation",
    "TableRows",
    "assert_valid_omop_cohort_export",
    "check_omop_cohort_export",
    "omop_row_fingerprint",
    "validate_cohort_export",
    "validate_omop_cohort_export",
]
