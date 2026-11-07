"""Referential-integrity-preserving surrogate map for linked tables (section 4.2).

De-identifying a single table in isolation breaks the foreign keys that join it
to its siblings: if ``demographics``, ``encounters`` and ``labs`` each replace
``patient_id`` independently, one patient receives three different surrogates
and the tables no longer join. This module coordinates the replacement so that
referential integrity survives de-identification.

Every column enrolled in a *key space* is pseudonymized through
:class:`openmed.core.surrogate_vault.SurrogateVault`, so an identical source
value maps to an identical surrogate across all tables and every foreign key
still joins. Each subject additionally receives one deterministic day offset
from :func:`openmed.core.date_shift.stable_offset_for`, applied to every date
in every table for that subject, so absolute dates change while intra-subject
intervals (for example a length of stay) are preserved exactly.

Transformation is deterministic and offline: the same tables, vault secret and
date-shift secret always yield the same surrogates and offsets. The emitted
manifest references vault entries by privacy-safe HMAC proof only; the
raw-to-surrogate mapping is never returned, logged, or written.

When quasi-identifiers are declared, the module builds one subject-level joined
view, detects privacy classes that become unsafe only after linkage, and runs a
single coordinated generalization/suppression search. The selected values are
then applied back to their source tables. Subject suppression is all-or-nothing
across the linked dataset, so privacy enforcement cannot create dangling keys.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from openmed.core.date_shift import DEFAULT_DATE_SHIFT_MAX_DAYS, stable_offset_for
from openmed.core.surrogate_vault import SurrogateSource, SurrogateVault
from openmed.risk.kanon import enforce_kanon, kanon_report

from .generalize import DEFAULT_TARGET_K
from .hierarchies import (
    SUPPORTED_COLUMN_TYPES,
    HierarchyError,
    build_enforcement_hierarchies,
)

RELATIONAL_ADVISORY = (
    "Linked-table de-identification pseudonymizes shared join keys consistently "
    "through the surrogate vault and shifts every date for a subject by one "
    "deterministic offset. Declared quasi-identifiers are enforced against a "
    "coordinated subject-level joined view, preserving foreign-key joins and "
    "intra-subject intervals. It is de-identification support tooling, not an "
    "automatic clinical decision system."
)

Table = Sequence[Mapping[str, Any]]
Tables = Mapping[str, Table]

_LANG = "en"
_SUBJECT_FIELD = "__openmed_relational_subject__"


class RelationalSchemaError(ValueError):
    """The relational schema is inconsistent with the supplied tables."""


class DanglingForeignKeyError(ValueError):
    """A foreign key no longer joins after transformation.

    Raised instead of emitting a dataset in which any non-null child key has no
    matching parent key. The message reports the relationship and the count of
    offending rows only; no key value (raw or surrogate) is included.
    """


class RelationalPrivacyError(ValueError):
    """A declared joined-view privacy policy cannot be satisfied safely."""


class CrossTableLinkageWarning(UserWarning):
    """A joined view exposed subjects that individual tables did not."""


@dataclass(frozen=True)
class ColumnRef:
    """A ``(table, column)`` coordinate within the linked dataset."""

    table: str
    column: str

    def __post_init__(self) -> None:
        if not self.table:
            raise RelationalSchemaError("table must be non-empty")
        if not self.column:
            raise RelationalSchemaError("column must be non-empty")


@dataclass(frozen=True)
class KeySpace:
    """A logical join key shared by one or more table columns.

    Every column enrolled in a key space is pseudonymized under the single vault
    label ``name``, so identical source values map to identical surrogates
    across tables and any foreign key spanning those columns still joins.
    """

    name: str
    columns: tuple[ColumnRef, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise RelationalSchemaError("key space name must be non-empty")
        if not self.columns:
            raise RelationalSchemaError(
                f"key space {self.name!r} must enrol at least one column"
            )
        if len(set(self.columns)) != len(self.columns):
            raise RelationalSchemaError(
                f"key space {self.name!r} lists a column more than once"
            )


@dataclass(frozen=True)
class ForeignKey:
    """A foreign key: ``child.column`` must reference ``parent.column``.

    Both endpoints must belong to the same key space so their shared source
    values receive the same surrogate; the transform verifies that every
    non-null child value still resolves to a parent key after replacement.
    """

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str

    def __post_init__(self) -> None:
        for label, value in (
            ("child_table", self.child_table),
            ("child_column", self.child_column),
            ("parent_table", self.parent_table),
            ("parent_column", self.parent_column),
        ):
            if not value:
                raise RelationalSchemaError(f"{label} must be non-empty")

    @property
    def child(self) -> ColumnRef:
        """Return the referencing ``(table, column)`` coordinate."""

        return ColumnRef(self.child_table, self.child_column)

    @property
    def parent(self) -> ColumnRef:
        """Return the referenced ``(table, column)`` coordinate."""

        return ColumnRef(self.parent_table, self.parent_column)


@dataclass(frozen=True)
class DateColumn:
    """A date column shifted by its row subject's deterministic offset.

    ``subject_column`` names the column in the same table that carries the
    subject key; its (raw) value selects the per-subject offset so every date
    for that subject moves together.
    """

    table: str
    column: str
    subject_column: str

    def __post_init__(self) -> None:
        if not self.table:
            raise RelationalSchemaError("date column table must be non-empty")
        if not self.column:
            raise RelationalSchemaError("date column must be non-empty")
        if not self.subject_column:
            raise RelationalSchemaError("date column subject_column must be non-empty")


@dataclass(frozen=True)
class QuasiIdentifier:
    """A subject-stable quasi-identifier in one linked table.

    Repeated rows for a subject may repeat the same value. A subject carrying
    multiple distinct values for this column is rejected because a scalar
    full-domain hierarchy cannot safely represent that longitudinal profile.

    Args:
        table: Source table name.
        column: Source column name.
        column_type: Declarative hierarchy family supported by
            :mod:`openmed.structured.hierarchies`.
    """

    table: str
    column: str
    column_type: str

    def __post_init__(self) -> None:
        if not self.table:
            raise RelationalSchemaError("quasi-identifier table must be non-empty")
        if not self.column:
            raise RelationalSchemaError("quasi-identifier column must be non-empty")
        if self.column_type not in SUPPORTED_COLUMN_TYPES:
            supported = ", ".join(sorted(SUPPORTED_COLUMN_TYPES))
            raise RelationalSchemaError(
                f"unknown quasi-identifier column type {self.column_type!r}; "
                f"supported: {supported}"
            )

    @property
    def ref(self) -> ColumnRef:
        """Return the source column coordinate."""

        return ColumnRef(self.table, self.column)

    @property
    def joined_name(self) -> str:
        """Return the schema-only field name used in the joined view."""

        return f"{self.table}.{self.column}"


@dataclass(frozen=True)
class RelationalSchema:
    """Join keys, foreign keys, dates, and QIs for a linked dataset."""

    key_spaces: tuple[KeySpace, ...]
    subject_key_space: str
    foreign_keys: tuple[ForeignKey, ...] = ()
    date_columns: tuple[DateColumn, ...] = ()
    quasi_identifiers: tuple[QuasiIdentifier, ...] = ()

    def __post_init__(self) -> None:
        if not self.key_spaces:
            raise RelationalSchemaError("schema must declare at least one key space")
        names = [space.name for space in self.key_spaces]
        if len(set(names)) != len(names):
            raise RelationalSchemaError("key space names must be unique")
        column_owner: dict[ColumnRef, str] = {}
        for space in self.key_spaces:
            for column in space.columns:
                if column in column_owner:
                    raise RelationalSchemaError(
                        f"column {column.table}.{column.column} is enrolled in "
                        f"key spaces {column_owner[column]!r} and {space.name!r}"
                    )
                column_owner[column] = space.name
        if self.subject_key_space not in set(names):
            raise RelationalSchemaError(
                f"subject_key_space {self.subject_key_space!r} is not a declared "
                "key space"
            )
        for fk in self.foreign_keys:
            child_space = column_owner.get(fk.child)
            parent_space = column_owner.get(fk.parent)
            if child_space is None:
                raise RelationalSchemaError(
                    f"foreign key child {fk.child_table}.{fk.child_column} is not "
                    "enrolled in any key space"
                )
            if parent_space is None:
                raise RelationalSchemaError(
                    f"foreign key parent {fk.parent_table}.{fk.parent_column} is "
                    "not enrolled in any key space"
                )
            if child_space != parent_space:
                raise RelationalSchemaError(
                    f"foreign key {fk.child_table}.{fk.child_column} -> "
                    f"{fk.parent_table}.{fk.parent_column} spans distinct key "
                    f"spaces {child_space!r} and {parent_space!r}; both endpoints "
                    "must share one key space to preserve the join"
                )
        date_targets = {(dc.table, dc.column) for dc in self.date_columns}
        for dc in self.date_columns:
            if ColumnRef(dc.table, dc.column) in column_owner:
                raise RelationalSchemaError(
                    f"date column {dc.table}.{dc.column} is also a key-space "
                    "column; a column cannot be both a join key and a date"
                )
        if len(date_targets) != len(self.date_columns):
            raise RelationalSchemaError("a date column is declared more than once")

        quasi_identifier_refs = [qi.ref for qi in self.quasi_identifiers]
        if len(set(quasi_identifier_refs)) != len(quasi_identifier_refs):
            raise RelationalSchemaError(
                "a relational quasi-identifier is declared more than once"
            )
        joined_names = [qi.joined_name for qi in self.quasi_identifiers]
        if len(set(joined_names)) != len(joined_names):
            raise RelationalSchemaError(
                "relational quasi-identifier table/column names are ambiguous"
            )
        for qi in self.quasi_identifiers:
            if qi.ref in column_owner:
                raise RelationalSchemaError(
                    f"quasi-identifier {qi.joined_name} is also a key-space column; "
                    "join keys must be pseudonymized rather than generalized"
                )

    def column_owner(self) -> dict[ColumnRef, str]:
        """Return the key-space name owning each enrolled column."""

        owner: dict[ColumnRef, str] = {}
        for space in self.key_spaces:
            for column in space.columns:
                owner[column] = space.name
        return owner


@dataclass(frozen=True)
class LinkageRiskManifest:
    """Raw-free evidence from the coordinated joined-view leakage gate."""

    target_k: int
    subject_count: int
    released_subject_count: int
    initial_joined_k: int
    achieved_joined_k: int
    cross_table_risk_detected: bool
    cross_table_risk_subject_count: int
    joined_singleton_subject_count: int
    suppressed_subject_count: int
    per_table_initial_k: dict[str, int]
    per_table_achieved_k: dict[str, int]
    generalization_levels: dict[str, int]
    generalization_level_names: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize aggregate joined-view evidence without source values."""

        return {
            "target_k": self.target_k,
            "subject_count": self.subject_count,
            "released_subject_count": self.released_subject_count,
            "initial_joined_k": self.initial_joined_k,
            "achieved_joined_k": self.achieved_joined_k,
            "cross_table_risk_detected": self.cross_table_risk_detected,
            "cross_table_risk_subject_count": self.cross_table_risk_subject_count,
            "joined_singleton_subject_count": self.joined_singleton_subject_count,
            "suppressed_subject_count": self.suppressed_subject_count,
            "per_table_initial_k": dict(self.per_table_initial_k),
            "per_table_achieved_k": dict(self.per_table_achieved_k),
            "generalization_levels": dict(self.generalization_levels),
            "generalization_level_names": dict(self.generalization_level_names),
        }


@dataclass(frozen=True)
class SurrogateManifest:
    """Privacy-safe record of a linked-table de-identification run.

    ``entries`` maps each source HMAC proof to its surrogate HMAC proof exactly
    as :meth:`SurrogateVault.consistency_snapshot` produces them, so vault
    entries are referenced by pseudonymous proof only. ``subject_offsets`` is
    keyed by the subject *surrogate* (already public in the emitted tables), so
    no raw identifier and no raw-to-surrogate mapping appears anywhere.
    ``linkage_risk`` contains schema names, levels, and aggregate counts only.
    """

    key_id: str
    key_spaces: tuple[str, ...]
    entries: dict[str, str]
    subject_offsets: dict[str, int]
    orphaned_foreign_keys: int = 0
    linkage_risk: LinkageRiskManifest | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest into JSON-compatible, raw-free fields."""

        payload: dict[str, Any] = {
            "key_id": self.key_id,
            "key_spaces": list(self.key_spaces),
            "entries": dict(self.entries),
            "subject_offsets": dict(self.subject_offsets),
            "orphaned_foreign_keys": self.orphaned_foreign_keys,
        }
        if self.linkage_risk is not None:
            payload["linkage_risk"] = self.linkage_risk.to_dict()
        return payload


@dataclass(frozen=True)
class RelationalDeidentificationResult:
    """The de-identified tables and the privacy-safe mapping manifest."""

    tables: dict[str, list[dict[str, Any]]]
    manifest: SurrogateManifest = field(repr=False)


def deidentify_linked_tables(
    tables: Tables,
    schema: RelationalSchema,
    *,
    vault: SurrogateVault,
    date_shift_secret: str | bytes,
    date_shift_max_days: int = DEFAULT_DATE_SHIFT_MAX_DAYS,
    target_k: int = DEFAULT_TARGET_K,
    suppression_limit: int | None = None,
    suppression_rate: float = 0.0,
    clinical_code_hierarchies: (
        Mapping[ColumnRef, Mapping[str, Sequence[str]]] | None
    ) = None,
) -> RelationalDeidentificationResult:
    """De-identify linked tables while preserving foreign-key joins.

    Shared join keys are replaced consistently through ``vault`` and every date
    is shifted by its subject's single deterministic offset. Declared relational
    quasi-identifiers are evaluated once per subject across their joined view;
    the selected full-domain generalization is copied back to every occurrence
    in its source table. If the policy selects suppression, all rows for that
    subject are removed from every supplied table.

    Args:
        tables: Local row mappings keyed by table name.
        schema: Explicit key spaces, foreign keys, dates, and optional QIs.
        vault: Caller-controlled local surrogate vault.
        date_shift_secret: Secret key material for deterministic date offsets.
        date_shift_max_days: Maximum absolute date offset.
        target_k: Minimum subject equivalence-class size in the joined view.
        suppression_limit: Maximum number of whole subjects that may be removed.
        suppression_rate: Fractional whole-subject suppression cap.
        clinical_code_hierarchies: Caller-supplied clinical parent chains keyed
            by source column. No terminology data is bundled.

    Returns:
        De-identified linked tables plus a raw-free manifest.

    Raises:
        DanglingForeignKeyError: If transformed foreign keys do not join.
        RelationalPrivacyError: If the joined-view policy cannot be enforced.
        RelationalSchemaError: If the supplied schema and tables disagree.
    """

    _validate_tables(tables, schema)

    owner = schema.column_owner()
    columns_by_table: dict[str, list[ColumnRef]] = {}
    for column in owner:
        columns_by_table.setdefault(column.table, []).append(column)

    subject_column_by_table = _subject_columns_by_table(schema)

    date_columns_by_table: dict[str, list[DateColumn]] = {}
    for dc in schema.date_columns:
        date_columns_by_table.setdefault(dc.table, []).append(dc)

    # Pseudonymize every enrolled key value first so the vault holds a stable
    # surrogate before any table is rewritten. ``sources`` retains the raw
    # descriptors only long enough to snapshot privacy-safe proofs.
    sources: list[SurrogateSource] = []
    seen_sources: set[tuple[str, str]] = set()
    surrogate_by_space_value: dict[tuple[str, str], str] = {}
    subject_offsets: dict[str, int] = {}

    for table_name, table in tables.items():
        for column in columns_by_table.get(table_name, ()):
            space_name = owner[column]
            for row in table:
                value = _key_text(row.get(column.column))
                if value is None:
                    continue
                cache_key = (space_name, value)
                if cache_key not in surrogate_by_space_value:
                    surrogate_by_space_value[cache_key] = _key_surrogate(
                        vault,
                        key_space=space_name,
                        value=value,
                        is_subject=space_name == schema.subject_key_space,
                    )
                    if cache_key not in seen_sources:
                        sources.append(
                            SurrogateSource(
                                source_text=_namespaced_source(space_name, value),
                                label=space_name,
                                lang=_LANG,
                            )
                        )
                        seen_sources.add(cache_key)
                if space_name == schema.subject_key_space:
                    surrogate = surrogate_by_space_value[cache_key]
                    if surrogate not in subject_offsets:
                        subject_offsets[surrogate] = stable_offset_for(
                            value,
                            max_days=date_shift_max_days,
                            secret=date_shift_secret,
                        )

    deidentified: dict[str, list[dict[str, Any]]] = {}
    for table_name, table in tables.items():
        rewritten: list[dict[str, Any]] = []
        for row in table:
            new_row = dict(row)
            for column in columns_by_table.get(table_name, ()):
                value = _key_text(row.get(column.column))
                if value is None:
                    new_row[column.column] = None
                else:
                    new_row[column.column] = surrogate_by_space_value[
                        (owner[column], value)
                    ]
            for dc in date_columns_by_table.get(table_name, ()):
                offset = _row_offset(
                    row,
                    date_column=dc,
                    subject_column=subject_column_by_table.get(table_name),
                    surrogate_by_space_value=surrogate_by_space_value,
                    subject_offsets=subject_offsets,
                    subject_space=schema.subject_key_space,
                )
                new_row[dc.column] = _shift_date(row.get(dc.column), offset, dc)
            rewritten.append(new_row)
        deidentified[table_name] = rewritten

    linkage_risk: LinkageRiskManifest | None = None
    if schema.quasi_identifiers:
        deidentified, linkage_risk = _enforce_joined_view_privacy(
            deidentified,
            schema,
            subject_column_by_table=subject_column_by_table,
            target_k=target_k,
            suppression_limit=suppression_limit,
            suppression_rate=suppression_rate,
            clinical_code_hierarchies=clinical_code_hierarchies,
        )

    orphans = _check_referential_integrity(deidentified, schema)

    if linkage_risk is not None and linkage_risk.cross_table_risk_detected:
        warnings.warn(
            CrossTableLinkageWarning(
                "cross-table linkage risk: joined quasi-identifiers created "
                f"{linkage_risk.cross_table_risk_subject_count} subject privacy "
                "class member(s) below the target that were not exposed by any "
                "individual table; coordinated generalization/suppression raised "
                f"the joined-view k from {linkage_risk.initial_joined_k} to "
                f"{linkage_risk.achieved_joined_k}"
            ),
            stacklevel=2,
        )

    manifest = SurrogateManifest(
        key_id=vault.current_key_id,
        key_spaces=tuple(space.name for space in schema.key_spaces),
        entries=vault.consistency_snapshot(sources),
        subject_offsets=subject_offsets,
        orphaned_foreign_keys=orphans,
        linkage_risk=linkage_risk,
    )
    return RelationalDeidentificationResult(tables=deidentified, manifest=manifest)


def _subject_columns_by_table(schema: RelationalSchema) -> dict[str, str]:
    columns: dict[str, str] = {}
    for space in schema.key_spaces:
        if space.name != schema.subject_key_space:
            continue
        for column in space.columns:
            existing = columns.get(column.table)
            if existing is not None and existing != column.column:
                raise RelationalSchemaError(
                    f"table {column.table!r} declares more than one subject-key column"
                )
            columns[column.table] = column.column
    return columns


def _enforce_joined_view_privacy(
    tables: dict[str, list[dict[str, Any]]],
    schema: RelationalSchema,
    *,
    subject_column_by_table: Mapping[str, str],
    target_k: int,
    suppression_limit: int | None,
    suppression_rate: float,
    clinical_code_hierarchies: (Mapping[ColumnRef, Mapping[str, Sequence[str]]] | None),
) -> tuple[dict[str, list[dict[str, Any]]], LinkageRiskManifest]:
    """Enforce k on one row per subject and project the result to each table."""

    _validate_joined_policy(
        target_k=target_k,
        suppression_limit=suppression_limit,
        suppression_rate=suppression_rate,
    )
    joined_records = _build_subject_joined_view(
        tables,
        schema,
        subject_column_by_table=subject_column_by_table,
    )
    quasi_identifier_fields = [qi.joined_name for qi in schema.quasi_identifiers]
    fields_by_table: dict[str, list[str]] = {}
    for qi in schema.quasi_identifiers:
        fields_by_table.setdefault(qi.table, []).append(qi.joined_name)

    try:
        initial_report = kanon_report(
            joined_records,
            quasi_identifiers=quasi_identifier_fields,
        )
        initial_sizes = _class_sizes_by_subject(initial_report, joined_records)
        table_initial_reports = {
            table_name: kanon_report(joined_records, quasi_identifiers=fields)
            for table_name, fields in fields_by_table.items()
        }
        table_initial_sizes = {
            table_name: _class_sizes_by_subject(report, joined_records)
            for table_name, report in table_initial_reports.items()
        }
    except (TypeError, ValueError):
        raise RelationalPrivacyError(
            "joined-view privacy analysis failed for the declared quasi-identifiers"
        ) from None

    cross_table_risk_subjects = {
        subject
        for subject, joined_size in initial_sizes.items()
        if joined_size < target_k
        and all(
            table_sizes.get(subject, 0) >= target_k
            for table_sizes in table_initial_sizes.values()
        )
    }
    singleton_subjects = {
        subject
        for subject in cross_table_risk_subjects
        if initial_sizes.get(subject) == 1
    }

    column_types = {qi.joined_name: qi.column_type for qi in schema.quasi_identifiers}
    code_hierarchies = _joined_clinical_code_hierarchies(
        schema,
        clinical_code_hierarchies,
    )
    try:
        hierarchies = build_enforcement_hierarchies(
            column_types,
            joined_records,
            clinical_code_hierarchies=code_hierarchies,
        )
        enforcement = enforce_kanon(
            joined_records,
            quasi_identifiers=quasi_identifier_fields,
            hierarchies=hierarchies,
            target_k=target_k,
            suppression_limit=suppression_limit,
            suppression_rate=suppression_rate,
            remove_direct_identifiers=False,
        )
    except (HierarchyError, TypeError, ValueError):
        raise RelationalPrivacyError(
            "joined-view privacy policy could not be enforced using the "
            "declared hierarchies and suppression bound"
        ) from None

    released_records = [dict(record) for record in enforcement["records"]]
    released_by_subject = {
        str(record[_SUBJECT_FIELD]): record for record in released_records
    }
    all_subjects = {str(record[_SUBJECT_FIELD]) for record in joined_records}
    suppressed_subjects = all_subjects - set(released_by_subject)
    transformed = _project_joined_policy_to_tables(
        tables,
        schema,
        subject_column_by_table=subject_column_by_table,
        released_by_subject=released_by_subject,
        suppressed_subjects=suppressed_subjects,
    )

    try:
        per_table_achieved_k = {
            table_name: int(
                kanon_report(released_records, quasi_identifiers=fields)["k"]
            )
            for table_name, fields in fields_by_table.items()
        }
    except (TypeError, ValueError):
        raise RelationalPrivacyError(
            "joined-view privacy verification failed after enforcement"
        ) from None

    level_report = enforcement.get("generalization", {}).get("levels", {})
    generalization_levels = {
        qi.joined_name: int(level_report[qi.joined_name]["level"])
        for qi in schema.quasi_identifiers
    }
    generalization_level_names = {
        qi.joined_name: str(level_report[qi.joined_name].get("name") or "")
        for qi in schema.quasi_identifiers
    }
    linkage_risk = LinkageRiskManifest(
        target_k=target_k,
        subject_count=len(joined_records),
        released_subject_count=len(released_records),
        initial_joined_k=int(initial_report["k"]),
        achieved_joined_k=int(enforcement["kanon"]["k"]),
        cross_table_risk_detected=bool(cross_table_risk_subjects),
        cross_table_risk_subject_count=len(cross_table_risk_subjects),
        joined_singleton_subject_count=len(singleton_subjects),
        suppressed_subject_count=len(suppressed_subjects),
        per_table_initial_k={
            table_name: int(report["k"])
            for table_name, report in table_initial_reports.items()
        },
        per_table_achieved_k=per_table_achieved_k,
        generalization_levels=generalization_levels,
        generalization_level_names=generalization_level_names,
    )
    if linkage_risk.achieved_joined_k < target_k:
        raise RelationalPrivacyError(
            "joined-view privacy verification did not reach the target k"
        )
    return transformed, linkage_risk


def _validate_joined_policy(
    *,
    target_k: int,
    suppression_limit: int | None,
    suppression_rate: float,
) -> None:
    if type(target_k) is not int or target_k < 1:
        raise RelationalPrivacyError("target_k must be an integer >= 1")
    if suppression_limit is not None and (
        type(suppression_limit) is not int or suppression_limit < 0
    ):
        raise RelationalPrivacyError("suppression_limit must be an integer >= 0")
    if (
        isinstance(suppression_rate, bool)
        or not isinstance(suppression_rate, (int, float))
        or not math.isfinite(float(suppression_rate))
        or not 0.0 <= float(suppression_rate) <= 1.0
    ):
        raise RelationalPrivacyError("suppression_rate must be between 0.0 and 1.0")


def _build_subject_joined_view(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    schema: RelationalSchema,
    *,
    subject_column_by_table: Mapping[str, str],
) -> list[dict[str, Any]]:
    subjects: set[str] = set()
    for table_name, subject_column in subject_column_by_table.items():
        for row in tables[table_name]:
            subject = _key_text(row.get(subject_column))
            if subject is not None:
                subjects.add(subject)
    if not subjects:
        raise RelationalPrivacyError(
            "joined-view privacy enforcement requires at least one subject"
        )

    joined_by_subject = {
        subject: {_SUBJECT_FIELD: subject} for subject in sorted(subjects)
    }
    for qi in schema.quasi_identifiers:
        subject_column = subject_column_by_table.get(qi.table)
        if subject_column is None:
            raise RelationalSchemaError(
                f"quasi-identifier table {qi.table!r} must expose a column in "
                f"subject key space {schema.subject_key_space!r}"
            )
        value_by_subject: dict[str, Any] = {}
        unstable_subjects: set[str] = set()
        for row in tables[qi.table]:
            subject = _key_text(row.get(subject_column))
            if subject is None:
                raise RelationalSchemaError(
                    f"table {qi.table!r} has a quasi-identifier row without a "
                    "subject key"
                )
            value = row.get(qi.column)
            if value is None or (isinstance(value, str) and not value):
                raise RelationalSchemaError(
                    f"quasi-identifier {qi.joined_name} must be non-null for "
                    "every subject"
                )
            if subject in value_by_subject and not _same_qi_value(
                value_by_subject[subject], value
            ):
                unstable_subjects.add(subject)
            else:
                value_by_subject[subject] = value
        if unstable_subjects:
            raise RelationalSchemaError(
                f"quasi-identifier {qi.joined_name} has multiple distinct values "
                f"for {len(unstable_subjects)} subject(s); relational QIs must be "
                "subject-stable"
            )
        missing_count = len(subjects - set(value_by_subject))
        if missing_count:
            raise RelationalSchemaError(
                f"quasi-identifier {qi.joined_name} is missing for "
                f"{missing_count} subject(s) in the joined view"
            )
        for subject, value in value_by_subject.items():
            joined_by_subject[subject][qi.joined_name] = value
    return list(joined_by_subject.values())


def _same_qi_value(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    result = left == right
    return result if isinstance(result, bool) else False


def _class_sizes_by_subject(
    report: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for equivalence_class in report.get("equivalence_classes", ()):
        size = int(equivalence_class["size"])
        for member in equivalence_class.get("members", ()):
            position = int(member)
            sizes[str(records[position][_SUBJECT_FIELD])] = size
    return sizes


def _joined_clinical_code_hierarchies(
    schema: RelationalSchema,
    supplied: Mapping[ColumnRef, Mapping[str, Sequence[str]]] | None,
) -> dict[str, Mapping[str, Sequence[str]]]:
    if supplied is None:
        return {}
    declared = {qi.ref: qi for qi in schema.quasi_identifiers}
    unknown = set(supplied) - set(declared)
    if unknown:
        raise RelationalSchemaError(
            "clinical code hierarchies target undeclared relational columns"
        )
    return {declared[ref].joined_name: hierarchy for ref, hierarchy in supplied.items()}


def _project_joined_policy_to_tables(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    schema: RelationalSchema,
    *,
    subject_column_by_table: Mapping[str, str],
    released_by_subject: Mapping[str, Mapping[str, Any]],
    suppressed_subjects: set[str],
) -> dict[str, list[dict[str, Any]]]:
    quasi_identifiers_by_table: dict[str, list[QuasiIdentifier]] = {}
    for qi in schema.quasi_identifiers:
        quasi_identifiers_by_table.setdefault(qi.table, []).append(qi)

    transformed: dict[str, list[dict[str, Any]]] = {}
    for table_name, table in tables.items():
        subject_column = subject_column_by_table.get(table_name)
        if suppressed_subjects and subject_column is None and table:
            raise RelationalPrivacyError(
                "whole-subject suppression requires every non-empty linked table "
                "to expose its subject key"
            )
        rows: list[dict[str, Any]] = []
        for row in table:
            subject = (
                _key_text(row.get(subject_column))
                if subject_column is not None
                else None
            )
            if subject in suppressed_subjects:
                continue
            if subject_column is not None and subject is None:
                raise RelationalSchemaError(
                    f"table {table_name!r} has a row without a subject key"
                )
            rewritten = dict(row)
            for qi in quasi_identifiers_by_table.get(table_name, ()):
                if subject is None or subject not in released_by_subject:
                    raise RelationalPrivacyError(
                        "joined-view projection could not resolve a released subject"
                    )
                rewritten[qi.column] = released_by_subject[subject][qi.joined_name]
            rows.append(rewritten)
        transformed[table_name] = rows
    return transformed


def _validate_tables(tables: Tables, schema: RelationalSchema) -> None:
    required: dict[str, set[str]] = {}
    for space in schema.key_spaces:
        for column in space.columns:
            required.setdefault(column.table, set()).add(column.column)
    for fk in schema.foreign_keys:
        required.setdefault(fk.child_table, set()).add(fk.child_column)
        required.setdefault(fk.parent_table, set()).add(fk.parent_column)
    for dc in schema.date_columns:
        required.setdefault(dc.table, set()).update({dc.column, dc.subject_column})
    for qi in schema.quasi_identifiers:
        required.setdefault(qi.table, set()).add(qi.column)

    for table_name, columns in required.items():
        if table_name not in tables:
            raise RelationalSchemaError(
                f"schema references table {table_name!r} which was not supplied"
            )
        table = tables[table_name]
        for row in table:
            missing = columns - set(row)
            if missing:
                raise RelationalSchemaError(
                    f"table {table_name!r} is missing column(s) "
                    f"{sorted(missing)!r} required by the schema"
                )


def _row_offset(
    row: Mapping[str, Any],
    *,
    date_column: DateColumn,
    subject_column: str | None,
    surrogate_by_space_value: Mapping[tuple[str, str], str],
    subject_offsets: Mapping[str, int],
    subject_space: str,
) -> int:
    subject_value = _key_text(row.get(date_column.subject_column))
    if subject_value is None:
        raise RelationalSchemaError(
            f"row in table {date_column.table!r} has a date in "
            f"{date_column.column!r} but no subject key in "
            f"{date_column.subject_column!r}"
        )
    surrogate = surrogate_by_space_value.get((subject_space, subject_value))
    if surrogate is None:
        raise RelationalSchemaError(
            f"subject column {date_column.table}.{date_column.subject_column} is "
            f"not enrolled in the subject key space {subject_space!r}"
        )
    return subject_offsets[surrogate]


def _check_referential_integrity(
    tables: Mapping[str, list[dict[str, Any]]],
    schema: RelationalSchema,
) -> int:
    orphaned = 0
    for fk in schema.foreign_keys:
        parent_keys = {
            _key_text(row.get(fk.parent_column)) for row in tables[fk.parent_table]
        }
        parent_keys.discard(None)
        dangling = 0
        for row in tables[fk.child_table]:
            value = _key_text(row.get(fk.child_column))
            if value is not None and value not in parent_keys:
                dangling += 1
        if dangling:
            orphaned += dangling
            warnings.warn(
                f"foreign key {fk.child_table}.{fk.child_column} -> "
                f"{fk.parent_table}.{fk.parent_column} left {dangling} dangling "
                "row(s) after de-identification",
                stacklevel=2,
            )
    if orphaned:
        raise DanglingForeignKeyError(
            f"refusing to emit dataset: {orphaned} foreign-key row(s) do not "
            "join after de-identification"
        )
    return orphaned


_NAMESPACE_SEPARATOR = "\x00"


def _namespaced_source(key_space: str, value: str) -> str:
    """Return the vault source key binding ``value`` to its key space.

    ``core.labels.normalize_label`` collapses every custom key-space name to
    ``OTHER``, so the vault ``label`` alone cannot separate two key spaces that
    share a raw value. Namespacing the value moves that distinction into the
    HMAC'd source text, keeping surrogates per-key-space injective and
    independent of the order tables are processed in.
    """

    return f"{key_space}{_NAMESPACE_SEPARATOR}{value}"


def _key_surrogate(
    vault: SurrogateVault,
    *,
    key_space: str,
    value: str,
    is_subject: bool = False,
) -> str:
    source_text = _namespaced_source(key_space, value)
    if is_subject:
        return vault.resolve_subject(
            value,
            aliases=(SurrogateSource(source_text, key_space, _LANG),),
        )

    token = vault.text_hash(source_text).rsplit(":", 1)[-1][:12]
    stem = _surrogate_stem(key_space)

    def create(attempt: int) -> str:
        if attempt == 0:
            return f"{stem}_{token}"
        return f"{stem}_{token}_{attempt}"

    return vault.get_or_create(
        source_text,
        label=key_space,
        lang=_LANG,
        create_surrogate=create,
    )


def _surrogate_stem(key_space: str) -> str:
    cleaned = "".join(
        character if character.isalnum() else "_" for character in key_space
    )
    stem = cleaned.strip("_").lower()
    return stem or "key"


def _key_text(value: Any) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text or None


def _shift_date(value: Any, offset_days: int, date_column: DateColumn) -> Any:
    if value is None or value == "":
        return value
    try:
        if isinstance(value, datetime):
            return value + timedelta(days=offset_days)
        if isinstance(value, date):
            return value + timedelta(days=offset_days)
        if isinstance(value, str):
            parsed, is_datetime = _parse_iso(value, date_column)
            shifted = parsed + timedelta(days=offset_days)
            return shifted.isoformat() if is_datetime else shifted.date().isoformat()
    except OverflowError:
        raise RelationalSchemaError(
            f"date column {date_column.table}.{date_column.column} shifted by "
            f"{offset_days} day(s) falls outside the representable date range"
        ) from None
    raise RelationalSchemaError(
        f"date column {date_column.table}.{date_column.column} holds a value that "
        "is not a date, datetime, or ISO-8601 string"
    )


def _parse_iso(value: str, date_column: DateColumn) -> tuple[datetime, bool]:
    # Try a date-only parse first: ``datetime.fromisoformat`` also accepts bare
    # dates (Python 3.11+), which would wrongly promote a date column to a
    # datetime. ``date.fromisoformat`` rejects any string carrying a time.
    try:
        return datetime.combine(date.fromisoformat(value), datetime.min.time()), False
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(value), True
    except ValueError:
        raise RelationalSchemaError(
            f"date column {date_column.table}.{date_column.column} holds "
            f"{value!r}, which is not an ISO-8601 date or datetime"
        ) from None


__all__ = [
    "RELATIONAL_ADVISORY",
    "ColumnRef",
    "CrossTableLinkageWarning",
    "DanglingForeignKeyError",
    "DateColumn",
    "ForeignKey",
    "KeySpace",
    "LinkageRiskManifest",
    "QuasiIdentifier",
    "RelationalDeidentificationResult",
    "RelationalPrivacyError",
    "RelationalSchema",
    "RelationalSchemaError",
    "SurrogateManifest",
    "deidentify_linked_tables",
]
