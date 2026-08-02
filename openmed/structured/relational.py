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

This layer is deliberately scoped to relational / foreign-key integrity across
linked tables. It consumes the existing cross-document surrogate vault rather
than reimplementing one, and does not perform quasi-identifier generalization
or suppression.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from openmed.core.date_shift import DEFAULT_DATE_SHIFT_MAX_DAYS, stable_offset_for
from openmed.core.surrogate_vault import SurrogateSource, SurrogateVault

RELATIONAL_ADVISORY = (
    "Linked-table de-identification pseudonymizes shared join keys consistently "
    "through the surrogate vault and shifts every date for a subject by one "
    "deterministic offset, preserving foreign-key joins and intra-subject "
    "intervals. It is de-identification support tooling, not a re-identification "
    "risk assessment."
)

Table = Sequence[Mapping[str, Any]]
Tables = Mapping[str, Table]

_LANG = "en"


class RelationalSchemaError(ValueError):
    """The relational schema is inconsistent with the supplied tables."""


class DanglingForeignKeyError(ValueError):
    """A foreign key no longer joins after transformation.

    Raised instead of emitting a dataset in which any non-null child key has no
    matching parent key. The message reports the relationship and the count of
    offending rows only; no key value (raw or surrogate) is included.
    """


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
class RelationalSchema:
    """The join keys, foreign-key graph and date columns of a linked dataset."""

    key_spaces: tuple[KeySpace, ...]
    subject_key_space: str
    foreign_keys: tuple[ForeignKey, ...] = ()
    date_columns: tuple[DateColumn, ...] = ()

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

    def column_owner(self) -> dict[ColumnRef, str]:
        """Return the key-space name owning each enrolled column."""

        owner: dict[ColumnRef, str] = {}
        for space in self.key_spaces:
            for column in space.columns:
                owner[column] = space.name
        return owner


@dataclass(frozen=True)
class SurrogateManifest:
    """Privacy-safe record of a linked-table de-identification run.

    ``entries`` maps each source HMAC proof to its surrogate HMAC proof exactly
    as :meth:`SurrogateVault.consistency_snapshot` produces them, so vault
    entries are referenced by pseudonymous proof only. ``subject_offsets`` is
    keyed by the subject *surrogate* (already public in the emitted tables), so
    no raw identifier and no raw-to-surrogate mapping appears anywhere.
    """

    key_id: str
    key_spaces: tuple[str, ...]
    entries: dict[str, str]
    subject_offsets: dict[str, int]
    orphaned_foreign_keys: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest into JSON-compatible, raw-free fields."""

        return {
            "key_id": self.key_id,
            "key_spaces": list(self.key_spaces),
            "entries": dict(self.entries),
            "subject_offsets": dict(self.subject_offsets),
            "orphaned_foreign_keys": self.orphaned_foreign_keys,
        }


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
) -> RelationalDeidentificationResult:
    """De-identify linked tables while preserving foreign-key joins.

    Shared join keys are replaced consistently through ``vault`` and every date
    is shifted by its subject's single deterministic offset. The result is
    refused (``DanglingForeignKeyError``) if any foreign key fails to join after
    replacement. The returned manifest references vault entries by HMAC proof
    only; the raw-to-surrogate mapping is never emitted.
    """

    _validate_tables(tables, schema)

    owner = schema.column_owner()
    columns_by_table: dict[str, list[ColumnRef]] = {}
    for column in owner:
        columns_by_table.setdefault(column.table, []).append(column)

    subject_column_by_table: dict[str, str] = {}
    for space in schema.key_spaces:
        if space.name != schema.subject_key_space:
            continue
        for column in space.columns:
            subject_column_by_table[column.table] = column.column

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

    orphans = _check_referential_integrity(deidentified, schema)

    manifest = SurrogateManifest(
        key_id=vault.current_key_id,
        key_spaces=tuple(space.name for space in schema.key_spaces),
        entries=vault.consistency_snapshot(sources),
        subject_offsets=subject_offsets,
        orphaned_foreign_keys=orphans,
    )
    return RelationalDeidentificationResult(tables=deidentified, manifest=manifest)


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
    "DanglingForeignKeyError",
    "DateColumn",
    "ForeignKey",
    "KeySpace",
    "RelationalDeidentificationResult",
    "RelationalSchema",
    "RelationalSchemaError",
    "SurrogateManifest",
    "deidentify_linked_tables",
]
