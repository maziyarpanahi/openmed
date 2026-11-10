"""Caller-owned PostgreSQL redaction for selected text columns.

The adapter deliberately targets the Python DB-API contract instead of
importing a PostgreSQL driver.  Callers provide an open connection, choose the
columns to transform, and retain control of the surrounding transaction.  SQL
values are always bound parameters; table and column names are quoted as SQL
identifiers because DB-API parameters cannot represent identifiers.

No connection is opened, closed, or committed by this module.  A failure
rolls back the caller-owned connection and raises :class:`PostgresRedactionError`
with a value-free message.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

Deidentifier = Callable[..., Any]

_DEFAULT_BATCH_SIZE = 500
_DEFAULT_POLICY = "hipaa_safe_harbor"
_DEFAULT_KEY_COLUMN = "id"


class PostgresRedactionError(RuntimeError):
    """Raised when a PostgreSQL redaction run fails and is rolled back."""


@dataclass(frozen=True)
class PostgresRedactionConfig:
    """Configuration for one deterministic PostgreSQL redaction run.

    ``key_column`` must identify each row uniquely and remain unchanged by the
    selected-column redaction.  ``extra_kwargs`` are forwarded to the
    deidentifier after the named options have been validated.
    """

    table: str
    text_columns: tuple[str, ...]
    key_column: str = _DEFAULT_KEY_COLUMN
    schema: str | None = None
    batch_size: int = _DEFAULT_BATCH_SIZE
    method: str = "mask"
    policy: str = _DEFAULT_POLICY
    model_name: str | None = None
    confidence_threshold: float = 0.7
    seed: int = 0
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize and validate configuration before any SQL is executed."""

        object.__setattr__(self, "table", _validate_identifier_path(self.table))
        object.__setattr__(
            self,
            "text_columns",
            _normalize_columns(self.text_columns, field_name="text_columns"),
        )
        object.__setattr__(
            self,
            "key_column",
            _validate_identifier(self.key_column, field_name="key_column"),
        )
        if self.key_column in self.text_columns:
            raise ValueError("key_column must not be a text column")
        if self.schema is not None:
            object.__setattr__(
                self,
                "schema",
                _validate_identifier(self.schema, field_name="schema"),
            )
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be an integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(self.extra_kwargs, Mapping):
            raise TypeError("extra_kwargs must be a mapping")
        object.__setattr__(self, "extra_kwargs", dict(self.extra_kwargs))

    def to_deidentify_kwargs(self) -> dict[str, Any]:
        """Return the fixed, reproducible options for the deidentifier."""

        kwargs: dict[str, Any] = {
            "method": self.method,
            "policy": self.policy,
            "confidence_threshold": self.confidence_threshold,
            "seed": self.seed,
        }
        if self.model_name is not None:
            kwargs["model_name"] = self.model_name

        collisions = sorted(kwargs.keys() & self.extra_kwargs.keys())
        if collisions:
            fields = ", ".join(collisions)
            raise ValueError(f"extra_kwargs cannot override named fields: {fields}")
        kwargs.update(self.extra_kwargs)
        return kwargs


@dataclass(frozen=True)
class PostgresRedactionResult:
    """PHI-free counts returned after a successful redaction run."""

    rows_processed: int = 0
    rows_updated: int = 0
    cells_processed: int = 0
    cells_redacted: int = 0
    spans_redacted: int = 0
    batches_processed: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable count-only result."""

        return {
            "rows_processed": self.rows_processed,
            "rows_updated": self.rows_updated,
            "cells_processed": self.cells_processed,
            "cells_redacted": self.cells_redacted,
            "spans_redacted": self.spans_redacted,
            "batches_processed": self.batches_processed,
        }

    def __getitem__(self, key: str) -> int:
        """Allow result counts to be accessed like a mapping."""

        aliases = {
            "total_rows": "rows_processed",
            "updated_rows": "rows_updated",
            "processed_cells": "cells_processed",
            "redacted_cells": "cells_redacted",
        }
        return self.to_dict()[aliases.get(key, key)]


class PostgresRedactionAdapter:
    """Redact selected columns using a caller-owned DB-API connection.

    The adapter reads rows in deterministic key order, sends one parameterized
    ``executemany`` update per changed batch, and never commits.  The caller
    can commit the successful run or include it in a larger transaction.

    The constructor accepts either a :class:`PostgresRedactionConfig` or the
    equivalent ``table`` and ``text_columns`` arguments for convenience.
    """

    def __init__(
        self,
        connection: Any,
        table: str | PostgresRedactionConfig | None = None,
        text_columns: Sequence[str] | None = None,
        *,
        config: PostgresRedactionConfig | None = None,
        schema: str | None = None,
        key_column: str = _DEFAULT_KEY_COLUMN,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        method: str = "mask",
        policy: str = _DEFAULT_POLICY,
        model_name: str | None = None,
        confidence_threshold: float = 0.7,
        seed: int = 0,
        extra_kwargs: Mapping[str, Any] | None = None,
        deidentifier: Deidentifier | None = None,
    ) -> None:
        if isinstance(table, PostgresRedactionConfig):
            if config is not None:
                raise TypeError("provide config only once")
            config = table
            table = None

        if config is not None:
            if table is not None or text_columns is not None:
                raise TypeError("config cannot be combined with table or text_columns")
            self.config = config
        else:
            if table is None or text_columns is None:
                raise TypeError("table and text_columns are required")
            self.config = PostgresRedactionConfig(
                table=table,
                text_columns=tuple(text_columns),
                schema=schema,
                key_column=key_column,
                batch_size=batch_size,
                method=method,
                policy=policy,
                model_name=model_name,
                confidence_threshold=confidence_threshold,
                seed=seed,
                extra_kwargs=extra_kwargs or {},
            )

        if not callable(getattr(connection, "cursor", None)):
            raise TypeError("connection must provide a DB-API cursor() method")
        if not callable(getattr(connection, "rollback", None)):
            raise TypeError("connection must provide a DB-API rollback() method")

        self.connection = connection
        self._deidentifier = deidentifier

    def redact(self) -> PostgresRedactionResult:
        """Redact the configured table and return aggregate counts only."""

        deidentify_kwargs = self.config.to_deidentify_kwargs()
        if self._deidentifier is None:
            deidentifier = _default_deidentifier()
            deidentify_kwargs["loader"] = _cached_model_loader()
        else:
            deidentifier = self._deidentifier
        select_sql = _select_sql(self.config)
        update_sql = _update_sql(self.config)
        cursor: Any | None = None

        rows_processed = 0
        rows_updated = 0
        cells_processed = 0
        cells_redacted = 0
        spans_redacted = 0
        batches_processed = 0
        offset = 0

        try:
            cursor = self.connection.cursor()
            while True:
                cursor.execute(
                    select_sql,
                    (self.config.batch_size, offset),
                )
                rows = cursor.fetchall()
                if not rows:
                    break

                batches_processed += 1
                rows_processed += len(rows)
                update_parameters: list[tuple[Any, ...]] = []
                description = getattr(cursor, "description", None)
                column_positions = _description_positions(
                    description,
                    (self.config.key_column, *self.config.text_columns),
                )

                for row in rows:
                    key_value = _row_value(
                        row,
                        self.config.key_column,
                        column_positions[self.config.key_column],
                    )
                    redacted_values: list[Any] = []
                    row_changed = False
                    row_spans = 0

                    for column in self.config.text_columns:
                        value = _row_value(row, column, column_positions[column])
                        if value is None:
                            redacted_values.append(None)
                            continue
                        if not isinstance(value, str):
                            raise TypeError(
                                "selected PostgreSQL columns must contain text or null"
                            )

                        cells_processed += 1
                        redacted, span_count = _redact_value(
                            value,
                            deidentifier,
                            deidentify_kwargs,
                        )
                        redacted_values.append(redacted)
                        if redacted != value:
                            row_changed = True
                            cells_redacted += 1
                            row_spans += span_count or 1

                    if row_changed:
                        rows_updated += 1
                        spans_redacted += row_spans
                        update_parameters.append(
                            (*redacted_values, key_value),
                        )

                if update_parameters:
                    cursor.executemany(update_sql, update_parameters)

                offset += len(rows)
                if len(rows) < self.config.batch_size:
                    break

            return PostgresRedactionResult(
                rows_processed=rows_processed,
                rows_updated=rows_updated,
                cells_processed=cells_processed,
                cells_redacted=cells_redacted,
                spans_redacted=spans_redacted,
                batches_processed=batches_processed,
            )
        except Exception:
            rollback_confirmed = _rollback_safely(self.connection)
            message = (
                "PostgreSQL redaction failed; transaction rolled back"
                if rollback_confirmed
                else (
                    "PostgreSQL redaction failed; transaction rollback "
                    "could not be confirmed"
                )
            )
            raise PostgresRedactionError(message) from None
        finally:
            _close_safely(cursor)

    run = redact
    redact_table = redact


def redact_postgres_table(
    connection: Any,
    table: str | None = None,
    text_columns: Sequence[str] | None = None,
    *,
    config: PostgresRedactionConfig | None = None,
    columns: Sequence[str] | None = None,
    schema: str | None = None,
    key_column: str = _DEFAULT_KEY_COLUMN,
    primary_key: str | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    method: str = "mask",
    policy: str = _DEFAULT_POLICY,
    model_name: str | None = None,
    confidence_threshold: float = 0.7,
    seed: int = 0,
    extra_kwargs: Mapping[str, Any] | None = None,
    deidentifier: Deidentifier | None = None,
) -> PostgresRedactionResult:
    """Redact selected PostgreSQL text columns in bounded batches.

    Args:
        connection: Existing DB-API connection. It remains open and is never
            committed by this function.
        table: Table name, optionally qualified as ``schema.table``.
        text_columns: Explicit text columns to redact. ``columns`` is a
            backwards-friendly alias.
        config: Optional complete :class:`PostgresRedactionConfig`.
        schema: Optional schema when ``table`` is unqualified.
        key_column: Unique, unchanged column used for deterministic ordering.
        primary_key: Alias for ``key_column``.
        batch_size: Maximum number of rows selected per database round trip.
        deidentifier: Optional local callable, primarily useful for injected
            local models and offline tests.

    Returns:
        A count-only :class:`PostgresRedactionResult`.

    Raises:
        PostgresRedactionError: If database or redaction processing fails. The
            supplied connection is rolled back before this exception is raised.
        ValueError: If the table, columns, or batch configuration is invalid.
    """

    if columns is not None:
        if text_columns is not None:
            raise TypeError("provide text_columns or columns, not both")
        text_columns = columns
    if primary_key is not None:
        if key_column != _DEFAULT_KEY_COLUMN:
            raise TypeError("provide key_column or primary_key, not both")
        key_column = primary_key

    return PostgresRedactionAdapter(
        connection,
        table,
        text_columns,
        config=config,
        schema=schema,
        key_column=key_column,
        batch_size=batch_size,
        method=method,
        policy=policy,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        seed=seed,
        extra_kwargs=extra_kwargs,
        deidentifier=deidentifier,
    ).redact()


redact_postgres = redact_postgres_table
redact_postgresql_table = redact_postgres_table


def _select_sql(config: PostgresRedactionConfig) -> str:
    selected = ", ".join(
        _quote_identifier(column)
        for column in (config.key_column, *config.text_columns)
    )
    return (
        f"SELECT {selected} FROM {_quote_table(config)} "
        f"ORDER BY {_quote_identifier(config.key_column)} LIMIT %s OFFSET %s"
    )


def _update_sql(config: PostgresRedactionConfig) -> str:
    assignments = ", ".join(
        f"{_quote_identifier(column)} = %s" for column in config.text_columns
    )
    return (
        f"UPDATE {_quote_table(config)} SET {assignments} "
        f"WHERE {_quote_identifier(config.key_column)} = %s"
    )


def _quote_table(config: PostgresRedactionConfig) -> str:
    if config.schema is not None:
        if "." in config.table:
            raise ValueError("schema must not be repeated in table")
        return f"{_quote_identifier(config.schema)}.{_quote_identifier(config.table)}"

    parts = config.table.split(".")
    if len(parts) == 1:
        return _quote_identifier(parts[0])
    if len(parts) == 2:
        return ".".join(_quote_identifier(part) for part in parts)
    raise ValueError("table must be a table name or schema.table")


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _validate_identifier(identifier: str, *, field_name: str) -> str:
    if not isinstance(identifier, str):
        raise TypeError(f"{field_name} must be a string")
    value = identifier.strip()
    if not value or "\x00" in value:
        raise ValueError(f"{field_name} must be a non-empty SQL identifier")
    return value


def _validate_identifier_path(identifier: str) -> str:
    value = _validate_identifier(identifier, field_name="table")
    parts = value.split(".")
    if len(parts) > 2 or any(not part for part in parts):
        raise ValueError("table must be a table name or schema.table")
    for part in parts:
        _validate_identifier(part, field_name="table")
    return value


def _normalize_columns(
    columns: Sequence[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(columns, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of column names")
    normalized: list[str] = []
    seen: set[str] = set()
    try:
        values = tuple(columns)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence of column names") from exc
    for column in values:
        value = _validate_identifier(column, field_name=field_name)
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one column")
    return tuple(normalized)


def _description_positions(
    description: Any,
    columns: Sequence[str],
) -> dict[str, int]:
    positions = {column: index for index, column in enumerate(columns)}
    if not description:
        return positions

    described: dict[str, int] = {}
    for index, item in enumerate(description):
        name = (
            item[0] if isinstance(item, (tuple, list)) else getattr(item, "name", None)
        )
        if isinstance(name, str):
            described[name] = index
    for column in columns:
        if column in described:
            positions[column] = described[column]
    return positions


def _row_value(row: Any, column: str, position: int) -> Any:
    mapping = row if isinstance(row, Mapping) else getattr(row, "_mapping", None)
    if isinstance(mapping, Mapping):
        if column in mapping:
            return mapping[column]
        for name, value in mapping.items():
            if isinstance(name, str) and name.lower() == column.lower():
                return value
        raise TypeError("database row is missing a selected column")

    if hasattr(row, "_asdict"):
        values = row._asdict()
        if column in values:
            return values[column]

    try:
        return row[position]
    except (IndexError, KeyError, TypeError) as exc:
        raise TypeError("database row does not match the selected columns") from exc


def _redact_value(
    value: str,
    deidentifier: Deidentifier,
    kwargs: Mapping[str, Any],
) -> tuple[str, int]:
    result = deidentifier(value, **dict(kwargs))
    spans = _result_span_count(result)
    if isinstance(result, str):
        return result, spans
    if isinstance(result, Mapping):
        redacted = result.get("deidentified_text")
    else:
        redacted = getattr(result, "deidentified_text", None)
    if not isinstance(redacted, str):
        raise TypeError("deidentifier must return deidentified_text as text")
    return redacted, spans


def _result_span_count(result: Any) -> int:
    if isinstance(result, Mapping):
        entities = result.get("pii_entities")
    else:
        entities = getattr(result, "pii_entities", None)
    if entities is None:
        return 0
    try:
        return max(0, len(entities))
    except TypeError:
        return 0


def _rollback_safely(connection: Any) -> bool:
    try:
        connection.rollback()
    except Exception:
        return False
    return True


def _close_safely(cursor: Any | None) -> None:
    if cursor is None:
        return
    try:
        cursor.close()
    except Exception:
        pass


def _default_deidentifier() -> Deidentifier:
    from openmed.core.pii import deidentify

    return deidentify


@lru_cache(maxsize=1)
def _cached_model_loader() -> Any:
    from openmed.core import ModelLoader

    return ModelLoader()


__all__ = [
    "Deidentifier",
    "PostgresRedactionAdapter",
    "PostgresRedactionConfig",
    "PostgresRedactionError",
    "PostgresRedactionResult",
    "redact_postgres",
    "redact_postgres_table",
    "redact_postgresql_table",
]
