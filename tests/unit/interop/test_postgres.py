"""Offline tests for the caller-owned PostgreSQL redaction adapter."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from openmed.interop.postgres import (
    PostgresRedactionError,
    PostgresRedactionResult,
    redact_postgres_table,
)


@dataclass
class _Description:
    name: str


class _FakeCursor:
    def __init__(self, connection: "_FakeConnection") -> None:
        self.connection = connection
        self.description: list[_Description] = []
        self._selected: list[tuple[Any, ...]] = []
        self.closed = False

    def execute(self, statement: str, parameters: tuple[Any, ...]) -> None:
        self.connection.execute_calls.append((statement, parameters))
        if not statement.startswith("SELECT"):
            raise AssertionError("the adapter should use executemany for updates")

        limit, offset = parameters
        columns = ("record_id", "note", "summary")
        self.description = [_Description(name) for name in columns]
        selected = self.connection.rows[offset : offset + limit]
        self._selected = [tuple(row[column] for column in columns) for row in selected]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._selected)

    def executemany(
        self,
        statement: str,
        parameters: list[tuple[Any, ...]],
    ) -> None:
        self.connection.executemany_calls.append((statement, parameters))
        for note, summary, record_id in parameters:
            for row in self.connection.rows:
                if row["record_id"] == record_id:
                    row["note"] = note
                    row["summary"] = summary

    def close(self) -> None:
        self.closed = True


class _FakeConnection:
    def __init__(self) -> None:
        self.rows = [
            {
                "record_id": 1,
                "note": "synthetic note alpha",
                "summary": "synthetic summary alpha",
            },
            {
                "record_id": 2,
                "note": "synthetic note beta",
                "summary": "synthetic summary beta",
            },
            {
                "record_id": 3,
                "note": "synthetic note gamma",
                "summary": None,
            },
        ]
        self.initial_rows = deepcopy(self.rows)
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.executemany_calls: list[tuple[str, list[tuple[Any, ...]]]] = []
        self.rollback_calls = 0
        self.rollback_error: Exception | None = None
        self.commit_calls = 0
        self.cursors: list[_FakeCursor] = []

    def cursor(self) -> _FakeCursor:
        cursor = _FakeCursor(self)
        self.cursors.append(cursor)
        return cursor

    def rollback(self) -> None:
        self.rollback_calls += 1
        if self.rollback_error is not None:
            raise self.rollback_error
        self.rows = deepcopy(self.initial_rows)

    def commit(self) -> None:
        self.commit_calls += 1


def _fake_deidentifier(text: str, **_: Any) -> SimpleNamespace:
    transformed = text.replace("alpha", "[TOKEN]").replace("gamma", "[TOKEN]")
    return SimpleNamespace(
        deidentified_text=transformed,
        pii_entities=[object()] if transformed != text else [],
    )


def test_redacts_selected_columns_in_parameterized_batches_without_owning_transaction():
    connection = _FakeConnection()

    result = redact_postgres_table(
        connection,
        table="clinical_notes",
        text_columns=["note", "summary"],
        key_column="record_id",
        batch_size=2,
        deidentifier=_fake_deidentifier,
    )

    assert result == PostgresRedactionResult(
        rows_processed=3,
        rows_updated=2,
        cells_processed=5,
        cells_redacted=3,
        spans_redacted=3,
        batches_processed=2,
    )
    assert connection.rows[0]["note"] == "synthetic note [TOKEN]"
    assert connection.rows[0]["summary"] == "synthetic summary [TOKEN]"
    assert connection.rows[1]["note"] == "synthetic note beta"
    assert connection.rows[2]["note"] == "synthetic note [TOKEN]"
    assert connection.rows[2]["summary"] is None
    assert connection.rollback_calls == 0
    assert connection.commit_calls == 0
    assert connection.cursors[0].closed is True

    select_sql, select_parameters = connection.execute_calls[0]
    assert 'SELECT "record_id", "note", "summary"' in select_sql
    assert 'FROM "clinical_notes"' in select_sql
    assert "%s" in select_sql
    assert select_parameters == (2, 0)
    assert all(
        "synthetic" not in statement for statement, _ in connection.execute_calls
    )
    assert all(
        "synthetic" not in statement for statement, _ in connection.executemany_calls
    )


def test_result_is_count_only_and_supports_common_summary_aliases():
    result = PostgresRedactionResult(
        rows_processed=4,
        rows_updated=3,
        cells_processed=8,
        cells_redacted=5,
        spans_redacted=6,
        batches_processed=2,
    )

    assert result.to_dict() == {
        "rows_processed": 4,
        "rows_updated": 3,
        "cells_processed": 8,
        "cells_redacted": 5,
        "spans_redacted": 6,
        "batches_processed": 2,
    }
    assert result["total_rows"] == 4
    assert result["redacted_cells"] == 5


def test_failure_rolls_back_and_does_not_expose_cell_value():
    connection = _FakeConnection()
    sensitive_value = "synthetic secret delta"

    def failing_deidentifier(text: str, **_: Any) -> str:
        if text == sensitive_value:
            raise RuntimeError(f"driver detail: {text}")
        return text.replace("alpha", "[TOKEN]")

    connection.rows[1]["note"] = sensitive_value

    with pytest.raises(PostgresRedactionError) as exc_info:
        redact_postgres_table(
            connection,
            table="clinical_notes",
            text_columns=["note"],
            key_column="record_id",
            batch_size=1,
            deidentifier=failing_deidentifier,
        )

    assert sensitive_value not in str(exc_info.value)
    assert connection.rollback_calls == 1
    assert connection.commit_calls == 0
    assert connection.rows == connection.initial_rows
    assert connection.cursors[0].closed is True


def test_rollback_failure_is_value_free_and_not_reported_as_successful():
    connection = _FakeConnection()
    sensitive_value = "synthetic secret rollback"
    connection.rows[1]["note"] = sensitive_value
    connection.rollback_error = RuntimeError(f"rollback detail: {sensitive_value}")

    def failing_deidentifier(text: str, **_: Any) -> str:
        if text == sensitive_value:
            raise RuntimeError(f"driver detail: {text}")
        return text

    with pytest.raises(PostgresRedactionError) as exc_info:
        redact_postgres_table(
            connection,
            table="clinical_notes",
            text_columns=["note"],
            key_column="record_id",
            batch_size=1,
            deidentifier=failing_deidentifier,
        )

    assert str(exc_info.value) == (
        "PostgreSQL redaction failed; transaction rollback could not be confirmed"
    )
    assert sensitive_value not in str(exc_info.value)
    assert connection.rollback_calls == 1
    assert connection.commit_calls == 0
    assert connection.rows != connection.initial_rows
    assert connection.cursors[0].closed is True


def test_config_accepts_qualified_table_and_column_alias():
    connection = _FakeConnection()

    result = redact_postgres_table(
        connection,
        table="clinical.notes",
        columns=["note"],
        key_column="record_id",
        batch_size=10,
        deidentifier=lambda text, **_: text,
    )

    assert result.rows_processed == 3
    assert 'FROM "clinical"."notes"' in connection.execute_calls[0][0]
