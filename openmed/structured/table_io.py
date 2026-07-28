"""Deterministic local table I/O for structured privacy workflows."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any

SUPPORTED_TABLE_SUFFIXES = frozenset({".csv", ".tsv", ".jsonl", ".ndjson", ".parquet"})

__all__ = ["SUPPORTED_TABLE_SUFFIXES", "read_table", "write_table"]


def read_table(path: str | Path) -> list[dict[str, Any]]:
    """Read a complete local table for final assessment or anonymization."""

    resolved = Path(path)
    suffix = _validate_suffix(resolved)
    if suffix in {".csv", ".tsv"}:
        return _read_delimited(
            resolved,
            delimiter="\t" if suffix == ".tsv" else ",",
        )
    if suffix in {".jsonl", ".ndjson"}:
        return _read_jsonl(resolved)
    return _read_parquet(resolved)


def write_table(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a structured release without overwriting by default."""

    resolved = Path(path)
    suffix = _validate_suffix(resolved)
    rows = _materialize_rows(records)
    if not rows:
        raise ValueError("Refusing to write an empty release table")
    if (resolved.exists() or resolved.is_symlink()) and not overwrite:
        raise FileExistsError(f"Output already exists: {resolved}")
    if not resolved.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {resolved.parent}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.",
        suffix=".tmp",
        dir=resolved.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if suffix in {".csv", ".tsv"}:
            _write_delimited(
                temporary,
                rows,
                delimiter="\t" if suffix == ".tsv" else ",",
            )
        elif suffix in {".jsonl", ".ndjson"}:
            _write_jsonl(temporary, rows)
        else:
            _write_parquet(temporary, rows)
        os.replace(temporary, resolved)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return resolved


def _read_delimited(path: Path, *, delimiter: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            reader = csv.DictReader(handle, delimiter=delimiter, strict=True)
            if reader.fieldnames is None:
                raise ValueError("Delimited input must include a header row")
            fields = _validated_field_names(
                reader.fieldnames,
                source="Delimited input header",
            )
            rows: list[dict[str, Any]] = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(
                        f"Delimited input row {row_number} has more cells "
                        "than its header"
                    )
                if any(row[field] is None for field in fields):
                    raise ValueError(
                        f"Delimited input row {row_number} has fewer cells "
                        "than its header"
                    )
                rows.append({field: row[field] for field in fields})
            return rows
        except csv.Error:
            raise ValueError("Delimited input is malformed") from None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = _strict_json_loads(line)
            except _DuplicateJsonKeyError:
                raise ValueError(
                    f"JSONL row {line_number} contains duplicate object keys"
                ) from None
            except (_NonFiniteJsonNumberError, json.JSONDecodeError):
                raise ValueError(f"JSONL row {line_number} is invalid") from None
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row {line_number} must be an object")
            row = _materialize_row(
                value,
                row_index=line_number,
                format_name="JSONL",
                allow_arrow_scalars=False,
            )
            rows.append(row)
    _validate_nonempty_schema(rows, source="JSONL input")
    return rows


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Parquet table I/O requires pyarrow. Install openmed[columnar]."
        ) from exc
    try:
        table = pq.read_table(path)
    except pa.ArrowException:
        raise ValueError("Parquet input could not be decoded safely") from None
    fields = _validated_field_names(
        table.column_names,
        source="Parquet input schema",
    )
    _validate_arrow_temporal_precision(table.schema)
    try:
        rows = table.to_pylist()
    except pa.ArrowException:
        raise ValueError("Parquet input could not be decoded safely") from None
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("Parquet input must decode to row mappings")
    canonical = [
        _materialize_row(
            row,
            row_index=row_index,
            format_name="Parquet",
            allow_arrow_scalars=True,
        )
        for row_index, row in enumerate(rows, start=1)
    ]
    _validate_parquet_column_families(canonical, fields)
    return canonical


def _write_delimited(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    delimiter: str,
) -> None:
    fields = _field_order(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter=delimiter,
            extrasaction="raise",
        )
        writer.writeheader()
        for row_index, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    field: _delimited_value(
                        row.get(field),
                        row_index=row_index,
                        column_index=column_index,
                    )
                    for column_index, field in enumerate(fields, start=1)
                }
            )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows, start=1):
            canonical = {
                field: _canonical_scalar(
                    value,
                    row_index=row_index,
                    column_index=column_index,
                    format_name="JSONL",
                    allow_arrow_scalars=False,
                )
                for column_index, (field, value) in enumerate(
                    row.items(),
                    start=1,
                )
            }
            handle.write(
                json.dumps(
                    canonical,
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Parquet table I/O requires pyarrow. Install openmed[columnar]."
        ) from exc
    fields = _field_order(rows)
    canonical = [
        {
            field: _canonical_scalar(
                row.get(field),
                row_index=row_index,
                column_index=column_index,
                format_name="Parquet",
                allow_arrow_scalars=True,
            )
            for column_index, field in enumerate(fields, start=1)
        }
        for row_index, row in enumerate(rows, start=1)
    ]
    _validate_parquet_column_families(canonical, fields)
    try:
        # Every row carries the complete union schema. PyArrow otherwise infers
        # fields from the first row and can silently discard later-row columns.
        table = pa.Table.from_pylist(canonical)
        pq.write_table(table, path)
    except (pa.ArrowException, OverflowError):
        raise ValueError(
            "Parquet scalar values could not be represented safely"
        ) from None


def _field_order(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    if not fields:
        raise ValueError("Release rows must contain at least one column")
    return fields


def _delimited_value(
    value: Any,
    *,
    row_index: int,
    column_index: int,
) -> str | int | float | bool:
    if value is None:
        return ""
    if type(value) is float and not math.isfinite(value):
        raise ValueError(
            "Delimited output contains a non-finite number at "
            f"row {row_index}, column {column_index}"
        )
    if type(value) in {str, int, float, bool}:
        return value
    raise TypeError(
        "CSV/TSV release output supports scalar values only: null, boolean, "
        f"finite numeric, and string values; unsupported value at row "
        f"{row_index}, column {column_index}"
    )


def _materialize_rows(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(records, Sequence) or isinstance(
        records, (str, bytes, bytearray)
    ):
        raise TypeError("records must be a sequence of row mappings")
    if not all(isinstance(row, Mapping) for row in records):
        raise TypeError("records must contain only row mappings")
    return [
        _materialize_row(
            row,
            row_index=row_index,
            format_name="Release output",
            allow_arrow_scalars=True,
            validate_values=False,
        )
        for row_index, row in enumerate(records, start=1)
    ]


def _materialize_row(
    row: Mapping[Any, Any],
    *,
    row_index: int,
    format_name: str,
    allow_arrow_scalars: bool,
    validate_values: bool = True,
) -> dict[str, Any]:
    materialized: dict[str, Any] = {}
    for column_index, (field, value) in enumerate(row.items(), start=1):
        if type(field) is not str:
            raise TypeError(
                f"{format_name} row {row_index} contains a non-string column name"
            )
        if not field.strip():
            raise ValueError(
                f"{format_name} row {row_index} contains an empty column name"
            )
        materialized[field] = (
            _canonical_scalar(
                value,
                row_index=row_index,
                column_index=column_index,
                format_name=format_name,
                allow_arrow_scalars=allow_arrow_scalars,
            )
            if validate_values
            else value
        )
    return materialized


def _canonical_scalar(
    value: Any,
    *,
    row_index: int,
    column_index: int,
    format_name: str,
    allow_arrow_scalars: bool,
) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(
                f"{format_name} contains a non-finite number at row "
                f"{row_index}, column {column_index}"
            )
        return 0.0 if value == 0.0 else value
    if allow_arrow_scalars:
        if type(value) is Decimal:
            if not value.is_finite():
                raise ValueError(
                    f"{format_name} contains a non-finite decimal at row "
                    f"{row_index}, column {column_index}"
                )
            return _canonical_decimal(value)
        if type(value) is datetime:
            _datetime_family(value, row_index=row_index, column_index=column_index)
            return value
        if type(value) is date:
            return value
        if type(value) is time:
            if value.tzinfo is not None and value.utcoffset() is not None:
                raise ValueError(
                    f"{format_name} contains a timezone-aware time at row "
                    f"{row_index}, column {column_index}"
                )
            return value
        if type(value) is bytes:
            return value
    raise TypeError(
        f"{format_name} contains an unsupported scalar at row {row_index}, "
        f"column {column_index}"
    )


def _canonical_decimal(value: Decimal) -> Decimal:
    if value.is_zero():
        return Decimal(0)
    sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError("Cannot canonicalize a non-finite decimal")
    normalized_digits = list(digits)
    while normalized_digits and normalized_digits[-1] == 0:
        normalized_digits.pop()
        exponent += 1
    return Decimal((sign, tuple(normalized_digits), exponent))


def _validate_parquet_column_families(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    for column_index, field in enumerate(fields, start=1):
        families = {
            _parquet_scalar_family(
                row[field],
                row_index=row_index,
                column_index=column_index,
            )
            for row_index, row in enumerate(rows, start=1)
            if row.get(field) is not None
        }
        if len(families) > 1:
            raise TypeError(
                "Parquet columns cannot mix scalar types; incompatible values "
                f"found in column {column_index}"
            )


def _validate_arrow_temporal_precision(schema: Any) -> None:
    """Reject Arrow temporal types that Python conversion would truncate."""

    try:
        fields = iter(schema)
    except TypeError:
        return
    for field in fields:
        data_type = getattr(field, "type", None)
        if type(data_type).__name__ in {"TimestampType", "Time64Type"} and (
            getattr(data_type, "unit", None) == "ns"
        ):
            raise ValueError(
                "Parquet temporal columns with sub-microsecond precision are "
                "unsupported"
            )


def _parquet_scalar_family(
    value: Any,
    *,
    row_index: int,
    column_index: int,
) -> tuple[str, str | None]:
    if type(value) is datetime:
        return _datetime_family(
            value,
            row_index=row_index,
            column_index=column_index,
        )
    return type(value).__name__, None


def _datetime_family(
    value: datetime,
    *,
    row_index: int,
    column_index: int,
) -> tuple[str, str | None]:
    if value.tzinfo is None:
        return "datetime", None
    offset = value.utcoffset()
    if offset is None:
        raise ValueError(
            "Parquet contains a datetime with an indeterminate timezone at "
            f"row {row_index}, column {column_index}"
        )
    zone = getattr(value.tzinfo, "key", None) or str(value.tzinfo)
    return "datetime", zone


def _validated_field_names(
    fields: Sequence[Any],
    *,
    source: str,
) -> tuple[str, ...]:
    names = tuple(fields)
    if not names:
        raise ValueError(f"{source} must include at least one column")
    if any(type(field) is not str or not field.strip() for field in names):
        raise ValueError(f"{source} contains an empty column name")
    if len(names) != len(set(names)):
        raise ValueError(f"{source} contains duplicate column names")
    return names


def _validate_nonempty_schema(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> None:
    if rows and not any(row for row in rows):
        raise ValueError(f"{source} must include at least one column")


class _DuplicateJsonKeyError(ValueError):
    pass


class _NonFiniteJsonNumberError(ValueError):
    pass


def _json_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field, value in pairs:
        if field in result:
            raise _DuplicateJsonKeyError
        result[field] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise _NonFiniteJsonNumberError


def _strict_json_loads(data: str) -> Any:
    """Decode JSON while rejecting duplicate keys and non-finite numbers."""

    value = json.loads(
        data,
        object_pairs_hook=_json_object_without_duplicates,
        parse_constant=_reject_json_constant,
    )
    _reject_non_finite_json_numbers(value)
    return value


def _reject_non_finite_json_numbers(value: Any) -> None:
    if type(value) is float and not math.isfinite(value):
        raise _NonFiniteJsonNumberError
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_non_finite_json_numbers(item)
    elif isinstance(value, list):
        for item in value:
            _reject_non_finite_json_numbers(item)


def _validate_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_TABLE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_TABLE_SUFFIXES))
        raise ValueError(f"Unsupported table format {suffix!r}; expected {supported}")
    return suffix
