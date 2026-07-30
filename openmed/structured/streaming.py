"""Streaming, bounded-memory de-identification for large CSV and Parquet tables.

Warehouse-scale clinical tables do not fit in memory, yet k-anonymity is a
global property: whether a record is suppressed depends on how many other
records share its quasi-identifier class. :func:`stream_deidentify_table`
resolves that tension with a two-pass design built on
:class:`openmed.risk.kanon.StreamingKanonState`:

* Pass one iterates the source in chunks (CSV rows or Parquet row groups) and
  folds each row into a compact global equivalence-class census keyed by the
  generalized quasi-identifier tuple. The working set scales with the number of
  distinct classes, never with the row count, and a ``memory_ceiling`` caps it.
* Pass two re-iterates the same source, drops rows whose global class is below
  ``target_k``, generalizes the surviving quasi-identifiers, routes free-text
  cells through :func:`openmed.core.pii.deidentify`, and writes the release
  incrementally so neither a whole column nor the whole table is ever buffered.

Because the suppression decision is derived from the global census rather than
from any single chunk, the released output is identical no matter how the rows
are batched, and re-measuring the release with :func:`openmed.risk.kanon_report`
reproduces the reported ``released_k``. Only schema names and aggregate counts
are returned or retained; no cell value is written to logs or spill files.
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from openmed.risk.kanon import (
    MemoryCeilingError,
    StreamingKanonDecision,
    StreamingKanonState,
)

from .table_io import (
    _canonical_scalar,
    _delimited_value,
    _field_order,
    _materialize_row,
    _validate_arrow_temporal_precision,
    _validate_parquet_column_families,
    _validated_field_names,
)

SUPPORTED_STREAMING_SUFFIXES = frozenset({".csv", ".tsv", ".parquet"})
DEFAULT_CHUNK_SIZE = 4_096
DEFAULT_MEMORY_CEILING = 64 * 1024 * 1024

Deidentifier = Callable[..., Any]

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_MEMORY_CEILING",
    "SUPPORTED_STREAMING_SUFFIXES",
    "MemoryCeilingError",
    "StreamingKanonDecision",
    "stream_deidentify_table",
]


def stream_deidentify_table(
    input_path: str | Path,
    output_path: str | Path,
    *,
    quasi_identifiers: Sequence[str],
    free_text_columns: Sequence[str] = (),
    target_k: int = 2,
    generalization: Mapping[str, int] | None = None,
    hierarchies: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    remove_direct_identifiers: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    memory_ceiling: int | None = DEFAULT_MEMORY_CEILING,
    overwrite: bool = False,
    deidentifier: Deidentifier | None = None,
    deidentify_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stream a CSV/Parquet table through bounded-memory k-anonymity de-id.

    Args:
        input_path: Source ``.csv``/``.tsv`` or ``.parquet`` table.
        output_path: Destination table. Its suffix selects the release format
            and may differ from the input's.
        quasi_identifiers: Columns whose combination forms the equivalence-class
            key. Classes with a global size below ``target_k`` are suppressed.
        free_text_columns: Columns whose string cells are individually routed
            through de-identification during the second pass.
        target_k: Minimum released equivalence-class size.
        generalization: Optional per-quasi-identifier generalization level
            index. Defaults to the exact (identity) level for every column.
        hierarchies: Optional explicit generalization hierarchies forwarded to
            :class:`openmed.risk.kanon.StreamingKanonState`.
        remove_direct_identifiers: Drop columns recognized as direct
            identifiers from the release.
        chunk_size: Rows per CSV batch / Parquet row-group batch. Bounds the
            second-pass write buffer.
        memory_ceiling: Byte ceiling on the first-pass class census. ``None``
            disables the guard. Raises :class:`MemoryCeilingError` if exceeded.
        overwrite: Allow replacing an existing ``output_path``.
        deidentifier: Optional de-identification callable (primarily for tests).
            Defaults to :func:`openmed.core.pii.deidentify`.
        deidentify_kwargs: Extra keyword arguments forwarded per free-text cell.

    Returns:
        An aggregate-only report: schema names, generalization node, and record
        counts. No cell value is included.
    """

    resolved_input = Path(input_path)
    resolved_output = Path(output_path)
    input_suffix = _validate_streaming_suffix(resolved_input, role="input")
    output_suffix = _validate_streaming_suffix(resolved_output, role="output")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if (resolved_output.exists() or resolved_output.is_symlink()) and not overwrite:
        raise FileExistsError(f"Output already exists: {resolved_output}")
    if not resolved_output.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {resolved_output.parent}"
        )

    columns = _read_columns(resolved_input, input_suffix)
    free_text = _validated_subset(free_text_columns, columns, name="free_text_columns")
    state = StreamingKanonState(
        quasi_identifiers,
        target_k=target_k,
        generalization=generalization,
        hierarchies=hierarchies,
        remove_direct_identifiers=remove_direct_identifiers,
        memory_ceiling=memory_ceiling,
    )
    missing_qi = sorted(set(state.quasi_identifiers) - set(columns))
    if missing_qi:
        raise ValueError(f"quasi_identifiers not present in input schema: {missing_qi}")

    # Pass one: build the global census. Only the class counter is retained.
    for batch in _iter_batches(resolved_input, input_suffix, chunk_size):
        for row in batch:
            state.observe(row)
    decision = state.resolve()

    output_columns = [
        column
        for column in columns
        if not (remove_direct_identifiers and _is_direct_identifier(column))
    ]
    redact = _redactor(deidentifier, deidentify_kwargs)

    # Pass two: re-stream, suppress sub-k classes, generalize and redact the
    # rest, and write incrementally.
    def released_rows() -> Iterator[dict[str, Any]]:
        for batch in _iter_batches(resolved_input, input_suffix, chunk_size):
            for row in batch:
                generalized = state.generalize(row)
                if not state.is_released(state.class_key(generalized)):
                    continue
                yield _project_row(generalized, output_columns, free_text, redact)

    _write_stream(
        resolved_output,
        output_suffix,
        output_columns,
        released_rows(),
        chunk_size=chunk_size,
    )

    return {
        "schema_version": "1.0",
        "input_format": input_suffix.lstrip("."),
        "output_format": output_suffix.lstrip("."),
        "chunk_size": chunk_size,
        "memory_ceiling": memory_ceiling,
        "quasi_identifiers": list(state.quasi_identifiers),
        "free_text_columns": list(free_text),
        "released_columns": output_columns,
        "generalization_node": state.generalization_node,
        "target_k": int(target_k),
        "decision": decision.to_dict(),
    }


def _project_row(
    generalized: Mapping[str, Any],
    output_columns: Sequence[str],
    free_text: Sequence[str],
    redact: Callable[[Any], Any],
) -> dict[str, Any]:
    free_text_set = set(free_text)
    projected: dict[str, Any] = {}
    for column in output_columns:
        value = generalized.get(column)
        projected[column] = redact(value) if column in free_text_set else value
    return projected


# ---------------------------------------------------------------------------
# Bounded readers
# ---------------------------------------------------------------------------


def _read_columns(path: Path, suffix: str) -> tuple[str, ...]:
    if suffix in {".csv", ".tsv"}:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t" if suffix == ".tsv" else ",")
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError("Delimited input must include a header row") from None
            return _validated_field_names(header, source="Delimited input header")
    return _read_parquet_columns(path)


def _read_parquet_columns(path: Path) -> tuple[str, ...]:
    pa, pq = _import_pyarrow()
    try:
        parquet_file = pq.ParquetFile(path)
    except pa.ArrowException:
        raise ValueError("Parquet input could not be decoded safely") from None
    schema = parquet_file.schema_arrow
    _validate_arrow_temporal_precision(schema)
    return _validated_field_names(schema.names, source="Parquet input schema")


def _iter_batches(
    path: Path,
    suffix: str,
    chunk_size: int,
) -> Iterator[list[dict[str, Any]]]:
    if suffix in {".csv", ".tsv"}:
        yield from _iter_delimited_batches(
            path,
            chunk_size,
            delimiter="\t" if suffix == ".tsv" else ",",
        )
    else:
        yield from _iter_parquet_batches(path, chunk_size)


def _iter_delimited_batches(
    path: Path,
    chunk_size: int,
    *,
    delimiter: str,
) -> Iterator[list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            reader = csv.DictReader(handle, delimiter=delimiter, strict=True)
            if reader.fieldnames is None:
                raise ValueError("Delimited input must include a header row")
            fields = _validated_field_names(
                reader.fieldnames,
                source="Delimited input header",
            )
            batch: list[dict[str, Any]] = []
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
                batch.append({field: row[field] for field in fields})
                if len(batch) >= chunk_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        except csv.Error:
            raise ValueError("Delimited input is malformed") from None


def _iter_parquet_batches(
    path: Path,
    chunk_size: int,
) -> Iterator[list[dict[str, Any]]]:
    pa, pq = _import_pyarrow()
    try:
        parquet_file = pq.ParquetFile(path)
    except pa.ArrowException:
        raise ValueError("Parquet input could not be decoded safely") from None
    _validate_arrow_temporal_precision(parquet_file.schema_arrow)
    columns = _validated_field_names(
        parquet_file.schema_arrow.names,
        source="Parquet input schema",
    )
    row_index = 0
    try:
        for record_batch in parquet_file.iter_batches(batch_size=chunk_size):
            batch: list[dict[str, Any]] = []
            for row in record_batch.to_pylist():
                row_index += 1
                if not isinstance(row, Mapping):
                    raise ValueError("Parquet batches must yield row mappings")
                batch.append(
                    _materialize_row(
                        row,
                        row_index=row_index,
                        format_name="Parquet",
                        allow_arrow_scalars=True,
                    )
                )
            if batch:
                _validate_parquet_column_families(batch, columns)
                yield batch
    except pa.ArrowException:
        raise ValueError("Parquet input could not be decoded safely") from None


# ---------------------------------------------------------------------------
# Bounded writers
# ---------------------------------------------------------------------------


def _write_stream(
    path: Path,
    suffix: str,
    columns: Sequence[str],
    rows: Iterator[Mapping[str, Any]],
    *,
    chunk_size: int,
) -> None:
    if not columns:
        raise ValueError("Release output must contain at least one column")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    wrote_any = False
    try:
        if suffix in {".csv", ".tsv"}:
            wrote_any = _write_delimited_stream(
                temporary,
                columns,
                rows,
                delimiter="\t" if suffix == ".tsv" else ",",
            )
        else:
            wrote_any = _write_parquet_stream(
                temporary,
                columns,
                rows,
                chunk_size=chunk_size,
            )
        if not wrote_any:
            raise ValueError(
                "Streaming de-identification suppressed every record; refusing "
                "to write an empty release"
            )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_delimited_stream(
    path: Path,
    columns: Sequence[str],
    rows: Iterator[Mapping[str, Any]],
    *,
    delimiter: str,
) -> bool:
    wrote_any = False
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(columns),
            delimiter=delimiter,
            extrasaction="raise",
        )
        writer.writeheader()
        for row_index, row in enumerate(rows, start=1):
            wrote_any = True
            writer.writerow(
                {
                    field: _delimited_value(
                        row.get(field),
                        row_index=row_index,
                        column_index=column_index,
                    )
                    for column_index, field in enumerate(columns, start=1)
                }
            )
    return wrote_any


def _write_parquet_stream(
    path: Path,
    columns: Sequence[str],
    rows: Iterator[Mapping[str, Any]],
    *,
    chunk_size: int,
) -> bool:
    pa, pq = _import_pyarrow()
    ordered = list(columns)
    writer = None
    schema = None
    wrote_any = False
    row_index = 0

    def canonical_batch(buffer: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        nonlocal row_index
        canonical: list[dict[str, Any]] = []
        for row in buffer:
            row_index += 1
            canonical.append(
                {
                    field: _canonical_scalar(
                        row.get(field),
                        row_index=row_index,
                        column_index=column_index,
                        format_name="Parquet",
                        allow_arrow_scalars=True,
                    )
                    for column_index, field in enumerate(ordered, start=1)
                }
            )
        _validate_parquet_column_families(canonical, ordered)
        return canonical

    try:
        buffer: list[Mapping[str, Any]] = []
        for row in rows:
            buffer.append(row)
            if len(buffer) >= chunk_size:
                canonical = canonical_batch(buffer)
                table = pa.Table.from_pylist(canonical)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(path, schema)
                writer.write_table(table.cast(schema))
                wrote_any = True
                buffer = []
        if buffer:
            canonical = canonical_batch(buffer)
            table = pa.Table.from_pylist(canonical)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(path, schema)
            writer.write_table(table.cast(schema))
            wrote_any = True
    except (pa.ArrowException, OverflowError):
        raise ValueError(
            "Parquet scalar values could not be represented safely"
        ) from None
    finally:
        if writer is not None:
            writer.close()
    return wrote_any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redactor(
    deidentifier: Deidentifier | None,
    deidentify_kwargs: Mapping[str, Any] | None,
) -> Callable[[Any], Any]:
    redact = deidentifier or _load_deidentifier()
    kwargs = dict(deidentify_kwargs or {})

    def apply(value: Any) -> Any:
        if not isinstance(value, str) or value == "":
            return value
        result = redact(value, **kwargs)
        if isinstance(result, str):
            return result
        text = getattr(result, "deidentified_text", None)
        if isinstance(text, str):
            return text
        raise TypeError(
            "deidentifier must return a string or an object with deidentified_text"
        )

    return apply


def _load_deidentifier() -> Deidentifier:
    from openmed.core.pii import deidentify

    return deidentify


def _is_direct_identifier(column: str) -> bool:
    from openmed.risk.reid import _field_is_direct_identifier

    return _field_is_direct_identifier(column)


def _validated_subset(
    columns: Sequence[str],
    available: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    if isinstance(columns, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of column names, not a string")
    selected = tuple(dict.fromkeys(str(column) for column in columns))
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"{name} not present in input schema: {missing}")
    return selected


def _validate_streaming_suffix(path: Path, *, role: str) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_STREAMING_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_STREAMING_SUFFIXES))
        raise ValueError(
            f"Unsupported streaming {role} format {suffix!r}; expected {supported}"
        )
    return suffix


def _import_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Streaming Parquet de-identification requires pyarrow. "
            "Install openmed[columnar]."
        ) from exc
    return pa, pq
