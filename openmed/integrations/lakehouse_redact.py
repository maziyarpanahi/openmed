"""Column-level redaction jobs for local lakehouse-style Parquet tables.

The public entrypoint, :func:`redact_lakehouse_table`, scans partitioned
Parquet data files, de-identifies configured free-text columns with
``openmed.processing.batch.process_batch``, and writes an OpenMed-managed
snapshot without mutating the source table data files.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from openmed.processing.batch import process_batch

LakehouseTableFormat = Literal["parquet", "delta", "iceberg"]
ProcessBatchCallable = Callable[..., Any]
ProgressCallback = Callable[["LakehouseRedactionProgress"], None]

_DEFAULT_PII_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
_CHECKPOINT_SCHEMA_VERSION = 1
_MANIFEST_SCHEMA_VERSION = 1
_METADATA_DIR = "_openmed_lakehouse"
_SNAPSHOTS_DIR = "snapshots"
_CHECKPOINTS_DIR = "checkpoints"
_IGNORED_DIRECTORIES = {_METADATA_DIR, "_delta_log", "metadata"}
_UNPARTITIONED = "__unpartitioned__"


@dataclass(frozen=True)
class LakehouseRedactionProgress:
    """PHI-free progress emitted after each completed partition."""

    partition_index: int
    partition_count: int
    partition_id: str
    files_in_partition: int
    rows_in_partition: int
    rows_completed: int
    total_rows: int
    resumed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable progress mapping."""
        return {
            "partition_index": self.partition_index,
            "partition_count": self.partition_count,
            "partition_id": self.partition_id,
            "files_in_partition": self.files_in_partition,
            "rows_in_partition": self.rows_in_partition,
            "rows_completed": self.rows_completed,
            "total_rows": self.total_rows,
            "resumed": self.resumed,
        }


@dataclass(frozen=True)
class LakehouseRedactionResult:
    """Result returned by :func:`redact_lakehouse_table`."""

    table_path: Path
    snapshot_id: str
    snapshot_path: Path | None
    manifest: dict[str, Any]
    dry_run: bool = False
    manifest_path: Path | None = None
    checkpoint_path: Path | None = None
    metadata_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result mapping."""
        return {
            "table_path": str(self.table_path),
            "snapshot_id": self.snapshot_id,
            "snapshot_path": str(self.snapshot_path) if self.snapshot_path else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "checkpoint_path": (
                str(self.checkpoint_path) if self.checkpoint_path else None
            ),
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "dry_run": self.dry_run,
            "manifest": self.manifest,
        }


@dataclass(frozen=True)
class _DataFile:
    path: Path
    relative_path: PurePosixPath
    file_id: str
    partition_id: str
    partition_columns: tuple[str, ...]
    row_count: int


@dataclass(frozen=True)
class _PartitionWork:
    partition_id: str
    files: tuple[_DataFile, ...]
    partition_columns: tuple[str, ...]
    row_count: int


@dataclass
class _ColumnSummary:
    processed_cells: int = 0
    affected_cells: int = 0
    span_count: int = 0
    per_label_counts: Counter[str] = field(default_factory=Counter)

    def add(self, *, changed: bool, labels: Counter[str]) -> None:
        self.processed_cells += 1
        if changed or labels:
            self.affected_cells += 1
        self.span_count += sum(labels.values())
        self.per_label_counts.update(labels)

    def merge(self, other: "_ColumnSummary") -> None:
        self.processed_cells += other.processed_cells
        self.affected_cells += other.affected_cells
        self.span_count += other.span_count
        self.per_label_counts.update(other.per_label_counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed_cells": self.processed_cells,
            "affected_cells": self.affected_cells,
            "span_count": self.span_count,
            "per_label_counts": dict(sorted(self.per_label_counts.items())),
        }


@dataclass
class _FileSummary:
    file_id: str
    row_count: int = 0
    processed_cells: int = 0
    affected_cells: int = 0
    affected_rows: set[int] = field(default_factory=set)
    span_count: int = 0
    per_label_counts: Counter[str] = field(default_factory=Counter)
    columns: dict[str, _ColumnSummary] = field(default_factory=dict)
    span_offsets: list[dict[str, Any]] = field(default_factory=list)

    def update_cell(
        self,
        *,
        row_index: int,
        column: str,
        changed: bool,
        labels: Counter[str],
        spans: Sequence[dict[str, Any]],
    ) -> None:
        if column not in self.columns:
            self.columns[column] = _ColumnSummary()
        self.columns[column].add(changed=changed, labels=labels)
        self.processed_cells += 1
        if changed or labels:
            self.affected_cells += 1
            self.affected_rows.add(row_index)
        self.span_count += sum(labels.values())
        self.per_label_counts.update(labels)
        self.span_offsets.extend(spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "row_count": self.row_count,
            "processed_cells": self.processed_cells,
            "affected_cells": self.affected_cells,
            "affected_rows": len(self.affected_rows),
            "span_count": self.span_count,
            "per_label_counts": dict(sorted(self.per_label_counts.items())),
            "column_counts": _column_counts_to_dict(self.columns),
            "span_offsets": self.span_offsets,
        }


@dataclass
class _PartitionSummary:
    partition_id: str
    file_count: int = 0
    row_count: int = 0
    processed_cells: int = 0
    affected_cells: int = 0
    affected_rows: int = 0
    span_count: int = 0
    per_label_counts: Counter[str] = field(default_factory=Counter)
    columns: dict[str, _ColumnSummary] = field(default_factory=dict)
    files: list[_FileSummary] = field(default_factory=list)

    def add_file(self, summary: _FileSummary) -> None:
        self.files.append(summary)
        self.file_count += 1
        self.row_count += summary.row_count
        self.processed_cells += summary.processed_cells
        self.affected_cells += summary.affected_cells
        self.affected_rows += len(summary.affected_rows)
        self.span_count += summary.span_count
        self.per_label_counts.update(summary.per_label_counts)
        for column, column_summary in summary.columns.items():
            if column not in self.columns:
                self.columns[column] = _ColumnSummary()
            self.columns[column].merge(column_summary)

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "file_count": self.file_count,
            "row_count": self.row_count,
            "processed_cells": self.processed_cells,
            "affected_cells": self.affected_cells,
            "affected_rows": self.affected_rows,
            "span_count": self.span_count,
            "per_label_counts": dict(sorted(self.per_label_counts.items())),
            "column_counts": _column_counts_to_dict(self.columns),
            "files": [summary.to_dict() for summary in self.files],
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "_PartitionSummary":
        summary = cls(partition_id=str(data["partition_id"]))
        summary.file_count = int(data.get("file_count", 0))
        summary.row_count = int(data.get("row_count", 0))
        summary.processed_cells = int(data.get("processed_cells", 0))
        summary.affected_cells = int(data.get("affected_cells", 0))
        summary.affected_rows = int(data.get("affected_rows", 0))
        summary.span_count = int(data.get("span_count", 0))
        summary.per_label_counts.update(
            {
                str(label): int(count)
                for label, count in data.get("per_label_counts", {}).items()
            }
        )
        for column, payload in data.get("column_counts", {}).items():
            column_summary = _ColumnSummary()
            column_summary.processed_cells = int(payload.get("processed_cells", 0))
            column_summary.affected_cells = int(payload.get("affected_cells", 0))
            column_summary.span_count = int(payload.get("span_count", 0))
            column_summary.per_label_counts.update(
                {
                    str(label): int(count)
                    for label, count in payload.get("per_label_counts", {}).items()
                }
            )
            summary.columns[str(column)] = column_summary
        return summary


@dataclass
class _ManifestAccumulator:
    table_format: LakehouseTableFormat
    snapshot_id: str
    text_columns: tuple[str, ...]
    partition_columns: tuple[str, ...]
    partition_count: int
    file_count: int
    total_rows_expected: int
    dry_run: bool
    rows_processed: int = 0
    affected_rows: int = 0
    processed_cells: int = 0
    affected_cells: int = 0
    span_count: int = 0
    per_label_counts: Counter[str] = field(default_factory=Counter)
    columns: dict[str, _ColumnSummary] = field(default_factory=dict)
    partitions: list[_PartitionSummary] = field(default_factory=list)

    def add_partition(self, summary: _PartitionSummary) -> None:
        self.partitions.append(summary)
        self.rows_processed += summary.row_count
        self.affected_rows += summary.affected_rows
        self.processed_cells += summary.processed_cells
        self.affected_cells += summary.affected_cells
        self.span_count += summary.span_count
        self.per_label_counts.update(summary.per_label_counts)
        for column, column_summary in summary.columns.items():
            if column not in self.columns:
                self.columns[column] = _ColumnSummary()
            self.columns[column].merge(column_summary)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "table_format": self.table_format,
            "snapshot_id": self.snapshot_id,
            "dry_run": self.dry_run,
            "text_columns": list(self.text_columns),
            "partition_columns": list(self.partition_columns),
            "partition_count": self.partition_count,
            "file_count": self.file_count,
            "total_rows_expected": self.total_rows_expected,
            "total_rows": self.rows_processed,
            "affected_rows": self.affected_rows,
            "processed_cells": self.processed_cells,
            "affected_cells": self.affected_cells,
            "affected_columns": [
                column
                for column, summary in sorted(self.columns.items())
                if summary.affected_cells
            ],
            "span_count": self.span_count,
            "per_label_counts": dict(sorted(self.per_label_counts.items())),
            "column_counts": _column_counts_to_dict(self.columns),
            "partitions": [
                summary.to_dict()
                for summary in sorted(
                    self.partitions,
                    key=lambda partition: partition.partition_id,
                )
            ],
            "raw_cell_values_included": False,
            "raw_partition_values_included": False,
            "source_paths_included": False,
        }


def redact_lakehouse_table(
    table_path: str | Path,
    text_columns: Sequence[str],
    *,
    snapshot_path: str | Path | None = None,
    snapshot_id: str | None = None,
    table_format: LakehouseTableFormat | None = None,
    manifest_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    dry_run: bool = False,
    resume: bool = False,
    model_name: str = _DEFAULT_PII_MODEL,
    method: str = "mask",
    policy: str | None = None,
    confidence_threshold: float = 0.7,
    config: Any | None = None,
    lang: str = "en",
    keep_year: bool = True,
    date_shift_days: int | None = None,
    use_safety_sweep: bool = True,
    batch_size: int = 512,
    on_progress: ProgressCallback | None = None,
    process_batch_fn: ProcessBatchCallable | None = None,
) -> LakehouseRedactionResult:
    """Redact text columns in a local lakehouse-style Parquet table.

    The table is treated as a directory of Parquet data files, optionally
    partitioned with Hive-style ``column=value`` path segments. Delta Lake and
    Iceberg metadata directories are ignored; the job writes an OpenMed-managed
    snapshot directory with the same relative data-file layout and a PHI-free
    redaction manifest.

    Args:
        table_path: Source table root containing Parquet data files.
        text_columns: Free-text column names to pass through ``process_batch``.
        snapshot_path: Optional destination data directory for the new snapshot.
        snapshot_id: Optional stable snapshot identifier. A timestamped id is
            generated when omitted.
        table_format: Optional table format label. When omitted, the function
            infers ``delta`` from ``_delta_log``, ``iceberg`` from ``metadata``,
            and otherwise uses ``parquet``.
        manifest_path: Optional path for the PHI-free redaction manifest.
        checkpoint_path: Optional PHI-free per-partition checkpoint path.
        dry_run: Build an affected row/column plan without writing outputs.
        resume: Reuse completed partitions from ``checkpoint_path`` and
            ``snapshot_path``.
        model_name: PII detection model forwarded to ``process_batch``.
        method: De-identification method forwarded to ``process_batch``.
        policy: Optional de-identification policy.
        confidence_threshold: Minimum confidence for redaction.
        config: Optional OpenMed configuration.
        lang: Language hint.
        keep_year: Keep year in date redaction where applicable.
        date_shift_days: Optional fixed date shift.
        use_safety_sweep: Enable deterministic structured-identifier sweep.
        batch_size: Maximum record batch and redaction batch size.
        on_progress: Optional PHI-free progress callback.
        process_batch_fn: Test seam for replacing ``process_batch``.

    Returns:
        ``LakehouseRedactionResult`` with snapshot metadata and manifest.
    """
    source = Path(table_path)
    if not source.exists():
        raise FileNotFoundError(f"Lakehouse table not found: {source}")
    if not source.is_dir():
        raise ValueError(f"Lakehouse table path must be a directory: {source}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if dry_run and resume:
        raise ValueError("resume is only supported for write runs")

    pyarrow, parquet = _import_pyarrow()
    normalized_columns = _normalize_columns(text_columns, name="text_columns")
    resolved_table_format = table_format or _infer_table_format(source)
    resolved_snapshot_id = snapshot_id or _default_snapshot_id(source)
    metadata_dir = source / _METADATA_DIR / _SNAPSHOTS_DIR / resolved_snapshot_id
    resolved_snapshot_path = (
        Path(snapshot_path) if snapshot_path is not None else metadata_dir / "data"
    )
    resolved_manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else metadata_dir / "redaction-manifest.json"
    )
    resolved_checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else source / _METADATA_DIR / _CHECKPOINTS_DIR / f"{resolved_snapshot_id}.json"
    )
    resolved_metadata_path = metadata_dir / "snapshot.json"

    if source.resolve() == resolved_snapshot_path.resolve():
        raise ValueError("snapshot_path must not overwrite the source table")

    files = _discover_data_files(source, parquet)
    if not files:
        raise ValueError(f"No Parquet data files found in lakehouse table: {source}")
    _validate_files(files, normalized_columns, pyarrow, parquet)
    partitions = _partition_work(files)
    partition_columns = tuple(
        sorted(
            {
                column
                for partition in partitions
                for column in partition.partition_columns
            }
        )
    )
    total_rows = sum(partition.row_count for partition in partitions)

    if not dry_run:
        if resolved_snapshot_path.exists() and not resume:
            raise FileExistsError(
                "snapshot_path already exists; pass resume=True with the same "
                "checkpoint_path or choose a new snapshot_id"
            )
        resolved_snapshot_path.mkdir(parents=True, exist_ok=True)

    state = (
        _load_checkpoint(resolved_checkpoint_path, resolved_snapshot_id)
        if resume
        else {}
    )
    completed_partitions = set(state.get("completed_partitions", []))
    resumed_summaries = {
        str(partition_id): _PartitionSummary.from_mapping(summary)
        for partition_id, summary in state.get("partitions", {}).items()
    }

    manifest = _ManifestAccumulator(
        table_format=resolved_table_format,
        snapshot_id=resolved_snapshot_id,
        text_columns=normalized_columns,
        partition_columns=partition_columns,
        partition_count=len(partitions),
        file_count=len(files),
        total_rows_expected=total_rows,
        dry_run=dry_run,
    )
    redaction_kwargs = {
        "operation": "deidentify",
        "batch_size": batch_size,
        "method": method,
        "policy": policy,
        "confidence_threshold": confidence_threshold,
        "config": config,
        "lang": lang,
        "keep_year": keep_year,
        "date_shift_days": date_shift_days,
        "use_safety_sweep": use_safety_sweep,
    }
    batch_fn = process_batch_fn or process_batch

    for partition_index, partition in enumerate(partitions):
        if (
            resume
            and partition.partition_id in completed_partitions
            and partition.partition_id in resumed_summaries
            and _partition_outputs_exist(partition, resolved_snapshot_path)
        ):
            partition_summary = resumed_summaries[partition.partition_id]
            manifest.add_partition(partition_summary)
            _emit_progress(
                on_progress,
                partition_index=partition_index,
                partition_count=len(partitions),
                partition=partition,
                rows_completed=manifest.rows_processed,
                total_rows=total_rows,
                resumed=True,
            )
            continue

        partition_summary = _process_partition(
            partition,
            source_snapshot_path=resolved_snapshot_path,
            text_columns=normalized_columns,
            model_name=model_name,
            redaction_kwargs=redaction_kwargs,
            process_batch_fn=batch_fn,
            batch_size=batch_size,
            dry_run=dry_run,
            pyarrow=pyarrow,
            parquet=parquet,
        )
        manifest.add_partition(partition_summary)
        if not dry_run:
            completed_partitions.add(partition.partition_id)
            resumed_summaries[partition.partition_id] = partition_summary
            _write_checkpoint(
                resolved_checkpoint_path,
                snapshot_id=resolved_snapshot_id,
                completed_partitions=completed_partitions,
                partitions=resumed_summaries,
                rows_completed=manifest.rows_processed,
                total_rows=total_rows,
            )
            _emit_progress(
                on_progress,
                partition_index=partition_index,
                partition_count=len(partitions),
                partition=partition,
                rows_completed=manifest.rows_processed,
                total_rows=total_rows,
                resumed=False,
            )

    manifest_payload = manifest.to_dict()
    if dry_run:
        return LakehouseRedactionResult(
            table_path=source,
            snapshot_id=resolved_snapshot_id,
            snapshot_path=None,
            manifest=manifest_payload,
            dry_run=True,
        )

    _write_json(resolved_manifest_path, manifest_payload)
    _write_snapshot_metadata(
        resolved_metadata_path,
        snapshot_id=resolved_snapshot_id,
        table_format=resolved_table_format,
        manifest_path=resolved_manifest_path,
    )
    return LakehouseRedactionResult(
        table_path=source,
        snapshot_id=resolved_snapshot_id,
        snapshot_path=resolved_snapshot_path,
        manifest=manifest_payload,
        dry_run=False,
        manifest_path=resolved_manifest_path,
        checkpoint_path=resolved_checkpoint_path,
        metadata_path=resolved_metadata_path,
    )


redact_lakehouse = redact_lakehouse_table


def _discover_data_files(root: Path, parquet: Any) -> tuple[_DataFile, ...]:
    files: list[_DataFile] = []
    for path in sorted(root.rglob("*.parquet")):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if _should_skip(relative):
            continue
        metadata = parquet.ParquetFile(path).metadata
        partition_columns = _partition_columns(relative)
        partition_id = _digest(_partition_key(relative))
        files.append(
            _DataFile(
                path=path,
                relative_path=relative,
                file_id=_digest(str(relative)),
                partition_id=partition_id,
                partition_columns=partition_columns,
                row_count=int(metadata.num_rows),
            )
        )
    return tuple(files)


def _validate_files(
    files: Sequence[_DataFile],
    text_columns: Sequence[str],
    pyarrow: Any,
    parquet: Any,
) -> None:
    for data_file in files:
        schema = parquet.ParquetFile(data_file.path).schema_arrow
        missing = [column for column in text_columns if column not in schema.names]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing text column(s) in data file: {joined}")
        for column in text_columns:
            field_type = schema.field(column).type
            if not (
                pyarrow.types.is_string(field_type)
                or pyarrow.types.is_large_string(field_type)
                or pyarrow.types.is_null(field_type)
            ):
                raise ValueError(
                    f"Lakehouse text column must be string-typed: {column}"
                )


def _partition_work(files: Sequence[_DataFile]) -> tuple[_PartitionWork, ...]:
    grouped: dict[str, list[_DataFile]] = {}
    for data_file in files:
        grouped.setdefault(data_file.partition_id, []).append(data_file)

    partitions: list[_PartitionWork] = []
    for partition_id in sorted(grouped):
        partition_files = tuple(
            sorted(grouped[partition_id], key=lambda item: str(item.relative_path))
        )
        partition_columns = tuple(
            sorted(
                {
                    column
                    for data_file in partition_files
                    for column in data_file.partition_columns
                }
            )
        )
        partitions.append(
            _PartitionWork(
                partition_id=partition_id,
                files=partition_files,
                partition_columns=partition_columns,
                row_count=sum(data_file.row_count for data_file in partition_files),
            )
        )
    return tuple(partitions)


def _process_partition(
    partition: _PartitionWork,
    *,
    source_snapshot_path: Path,
    text_columns: Sequence[str],
    model_name: str,
    redaction_kwargs: Mapping[str, Any],
    process_batch_fn: ProcessBatchCallable,
    batch_size: int,
    dry_run: bool,
    pyarrow: Any,
    parquet: Any,
) -> _PartitionSummary:
    summary = _PartitionSummary(partition_id=partition.partition_id)
    for data_file in partition.files:
        file_summary = _process_file(
            data_file,
            source_snapshot_path=source_snapshot_path,
            text_columns=text_columns,
            model_name=model_name,
            redaction_kwargs=redaction_kwargs,
            process_batch_fn=process_batch_fn,
            batch_size=batch_size,
            dry_run=dry_run,
            pyarrow=pyarrow,
            parquet=parquet,
        )
        summary.add_file(file_summary)
    return summary


def _process_file(
    data_file: _DataFile,
    *,
    source_snapshot_path: Path,
    text_columns: Sequence[str],
    model_name: str,
    redaction_kwargs: Mapping[str, Any],
    process_batch_fn: ProcessBatchCallable,
    batch_size: int,
    dry_run: bool,
    pyarrow: Any,
    parquet: Any,
) -> _FileSummary:
    source = parquet.ParquetFile(data_file.path)
    schema = source.schema_arrow
    destination = source_snapshot_path / Path(data_file.relative_path.as_posix())
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()

    summary = _FileSummary(file_id=data_file.file_id, row_count=data_file.row_count)
    writer = None
    try:
        if not dry_run:
            writer = parquet.ParquetWriter(destination, schema)
        row_offset = 0
        for record_batch in source.iter_batches(batch_size=batch_size):
            rows = pyarrow.Table.from_batches([record_batch], schema=schema).to_pylist()
            _process_rows(
                rows,
                text_columns,
                summary,
                row_offset=row_offset,
                model_name=model_name,
                redaction_kwargs=redaction_kwargs,
                process_batch_fn=process_batch_fn,
            )
            if writer is not None:
                writer.write_table(pyarrow.Table.from_pylist(rows, schema=schema))
            row_offset += len(rows)
        if writer is None and not dry_run:
            writer = parquet.ParquetWriter(destination, schema)
    finally:
        if writer is not None:
            writer.close()
    return summary


def _process_rows(
    rows: list[dict[str, Any]],
    text_columns: Sequence[str],
    summary: _FileSummary,
    *,
    row_offset: int,
    model_name: str,
    redaction_kwargs: Mapping[str, Any],
    process_batch_fn: ProcessBatchCallable,
) -> None:
    text_values: list[str] = []
    ids: list[str] = []
    cells: list[tuple[int, str, str]] = []
    for local_row_index, row in enumerate(rows):
        row_index = row_offset + local_row_index
        for column in text_columns:
            value = row.get(column)
            if value is None:
                continue
            text = value if isinstance(value, str) else str(value)
            if text == "":
                continue
            ids.append(f"file_{summary.file_id}_row_{row_index}_col_{_digest(column)}")
            text_values.append(text)
            cells.append((local_row_index, column, text))

    if not text_values:
        return

    result = process_batch_fn(
        text_values,
        model_name=model_name,
        ids=ids,
        **dict(redaction_kwargs),
    )
    items = list(getattr(result, "items", []) or [])
    if len(items) != len(cells):
        raise ValueError(
            "process_batch returned "
            f"{len(items)} result(s) for {len(cells)} text cell(s)"
        )

    for offset, ((local_row_index, column, original_text), item) in enumerate(
        zip(cells, items)
    ):
        if getattr(item, "error", None):
            raise RuntimeError(
                f"Lakehouse redaction failed for text column {column!r}: {item.error}"
            )
        cell_result = getattr(item, "result", item)
        redacted = _deidentified_text(cell_result)
        rows[local_row_index][column] = redacted
        row_index = row_offset + cells[offset][0]
        labels = _label_counts(cell_result)
        spans = _span_offsets(cell_result, row_index=row_index, column=column)
        summary.update_cell(
            row_index=row_index,
            column=column,
            changed=redacted != original_text,
            labels=labels,
            spans=spans,
        )


def _partition_outputs_exist(
    partition: _PartitionWork,
    snapshot_path: Path,
) -> bool:
    return all(
        (snapshot_path / Path(data_file.relative_path.as_posix())).exists()
        for data_file in partition.files
    )


def _write_checkpoint(
    path: Path,
    *,
    snapshot_id: str,
    completed_partitions: set[str],
    partitions: Mapping[str, _PartitionSummary],
    rows_completed: int,
    total_rows: int,
) -> None:
    payload = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "completed_partitions": sorted(completed_partitions),
        "rows_completed": rows_completed,
        "total_rows": total_rows,
        "partitions": {
            partition_id: partitions[partition_id].to_dict()
            for partition_id in sorted(partitions)
        },
        "raw_cell_values_included": False,
        "raw_partition_values_included": False,
        "source_paths_included": False,
    }
    _write_json(path, payload)


def _load_checkpoint(path: Path, snapshot_id: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", 0)) != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported lakehouse redaction checkpoint schema")
    if str(data.get("snapshot_id", "")) != snapshot_id:
        raise ValueError("Checkpoint snapshot_id does not match the current run")
    return data


def _write_snapshot_metadata(
    path: Path,
    *,
    snapshot_id: str,
    table_format: LakehouseTableFormat,
    manifest_path: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "table_format": table_format,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_directory": "data",
        "manifest_file": manifest_path.name,
        "raw_cell_values_included": False,
        "raw_partition_values_included": False,
        "source_paths_included": False,
    }
    _write_json(path, payload)


def _emit_progress(
    callback: ProgressCallback | None,
    *,
    partition_index: int,
    partition_count: int,
    partition: _PartitionWork,
    rows_completed: int,
    total_rows: int,
    resumed: bool,
) -> None:
    if callback is None:
        return
    callback(
        LakehouseRedactionProgress(
            partition_index=partition_index,
            partition_count=partition_count,
            partition_id=partition.partition_id,
            files_in_partition=len(partition.files),
            rows_in_partition=partition.row_count,
            rows_completed=rows_completed,
            total_rows=total_rows,
            resumed=resumed,
        )
    )


def _normalize_columns(
    columns: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for column in columns:
        value = str(column).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    if not normalized:
        raise ValueError(f"At least one {name} value is required")
    return tuple(normalized)


def _import_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pyarrow
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Lakehouse redaction requires pyarrow. Install openmed[columnar] "
            "or install pyarrow directly."
        ) from exc
    return pyarrow, parquet


def _infer_table_format(root: Path) -> LakehouseTableFormat:
    if (root / "_delta_log").is_dir():
        return "delta"
    if (root / "metadata").is_dir():
        return "iceberg"
    return "parquet"


def _default_snapshot_id(root: Path) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"openmed-redacted-{stamp}-{_digest(str(root.resolve()))[:8]}"


def _should_skip(relative: PurePosixPath) -> bool:
    for part in relative.parts[:-1]:
        if part in _IGNORED_DIRECTORIES or part.startswith("."):
            return True
    return False


def _partition_key(relative: PurePosixPath) -> str:
    parent = PurePosixPath(*relative.parts[:-1])
    if not parent.parts:
        return _UNPARTITIONED
    return parent.as_posix()


def _partition_columns(relative: PurePosixPath) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()
    for part in relative.parts[:-1]:
        if "=" not in part:
            continue
        column, _value = part.split("=", 1)
        column = column.strip()
        if column and column not in seen:
            seen.add(column)
            columns.append(column)
    return tuple(columns)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _deidentified_text(result: Any) -> str:
    if hasattr(result, "deidentified_text"):
        return str(result.deidentified_text)
    if isinstance(result, Mapping) and "deidentified_text" in result:
        return str(result["deidentified_text"])
    if isinstance(result, str):
        return result
    raise TypeError("process_batch results must expose deidentified_text")


def _entities(result: Any) -> Sequence[Any]:
    if hasattr(result, "pii_entities"):
        return tuple(result.pii_entities or ())
    if isinstance(result, Mapping):
        value = result.get("pii_entities", ())
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return tuple(value)
    return ()


def _entity_label(entity: Any) -> str:
    if isinstance(entity, Mapping):
        return str(
            entity.get("canonical_label")
            or entity.get("label")
            or entity.get("entity_type")
            or "UNKNOWN"
        )
    return str(
        getattr(entity, "canonical_label", None)
        or getattr(entity, "label", None)
        or getattr(entity, "entity_type", None)
        or "UNKNOWN"
    )


def _entity_int(entity: Any, field: str) -> int | None:
    value = (
        entity.get(field)
        if isinstance(entity, Mapping)
        else getattr(entity, field, None)
    )
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _label_counts(result: Any) -> Counter[str]:
    labels: Counter[str] = Counter()
    for entity in _entities(result):
        labels[_entity_label(entity)] += 1
    return labels


def _span_offsets(
    result: Any,
    *,
    row_index: int,
    column: str,
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for entity in _entities(result):
        start = _entity_int(entity, "start")
        end = _entity_int(entity, "end")
        if start is None or end is None:
            continue
        spans.append(
            {
                "row_index": row_index,
                "column": column,
                "start": start,
                "end": end,
                "label": _entity_label(entity),
            }
        )
    return spans


def _column_counts_to_dict(
    columns: Mapping[str, _ColumnSummary],
) -> dict[str, dict[str, Any]]:
    return {column: columns[column].to_dict() for column in sorted(columns)}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
