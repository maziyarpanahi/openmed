"""Streaming Parquet/ORC redaction with QI risk profiling.

The public entrypoint, :func:`redact_columnar_dataset`, processes one Parquet
row group or ORC stripe at a time. Configured free-text columns are sent through
``openmed.processing.batch.process_batch`` with the de-identification operation,
while non-text quasi-identifier columns are profiled into PHI-free aggregate
k-anonymity style counts.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from openmed.processing.batch import process_batch
from openmed.risk.reid import _normalize_qi_value

ColumnarFormat = Literal["parquet", "orc"]
GroupKind = Literal["row_group", "stripe"]
ProgressCallback = Callable[["ColumnarProgress"], None]
ProcessBatchCallable = Callable[..., Any]

_DEFAULT_PII_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
_PROGRESS_SCHEMA_VERSION = 1
_MISSING_VALUE = "<missing>"
_EMPTY_VALUE = "<empty>"


@dataclass(frozen=True)
class ColumnarProgress:
    """PHI-free progress snapshot emitted after each row group or stripe."""

    input_format: ColumnarFormat
    group_kind: GroupKind
    group_index: int
    group_count: int
    rows_in_group: int
    rows_completed: int
    total_rows: int
    resumed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable progress mapping."""
        return {
            "input_format": self.input_format,
            "group_kind": self.group_kind,
            "group_index": self.group_index,
            "group_count": self.group_count,
            "rows_in_group": self.rows_in_group,
            "rows_completed": self.rows_completed,
            "total_rows": self.total_rows,
            "resumed": self.resumed,
        }


@dataclass(frozen=True)
class ColumnarRedactionResult:
    """Result returned by :func:`redact_columnar_dataset`."""

    output_path: Path
    manifest: dict[str, Any]
    qi_report: dict[str, Any]
    manifest_path: Path | None = None
    qi_report_path: Path | None = None
    progress_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result mapping."""
        return {
            "output_path": str(self.output_path),
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "qi_report_path": str(self.qi_report_path) if self.qi_report_path else None,
            "progress_path": str(self.progress_path) if self.progress_path else None,
            "manifest": self.manifest,
            "qi_report": self.qi_report,
        }


@dataclass
class _GroupSummary:
    index: int
    rows: int = 0
    batch_count: int = 0
    processed_cells: int = 0
    redacted_cells: int = 0
    total_spans: int = 0
    per_label_counts: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "rows": self.rows,
            "batch_count": self.batch_count,
            "processed_cells": self.processed_cells,
            "redacted_cells": self.redacted_cells,
            "total_spans": self.total_spans,
            "per_label_counts": dict(sorted(self.per_label_counts.items())),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "_GroupSummary":
        summary = cls(index=int(data["index"]))
        summary.rows = int(data.get("rows", 0))
        summary.batch_count = int(data.get("batch_count", 0))
        summary.processed_cells = int(data.get("processed_cells", 0))
        summary.redacted_cells = int(data.get("redacted_cells", 0))
        summary.total_spans = int(data.get("total_spans", 0))
        summary.per_label_counts.update(
            {
                str(key): int(value)
                for key, value in data.get("per_label_counts", {}).items()
            }
        )
        return summary


@dataclass
class _ManifestAccumulator:
    input_format: ColumnarFormat
    group_kind: GroupKind
    text_columns: tuple[str, ...]
    quasi_identifier_columns: tuple[str, ...]
    group_count: int
    total_rows_expected: int
    low_k_threshold: int
    rows_processed: int = 0
    processed_cells: int = 0
    redacted_cells: int = 0
    total_spans: int = 0
    peak_record_batch_rows: int = 0
    per_label_counts: Counter[str] = field(default_factory=Counter)
    groups: dict[int, _GroupSummary] = field(default_factory=dict)

    def add_group(self, summary: _GroupSummary) -> None:
        self.groups[summary.index] = summary
        self.rows_processed += summary.rows
        self.processed_cells += summary.processed_cells
        self.redacted_cells += summary.redacted_cells
        self.total_spans += summary.total_spans
        self.per_label_counts.update(summary.per_label_counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "input_format": self.input_format,
            "group_kind": self.group_kind,
            "row_group_count": self.group_count,
            "total_rows_expected": self.total_rows_expected,
            "total_rows": self.rows_processed,
            "text_columns": list(self.text_columns),
            "quasi_identifier_columns": list(self.quasi_identifier_columns),
            "processed_cells": self.processed_cells,
            "redacted_cells": self.redacted_cells,
            "total_spans": self.total_spans,
            "per_label_counts": dict(sorted(self.per_label_counts.items())),
            "peak_record_batch_rows": self.peak_record_batch_rows,
            "low_k_threshold": self.low_k_threshold,
            "groups": [self.groups[index].to_dict() for index in sorted(self.groups)],
            "raw_cell_values_included": False,
        }


class _QIRiskAccumulator:
    def __init__(
        self,
        quasi_identifier_columns: Sequence[str],
        *,
        low_k_threshold: int,
    ) -> None:
        self.columns = tuple(quasi_identifier_columns)
        self.low_k_threshold = low_k_threshold
        self.record_count = 0
        self.class_counts: Counter[str] = Counter()

    def add_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            self.record_count += 1
            self.class_counts[self._class_digest(row)] += 1

    def to_report(self) -> dict[str, Any]:
        sizes = list(self.class_counts.values())
        distribution = Counter(sizes)
        k_min = min(sizes) if sizes else 0
        low_sizes = [size for size in sizes if size < self.low_k_threshold]
        singleton_count = distribution.get(1, 0)
        risk_flags: list[dict[str, Any]] = []
        if low_sizes:
            risk_flags.append(
                {
                    "type": "low_equivalence_class_size",
                    "threshold": self.low_k_threshold,
                    "class_count": len(low_sizes),
                    "record_count": sum(low_sizes),
                    "min_class_size": min(low_sizes),
                }
            )
        return {
            "schema_version": 1,
            "record_count": self.record_count,
            "quasi_identifier_columns": list(self.columns),
            "k_min": k_min,
            "class_count": len(self.class_counts),
            "class_size_distribution": [
                [size, distribution[size]] for size in sorted(distribution)
            ],
            "low_k_threshold": self.low_k_threshold,
            "low_k_class_count": len(low_sizes),
            "low_k_record_count": sum(low_sizes),
            "singleton_class_count": singleton_count,
            "singleton_record_count": singleton_count,
            "risk_flags": risk_flags,
            "normalization": "openmed.risk.reid quasi-identifier normalization",
            "raw_cell_values_included": False,
        }

    def _class_digest(self, row: Mapping[str, Any]) -> str:
        values = [
            [column, _normalize_columnar_qi_value(column, row.get(column))]
            for column in self.columns
        ]
        payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


def redact_columnar_dataset(
    path: str | Path,
    text_columns: Sequence[str],
    *,
    output_path: str | Path | None = None,
    quasi_identifier_columns: Sequence[str] | None = None,
    manifest_path: str | Path | None = None,
    qi_report_path: str | Path | None = None,
    progress_path: str | Path | None = None,
    resume: bool = False,
    low_k_threshold: int = 5,
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
) -> ColumnarRedactionResult:
    """Redact free-text columns and profile QI risk in Parquet/ORC files.

    Args:
        path: Input ``.parquet`` or ``.orc`` file.
        text_columns: Column names to de-identify through ``process_batch``.
        output_path: Destination file. Defaults to
            ``<input-stem>.redacted<suffix>``.
        quasi_identifier_columns: Structured columns to profile. When omitted,
            every non-text column is profiled.
        manifest_path: Optional JSON file for the PHI-free redaction manifest.
        qi_report_path: Optional JSON file for the PHI-free QI report.
        progress_path: Optional JSON checkpoint updated after each group.
        resume: Reuse completed checkpointed row-group/stripe parts when present.
        low_k_threshold: Equivalence classes smaller than this threshold are
            flagged in the QI report.
        model_name: PII detection model forwarded to ``process_batch``.
        method: De-identification method forwarded to ``process_batch``.
        policy: Optional de-identification policy.
        confidence_threshold: Minimum confidence for redaction.
        config: Optional OpenMed configuration.
        lang: Language hint.
        keep_year: Keep year in date redaction where applicable.
        date_shift_days: Optional fixed date shift.
        use_safety_sweep: Enable deterministic structured-identifier sweep.
        batch_size: Maximum row batch size and redaction batch size.
        on_progress: Optional PHI-free progress callback.
        process_batch_fn: Test seam for replacing ``process_batch``.

    Returns:
        ``ColumnarRedactionResult`` with output path, manifest, and QI report.
    """
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Columnar input not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Columnar input path must be a file: {input_path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if low_k_threshold <= 0:
        raise ValueError("low_k_threshold must be positive")

    input_format = _infer_format(input_path)
    destination = (
        Path(output_path)
        if output_path is not None
        else input_path.with_name(f"{input_path.stem}.redacted{input_path.suffix}")
    )
    if input_path.resolve() == destination.resolve():
        raise ValueError("output_path must not overwrite the input dataset")

    pyarrow, parquet, orc = _import_pyarrow()
    metadata = _input_metadata(input_format, input_path, parquet, orc)
    normalized_text_columns = _normalize_columns(text_columns, name="text_columns")
    _validate_columns(metadata.schema_names, normalized_text_columns, kind="text")
    _validate_text_column_types(metadata.schema, normalized_text_columns, pyarrow)

    if quasi_identifier_columns is None:
        normalized_qi_columns = tuple(
            column
            for column in metadata.schema_names
            if column not in normalized_text_columns
        )
    else:
        normalized_qi_columns = _normalize_columns(
            quasi_identifier_columns,
            name="quasi_identifier_columns",
            allow_empty=True,
        )
        _validate_columns(
            metadata.schema_names,
            normalized_qi_columns,
            kind="quasi-identifier",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    part_dir = _part_dir(destination)
    checkpoint = Path(progress_path) if progress_path is not None else None
    if not resume:
        if part_dir.exists():
            shutil.rmtree(part_dir)
        if checkpoint is not None and checkpoint.exists():
            checkpoint.unlink()
    part_dir.mkdir(parents=True, exist_ok=True)

    state = _load_progress(checkpoint) if resume and checkpoint is not None else {}
    completed_groups = {
        int(index)
        for index in state.get("completed_groups", [])
        if str(index).isdigit()
    }
    group_summaries = {
        int(index): _GroupSummary.from_mapping(group)
        for index, group in state.get("groups", {}).items()
        if str(index).isdigit()
    }

    manifest = _ManifestAccumulator(
        input_format=input_format,
        group_kind=metadata.group_kind,
        text_columns=normalized_text_columns,
        quasi_identifier_columns=normalized_qi_columns,
        group_count=metadata.group_count,
        total_rows_expected=metadata.total_rows,
        low_k_threshold=low_k_threshold,
    )
    qi = _QIRiskAccumulator(
        normalized_qi_columns,
        low_k_threshold=low_k_threshold,
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

    reader = metadata.reader
    for group_index in range(metadata.group_count):
        part_path = _part_path(part_dir, input_format, group_index)
        if (
            resume
            and group_index in completed_groups
            and group_index in group_summaries
            and part_path.exists()
        ):
            _restore_qi_from_part(
                input_format,
                part_path,
                qi,
                metadata.schema,
                batch_size,
                parquet,
                orc,
                pyarrow,
            )
            group_summary = group_summaries[group_index]
            manifest.add_group(group_summary)
            _emit_progress(
                on_progress,
                input_format=input_format,
                group_kind=metadata.group_kind,
                group_index=group_index,
                group_count=metadata.group_count,
                rows_in_group=group_summary.rows,
                rows_completed=manifest.rows_processed,
                total_rows=metadata.total_rows,
                resumed=True,
            )
            continue

        group_summary = _process_group(
            input_format,
            reader,
            group_index,
            part_path,
            schema=metadata.schema,
            text_columns=normalized_text_columns,
            manifest=manifest,
            qi=qi,
            model_name=model_name,
            redaction_kwargs=redaction_kwargs,
            process_batch_fn=batch_fn,
            batch_size=batch_size,
            pyarrow=pyarrow,
            parquet=parquet,
            orc=orc,
        )
        manifest.add_group(group_summary)
        group_summaries[group_index] = group_summary
        completed_groups.add(group_index)
        _write_progress(
            checkpoint,
            input_path=input_path,
            output_path=destination,
            input_format=input_format,
            group_kind=metadata.group_kind,
            group_count=metadata.group_count,
            completed_groups=completed_groups,
            groups=group_summaries,
            rows_completed=manifest.rows_processed,
            total_rows=metadata.total_rows,
        )
        _emit_progress(
            on_progress,
            input_format=input_format,
            group_kind=metadata.group_kind,
            group_index=group_index,
            group_count=metadata.group_count,
            rows_in_group=group_summary.rows,
            rows_completed=manifest.rows_processed,
            total_rows=metadata.total_rows,
            resumed=False,
        )

    _combine_parts(
        input_format,
        destination,
        part_dir,
        metadata.schema,
        metadata.group_count,
        batch_size,
        pyarrow,
        parquet,
        orc,
    )
    if part_dir.exists():
        shutil.rmtree(part_dir)

    manifest_payload = manifest.to_dict()
    qi_payload = qi.to_report()
    manifest_destination = Path(manifest_path) if manifest_path is not None else None
    qi_destination = Path(qi_report_path) if qi_report_path is not None else None
    if manifest_destination is not None:
        _write_json(manifest_destination, manifest_payload)
    if qi_destination is not None:
        _write_json(qi_destination, qi_payload)

    return ColumnarRedactionResult(
        output_path=destination,
        manifest=manifest_payload,
        qi_report=qi_payload,
        manifest_path=manifest_destination,
        qi_report_path=qi_destination,
        progress_path=checkpoint,
    )


redact_columnar = redact_columnar_dataset


@dataclass(frozen=True)
class _InputMetadata:
    input_format: ColumnarFormat
    group_kind: GroupKind
    schema: Any
    schema_names: tuple[str, ...]
    group_count: int
    total_rows: int
    reader: Any


def _infer_format(path: Path) -> ColumnarFormat:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".orc":
        return "orc"
    raise ValueError("Unsupported columnar format. Expected .parquet or .orc.")


def _import_pyarrow() -> tuple[Any, Any, Any]:
    try:
        import pyarrow as pyarrow
        import pyarrow.orc as orc
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Columnar redaction requires pyarrow. Install openmed[columnar] "
            "or install pyarrow directly."
        ) from exc
    return pyarrow, parquet, orc


def _input_metadata(
    input_format: ColumnarFormat,
    input_path: Path,
    parquet: Any,
    orc: Any,
) -> _InputMetadata:
    if input_format == "parquet":
        reader = parquet.ParquetFile(input_path)
        schema = reader.schema_arrow
        return _InputMetadata(
            input_format=input_format,
            group_kind="row_group",
            schema=schema,
            schema_names=tuple(schema.names),
            group_count=reader.num_row_groups,
            total_rows=int(reader.metadata.num_rows),
            reader=reader,
        )

    reader = orc.ORCFile(input_path)
    schema = reader.schema
    return _InputMetadata(
        input_format=input_format,
        group_kind="stripe",
        schema=schema,
        schema_names=tuple(schema.names),
        group_count=reader.nstripes,
        total_rows=int(reader.nrows),
        reader=reader,
    )


def _normalize_columns(
    columns: Sequence[str],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for column in columns:
        value = str(column).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    if not normalized and not allow_empty:
        raise ValueError(f"At least one {name} value is required")
    return tuple(normalized)


def _validate_columns(
    available: Sequence[str],
    columns: Sequence[str],
    *,
    kind: str,
) -> None:
    missing = [column for column in columns if column not in available]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing {kind} column(s): {joined}")


def _validate_text_column_types(
    schema: Any,
    text_columns: Sequence[str],
    pyarrow: Any,
) -> None:
    for column in text_columns:
        field_type = schema.field(column).type
        if not (
            pyarrow.types.is_string(field_type)
            or pyarrow.types.is_large_string(field_type)
            or pyarrow.types.is_null(field_type)
        ):
            raise ValueError(f"Columnar text column must be string-typed: {column}")


def _process_group(
    input_format: ColumnarFormat,
    reader: Any,
    group_index: int,
    part_path: Path,
    *,
    schema: Any,
    text_columns: Sequence[str],
    manifest: _ManifestAccumulator,
    qi: _QIRiskAccumulator,
    model_name: str,
    redaction_kwargs: Mapping[str, Any],
    process_batch_fn: ProcessBatchCallable,
    batch_size: int,
    pyarrow: Any,
    parquet: Any,
    orc: Any,
) -> _GroupSummary:
    summary = _GroupSummary(index=group_index)
    if input_format == "parquet":
        writer = parquet.ParquetWriter(part_path, schema)
        try:
            for record_batch in reader.iter_batches(
                batch_size=batch_size,
                row_groups=[group_index],
            ):
                rows = pyarrow.Table.from_batches(
                    [record_batch], schema=schema
                ).to_pylist()
                _process_rows(
                    rows,
                    text_columns,
                    summary,
                    qi,
                    model_name,
                    redaction_kwargs,
                    process_batch_fn,
                )
                manifest.peak_record_batch_rows = max(
                    manifest.peak_record_batch_rows,
                    len(rows),
                )
                writer.write_table(pyarrow.Table.from_pylist(rows, schema=schema))
        finally:
            writer.close()
        return summary

    writer = orc.ORCWriter(part_path, batch_size=batch_size)
    try:
        for record_batch in _stripe_batches(
            reader.read_stripe(group_index), batch_size
        ):
            rows = pyarrow.Table.from_batches([record_batch], schema=schema).to_pylist()
            _process_rows(
                rows,
                text_columns,
                summary,
                qi,
                model_name,
                redaction_kwargs,
                process_batch_fn,
            )
            manifest.peak_record_batch_rows = max(
                manifest.peak_record_batch_rows,
                len(rows),
            )
            writer.write(pyarrow.Table.from_pylist(rows, schema=schema))
    finally:
        writer.close()
    return summary


def _process_rows(
    rows: list[dict[str, Any]],
    text_columns: Sequence[str],
    summary: _GroupSummary,
    qi: _QIRiskAccumulator,
    model_name: str,
    redaction_kwargs: Mapping[str, Any],
    process_batch_fn: ProcessBatchCallable,
) -> None:
    text_values: list[str] = []
    ids: list[str] = []
    cells: list[tuple[int, str, str]] = []
    for row_index, row in enumerate(rows):
        for column in text_columns:
            value = row.get(column)
            if value is None:
                continue
            text = value if isinstance(value, str) else str(value)
            if text == "":
                continue
            ids.append(
                f"row_{summary.index}_{summary.batch_count}_{row_index}_{column}"
            )
            text_values.append(text)
            cells.append((row_index, column, text))

    if text_values:
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
        for (row_index, column, original_text), item in zip(cells, items):
            if getattr(item, "error", None):
                raise RuntimeError(
                    f"Columnar redaction failed for text column {column!r}: "
                    f"{item.error}"
                )
            cell_result = getattr(item, "result", item)
            redacted = _deidentified_text(cell_result)
            rows[row_index][column] = redacted
            _update_summary(summary, original_text, cell_result, redacted)

    qi.add_rows(rows)
    summary.rows += len(rows)
    summary.batch_count += 1


def _deidentified_text(result: Any) -> str:
    if hasattr(result, "deidentified_text"):
        return str(result.deidentified_text)
    if isinstance(result, Mapping) and "deidentified_text" in result:
        return str(result["deidentified_text"])
    if isinstance(result, str):
        return result
    raise TypeError("process_batch results must expose deidentified_text")


def _update_summary(
    summary: _GroupSummary,
    original_text: str,
    result: Any,
    redacted_text: str,
) -> None:
    summary.processed_cells += 1
    if redacted_text != original_text:
        summary.redacted_cells += 1

    label_counts: Counter[str] = Counter()
    for entity in _entities(result):
        label = _entity_label(entity)
        label_counts[label] += 1
    summary.per_label_counts.update(label_counts)
    summary.total_spans += sum(label_counts.values())


def _entities(result: Any) -> Sequence[Any]:
    if hasattr(result, "pii_entities"):
        return tuple(result.pii_entities or ())
    if isinstance(result, Mapping):
        value = result.get("pii_entities", ())
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
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


def _normalize_columnar_qi_value(column: str, value: Any) -> str:
    if value is None:
        return _MISSING_VALUE
    normalized = _normalize_qi_value(column, value)
    return normalized if normalized else _EMPTY_VALUE


def _part_dir(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.openmed-parts")


def _part_path(part_dir: Path, input_format: ColumnarFormat, group_index: int) -> Path:
    return part_dir / f"part-{group_index:06d}.{input_format}"


def _write_progress(
    path: Path | None,
    *,
    input_path: Path,
    output_path: Path,
    input_format: ColumnarFormat,
    group_kind: GroupKind,
    group_count: int,
    completed_groups: set[int],
    groups: Mapping[int, _GroupSummary],
    rows_completed: int,
    total_rows: int,
) -> None:
    if path is None:
        return
    payload = {
        "schema_version": _PROGRESS_SCHEMA_VERSION,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_format": input_format,
        "group_kind": group_kind,
        "group_count": group_count,
        "completed_groups": sorted(completed_groups),
        "rows_completed": rows_completed,
        "total_rows": total_rows,
        "groups": {str(index): groups[index].to_dict() for index in sorted(groups)},
        "raw_cell_values_included": False,
    }
    _write_json(path, payload)


def _load_progress(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", 0)) != _PROGRESS_SCHEMA_VERSION:
        raise ValueError("Unsupported columnar redaction progress schema")
    return data


def _emit_progress(callback: ProgressCallback | None, **kwargs: Any) -> None:
    if callback is not None:
        callback(ColumnarProgress(**kwargs))


def _restore_qi_from_part(
    input_format: ColumnarFormat,
    part_path: Path,
    qi: _QIRiskAccumulator,
    schema: Any,
    batch_size: int,
    parquet: Any,
    orc: Any,
    pyarrow: Any,
) -> None:
    for rows in _iter_part_rows(
        input_format,
        part_path,
        schema,
        batch_size,
        parquet,
        orc,
        pyarrow,
    ):
        qi.add_rows(rows)


def _combine_parts(
    input_format: ColumnarFormat,
    output_path: Path,
    part_dir: Path,
    schema: Any,
    group_count: int,
    batch_size: int,
    pyarrow: Any,
    parquet: Any,
    orc: Any,
) -> None:
    if input_format == "parquet":
        writer = parquet.ParquetWriter(output_path, schema)
        try:
            for group_index in range(group_count):
                part_path = _part_path(part_dir, input_format, group_index)
                for rows in _iter_part_rows(
                    input_format,
                    part_path,
                    schema,
                    batch_size,
                    parquet,
                    orc,
                    pyarrow,
                ):
                    writer.write_table(pyarrow.Table.from_pylist(rows, schema=schema))
        finally:
            writer.close()
        return

    writer = orc.ORCWriter(output_path, batch_size=batch_size)
    try:
        wrote = False
        for group_index in range(group_count):
            part_path = _part_path(part_dir, input_format, group_index)
            for rows in _iter_part_rows(
                input_format,
                part_path,
                schema,
                batch_size,
                parquet,
                orc,
                pyarrow,
            ):
                writer.write(pyarrow.Table.from_pylist(rows, schema=schema))
                wrote = True
        if not wrote:
            writer.write(pyarrow.Table.from_pylist([], schema=schema))
    finally:
        writer.close()


def _iter_part_rows(
    input_format: ColumnarFormat,
    part_path: Path,
    schema: Any,
    batch_size: int,
    parquet: Any,
    orc: Any,
    pyarrow: Any,
) -> Iterator[list[dict[str, Any]]]:
    if input_format == "parquet":
        reader = parquet.ParquetFile(part_path)
        for batch in reader.iter_batches(batch_size=batch_size):
            yield pyarrow.Table.from_batches([batch], schema=schema).to_pylist()
        return

    reader = orc.ORCFile(part_path)
    for stripe_index in range(reader.nstripes):
        for batch in _stripe_batches(reader.read_stripe(stripe_index), batch_size):
            yield pyarrow.Table.from_batches([batch], schema=schema).to_pylist()


def _stripe_batches(stripe: Any, batch_size: int) -> Sequence[Any]:
    to_batches = getattr(stripe, "to_batches", None)
    if callable(to_batches):
        return tuple(to_batches(max_chunksize=batch_size))

    if hasattr(stripe, "num_rows"):
        batches = []
        for offset in range(0, stripe.num_rows, batch_size):
            batches.append(stripe.slice(offset, batch_size))
        return tuple(batches)

    raise TypeError("ORC stripe reads must return a Table or RecordBatch")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
