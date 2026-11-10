"""Prefect task and flow for batch de-identification of local datasets."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Union

try:
    from prefect import flow, task
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "Prefect support requires the 'prefect' extra. "
        "Install with `pip install openmed[prefect]`."
    ) from exc

DatasetRedactor = Callable[..., Any]
PathLike = Union[str, Path]


@task(name="openmed-deidentify-file")
def deidentify_file_task(
    path: PathLike,
    text_columns: Sequence[str],
    *,
    output_path: PathLike | None = None,
    policy: str | None = None,
    method: str = "mask",
    confidence_threshold: float = 0.7,
    **redact_kwargs: Any,
) -> dict[str, Any]:
    """Redact one local dataset file and return a PHI-free result summary.

    The task wraps :func:`openmed.processing.batch.redact_dataset`, so it
    supports the same ``.csv``, ``.jsonl``/``.ndjson``, and ``.parquet``
    inputs and writes the redacted copy next to the input by default.

    Args:
        path: Input dataset path.
        text_columns: Free-text column names to de-identify.
        output_path: Destination path. Defaults to ``<stem>.redacted<suffix>``.
        policy: Optional de-identification policy profile name.
        method: De-identification method forwarded to ``deidentify``.
        confidence_threshold: Minimum confidence for redaction.
        **redact_kwargs: Additional keyword arguments forwarded to
            :func:`openmed.processing.batch.redact_dataset`.

    Returns:
        PHI-free mapping with ``files_processed``, ``rows_processed``,
        ``cells_redacted``, and ``spans_redacted`` counts usable by
        downstream tasks.
    """
    redact_dataset = _load_redact_dataset()
    result = redact_dataset(
        path,
        text_columns,
        output_path=output_path,
        policy=policy,
        method=method,
        confidence_threshold=confidence_threshold,
        **redact_kwargs,
    )
    summary = result.summary
    return {
        "files_processed": 1,
        "rows_processed": summary.total_rows,
        "cells_redacted": summary.redacted_cells,
        "spans_redacted": summary.total_spans,
    }


@flow(name="openmed-deidentify-dataset")
def deidentify_dataset_flow(
    input_paths: Sequence[PathLike],
    text_columns: Sequence[str],
    *,
    output_dir: PathLike | None = None,
    policy: str | None = None,
    method: str = "mask",
    confidence_threshold: float = 0.7,
    **redact_kwargs: Any,
) -> dict[str, Any]:
    """Fan :func:`deidentify_file_task` over *input_paths* and aggregate.

    Args:
        input_paths: Dataset files to redact, one task run per file.
        text_columns: Free-text column names to de-identify in every file.
        output_dir: Optional directory for redacted copies. When omitted,
            each redacted file is written next to its input.
        policy: Optional de-identification policy profile name.
        method: De-identification method forwarded to ``deidentify``.
        confidence_threshold: Minimum confidence for redaction.
        **redact_kwargs: Additional keyword arguments forwarded to
            :func:`openmed.processing.batch.redact_dataset` for every file.

    Returns:
        PHI-free mapping with aggregate ``files_processed``,
        ``rows_processed``, ``cells_redacted``, and ``spans_redacted``
        counts plus the per-file summaries under ``files``.
    """
    paths = tuple(input_paths)
    output_paths = [
        _resolve_output_path(input_path, output_dir) for input_path in paths
    ]
    _validate_unique_output_paths(output_paths)
    file_summaries = [
        deidentify_file_task(
            input_path,
            text_columns,
            output_path=output_path,
            policy=policy,
            method=method,
            confidence_threshold=confidence_threshold,
            **redact_kwargs,
        )
        for input_path, output_path in zip(paths, output_paths, strict=True)
    ]
    return {
        "files_processed": len(file_summaries),
        "rows_processed": sum(item["rows_processed"] for item in file_summaries),
        "cells_redacted": sum(item["cells_redacted"] for item in file_summaries),
        "spans_redacted": sum(item["spans_redacted"] for item in file_summaries),
        "files": file_summaries,
    }


def _resolve_output_path(
    input_path: PathLike,
    output_dir: PathLike | None,
) -> Path | None:
    if output_dir is None:
        return None
    source = Path(input_path)
    return Path(output_dir) / f"{source.stem}.redacted{source.suffix}"


def _validate_unique_output_paths(output_paths: Sequence[Path | None]) -> None:
    resolved = [path.resolve() for path in output_paths if path is not None]
    if len(resolved) != len(set(resolved)):
        raise ValueError(
            "output_dir produces duplicate redacted output names; "
            "use unique input basenames or omit output_dir"
        )


def _load_redact_dataset() -> DatasetRedactor:
    from openmed.processing.batch import redact_dataset

    return redact_dataset


__all__ = [
    "DatasetRedactor",
    "deidentify_dataset_flow",
    "deidentify_file_task",
]
