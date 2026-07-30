"""Dask DataFrame accessors for OpenMed de-identification workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import RLock
from typing import Any

try:
    import dask.dataframe as dd
    import pandas as pd
    from dask.dataframe.extensions import (
        register_dataframe_accessor,
        register_series_accessor,
    )
except ImportError as exc:  # pragma: no cover - exercised by packaging users
    raise ImportError(
        "Dask DataFrame support requires the 'dask' extra. "
        "Install with `pip install openmed[dask]`."
    ) from exc

ProcessBatch = Callable[..., Any]

_PROCESS_BATCH_CACHE: dict[str, ProcessBatch] = {}
_CACHE_LOCK = RLock()


def map_partitions_deidentify(
    collection: Any,
    target_columns: Sequence[str] | str | None = None,
    *,
    columns: Sequence[str] | str | None = None,
    method: str = "mask",
    policy: str | None = None,
    model_name: str = "disease_detection_superclinical",
    process_batch_fn: ProcessBatch | None = None,
    meta: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Return a Dask collection with selected text values de-identified.

    Args:
        collection: A ``dask.dataframe.DataFrame`` or ``Series``.
        target_columns: Column name or names to redact for DataFrames. Series
            inputs redact the series itself and should leave this unset.
        columns: Alias for ``target_columns``.
        method: De-identification method forwarded to ``process_batch``.
        policy: Optional policy profile forwarded to ``process_batch``.
        model_name: OpenMed model registry key or model identifier.
        process_batch_fn: Optional replacement for
            :func:`openmed.processing.process_batch`, primarily for tests.
        meta: Optional Dask metadata. Defaults to the input collection metadata.
        **kwargs: Additional keyword arguments forwarded to ``process_batch``.

    Returns:
        A Dask DataFrame or Series with the same metadata and divisions.
    """

    selected_columns = _resolve_target_columns(target_columns, columns)
    meta = collection._meta.copy() if meta is None else meta

    if isinstance(collection, dd.DataFrame):
        selected = _validate_dataframe_columns(collection, selected_columns)
        return collection.map_partitions(
            _deidentify_dataframe_partition,
            selected,
            method,
            policy,
            model_name,
            process_batch_fn,
            kwargs,
            meta=meta,
        )

    if isinstance(collection, dd.Series):
        if selected_columns is not None:
            raise ValueError("target_columns is only valid for Dask DataFrames")
        return collection.map_partitions(
            _deidentify_series_partition,
            method,
            policy,
            model_name,
            process_batch_fn,
            kwargs,
            meta=meta,
        )

    raise TypeError("collection must be a dask DataFrame or Series")


deidentify_partitions = map_partitions_deidentify


@register_dataframe_accessor("deid")
class OpenMedDaskDataFrameAccessor:
    """OpenMed helpers attached to ``dask.dataframe.DataFrame.deid``."""

    def __init__(self, dask_obj: Any) -> None:
        self._obj = dask_obj

    def deidentify(
        self,
        target_columns: Sequence[str] | str | None = None,
        *,
        columns: Sequence[str] | str | None = None,
        method: str = "mask",
        policy: str | None = None,
        model_name: str = "disease_detection_superclinical",
        process_batch_fn: ProcessBatch | None = None,
        meta: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return a redacted Dask DataFrame for selected free-text columns."""

        return map_partitions_deidentify(
            self._obj,
            target_columns,
            columns=columns,
            method=method,
            policy=policy,
            model_name=model_name,
            process_batch_fn=process_batch_fn,
            meta=meta,
            **kwargs,
        )

    def map_partitions(
        self,
        target_columns: Sequence[str] | str | None = None,
        *,
        columns: Sequence[str] | str | None = None,
        method: str = "mask",
        policy: str | None = None,
        model_name: str = "disease_detection_superclinical",
        process_batch_fn: ProcessBatch | None = None,
        meta: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Alias for :meth:`deidentify` named for Dask partition workflows."""

        return self.deidentify(
            target_columns,
            columns=columns,
            method=method,
            policy=policy,
            model_name=model_name,
            process_batch_fn=process_batch_fn,
            meta=meta,
            **kwargs,
        )


@register_series_accessor("deid")
class OpenMedDaskSeriesAccessor:
    """OpenMed helpers attached to ``dask.dataframe.Series.deid``."""

    def __init__(self, dask_obj: Any) -> None:
        self._obj = dask_obj

    def deidentify(
        self,
        *,
        method: str = "mask",
        policy: str | None = None,
        model_name: str = "disease_detection_superclinical",
        process_batch_fn: ProcessBatch | None = None,
        meta: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return a redacted Dask Series."""

        return map_partitions_deidentify(
            self._obj,
            method=method,
            policy=policy,
            model_name=model_name,
            process_batch_fn=process_batch_fn,
            meta=meta,
            **kwargs,
        )

    def map_partitions(
        self,
        *,
        method: str = "mask",
        policy: str | None = None,
        model_name: str = "disease_detection_superclinical",
        process_batch_fn: ProcessBatch | None = None,
        meta: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Alias for :meth:`deidentify` named for Dask partition workflows."""

        return self.deidentify(
            method=method,
            policy=policy,
            model_name=model_name,
            process_batch_fn=process_batch_fn,
            meta=meta,
            **kwargs,
        )


def _deidentify_dataframe_partition(
    partition: Any,
    target_columns: tuple[str, ...],
    method: str,
    policy: str | None,
    model_name: str,
    process_batch_fn: ProcessBatch | None,
    process_kwargs: dict[str, Any],
) -> Any:
    redacted = partition.copy(deep=True)
    if redacted.empty:
        return redacted

    texts: list[str] = []
    positions: list[tuple[int, int]] = []
    column_positions = {
        column: redacted.columns.get_loc(column) for column in target_columns
    }

    for column in target_columns:
        column_position = column_positions[column]
        for row_position, value in enumerate(redacted[column].tolist()):
            if _should_deidentify(value):
                texts.append(value)
                positions.append((row_position, column_position))

    if not texts:
        return redacted

    replacements = _run_partition_batch(
        texts,
        method=method,
        policy=policy,
        model_name=model_name,
        process_batch_fn=process_batch_fn,
        process_kwargs=process_kwargs,
    )
    for replacement, (row_position, column_position) in zip(replacements, positions):
        redacted.iat[row_position, column_position] = replacement

    return redacted


def _deidentify_series_partition(
    partition: Any,
    method: str,
    policy: str | None,
    model_name: str,
    process_batch_fn: ProcessBatch | None,
    process_kwargs: dict[str, Any],
) -> Any:
    redacted = partition.copy(deep=True)
    if redacted.empty:
        return redacted

    texts: list[str] = []
    positions: list[int] = []
    for row_position, value in enumerate(redacted.tolist()):
        if _should_deidentify(value):
            texts.append(value)
            positions.append(row_position)

    if not texts:
        return redacted

    replacements = _run_partition_batch(
        texts,
        method=method,
        policy=policy,
        model_name=model_name,
        process_batch_fn=process_batch_fn,
        process_kwargs=process_kwargs,
    )
    for replacement, row_position in zip(replacements, positions):
        redacted.iloc[row_position] = replacement

    return redacted


def _run_partition_batch(
    texts: Sequence[str],
    *,
    method: str,
    policy: str | None,
    model_name: str,
    process_batch_fn: ProcessBatch | None,
    process_kwargs: dict[str, Any],
) -> list[str]:
    process_batch = _get_process_batch(process_batch_fn)
    kwargs = dict(process_kwargs)
    kwargs["operation"] = "deidentify"
    kwargs["method"] = method
    kwargs["model_name"] = model_name
    kwargs.setdefault("batch_size", max(1, len(texts)))
    if policy is not None:
        kwargs["policy"] = policy

    batch_result = process_batch(list(texts), **kwargs)
    return _deidentified_texts(batch_result, expected=len(texts))


def _get_process_batch(process_batch_fn: ProcessBatch | None) -> ProcessBatch:
    if process_batch_fn is not None:
        return process_batch_fn

    cache_key = "openmed.processing.process_batch"
    with _CACHE_LOCK:
        if cache_key not in _PROCESS_BATCH_CACHE:
            from openmed.processing import process_batch

            _PROCESS_BATCH_CACHE[cache_key] = process_batch
        return _PROCESS_BATCH_CACHE[cache_key]


def _deidentified_texts(batch_result: Any, *, expected: int) -> list[str]:
    items = getattr(batch_result, "items", batch_result)
    if len(items) != expected:
        raise ValueError(
            f"process_batch returned {len(items)} results for {expected} inputs"
        )

    return [_deidentified_text(item) for item in items]


def _deidentified_text(item: Any) -> str:
    if hasattr(item, "success") and not item.success:
        raise RuntimeError("Dask de-identification failed for one or more cells")

    result = getattr(item, "result", item)
    if isinstance(result, str):
        return result

    try:
        return str(result.deidentified_text)
    except AttributeError as exc:
        raise TypeError(
            "process_batch results must contain strings or deidentified_text"
        ) from exc


def _resolve_target_columns(
    target_columns: Sequence[str] | str | None,
    columns: Sequence[str] | str | None,
) -> tuple[str, ...] | None:
    if target_columns is not None and columns is not None:
        raise ValueError("pass only one of target_columns or columns")
    selected = target_columns if target_columns is not None else columns
    if selected is None:
        return None
    return _normalize_columns(selected)


def _validate_dataframe_columns(
    collection: Any,
    target_columns: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if target_columns is None:
        raise ValueError("target_columns is required for Dask DataFrames")

    missing = [column for column in target_columns if column not in collection.columns]
    if missing:
        raise KeyError(f"DataFrame is missing columns: {', '.join(missing)}")
    return target_columns


def _normalize_columns(columns: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(columns, str):
        normalized = (columns,)
    else:
        normalized = tuple(str(column) for column in columns)

    if not normalized:
        raise ValueError("target_columns must include at least one column name")
    return normalized


def _should_deidentify(value: Any) -> bool:
    if not isinstance(value, str) or value == "":
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def clear_worker_pipeline_cache() -> None:
    """Clear the per-process process_batch cache used by Dask workers."""

    with _CACHE_LOCK:
        _PROCESS_BATCH_CACHE.clear()


__all__ = [
    "OpenMedDaskDataFrameAccessor",
    "OpenMedDaskSeriesAccessor",
    "ProcessBatch",
    "clear_worker_pipeline_cache",
    "deidentify_partitions",
    "map_partitions_deidentify",
]
