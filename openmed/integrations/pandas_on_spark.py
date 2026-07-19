"""Pandas API on Spark accessors for distributed de-identification."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import RLock
from typing import Any

try:
    import pandas as pd
    import pyspark.pandas as ps
    from pyspark.pandas.extensions import (
        register_dataframe_accessor,
        register_series_accessor,
    )
except ImportError as exc:  # pragma: no cover - exercised by packaging users
    raise ImportError(
        "Pandas-on-Spark support requires the 'spark' extra. "
        "Install with `pip install openmed[spark]`."
    ) from exc

Deidentifier = Callable[..., Any]

DEFAULT_PANDAS_ON_SPARK_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"

_CACHE_LOCK = RLock()
_PROCESS_BATCH: Deidentifier | None = None
_WORKER_LOADER: Any | None = None
_WORKER_LOADER_INITIALIZED = False


@register_dataframe_accessor("deid")
class OpenMedPandasOnSparkDataFrameAccessor:
    """OpenMed helpers attached to ``pyspark.pandas.DataFrame.deid``."""

    def __init__(self, pandas_obj: Any) -> None:
        self._obj = pandas_obj

    def deidentify(
        self,
        columns: Sequence[str] | str,
        *,
        method: str = "mask",
        policy: str | None = None,
        deidentifier: Deidentifier | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return a distributed DataFrame with selected text columns redacted.

        The signature mirrors the local Pandas accessor. Each selected column
        is transformed through a pandas UDF, and every Arrow row group is sent
        to :func:`openmed.processing.process_batch` as one batch.

        Args:
            columns: Free-text column names to redact.
            method: De-identification method forwarded to ``process_batch``.
            policy: Optional policy profile forwarded to ``process_batch``.
            deidentifier: Optional batch callable used primarily by tests and
                custom executor environments. It must accept a sequence of
                strings and return the ``process_batch`` result shape.
            **kwargs: Additional keyword arguments forwarded to
                ``process_batch``. ``model_name`` may be supplied here.

        Returns:
            A new pandas-on-Spark DataFrame with selected columns redacted.
        """

        selected_columns = _validate_columns(self._obj, columns)
        redacted = self._obj.copy()
        process_batch_fn, process_kwargs = _resolve_process_batch(
            deidentifier,
            kwargs,
        )

        for column in selected_columns:
            redacted[column] = _transform_series(
                redacted[column],
                method=method,
                policy=policy,
                process_batch_fn=process_batch_fn,
                process_kwargs=process_kwargs,
            )

        return redacted


@register_series_accessor("deid")
class OpenMedPandasOnSparkSeriesAccessor:
    """OpenMed helpers attached to ``pyspark.pandas.Series.deid``."""

    def __init__(self, pandas_obj: Any) -> None:
        self._obj = pandas_obj

    def deidentify(
        self,
        *,
        method: str = "mask",
        policy: str | None = None,
        deidentifier: Deidentifier | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return this distributed text Series with identifiers redacted."""

        process_batch_fn, process_kwargs = _resolve_process_batch(
            deidentifier,
            kwargs,
        )
        return _transform_series(
            self._obj,
            method=method,
            policy=policy,
            process_batch_fn=process_batch_fn,
            process_kwargs=process_kwargs,
        )


def _transform_series(
    series: Any,
    *,
    method: str,
    policy: str | None,
    process_batch_fn: Deidentifier | None,
    process_kwargs: dict[str, Any],
) -> Any:
    if not pd.api.types.is_string_dtype(series.dtype):
        raise TypeError("pandas-on-Spark de-identification requires a text column")

    return series.pandas_on_spark.transform_batch(
        _deidentify_series_batch,
        method,
        policy,
        process_batch_fn,
        process_kwargs,
    )


def _deidentify_series_batch(
    batch: Any,
    method: str,
    policy: str | None,
    process_batch_fn: Deidentifier | None,
    process_kwargs: dict[str, Any],
) -> ps.Series[str]:
    """Redact one Arrow-backed pandas Series batch on a Spark worker."""

    redacted = batch.copy(deep=True)
    texts: list[str] = []
    positions: list[int] = []
    for position, value in enumerate(redacted.tolist()):
        if _should_deidentify(value):
            texts.append(value)
            positions.append(position)

    if not texts:
        return redacted

    replacements = _run_worker_batch(
        texts,
        method=method,
        policy=policy,
        process_batch_fn=process_batch_fn,
        process_kwargs=process_kwargs,
    )
    for replacement, position in zip(replacements, positions):
        redacted.iloc[position] = replacement

    return redacted


def _run_worker_batch(
    texts: Sequence[str],
    *,
    method: str,
    policy: str | None,
    process_batch_fn: Deidentifier | None,
    process_kwargs: dict[str, Any],
) -> list[str]:
    batcher = process_batch_fn if process_batch_fn is not None else _get_process_batch()
    kwargs = dict(process_kwargs)
    model_name = str(kwargs.pop("model_name", DEFAULT_PANDAS_ON_SPARK_MODEL)).strip()
    if not model_name:
        raise ValueError("model_name must be a non-empty string")

    kwargs["operation"] = "deidentify"
    kwargs["method"] = method
    kwargs["model_name"] = model_name
    kwargs.setdefault("batch_size", max(1, len(texts)))
    if policy is not None:
        kwargs["policy"] = policy

    if process_batch_fn is None and "loader" not in kwargs and "config" not in kwargs:
        loader = _get_worker_loader()
        if loader is not None:
            kwargs["loader"] = loader

    batch_result = batcher(list(texts), **kwargs)
    return _deidentified_texts(batch_result, expected=len(texts))


def _resolve_process_batch(
    deidentifier: Deidentifier | None,
    kwargs: dict[str, Any],
) -> tuple[Deidentifier | None, dict[str, Any]]:
    process_kwargs = dict(kwargs)
    process_batch_fn = process_kwargs.pop("process_batch_fn", None)
    if deidentifier is not None and process_batch_fn is not None:
        raise ValueError("pass only one of deidentifier or process_batch_fn")
    if deidentifier is not None and not callable(deidentifier):
        raise TypeError("deidentifier must be callable")
    if process_batch_fn is not None and not callable(process_batch_fn):
        raise TypeError("process_batch_fn must be callable")
    return (
        deidentifier if deidentifier is not None else process_batch_fn,
        process_kwargs,
    )


def _get_process_batch() -> Deidentifier:
    global _PROCESS_BATCH

    with _CACHE_LOCK:
        if _PROCESS_BATCH is None:
            from openmed.processing import process_batch

            _PROCESS_BATCH = process_batch
        return _PROCESS_BATCH


def _get_worker_loader() -> Any | None:
    global _WORKER_LOADER, _WORKER_LOADER_INITIALIZED

    with _CACHE_LOCK:
        if _WORKER_LOADER_INITIALIZED:
            return _WORKER_LOADER

        try:
            from openmed.core import ModelLoader

            _WORKER_LOADER = ModelLoader()
        except ImportError:
            _WORKER_LOADER = None
        _WORKER_LOADER_INITIALIZED = True
        return _WORKER_LOADER


def _validate_columns(
    frame: Any,
    columns: Sequence[str] | str,
) -> tuple[str, ...]:
    selected = _normalize_columns(columns)
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise KeyError(f"DataFrame is missing columns: {', '.join(missing)}")
    return selected


def _normalize_columns(columns: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(columns, str):
        normalized = (columns,)
    else:
        normalized = tuple(str(column) for column in columns)

    if not normalized:
        raise ValueError("columns must include at least one column name")
    return normalized


def _should_deidentify(value: Any) -> bool:
    if not isinstance(value, str) or value == "":
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _deidentified_texts(batch_result: Any, *, expected: int) -> list[str]:
    items = getattr(batch_result, "items", batch_result)
    if len(items) != expected:
        raise ValueError(
            f"process_batch returned {len(items)} results for {expected} inputs"
        )
    return [_deidentified_text(item) for item in items]


def _deidentified_text(item: Any) -> str:
    if hasattr(item, "success") and not item.success:
        raise RuntimeError(
            "pandas-on-Spark de-identification failed for one or more cells"
        )

    result = getattr(item, "result", item)
    if isinstance(result, str):
        return result

    try:
        return str(result.deidentified_text)
    except AttributeError as exc:
        raise TypeError(
            "process_batch results must contain strings or deidentified_text"
        ) from exc


def clear_worker_pipeline_cache() -> None:
    """Clear driver-local worker cache state used by deterministic tests."""

    global _PROCESS_BATCH, _WORKER_LOADER, _WORKER_LOADER_INITIALIZED

    with _CACHE_LOCK:
        _PROCESS_BATCH = None
        _WORKER_LOADER = None
        _WORKER_LOADER_INITIALIZED = False


__all__ = [
    "DEFAULT_PANDAS_ON_SPARK_MODEL",
    "Deidentifier",
    "OpenMedPandasOnSparkDataFrameAccessor",
    "OpenMedPandasOnSparkSeriesAccessor",
    "clear_worker_pipeline_cache",
]
