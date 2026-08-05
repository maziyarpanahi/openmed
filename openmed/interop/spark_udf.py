"""PySpark pandas_udf adapter for batch column de-identification.

``pyspark`` is imported lazily inside functions, never at module scope, so
importing :mod:`openmed` or :mod:`openmed.interop` never imports it.

Executor model-loading: Spark reuses one Python worker process across many
batches within a partition. The OpenMed model must load once in that worker
process and be reused for every batch it handles, not reloaded per batch or
per row -- see :func:`_cached_model_loader`. The model is never broadcast
from the driver: broadcasting a loaded model forces the driver to hold and
pickle a copy of it, which can exhaust driver memory or fail outright for
models that do not serialize cleanly. Only the small, picklable ``policy``
and keyword-argument closure is shipped to executors; each worker loads its
own model instance locally on first use.

No raw PHI in driver logs: task failures propagate their exception message
back to the driver, which often lands in centralized log aggregation outside
this process's control. Do not interpolate raw input text into exception
messages or log statements anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from importlib import import_module as _import_module
from typing import Any


def _deidentify_series(
    texts: Any,
    *,
    policy: str = "hipaa_safe_harbor",
    deidentifier: Any = None,
    **kwargs: Any,
) -> Any:
    """Return a de-identified copy of the pandas Series *texts*."""

    import pandas as pd

    if deidentifier is None:
        deidentifier = _default_deidentifier()
        kwargs = {**kwargs, "loader": _cached_model_loader()}

    def _redact_one(text: str | None) -> str | None:
        if text is None or pd.isna(text):
            return None
        result = deidentifier(text, policy=policy, **kwargs)
        return _result_text(result)

    return texts.apply(_redact_one)


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return str(result.deidentified_text)
    except AttributeError as exc:
        raise TypeError(
            "deidentifier must return a string or an object with deidentified_text"
        ) from exc


def _default_deidentifier() -> Any:
    from openmed.core.pii import deidentify

    return deidentify


@lru_cache(maxsize=1)
def _cached_model_loader() -> Any:
    """Return one OpenMed :class:`ModelLoader`, cached for this process.

    Spark keeps a Python worker process alive across many UDF batches, so the
    ``lru_cache`` here means the model loads once on the first call in that
    process and every later batch handled by the same worker reuses it. This
    function must only ever run inside a worker (never on the driver, never
    captured by a UDF closure before that closure is serialized), which is
    why callers invoke it lazily from inside :func:`_deidentify_series`.
    """

    from openmed.core import ModelLoader

    return ModelLoader()


def make_deidentify_udf(*, policy: str = "hipaa_safe_harbor", **kwargs: Any) -> Any:
    """Return a pandas_udf that de-identifies a string column.

    The model is loaded lazily, once per Python worker process, via a
    process-local cached loader (see :func:`_cached_model_loader`) -- never on
    the driver and never captured in the UDF closure. Executor-side failures
    must not include raw input text in exception messages or logs, since
    those can propagate to driver logs and downstream log aggregators.
    """

    pandas_udf, StringType = _load_pandas_udf()
    pandas_module = _import_module("pandas")

    def _redact(texts: Any) -> Any:
        return _deidentify_series(texts, policy=policy, **kwargs)

    # PySpark 3.5 infers the pandas UDF type from runtime annotations and
    # rejects ``Any`` or postponed ``"pd.Series"`` annotations. Keep pandas
    # lazy while supplying the concrete types before applying the decorator.
    _redact.__annotations__ = {
        "texts": pandas_module.Series,
        "return": pandas_module.Series,
    }
    return pandas_udf(StringType())(_redact)


def deidentify_columns(
    df: Any,
    columns: Sequence[str],
    *,
    policy: str = "hipaa_safe_harbor",
    **kwargs: Any,
) -> Any:
    """Return *df* with *columns* de-identified via :func:`make_deidentify_udf`."""

    redact = make_deidentify_udf(policy=policy, **kwargs)
    for column in columns:
        df = df.withColumn(column, redact(df[column]))
    return df


def _load_pandas_udf() -> tuple[Any, Any]:
    try:
        functions_module = _import_module("pyspark.sql.functions")
        types_module = _import_module("pyspark.sql.types")
    except ImportError as exc:
        raise ImportError(
            "PySpark support requires the optional dependency; install "
            "openmed[spark] to use openmed.interop.spark_udf"
        ) from exc
    return functions_module.pandas_udf, types_module.StringType
