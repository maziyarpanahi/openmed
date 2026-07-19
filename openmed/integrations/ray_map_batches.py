"""Stateful Ray Data batch de-identification with shared model actors.

The callable class in this module is intended for
``ray.data.Dataset.map_batches``. Ray runs callable classes as actors, so each
actor constructs one OpenMed batch pipeline and reuses its loaded model across
all batches assigned to that worker.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol, TypeAlias

DEFAULT_RAY_PII_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"

BatchFormat: TypeAlias = Literal["pandas", "pyarrow"]
ActorPoolConcurrency: TypeAlias = int | tuple[int, int] | tuple[int, int, int]


class BatchPipeline(Protocol):
    """Protocol implemented by actor-local batch processing pipelines."""

    def process_texts(
        self,
        texts: Sequence[str],
        ids: Sequence[str] | None = None,
    ) -> Any:
        """De-identify a batch of text values."""


PipelineFactory: TypeAlias = Callable[..., BatchPipeline]


class RayDeidentifyBatch:
    """Callable Ray Data actor that de-identifies one column per batch.

    Ray constructs this class once per actor. The constructor eagerly builds
    and warms one OpenMed batch pipeline; subsequent ``__call__`` invocations
    reuse the same model resources.

    Args:
        column: Name of the text column to de-identify.
        policy_profile: Optional OpenMed policy profile passed as ``policy`` to
            batch de-identification.
        model_name: OpenMed model registry key or model identifier.
        method: De-identification method, such as ``"mask"``.
        process_batch_size: Maximum number of text values processed together
            inside the actor pipeline.
        process_batch_kwargs: Additional keyword arguments forwarded to the
            actor-local OpenMed batch processor.
        pipeline_factory: Optional factory used to construct the actor-local
            pipeline. This is primarily useful for offline tests.
    """

    def __init__(
        self,
        column: str,
        policy_profile: str | None = None,
        *,
        model_name: str = DEFAULT_RAY_PII_MODEL,
        method: str = "mask",
        process_batch_size: int = 512,
        process_batch_kwargs: Mapping[str, Any] | None = None,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        if not isinstance(column, str) or not column.strip():
            raise ValueError("column must be a non-empty string")

        kwargs = dict(process_batch_kwargs or {})
        reserved = {
            "batch_size",
            "method",
            "model_name",
            "operation",
            "policy",
            "policy_profile",
        }
        conflicts = sorted(reserved.intersection(kwargs))
        if conflicts:
            joined = ", ".join(conflicts)
            raise ValueError(
                "process_batch_kwargs contains reserved argument(s): " + joined
            )

        self.column = column
        self.policy_profile = policy_profile
        self.model_name = model_name
        self.method = method
        factory = pipeline_factory or _create_openmed_batch_pipeline
        self._pipeline = factory(
            model_name=model_name,
            method=method,
            policy_profile=policy_profile,
            process_batch_size=process_batch_size,
            process_batch_kwargs=kwargs,
        )

    def __call__(self, batch: Any) -> Any:
        """Return the batch with the configured text column de-identified."""

        values = _target_values(batch, self.column)
        positions: list[int] = []
        texts: list[str] = []
        for position, value in enumerate(values):
            if _is_missing(value) or value == "":
                continue
            if not isinstance(value, str):
                raise TypeError(
                    f"Ray batch column {self.column!r} must contain strings or nulls"
                )
            positions.append(position)
            texts.append(value)

        redacted_values = list(values)
        if texts:
            ids = [f"ray_batch_{position}" for position in positions]
            result = self._pipeline.process_texts(texts, ids=ids)
            redacted_texts = _deidentified_texts(result, expected=len(texts))
            for position, redacted in zip(positions, redacted_texts):
                redacted_values[position] = redacted

        if len(redacted_values) != len(values):  # pragma: no cover - invariant
            raise RuntimeError("Ray de-identification changed the batch row count")
        return _replace_target_values(batch, self.column, redacted_values)


RayMapBatchesDeidentifier = RayDeidentifyBatch


def map_batches_deidentify(
    dataset: Any,
    *,
    column: str,
    policy_profile: str | None = None,
    model_name: str = DEFAULT_RAY_PII_MODEL,
    method: str = "mask",
    batch_size: int | None = None,
    batch_format: BatchFormat = "pyarrow",
    concurrency: ActorPoolConcurrency = 1,
    process_batch_kwargs: Mapping[str, Any] | None = None,
    pipeline_factory: PipelineFactory | None = None,
    **ray_remote_args: Any,
) -> Any:
    """Apply actor-backed batch de-identification to a Ray Dataset.

    Args:
        dataset: A ``ray.data.Dataset`` instance.
        column: Name of the text column to de-identify.
        policy_profile: Optional OpenMed policy profile.
        model_name: OpenMed model registry key or model identifier.
        method: De-identification method, such as ``"mask"``.
        batch_size: Desired number of dataset rows in each Ray batch.
        batch_format: Either ``"pandas"`` or ``"pyarrow"``.
        concurrency: Fixed actor count, ``(min, max)`` autoscaling range, or
            ``(min, max, initial)`` autoscaling configuration.
        process_batch_kwargs: Extra OpenMed batch-processing arguments.
        pipeline_factory: Optional actor-local pipeline factory, primarily for
            offline tests.
        **ray_remote_args: Resource requirements forwarded to
            ``Dataset.map_batches``, such as ``num_cpus`` or ``num_gpus``.

    Returns:
        A lazy Ray Dataset with the same rows and non-target columns.
    """

    if batch_format not in {"pandas", "pyarrow"}:
        raise ValueError("batch_format must be 'pandas' or 'pyarrow'")
    if batch_size is not None and (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer or None")

    try:
        from ray.data import ActorPoolStrategy
    except ImportError as exc:  # pragma: no cover - packaging users
        raise ImportError(
            "Ray Data support requires Ray. Install with `pip install 'ray[data]'`."
        ) from exc

    compute = _actor_pool_strategy(ActorPoolStrategy, concurrency)
    constructor_kwargs = {
        "column": column,
        "policy_profile": policy_profile,
        "model_name": model_name,
        "method": method,
        "process_batch_size": batch_size or 512,
        "process_batch_kwargs": dict(process_batch_kwargs or {}),
        "pipeline_factory": pipeline_factory,
    }
    return dataset.map_batches(
        RayDeidentifyBatch,
        batch_size=batch_size,
        batch_format=batch_format,
        compute=compute,
        fn_constructor_kwargs=constructor_kwargs,
        zero_copy_batch=True,
        udf_modifying_row_count=False,
        **ray_remote_args,
    )


def _create_openmed_batch_pipeline(
    *,
    model_name: str,
    method: str,
    policy_profile: str | None,
    process_batch_size: int,
    process_batch_kwargs: Mapping[str, Any],
) -> BatchPipeline:
    """Create and eagerly warm one actor-local OpenMed batch pipeline."""

    from openmed.processing.batch import BatchProcessor

    processor = BatchProcessor(
        model_name=model_name,
        operation="deidentify",
        batch_size=process_batch_size,
        method=method,
        policy=policy_profile,
        **dict(process_batch_kwargs),
    )

    privacy_pipeline = processor._get_privacy_filter_pipeline()
    if privacy_pipeline is None:
        loader = processor._get_shared_loader()
        if loader is None:
            raise ImportError(
                "Ray de-identification requires model dependencies. "
                "Install with `pip install 'openmed[hf]'`."
            )

        from openmed.core.pii import _resolve_effective_pii_model

        effective_model = _resolve_effective_pii_model(
            model_name,
            str(processor.analyze_kwargs.get("lang", "en")),
        )
        loader.create_pipeline(
            effective_model,
            task="token-classification",
            aggregation_strategy=processor.aggregation_strategy,
            use_fast_tokenizer=True,
        )

    return processor


def _actor_pool_strategy(
    strategy_type: Callable[..., Any],
    concurrency: ActorPoolConcurrency,
) -> Any:
    if isinstance(concurrency, bool):
        raise ValueError("concurrency must use positive integer actor counts")
    if isinstance(concurrency, int):
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        return strategy_type(size=concurrency)
    if not isinstance(concurrency, tuple) or len(concurrency) not in {2, 3}:
        raise ValueError(
            "concurrency must be an int, (min, max), or (min, max, initial)"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in concurrency
    ):
        raise ValueError("concurrency values must be integers")

    minimum, maximum = concurrency[:2]
    if minimum < 1 or maximum < minimum:
        raise ValueError("concurrency requires 1 <= min <= max")
    if len(concurrency) == 2:
        return strategy_type(min_size=minimum, max_size=maximum)

    initial = concurrency[2]
    if initial < minimum or initial > maximum:
        raise ValueError("initial concurrency must be between min and max")
    return strategy_type(
        min_size=minimum,
        max_size=maximum,
        initial_size=initial,
    )


def _target_values(batch: Any, column: str) -> list[Any]:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - Ray installs pandas
        pd = None  # type: ignore[assignment]
    if pd is not None and isinstance(batch, pd.DataFrame):
        if column not in batch.columns:
            raise KeyError(f"Ray batch is missing column {column!r}")
        return batch[column].tolist()

    try:
        import pyarrow as pa
    except ImportError:  # pragma: no cover - Ray installs PyArrow
        pa = None  # type: ignore[assignment]
    if pa is not None and isinstance(batch, pa.Table):
        if column not in batch.column_names:
            raise KeyError(f"Ray batch is missing column {column!r}")
        return batch.column(column).to_pylist()

    raise TypeError("RayDeidentifyBatch expects a pandas DataFrame or pyarrow Table")


def _replace_target_values(batch: Any, column: str, values: Sequence[Any]) -> Any:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - Ray installs pandas
        pd = None  # type: ignore[assignment]
    if pd is not None and isinstance(batch, pd.DataFrame):
        result = batch.copy(deep=True)
        result[column] = list(values)
        return result

    try:
        import pyarrow as pa
    except ImportError:  # pragma: no cover - Ray installs PyArrow
        pa = None  # type: ignore[assignment]
    if pa is not None and isinstance(batch, pa.Table):
        index = batch.schema.get_field_index(column)
        field = batch.schema.field(index)
        replacement = pa.array(values, type=field.type)
        return batch.set_column(index, field, replacement)

    raise TypeError("RayDeidentifyBatch expects a pandas DataFrame or pyarrow Table")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - Ray installs pandas
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(missing)
    except ValueError:
        return False


def _deidentified_texts(result: Any, *, expected: int) -> list[str]:
    items = getattr(result, "items", result)
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        raise TypeError("OpenMed batch results must contain an item sequence")
    if len(items) != expected:
        raise ValueError(
            f"OpenMed returned {len(items)} result(s) for {expected} input(s)"
        )
    return [_deidentified_text(item) for item in items]


def _deidentified_text(item: Any) -> str:
    if getattr(item, "error", None):
        raise RuntimeError("Ray batch de-identification failed for one or more cells")
    if hasattr(item, "success") and not item.success:
        raise RuntimeError("Ray batch de-identification failed for one or more cells")

    value = getattr(item, "result", item)
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and "deidentified_text" in value:
        return str(value["deidentified_text"])
    if hasattr(value, "deidentified_text"):
        return str(value.deidentified_text)
    raise TypeError("OpenMed batch results must expose deidentified_text")


__all__ = [
    "ActorPoolConcurrency",
    "BatchFormat",
    "DEFAULT_RAY_PII_MODEL",
    "RayDeidentifyBatch",
    "RayMapBatchesDeidentifier",
    "map_batches_deidentify",
]
