"""Spark Structured Streaming sink helpers for OpenMed de-identification."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from openmed.core.policy import PolicyName, canonical_policy_name

DEFAULT_SPARK_POLICY = "hipaa_safe_harbor"
DEFAULT_BATCH_ID_COLUMN = "_openmed_batch_id"


class SparkPartitionDeidentifier(Protocol):
    """Per-partition de-identification interface used by the Spark sink."""

    def deidentify_many(
        self,
        texts: Sequence[str],
        column: "SparkDeidentifyColumn",
    ) -> Sequence[Any]:
        """Return redacted outputs for ``texts`` under ``column`` settings."""


PartitionDeidentifierFactory = Callable[[], SparkPartitionDeidentifier | Callable]
BatchWriter = Callable[[Any, int], Any]


@dataclass(frozen=True)
class SparkDeidentifyColumn:
    """Column-level de-identification settings for a Spark micro-batch."""

    name: str
    policy: str | PolicyName = DEFAULT_SPARK_POLICY
    output_column: str | None = None
    method: str = "mask"
    deidentify_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _non_empty_string(self.name, "name")
        output_column = (
            _non_empty_string(self.output_column, "output_column")
            if self.output_column is not None
            else None
        )
        method = _non_empty_string(self.method, "method")
        policy = canonical_policy_name(self.policy)
        kwargs = dict(self.deidentify_kwargs)
        for reserved in ("method", "policy", "text", "loader"):
            if reserved in kwargs:
                raise ValueError(
                    f"deidentify_kwargs must not include reserved key {reserved!r}"
                )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "output_column", output_column)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "deidentify_kwargs", kwargs)

    @property
    def target_column(self) -> str:
        """Return the column receiving redacted text."""

        return self.output_column or self.name


@dataclass(frozen=True)
class _SparkPartitionConfig:
    columns: tuple[SparkDeidentifyColumn, ...]
    model_name: str
    batch_size: int
    default_deidentify_kwargs: Mapping[str, Any]
    batch_id_column: str | None
    batch_id: int


class _OpenMedPartitionDeidentifier:
    """Default per-partition public API adapter with a shared model loader."""

    def __init__(
        self,
        *,
        model_name: str,
        default_deidentify_kwargs: Mapping[str, Any],
    ) -> None:
        self._model_name = model_name
        self._default_deidentify_kwargs = dict(default_deidentify_kwargs)
        self._loader: Any | None = None
        self._loader_initialized = False

    def deidentify_many(
        self,
        texts: Sequence[str],
        column: SparkDeidentifyColumn,
    ) -> Sequence[Any]:
        """De-identify text values with one reusable loader per partition."""

        from openmed.core.pii import deidentify

        loader = self._get_loader()
        kwargs = {
            **self._default_deidentify_kwargs,
            **dict(column.deidentify_kwargs),
        }
        return [
            deidentify(
                text,
                model_name=self._model_name,
                method=column.method,
                policy=column.policy,
                loader=loader,
                **kwargs,
            )
            for text in texts
        ]

    def _get_loader(self) -> Any | None:
        if self._loader_initialized:
            return self._loader

        self._loader_initialized = True
        try:
            from openmed.core import ModelLoader
        except ImportError:
            self._loader = None
            return None

        try:
            self._loader = ModelLoader()
        except ImportError:
            self._loader = None
        return self._loader


class SparkDeidentifySink:
    """foreachBatch sink that redacts text columns and writes a target table."""

    def __init__(
        self,
        *,
        columns: Sequence[str | Mapping[str, Any] | SparkDeidentifyColumn],
        target_table: str | None = None,
        checkpoint_location: str | None = None,
        model_name: str = "disease_detection_superclinical",
        batch_size: int = 64,
        batch_id_column: str | None = DEFAULT_BATCH_ID_COLUMN,
        output_format: str | None = None,
        write_options: Mapping[str, Any] | None = None,
        partition_by: Sequence[str] | None = None,
        skip_existing_batches: bool = True,
        deidentifier_factory: PartitionDeidentifierFactory | None = None,
        writer: BatchWriter | None = None,
        **deidentify_kwargs: Any,
    ) -> None:
        """Create a Spark Structured Streaming de-identification sink.

        Args:
            columns: Text columns to redact. Entries may be column names,
                dictionaries, or :class:`SparkDeidentifyColumn` instances.
            target_table: Spark table receiving redacted rows.
            checkpoint_location: Optional Structured Streaming checkpoint path.
            model_name: OpenMed model registry key or artifact identifier.
            batch_size: Rows processed together inside each partition.
            batch_id_column: Output metadata column used for replay idempotence.
                Set to ``None`` only when a custom ``writer`` handles replay.
            output_format: Optional DataFrameWriter format for ``saveAsTable``.
            write_options: Extra DataFrameWriter options.
            partition_by: Optional output table partitioning columns.
            skip_existing_batches: Skip a micro-batch when ``target_table``
                already contains the same ``batch_id_column`` value.
            deidentifier_factory: Optional per-partition factory for tests or
                custom runtime adapters. The factory is called once per Spark
                partition and may return an object with ``deidentify_many`` or
                a callable compatible with ``openmed.deidentify``.
            writer: Optional custom writer called as ``writer(df, batch_id)``.
            **deidentify_kwargs: Default kwargs forwarded to de-identification.
        """

        self.columns = _coerce_columns(columns)
        self.target_table = (
            _non_empty_string(target_table, "target_table")
            if target_table is not None
            else None
        )
        self.checkpoint_location = checkpoint_location
        self.model_name = _non_empty_string(model_name, "model_name")
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.batch_id_column = (
            _non_empty_string(batch_id_column, "batch_id_column")
            if batch_id_column is not None
            else None
        )
        self.output_format = output_format
        self.write_options = dict(write_options or {})
        self.partition_by = tuple(partition_by or ())
        self.skip_existing_batches = bool(skip_existing_batches)
        self.deidentifier_factory = deidentifier_factory
        self.writer = writer
        self.default_deidentify_kwargs = dict(deidentify_kwargs)

        if self.target_table is None and self.writer is None:
            raise ValueError("target_table is required when writer is not provided")
        if self.writer is not None and not callable(self.writer):
            raise TypeError("writer must be callable")
        for reserved in ("method", "policy", "text", "loader"):
            if reserved in self.default_deidentify_kwargs:
                raise ValueError(
                    f"deidentify kwargs must not include reserved key {reserved!r}"
                )

    def foreach_batch(self, batch_df: Any, batch_id: int) -> Any:
        """Structured Streaming ``foreachBatch`` callback."""

        return self.process_micro_batch(batch_df, batch_id)

    def process_micro_batch(self, batch_df: Any, batch_id: int) -> Any:
        """Redact and write one static DataFrame micro-batch."""

        batch_id = int(batch_id)
        if self._should_skip_batch(batch_df, batch_id):
            return None

        redacted_df = self.redact_dataframe(batch_df, batch_id)
        self._write(redacted_df, batch_id)
        return redacted_df

    def redact_dataframe(self, batch_df: Any, batch_id: int) -> Any:
        """Return ``batch_df`` with configured columns de-identified."""

        _validate_dataframe_columns(batch_df, self.columns, self.partition_by)
        spark = batch_df.sparkSession
        config = _SparkPartitionConfig(
            columns=self.columns,
            model_name=self.model_name,
            batch_size=self.batch_size,
            default_deidentify_kwargs=self.default_deidentify_kwargs,
            batch_id_column=self.batch_id_column,
            batch_id=int(batch_id),
        )
        broadcast_config = spark.sparkContext.broadcast(config)
        output_schema = _output_schema(
            batch_df.schema, self.columns, self.batch_id_column
        )
        deidentifier_factory = self.deidentifier_factory

        def redact_partition(rows: Iterable[Any]) -> Iterable[dict[str, Any]]:
            return _redact_partition(
                rows,
                broadcast_config.value,
                deidentifier_factory,
            )

        redacted_rdd = batch_df.rdd.mapPartitions(redact_partition)
        return spark.createDataFrame(redacted_rdd, schema=output_schema)

    def start(
        self,
        streaming_df: Any,
        *,
        query_name: str | None = None,
        output_mode: str = "append",
        trigger: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> Any:
        """Start a Structured Streaming query using this sink."""

        writer = streaming_df.writeStream.foreachBatch(self.foreach_batch)
        writer = writer.outputMode(output_mode)
        if self.checkpoint_location:
            writer = writer.option("checkpointLocation", self.checkpoint_location)
        if query_name:
            writer = writer.queryName(query_name)
        for key, value in dict(options or {}).items():
            writer = writer.option(key, value)
        if trigger:
            writer = writer.trigger(**dict(trigger))
        return writer.start()

    def _should_skip_batch(self, batch_df: Any, batch_id: int) -> bool:
        if (
            not self.skip_existing_batches
            or self.target_table is None
            or self.batch_id_column is None
        ):
            return False
        return _target_has_batch(
            batch_df.sparkSession,
            self.target_table,
            self.batch_id_column,
            batch_id,
        )

    def _write(self, redacted_df: Any, batch_id: int) -> None:
        if self.writer is not None:
            self.writer(redacted_df, batch_id)
            return

        writer = redacted_df.write.mode("append")
        if self.output_format:
            writer = writer.format(self.output_format)
        for key, value in self.write_options.items():
            writer = writer.option(key, value)
        if self.partition_by:
            writer = writer.partitionBy(*self.partition_by)
        writer.saveAsTable(self.target_table)


class SparkDeidentifyStreamBuilder:
    """Fluent builder for registering an OpenMed sink with ``writeStream``."""

    def __init__(self, streaming_df: Any) -> None:
        self._streaming_df = streaming_df
        self._columns: tuple[str | Mapping[str, Any] | SparkDeidentifyColumn, ...] = ()
        self._target_table: str | None = None
        self._checkpoint_location: str | None = None
        self._model_name = "disease_detection_superclinical"
        self._batch_size = 64
        self._batch_id_column: str | None = DEFAULT_BATCH_ID_COLUMN
        self._output_format: str | None = None
        self._write_options: dict[str, Any] = {}
        self._partition_by: tuple[str, ...] = ()
        self._skip_existing_batches = True
        self._deidentifier_factory: PartitionDeidentifierFactory | None = None
        self._writer: BatchWriter | None = None
        self._deidentify_kwargs: dict[str, Any] = {}
        self._query_name: str | None = None
        self._output_mode = "append"
        self._trigger: dict[str, Any] | None = None
        self._stream_options: dict[str, Any] = {}

    def columns(
        self,
        *columns: str | Mapping[str, Any] | SparkDeidentifyColumn,
    ) -> "SparkDeidentifyStreamBuilder":
        """Set the text columns to de-identify."""

        if (
            len(columns) == 1
            and isinstance(columns[0], Sequence)
            and not isinstance(
                columns[0],
                (str, bytes, SparkDeidentifyColumn, Mapping),
            )
        ):
            self._columns = tuple(columns[0])  # type: ignore[arg-type]
        else:
            self._columns = tuple(columns)
        return self

    def target_table(self, table: str) -> "SparkDeidentifyStreamBuilder":
        """Set the output Spark table."""

        self._target_table = table
        return self

    def checkpoint(self, location: str) -> "SparkDeidentifyStreamBuilder":
        """Set the Structured Streaming checkpoint path."""

        self._checkpoint_location = location
        return self

    def model_name(self, model_name: str) -> "SparkDeidentifyStreamBuilder":
        """Set the OpenMed model identifier."""

        self._model_name = model_name
        return self

    def batch_size(self, batch_size: int) -> "SparkDeidentifyStreamBuilder":
        """Set the per-partition processing batch size."""

        self._batch_size = batch_size
        return self

    def batch_id_column(self, column: str | None) -> "SparkDeidentifyStreamBuilder":
        """Set or disable the output batch ID column."""

        self._batch_id_column = column
        return self

    def format(self, output_format: str | None) -> "SparkDeidentifyStreamBuilder":
        """Set the output table writer format."""

        self._output_format = output_format
        return self

    def write_option(self, key: str, value: Any) -> "SparkDeidentifyStreamBuilder":
        """Set one DataFrameWriter option for the target table."""

        self._write_options[key] = value
        return self

    def partition_by(self, *columns: str) -> "SparkDeidentifyStreamBuilder":
        """Set output table partitioning columns."""

        self._partition_by = tuple(columns)
        return self

    def skip_existing_batches(self, enabled: bool) -> "SparkDeidentifyStreamBuilder":
        """Control target-table batch replay skipping."""

        self._skip_existing_batches = bool(enabled)
        return self

    def deidentifier_factory(
        self,
        factory: PartitionDeidentifierFactory | None,
    ) -> "SparkDeidentifyStreamBuilder":
        """Set a custom per-partition deidentifier factory."""

        self._deidentifier_factory = factory
        return self

    def writer(self, writer: BatchWriter | None) -> "SparkDeidentifyStreamBuilder":
        """Set a custom micro-batch writer."""

        self._writer = writer
        return self

    def deidentify_option(self, key: str, value: Any) -> "SparkDeidentifyStreamBuilder":
        """Set one default OpenMed de-identification option."""

        self._deidentify_kwargs[key] = value
        return self

    def query_name(self, name: str | None) -> "SparkDeidentifyStreamBuilder":
        """Set the Structured Streaming query name."""

        self._query_name = name
        return self

    def output_mode(self, mode: str) -> "SparkDeidentifyStreamBuilder":
        """Set the Structured Streaming output mode."""

        self._output_mode = mode
        return self

    def trigger(self, **kwargs: Any) -> "SparkDeidentifyStreamBuilder":
        """Set the Structured Streaming trigger options."""

        self._trigger = dict(kwargs)
        return self

    def stream_option(self, key: str, value: Any) -> "SparkDeidentifyStreamBuilder":
        """Set one DataStreamWriter option."""

        self._stream_options[key] = value
        return self

    def sink(self) -> SparkDeidentifySink:
        """Build the configured sink without starting the query."""

        return SparkDeidentifySink(
            columns=self._columns,
            target_table=self._target_table,
            checkpoint_location=self._checkpoint_location,
            model_name=self._model_name,
            batch_size=self._batch_size,
            batch_id_column=self._batch_id_column,
            output_format=self._output_format,
            write_options=self._write_options,
            partition_by=self._partition_by,
            skip_existing_batches=self._skip_existing_batches,
            deidentifier_factory=self._deidentifier_factory,
            writer=self._writer,
            **self._deidentify_kwargs,
        )

    def start(self) -> Any:
        """Start the configured Structured Streaming query."""

        return self.sink().start(
            self._streaming_df,
            query_name=self._query_name,
            output_mode=self._output_mode,
            trigger=self._trigger,
            options=self._stream_options,
        )


def deidentify_write_stream(streaming_df: Any) -> SparkDeidentifyStreamBuilder:
    """Return a fluent OpenMed de-identification writer for ``streaming_df``."""

    return SparkDeidentifyStreamBuilder(streaming_df)


def write_deidentified_stream(
    streaming_df: Any,
    *,
    target_table: str,
    columns: Sequence[str | Mapping[str, Any] | SparkDeidentifyColumn],
    checkpoint_location: str,
    query_name: str | None = None,
    output_mode: str = "append",
    trigger: Mapping[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
    **sink_kwargs: Any,
) -> Any:
    """Start a Structured Streaming query that writes de-identified rows."""

    sink = SparkDeidentifySink(
        columns=columns,
        target_table=target_table,
        checkpoint_location=checkpoint_location,
        **sink_kwargs,
    )
    return sink.start(
        streaming_df,
        query_name=query_name,
        output_mode=output_mode,
        trigger=trigger,
        options=options,
    )


def _coerce_columns(
    columns: Sequence[str | Mapping[str, Any] | SparkDeidentifyColumn],
) -> tuple[SparkDeidentifyColumn, ...]:
    if isinstance(columns, (str, bytes)):
        columns = (columns,)  # type: ignore[assignment]
    specs = tuple(_coerce_column(column) for column in columns)
    if not specs:
        raise ValueError("at least one de-identification column is required")

    targets: set[str] = set()
    for spec in specs:
        if spec.target_column in targets:
            raise ValueError(f"duplicate output column {spec.target_column!r}")
        targets.add(spec.target_column)
    return specs


def _coerce_column(
    value: str | Mapping[str, Any] | SparkDeidentifyColumn,
) -> SparkDeidentifyColumn:
    if isinstance(value, SparkDeidentifyColumn):
        return value
    if isinstance(value, str):
        return SparkDeidentifyColumn(name=value)
    if isinstance(value, Mapping):
        data = dict(value)
        name = data.pop("name", None)
        if name is None:
            name = data.pop("column", None)
        if name is None:
            raise ValueError("column spec mappings require 'name' or 'column'")
        output_column = data.pop("output_column", None)
        if output_column is None:
            output_column = data.pop("output", None)
        if output_column is None:
            output_column = data.pop("target", None)
        kwargs = data.pop("deidentify_kwargs", None)
        if kwargs is None:
            kwargs = data.pop("kwargs", {})
        return SparkDeidentifyColumn(
            name=name,
            policy=data.pop("policy", DEFAULT_SPARK_POLICY),
            output_column=output_column,
            method=data.pop("method", "mask"),
            deidentify_kwargs={**dict(kwargs), **data},
        )
    raise TypeError("columns must contain strings, mappings, or SparkDeidentifyColumn")


def _redact_partition(
    rows: Iterable[Any],
    config: _SparkPartitionConfig,
    factory: PartitionDeidentifierFactory | None,
) -> Iterable[dict[str, Any]]:
    deidentifier = (
        factory()
        if factory is not None
        else _OpenMedPartitionDeidentifier(
            model_name=config.model_name,
            default_deidentify_kwargs=config.default_deidentify_kwargs,
        )
    )
    chunk: list[dict[str, Any]] = []
    for row in rows:
        chunk.append(_row_to_dict(row))
        if len(chunk) >= config.batch_size:
            yield from _redact_rows(chunk, config, deidentifier)
            chunk = []
    if chunk:
        yield from _redact_rows(chunk, config, deidentifier)


def _redact_rows(
    rows: Sequence[Mapping[str, Any]],
    config: _SparkPartitionConfig,
    deidentifier: SparkPartitionDeidentifier | Callable,
) -> list[dict[str, Any]]:
    redacted_rows = [dict(row) for row in rows]
    for column in config.columns:
        positions: list[int] = []
        texts: list[str] = []
        for index, row in enumerate(rows):
            value = row.get(column.name)
            if value is None:
                redacted_rows[index][column.target_column] = None
                continue
            if not isinstance(value, str):
                raise TypeError(
                    f"column {column.name!r} must contain string values or nulls"
                )
            positions.append(index)
            texts.append(value)

        if not texts:
            continue

        outputs = _run_deidentifier(deidentifier, texts, column)
        if len(outputs) != len(texts):
            raise ValueError(
                f"deidentifier returned {len(outputs)} outputs for {len(texts)} inputs"
            )
        for row_index, output in zip(positions, outputs):
            redacted_rows[row_index][column.target_column] = output

    if config.batch_id_column is not None:
        for row in redacted_rows:
            row[config.batch_id_column] = config.batch_id
    return redacted_rows


def _run_deidentifier(
    deidentifier: SparkPartitionDeidentifier | Callable,
    texts: Sequence[str],
    column: SparkDeidentifyColumn,
) -> list[str]:
    if hasattr(deidentifier, "deidentify_many"):
        raw_outputs = deidentifier.deidentify_many(texts, column)
    elif callable(deidentifier):
        kwargs = dict(column.deidentify_kwargs)
        raw_outputs = [
            deidentifier(
                text,
                method=column.method,
                policy=column.policy,
                **kwargs,
            )
            for text in texts
        ]
    else:
        raise TypeError("deidentifier must be callable or expose deidentify_many")

    if hasattr(raw_outputs, "items"):
        raw_outputs = [
            _successful_batch_item_result(item)
            for item in raw_outputs.items  # type: ignore[union-attr]
        ]
    return [_deidentified_text(output) for output in raw_outputs]


def _successful_batch_item_result(item: Any) -> Any:
    if getattr(item, "success", True) is False:
        raise RuntimeError(getattr(item, "error", "batch de-identification failed"))
    return getattr(item, "result", item)


def _deidentified_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, Mapping) and "deidentified_text" in output:
        return str(output["deidentified_text"])
    text = getattr(output, "deidentified_text", None)
    if text is not None:
        return str(text)
    result = getattr(output, "result", None)
    if result is not None:
        return _deidentified_text(result)
    raise TypeError("deidentifier outputs must expose deidentified_text")


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    as_dict = getattr(row, "asDict", None)
    if callable(as_dict):
        return dict(as_dict(recursive=False))
    return dict(row)


def _output_schema(
    input_schema: Any,
    columns: Sequence[SparkDeidentifyColumn],
    batch_id_column: str | None,
) -> Any:
    from pyspark.sql.types import LongType, StringType, StructField, StructType

    fields = list(input_schema.fields)
    existing = {field.name for field in fields}
    for column in columns:
        if column.target_column not in existing:
            fields.append(StructField(column.target_column, StringType(), True))
            existing.add(column.target_column)
    if batch_id_column is not None and batch_id_column not in existing:
        fields.append(StructField(batch_id_column, LongType(), False))
    return StructType(fields)


def _validate_dataframe_columns(
    df: Any,
    columns: Sequence[SparkDeidentifyColumn],
    partition_by: Sequence[str],
) -> None:
    available = set(df.columns)
    for column in columns:
        if column.name not in available:
            raise ValueError(f"input DataFrame is missing column {column.name!r}")
    produced = available | {column.target_column for column in columns}
    for column in partition_by:
        if column not in produced:
            raise ValueError(f"partition column {column!r} is not in output DataFrame")


def _target_has_batch(
    spark: Any,
    table: str,
    batch_id_column: str,
    batch_id: int,
) -> bool:
    if not spark.catalog.tableExists(table):
        return False
    target = spark.table(table)
    if batch_id_column not in target.columns:
        return False

    from pyspark.sql import functions as functions

    return (
        target.where(functions.col(batch_id_column) == functions.lit(int(batch_id)))
        .limit(1)
        .count()
        > 0
    )


def _non_empty_string(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: int, name: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    return integer


__all__ = [
    "DEFAULT_BATCH_ID_COLUMN",
    "DEFAULT_SPARK_POLICY",
    "SparkDeidentifyColumn",
    "SparkDeidentifySink",
    "SparkDeidentifyStreamBuilder",
    "deidentify_write_stream",
    "write_deidentified_stream",
]
