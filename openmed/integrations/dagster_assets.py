"""Dagster ops and assets for in-memory dataset de-identification.

Dagster is an optional integration.  This module keeps the import lazy so the
core ``openmed`` package remains usable without the ``dagster`` extra.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

from openmed.core.policy import CANONICAL_POLICY_NAMES, canonical_policy_name
from openmed.processing import process_batch

try:  # pragma: no cover - exercised by the optional integration test
    import dagster as dg
except ImportError:  # pragma: no cover - the normal core-install path
    dg = None  # type: ignore[assignment]


DEFAULT_POLICY_PROFILE = "hipaa_safe_harbor"
DEFAULT_MODEL_NAME = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
DEFAULT_PARTITION_START_DATE = "2024-01-01"
SOURCE_DATASET_RESOURCE_KEY = "source_dataset"

_ProcessBatch = Callable[..., Any]
_CONFIG_ERROR = (
    "Dagster support requires the 'dagster' extra. "
    "Install with `pip install openmed[dagster]`."
)


def _require_dagster() -> Any:
    if dg is None:
        raise ImportError(_CONFIG_ERROR) from None
    return dg


def _resolve_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize orchestrator config and validate direct Python calls."""

    policy_profile = raw_config.get("policy_profile")
    policy = raw_config.get("policy")
    if policy_profile is not None and policy is not None and policy_profile != policy:
        raise ValueError("policy_profile and policy must match when both are set")

    selected_policy = policy_profile or policy or DEFAULT_POLICY_PROFILE
    return {
        "policy_profile": canonical_policy_name(selected_policy),
        "text_columns": tuple(_normalize_text_columns(raw_config.get("text_columns"))),
        "method": str(raw_config.get("method", "mask")),
        "model_name": str(raw_config.get("model_name", DEFAULT_MODEL_NAME)),
        "confidence_threshold": float(raw_config.get("confidence_threshold", 0.7)),
    }


def _normalize_text_columns(columns: Any) -> tuple[str, ...]:
    if isinstance(columns, str):
        values = (columns,)
    else:
        try:
            values = tuple(str(column) for column in columns)
        except TypeError as exc:
            raise ValueError("text_columns must be a non-empty sequence") from exc

    if not values or any(not column.strip() for column in values):
        raise ValueError("text_columns must be a non-empty sequence of names")
    if len(values) != len(set(values)):
        raise ValueError("text_columns must not contain duplicates")
    return values


def _copy_rows(source_dataset: Any) -> list[dict[str, Any]]:
    if isinstance(source_dataset, (str, bytes, Mapping)):
        raise TypeError("source_dataset must be a sequence of row mappings")
    try:
        values = list(source_dataset)
    except TypeError as exc:
        raise TypeError("source_dataset must be a sequence of row mappings") from exc

    rows: list[dict[str, Any]] = []
    for row in values:
        if not isinstance(row, Mapping):
            raise TypeError("source_dataset must contain only row mappings")
        rows.append(dict(row))
    return rows


def _batch_items(batch_result: Any) -> list[Any]:
    items = getattr(batch_result, "items", batch_result)
    try:
        return list(items)
    except TypeError as exc:
        raise TypeError("process_batch must return a sequence of results") from exc


def _redacted_text(item: Any) -> str:
    if getattr(item, "success", True) is False:
        raise RuntimeError("process_batch failed for a dataset cell")

    result = getattr(item, "result", item)
    if isinstance(result, str):
        return result

    redacted_text = getattr(result, "deidentified_text", None)
    if redacted_text is None:
        raise TypeError(
            "process_batch results must contain strings or deidentified_text"
        )
    return str(redacted_text)


def _entities(item: Any) -> tuple[Any, ...]:
    result = getattr(item, "result", item)
    return tuple(getattr(result, "pii_entities", ()) or ())


def _redact_rows(
    source_dataset: Any,
    *,
    config: Mapping[str, Any],
    process_batch_fn: _ProcessBatch | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Redact configured columns and return rows plus PHI-free metadata."""

    rows = _copy_rows(source_dataset)
    columns = tuple(config["text_columns"])
    missing = sorted(
        {column for row in rows for column in columns if column not in row}
    )
    if missing:
        raise KeyError(f"source_dataset is missing columns: {', '.join(missing)}")

    batch_fn = process_batch_fn or process_batch
    label_counts: Counter[str] = Counter()
    redacted_rows: set[int] = set()
    processed_cells = 0
    redacted_cells = 0
    redacted_spans = 0

    for column in columns:
        positions: list[int] = []
        texts: list[str] = []
        for row_index, row in enumerate(rows):
            value = row[column]
            if value is None:
                continue
            text = value if isinstance(value, str) else str(value)
            if not text:
                continue
            positions.append(row_index)
            texts.append(text)

        if not texts:
            continue

        batch_result = batch_fn(
            texts,
            operation="deidentify",
            method=config["method"],
            model_name=config["model_name"],
            policy=config["policy_profile"],
            confidence_threshold=config["confidence_threshold"],
        )
        items = _batch_items(batch_result)
        if len(items) != len(positions):
            raise ValueError(
                "process_batch returned a different number of results than inputs"
            )

        for row_index, original_text, item in zip(
            positions,
            texts,
            items,
            strict=True,
        ):
            redacted_text = _redacted_text(item)
            rows[row_index][column] = redacted_text
            processed_cells += 1
            entities = _entities(item)
            redacted_spans += len(entities)
            if redacted_text != original_text or entities:
                redacted_rows.add(row_index)
            if redacted_text != original_text:
                redacted_cells += 1
            for entity in entities:
                label = getattr(entity, "label", None) or getattr(
                    entity,
                    "entity_type",
                    None,
                )
                label_counts[str(label or "UNKNOWN")] += 1

    sorted_labels = dict(sorted(label_counts.items()))
    metadata = {
        "asset_type": "redacted_dataset",
        "policy_profile": config["policy_profile"],
        "text_columns": list(columns),
        "row_count": len(rows),
        "processed_rows": len(rows),
        "redacted_rows": len(redacted_rows),
        "processed_cells": processed_cells,
        "redacted_cells": redacted_cells,
        "redacted_spans": redacted_spans,
        "labels": sorted(sorted_labels),
        "per_label_counts": sorted_labels,
        "raw_text_included": False,
    }
    return rows, metadata


def _config_for_context(context: Any) -> dict[str, Any]:
    op_context = getattr(context, "op_execution_context", None) or context
    return _resolve_config(getattr(op_context, "op_config", {}) or {})


def _source_for_context(context: Any) -> Any:
    source = getattr(context.resources, SOURCE_DATASET_RESOURCE_KEY)
    partition_key = getattr(context, "partition_key", None)
    if partition_key is not None and isinstance(source, Mapping):
        partition_source = source.get(partition_key)
        if partition_source is not None:
            return partition_source
    if callable(source):
        return source(partition_key)
    return source


def _emit_metadata(context: Any, metadata: Mapping[str, Any]) -> None:
    context.add_output_metadata(dict(metadata))


if dg is not None:
    _POLICY_PROFILE_ENUM = dg.Enum(
        "OpenMedPolicyProfile",
        [dg.EnumValue(name) for name in CANONICAL_POLICY_NAMES],
    )
    _CONFIG_SCHEMA = {
        "policy_profile": dg.Field(_POLICY_PROFILE_ENUM, is_required=False),
        "policy": dg.Field(_POLICY_PROFILE_ENUM, is_required=False),
        "text_columns": dg.Field(dg.Array(dg.String), is_required=True),
        "method": dg.Field(dg.String, default_value="mask"),
        "model_name": dg.Field(dg.String, default_value=DEFAULT_MODEL_NAME),
        "confidence_threshold": dg.Field(dg.Float, default_value=0.7),
    }
    DAILY_PARTITIONS = dg.DailyPartitionsDefinition(
        start_date=DEFAULT_PARTITION_START_DATE,
    )

    @dg.op(config_schema=_CONFIG_SCHEMA, ins={"source_dataset": dg.In()})
    def deidentify_dataset_op(context, source_dataset: Any) -> list[dict[str, Any]]:
        """Redact configured columns from an in-memory dataset."""

        rows, metadata = _redact_rows(
            source_dataset,
            config=_config_for_context(context),
        )
        _emit_metadata(context, metadata)
        return rows

    @dg.asset(
        config_schema=_CONFIG_SCHEMA,
        required_resource_keys={SOURCE_DATASET_RESOURCE_KEY},
        partitions_def=DAILY_PARTITIONS,
    )
    def redacted_dataset(context) -> list[dict[str, Any]]:
        """Materialize a partitioned, redacted in-memory dataset asset."""

        rows, metadata = _redact_rows(
            _source_for_context(context),
            config=_config_for_context(context),
        )
        _emit_metadata(context, metadata)
        return rows
else:
    DAILY_PARTITIONS = None

    def deidentify_dataset_op(*args: Any, **kwargs: Any) -> Any:
        """Raise an actionable error when Dagster is not installed."""

        _require_dagster()

    def redacted_dataset(*args: Any, **kwargs: Any) -> Any:
        """Raise an actionable error when Dagster is not installed."""

        _require_dagster()


__all__ = [
    "DAILY_PARTITIONS",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_PARTITION_START_DATE",
    "DEFAULT_POLICY_PROFILE",
    "SOURCE_DATASET_RESOURCE_KEY",
    "deidentify_dataset_op",
    "redacted_dataset",
]
