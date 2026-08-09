"""Offline contract tests for the partition-safe Spark transform."""

from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import Any, Iterable

import pytest

from openmed.interop.spark import (
    SparkRedactionConfig,
    SparkRedactionError,
    SparkRedactionTransform,
    redact_partition,
)

_SYNTHETIC_VALUE = "synthetic-person-001@example.test"


def _fake_deidentifier(text: str, **kwargs: Any) -> SimpleNamespace:
    assert kwargs["method"] == "mask"
    assert kwargs["policy"] == "hipaa_safe_harbor"
    return SimpleNamespace(
        deidentified_text=text.replace(_SYNTHETIC_VALUE, "[EMAIL]"),
        pii_entities=[object()] if _SYNTHETIC_VALUE in text else [],
    )


def test_config_is_pickle_safe_and_normalizes_selected_columns() -> None:
    config = SparkRedactionConfig(
        text_columns=["note", "summary"],
        extra_kwargs={"use_safety_sweep": False},
    )

    restored = pickle.loads(pickle.dumps(config))

    assert restored == config
    assert restored.columns == ("note", "summary")
    assert restored.text_columns == restored.columns
    assert restored.extra_kwargs == {"use_safety_sweep": False}
    assert restored.to_deidentify_kwargs() == {
        "method": "mask",
        "policy": "hipaa_safe_harbor",
        "confidence_threshold": 0.7,
        "seed": 0,
        "consistent": True,
        "use_safety_sweep": False,
    }


def test_transform_serializes_only_immutable_configuration() -> None:
    transform = SparkRedactionTransform(
        columns=["note"],
        policy="strict_no_leak",
        seed=17,
    )

    restored = pickle.loads(pickle.dumps(transform))

    assert restored.config == transform.config
    assert restored.columns == ("note",)
    assert restored.config.policy == "strict_no_leak"
    assert restored.config.seed == 17


def test_partition_factory_is_worker_local_and_called_once_per_attempt() -> None:
    transform = SparkRedactionTransform(columns=["note"])
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        call_count = 0

        def worker(text: str, **_: Any) -> str:
            nonlocal call_count
            call_count += 1
            return f"[PARTITION_VALUE_{call_count}]"

        return worker

    rows = [{"id": 1, "note": _SYNTHETIC_VALUE}]
    first_partition = list(
        transform.redact_partition(
            rows,
            partition_id=2,
            deidentifier_factory=factory,
        )
    )
    replayed_partition = list(
        transform.redact_partition(
            rows,
            partition_id=2,
            deidentifier_factory=factory,
        )
    )

    assert factory_calls == 2
    assert first_partition == [{"id": 1, "note": "[PARTITION_VALUE_1]"}]
    assert replayed_partition == first_partition
    assert rows == [{"id": 1, "note": _SYNTHETIC_VALUE}]


def test_partition_redacts_only_selected_string_columns_and_preserves_nulls() -> None:
    config = SparkRedactionConfig(columns=["note", "summary"])
    rows = [
        {
            "record_id": "synthetic-001",
            "note": _SYNTHETIC_VALUE,
            "summary": None,
            "status": "keep",
        }
    ]

    output = list(
        redact_partition(rows, config, deidentifier_factory=lambda: _fake_deidentifier)
    )

    assert output == [
        {
            "record_id": "synthetic-001",
            "note": "[EMAIL]",
            "summary": None,
            "status": "keep",
        }
    ]
    assert rows[0]["note"] == _SYNTHETIC_VALUE


def test_partition_accepts_spark_row_like_mappings() -> None:
    class RowLike:
        def asDict(self, *, recursive: bool = False) -> dict[str, Any]:
            assert recursive is True
            return {"note": _SYNTHETIC_VALUE, "kind": "synthetic"}

    output = list(
        redact_partition(
            [RowLike()],
            SparkRedactionConfig(columns=["note"]),
            deidentifier_factory=lambda: _fake_deidentifier,
        )
    )

    assert output == [{"note": "[EMAIL]", "kind": "synthetic"}]


def test_partition_failure_does_not_expose_source_value() -> None:
    def failing_deidentifier(text: str, **_: Any) -> str:
        raise RuntimeError(f"worker detail: {text}")

    with pytest.raises(SparkRedactionError) as exc_info:
        list(
            redact_partition(
                [{"note": _SYNTHETIC_VALUE}],
                SparkRedactionConfig(columns=["note"]),
                deidentifier_factory=lambda: failing_deidentifier,
            )
        )

    assert _SYNTHETIC_VALUE not in str(exc_info.value)
    assert str(exc_info.value) == "partition deidentifier failed"


def test_partition_rejects_missing_or_non_string_selected_columns() -> None:
    config = SparkRedactionConfig(columns=["note"])

    with pytest.raises(SparkRedactionError, match="missing"):
        list(
            redact_partition(
                [{"other": "synthetic"}],
                config,
                deidentifier_factory=lambda: _fake_deidentifier,
            )
        )

    with pytest.raises(SparkRedactionError, match="strings or nulls"):
        list(
            redact_partition(
                [{"note": 7}],
                config,
                deidentifier_factory=lambda: _fake_deidentifier,
            )
        )


class _FakeRDD:
    def __init__(self, partitions: Iterable[list[dict[str, Any]]]) -> None:
        self.partitions = list(partitions)

    def mapPartitionsWithIndex(self, function):
        mapped = [
            list(function(partition_id, rows))
            for partition_id, rows in enumerate(self.partitions)
        ]
        return _FakeRDD(mapped)


class _FakeSparkSession:
    def createDataFrame(self, rdd: _FakeRDD, *, schema: Any) -> dict[str, Any]:
        return {"rows": rdd.partitions, "schema": schema}


class _FakeDataFrame:
    columns = ["id", "note", "status"]
    schema = "synthetic-schema"

    def __init__(self) -> None:
        self.rdd = _FakeRDD([[{"id": 1, "note": _SYNTHETIC_VALUE, "status": "ok"}]])
        self.sparkSession = _FakeSparkSession()


def test_apply_uses_map_partitions_without_importing_pyspark() -> None:
    dataframe = _FakeDataFrame()
    transform = SparkRedactionTransform(columns=["note"])

    result = transform.apply(
        dataframe,
        deidentifier_factory=lambda: _fake_deidentifier,
    )

    assert result == {
        "rows": [[{"id": 1, "note": "[EMAIL]", "status": "ok"}]],
        "schema": "synthetic-schema",
    }
