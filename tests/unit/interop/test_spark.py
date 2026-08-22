"""Offline contract tests for the partition-safe Spark transform."""

from __future__ import annotations

import pickle
from pathlib import Path
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


def test_config_snapshots_mutable_extra_kwargs_and_returns_fresh_values() -> None:
    supplied = {
        "token_language_tags": ["en"],
        "calibration_thresholds_path": Path("synthetic-thresholds.json"),
        "nested": {"ordered": [1, 2]},
    }
    config = SparkRedactionConfig(columns=["note"], extra_kwargs=supplied)

    supplied["token_language_tags"].append("fr")
    supplied["nested"]["ordered"].append(3)
    returned = config.extra_kwargs
    returned["token_language_tags"].append("de")
    returned["nested"]["ordered"].append(4)

    assert config.extra_kwargs == {
        "calibration_thresholds_path": Path("synthetic-thresholds.json"),
        "nested": {"ordered": [1, 2]},
        "token_language_tags": ["en"],
    }
    assert pickle.loads(pickle.dumps(config)) == config


def test_config_snapshot_is_stable_across_nested_mapping_order() -> None:
    first = SparkRedactionConfig(
        columns=["note"],
        extra_kwargs={"nested": {"beta": 2, "alpha": 1}},
    )
    second = SparkRedactionConfig(
        columns=["note"],
        extra_kwargs={"nested": {"alpha": 1, "beta": 2}},
    )

    assert first == second


def test_config_rejects_executable_or_stateful_extra_options() -> None:
    reduce_called = False

    class ExecutableValue:
        def __reduce__(self):
            nonlocal reduce_called
            reduce_called = True
            return (str, ("should-not-run",))

    with pytest.raises(TypeError, match="supported data types"):
        SparkRedactionConfig(
            columns=["note"],
            extra_kwargs={"unsupported": ExecutableValue()},
        )

    assert reduce_called is False

    for key in (
        "budget",
        "custom_recognizer",
        "lid_model",
        "transliterated_name_config",
    ):
        with pytest.raises(ValueError, match="reserved"):
            SparkRedactionConfig(columns=["note"], extra_kwargs={key: None})


def test_config_rejects_cycles_and_contains_corrupted_snapshot_state() -> None:
    cyclic: list[Any] = []
    cyclic.append(cyclic)

    with pytest.raises(TypeError, match="must not contain cycles"):
        SparkRedactionConfig(columns=["note"], extra_kwargs={"nested": cyclic})

    config = SparkRedactionConfig(columns=["note"])
    object.__setattr__(
        config,
        "_extra_items",
        (("unsafe", ("serialized", b"not-loaded")),),
    )

    with pytest.raises(TypeError, match="stored extra kwargs are invalid"):
        config.extra_kwargs


def test_config_validation_does_not_invoke_hostile_string_or_mapping_hooks() -> None:
    strip_called = False

    class HostileString(str):
        def strip(self, *args: Any, **kwargs: Any) -> str:
            nonlocal strip_called
            strip_called = True
            raise RuntimeError(f"sensitive detail: {_SYNTHETIC_VALUE}")

    with pytest.raises(ValueError, match="non-empty string names") as exc_info:
        SparkRedactionConfig(columns=[HostileString("note")])

    assert strip_called is False
    assert _SYNTHETIC_VALUE not in str(exc_info.value)

    class MalformedMapping(dict[str, Any]):
        def items(self):
            return [("malformed",)]

    with pytest.raises(TypeError, match="extra kwargs could not be read") as exc_info:
        SparkRedactionConfig(
            columns=["note"],
            extra_kwargs=MalformedMapping(),
        )

    assert _SYNTHETIC_VALUE not in str(exc_info.value)


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


def test_partition_normalizes_external_contract_errors() -> None:
    def leaking_factory():
        raise SparkRedactionError(f"factory detail: {_SYNTHETIC_VALUE}")

    with pytest.raises(SparkRedactionError) as factory_error:
        list(
            redact_partition(
                [],
                SparkRedactionConfig(columns=["note"]),
                deidentifier_factory=leaking_factory,
            )
        )

    assert str(factory_error.value) == (
        "Spark partition redaction failed for the configured columns"
    )
    assert _SYNTHETIC_VALUE not in str(factory_error.value)
    assert factory_error.value.__cause__ is None

    def leaking_rows():
        raise SparkRedactionError(f"iterator detail: {_SYNTHETIC_VALUE}")
        yield {}

    with pytest.raises(SparkRedactionError) as iterator_error:
        list(
            redact_partition(
                leaking_rows(),
                SparkRedactionConfig(columns=["note"]),
                deidentifier_factory=lambda: _fake_deidentifier,
            )
        )

    assert str(iterator_error.value) == (
        "Spark partition redaction failed for the configured columns"
    )
    assert _SYNTHETIC_VALUE not in str(iterator_error.value)
    assert iterator_error.value.__cause__ is None


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


def test_dataframe_contract_failure_does_not_expose_source_value() -> None:
    class ExplodingDataFrame:
        @property
        def columns(self):
            raise RuntimeError(f"driver detail: {_SYNTHETIC_VALUE}")

    with pytest.raises(TypeError) as exc_info:
        SparkRedactionTransform(columns=["note"]).apply(ExplodingDataFrame())

    assert _SYNTHETIC_VALUE not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_dataframe_normalizes_external_contract_errors() -> None:
    class LeakingRDD:
        def mapPartitionsWithIndex(self, function):
            raise SparkRedactionError(f"driver detail: {_SYNTHETIC_VALUE}")

    class LeakingDataFrame:
        columns = ["note"]
        schema = "synthetic-schema"
        rdd = LeakingRDD()
        sparkSession = _FakeSparkSession()

    with pytest.raises(SparkRedactionError) as exc_info:
        SparkRedactionTransform(columns=["note"]).apply(LeakingDataFrame())

    assert str(exc_info.value) == "Spark DataFrame redaction failed"
    assert _SYNTHETIC_VALUE not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
