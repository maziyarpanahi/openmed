from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")


def _spark_session(tmp_path: Path) -> Any:
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    try:
        return (
            SparkSession.builder.master("local[2]")
            .appName("openmed-pandas-on-spark-unit")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
            .config("spark.sql.warehouse.dir", str(tmp_path / "warehouse"))
            .getOrCreate()
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"local Spark session is unavailable: {exc}")


@pytest.fixture()
def spark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("PYSPARK_PYTHON", sys.executable)
    monkeypatch.setenv("PYSPARK_DRIVER_PYTHON", sys.executable)
    session = _spark_session(tmp_path)
    try:
        yield session
    finally:
        session.stop()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="local pandas-on-Spark cleanup is not stable on Windows",
)
def test_dataframe_accessor_redacts_synthetic_rows_in_partition_batches(
    spark: Any,
) -> None:
    pd = pytest.importorskip("pandas")
    ps = pytest.importorskip("pyspark.pandas")
    import openmed.integrations.pandas_on_spark  # noqa: F401

    batch_calls = spark.sparkContext.accumulator(0)

    def fake_process_batch(texts: list[str], **kwargs: Any) -> Any:
        from types import SimpleNamespace

        batch_calls.add(1)
        assert kwargs["operation"] == "deidentify"
        assert kwargs["method"] == "mask"
        assert kwargs["policy"] == "hipaa_safe_harbor"
        assert kwargs["lang"] == "en"
        return SimpleNamespace(
            items=[
                SimpleNamespace(
                    success=True,
                    result=SimpleNamespace(
                        deidentified_text=text.replace("Jane Roe", "[NAME]")
                        .replace("John Doe", "[NAME]")
                        .replace("555-0100", "[PHONE]")
                        .replace("555-0199", "[PHONE]")
                    ),
                )
                for text in texts
            ]
        )

    source = pd.DataFrame(
        {
            "record_id": ["a", "b", "c", "d"],
            "note": [
                "Patient Jane Roe called 555-0100.",
                "No identifiers here.",
                "John Doe left voicemail at 555-0199.",
                "Follow-up contains no seeded PHI.",
            ],
            "age": [42, 73, 55, 61],
        },
        index=pd.Index([10, 11, 20, 21], name="row_id"),
    )
    distributed = ps.from_pandas(source).spark.repartition(2)
    partition_count = distributed.to_spark().rdd.getNumPartitions()

    redacted = distributed.deid.deidentify(
        columns="note",
        policy="hipaa_safe_harbor",
        deidentifier=fake_process_batch,
        lang="en",
    )
    computed = redacted.to_pandas().sort_index()

    assert computed.index.tolist() == source.index.tolist()
    assert computed["record_id"].tolist() == source["record_id"].tolist()
    assert computed["age"].tolist() == source["age"].tolist()
    assert "Jane Roe" not in "\n".join(computed["note"])
    assert "John Doe" not in "\n".join(computed["note"])
    assert "555-0100" not in "\n".join(computed["note"])
    assert "555-0199" not in "\n".join(computed["note"])
    assert 0 < batch_calls.value <= partition_count
    assert batch_calls.value < len(source)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="local pandas-on-Spark cleanup is not stable on Windows",
)
def test_series_accessor_returns_redacted_series(spark: Any) -> None:
    pd = pytest.importorskip("pandas")
    ps = pytest.importorskip("pyspark.pandas")
    import openmed.integrations.pandas_on_spark  # noqa: F401

    source = pd.Series(
        [
            "Patient Jane Roe called 555-0100.",
            "John Doe left voicemail at 555-0199.",
        ],
        name="note",
    )
    distributed = ps.from_pandas(source)

    def fake_process_batch(texts: list[str], **kwargs: Any) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            items=[
                SimpleNamespace(
                    success=True,
                    result=SimpleNamespace(
                        deidentified_text=text.replace("Jane Roe", "[NAME]")
                        .replace("John Doe", "[NAME]")
                        .replace("555-0100", "[PHONE]")
                        .replace("555-0199", "[PHONE]")
                    ),
                )
                for text in texts
            ]
        )

    redacted = distributed.deid.deidentify(
        process_batch_fn=fake_process_batch,
    )

    computed = redacted.to_pandas()
    assert computed.name == source.name
    assert "Jane Roe" not in "\n".join(computed)
    assert "John Doe" not in "\n".join(computed)


def test_dataframe_accessor_signature_matches_local_pandas_accessor() -> None:
    pytest.importorskip("pyspark.pandas")
    from openmed.integrations.pandas_on_spark import (
        OpenMedPandasOnSparkDataFrameAccessor,
    )
    from openmed.interop.pandas_accessor import OpenMedDataFrameAccessor

    assert inspect.signature(
        OpenMedPandasOnSparkDataFrameAccessor.deidentify
    ) == inspect.signature(OpenMedDataFrameAccessor.deidentify)


def test_worker_loader_is_initialized_once(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pyspark.pandas")
    import openmed.core
    from openmed.integrations import pandas_on_spark

    loads = 0
    sentinel = object()

    def fake_loader() -> object:
        nonlocal loads
        loads += 1
        return sentinel

    pandas_on_spark.clear_worker_pipeline_cache()
    monkeypatch.setattr(openmed.core, "ModelLoader", fake_loader)

    assert pandas_on_spark._get_worker_loader() is sentinel
    assert pandas_on_spark._get_worker_loader() is sentinel
    assert loads == 1
    pandas_on_spark.clear_worker_pipeline_cache()


def test_dataframe_accessor_rejects_missing_and_non_text_columns(spark: Any) -> None:
    ps = pytest.importorskip("pyspark.pandas")
    import openmed.integrations.pandas_on_spark  # noqa: F401

    distributed = ps.DataFrame({"note": ["Patient Jane Roe"], "age": [42]})

    with pytest.raises(KeyError, match="missing"):
        distributed.deid.deidentify(columns="unknown")
    with pytest.raises(TypeError, match="text column"):
        distributed.deid.deidentify(columns="age")
