from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

import pytest

from openmed.integrations.spark_streaming import (
    DEFAULT_BATCH_ID_COLUMN,
    SparkDeidentifyColumn,
    SparkDeidentifySink,
    _coerce_columns,
    _redact_partition,
    _SparkPartitionConfig,
)


class ProbeDeidentifier:
    def deidentify_many(
        self,
        texts: Sequence[str],
        column: SparkDeidentifyColumn,
    ) -> list[str]:
        assert column.policy == "hipaa_safe_harbor"
        return [_redact_seeded_tokens(text) for text in texts]


def _redact_seeded_tokens(text: str) -> str:
    replacements = {
        "Jane Roe": "[NAME]",
        "John Doe": "[NAME]",
        "555-0101": "[PHONE]",
        "john@example.test": "[EMAIL]",
    }
    for token, replacement in replacements.items():
        text = text.replace(token, replacement)
    return text


def test_column_specs_accept_strings_and_mappings() -> None:
    columns = _coerce_columns(
        [
            "note",
            {
                "column": "comment",
                "output_column": "comment_redacted",
                "policy": "hipaa_safe_harbor",
                "method": "mask",
                "use_safety_sweep": False,
            },
        ]
    )

    assert columns[0] == SparkDeidentifyColumn(name="note")
    assert columns[1].name == "comment"
    assert columns[1].target_column == "comment_redacted"
    assert columns[1].deidentify_kwargs == {"use_safety_sweep": False}


def test_duplicate_output_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate output column"):
        _coerce_columns(
            [
                SparkDeidentifyColumn(name="note", output_column="redacted"),
                SparkDeidentifyColumn(name="comment", output_column="redacted"),
            ]
        )


def test_redact_partition_constructs_deidentifier_once_per_partition() -> None:
    load_count = 0

    def factory() -> ProbeDeidentifier:
        nonlocal load_count
        load_count += 1
        return ProbeDeidentifier()

    config = _SparkPartitionConfig(
        columns=(SparkDeidentifyColumn(name="note"),),
        model_name="stub",
        batch_size=2,
        default_deidentify_kwargs={"use_safety_sweep": False},
        batch_id_column=DEFAULT_BATCH_ID_COLUMN,
        batch_id=42,
    )
    rows = [
        {"id": 1, "note": "Patient Jane Roe called 555-0101"},
        {"id": 2, "note": "Patient John Doe emailed john@example.test"},
        {"id": 3, "note": None},
    ]

    output = list(_redact_partition(rows, config, factory))

    assert load_count == 1
    assert output == [
        {"id": 1, "note": "Patient [NAME] called [PHONE]", "_openmed_batch_id": 42},
        {
            "id": 2,
            "note": "Patient [NAME] emailed [EMAIL]",
            "_openmed_batch_id": 42,
        },
        {"id": 3, "note": None, "_openmed_batch_id": 42},
    ]


def _spark_session(tmp_path: Path) -> Any:
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    try:
        return (
            SparkSession.builder.master("local[2]")
            .appName("openmed-spark-streaming-unit")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
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
    reason="local Spark warehouse cleanup is not stable on Windows",
)
def test_structured_streaming_sink_redacts_rate_micro_batch_and_skips_replay(
    spark: Any,
    tmp_path: Path,
) -> None:
    from pyspark.sql import functions as functions

    table = f"om443_redacted_{uuid.uuid4().hex}"
    checkpoint = str(tmp_path / "checkpoint")
    load_count = spark.sparkContext.accumulator(0)

    def factory() -> Any:
        load_count.add(1)

        class WorkerProbeDeidentifier:
            def deidentify_many(
                self,
                texts: Sequence[str],
                column: SparkDeidentifyColumn,
            ) -> list[str]:
                assert column.policy == "hipaa_safe_harbor"
                replacements = {
                    "Jane Roe": "[NAME]",
                    "John Doe": "[NAME]",
                    "555-0101": "[PHONE]",
                    "john@example.test": "[EMAIL]",
                }
                redacted = []
                for text in texts:
                    for token, replacement in replacements.items():
                        text = text.replace(token, replacement)
                    redacted.append(text)
                return redacted

        return WorkerProbeDeidentifier()

    try:
        source = (
            spark.readStream.format("rate-micro-batch")
            .option("rowsPerBatch", 4)
            .option("numPartitions", 2)
            .load()
        )
    except Exception as exc:
        pytest.skip(f"rate-micro-batch source is unavailable: {exc}")

    notes = source.select(
        functions.col("value").cast("long").alias("id"),
        functions.when(
            functions.col("value") % 2 == 0,
            functions.lit("Patient Jane Roe called 555-0101"),
        )
        .otherwise(functions.lit("Patient John Doe emailed john@example.test"))
        .alias("note"),
    )

    sink = SparkDeidentifySink(
        columns=[SparkDeidentifyColumn(name="note", policy="hipaa_safe_harbor")],
        target_table=table,
        checkpoint_location=checkpoint,
        deidentifier_factory=factory,
        batch_size=2,
    )

    query = sink.start(notes, trigger={"once": True})
    try:
        assert query.awaitTermination(30)
    finally:
        if query.isActive:
            query.stop()

    output = spark.table(table).orderBy("id").collect()
    assert output
    assert all("Jane Roe" not in row.note for row in output)
    assert all("John Doe" not in row.note for row in output)
    assert all("555-0101" not in row.note for row in output)
    assert all("john@example.test" not in row.note for row in output)
    assert {row[DEFAULT_BATCH_ID_COLUMN] for row in output} == {0}
    assert load_count.value >= 1

    before_replay = [row.asDict() for row in spark.table(table).orderBy("id").collect()]
    replay_df = spark.createDataFrame(
        [
            (0, "Patient Jane Roe called 555-0101"),
            (1, "Patient John Doe emailed john@example.test"),
        ],
        "id long, note string",
    )
    assert sink.process_micro_batch(replay_df, 0) is None
    after_replay = [row.asDict() for row in spark.table(table).orderBy("id").collect()]
    assert after_replay == before_replay

    spark.sql(f"DROP TABLE IF EXISTS {table}")
