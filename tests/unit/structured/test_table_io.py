"""Tests for structured release table I/O."""

from __future__ import annotations

import json
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from openmed.structured import read_table, write_table

ROWS = [
    {"age": 30, "zip": "10001", "condition": "a"},
    {"age": 40, "zip": "20001", "condition": "b"},
]


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".jsonl", ".ndjson"])
def test_table_io_round_trip_supported_text_formats(
    tmp_path: Path,
    suffix: str,
) -> None:
    path = tmp_path / f"release{suffix}"

    written = write_table(path, ROWS)
    restored = read_table(written)

    assert written == path
    if suffix in {".csv", ".tsv"}:
        assert restored == [
            {"age": "30", "zip": "10001", "condition": "a"},
            {"age": "40", "zip": "20001", "condition": "b"},
        ]
    else:
        assert restored == ROWS


def test_table_writer_refuses_overwrite_by_default(tmp_path: Path) -> None:
    path = tmp_path / "release.jsonl"
    write_table(path, ROWS)

    with pytest.raises(FileExistsError):
        write_table(path, ROWS)

    write_table(path, list(reversed(ROWS)), overwrite=True)
    assert read_table(path) == list(reversed(ROWS))


def test_table_writer_does_not_replace_broken_symlink_without_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "release.jsonl"
    path.symlink_to(tmp_path / "missing-target.jsonl")

    with pytest.raises(FileExistsError):
        write_table(path, ROWS)

    assert path.is_symlink()


def test_failed_delimited_write_is_atomic_and_leaves_no_output(
    tmp_path: Path,
) -> None:
    path = tmp_path / "release.csv"

    with pytest.raises(TypeError, match="scalar values only"):
        write_table(path, [{"age": 30, "codes": ["a", "b"]}])

    assert not path.exists()
    assert not list(tmp_path.glob(".release.csv.*.tmp"))


def test_jsonl_reader_requires_object_rows(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(["not", "an", "object"]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="row 1"):
        read_table(path)


@pytest.mark.parametrize(
    ("suffix", "delimiter"),
    [(".csv", ","), (".tsv", "\t")],
)
def test_delimited_reader_rejects_duplicate_and_empty_headers(
    tmp_path: Path,
    suffix: str,
    delimiter: str,
) -> None:
    duplicate = tmp_path / f"duplicate{suffix}"
    duplicate.write_text(
        f"age{delimiter}age\n30{delimiter}40\n",
        encoding="utf-8",
    )
    empty = tmp_path / f"empty{suffix}"
    empty.write_text(
        f"age{delimiter}   {delimiter}condition\n"
        f"30{delimiter}value{delimiter}example\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate column names"):
        read_table(duplicate)
    with pytest.raises(ValueError, match="empty column name"):
        read_table(empty)


@pytest.mark.parametrize(
    ("suffix", "delimiter"),
    [(".csv", ","), (".tsv", "\t")],
)
def test_delimited_reader_rejects_overflow_cells_without_echoing_values(
    tmp_path: Path,
    suffix: str,
    delimiter: str,
) -> None:
    path = tmp_path / f"overflow{suffix}"
    canary = "sensitive-overflow-canary"
    path.write_text(
        f"age{delimiter}condition\n30{delimiter}example{delimiter}{canary}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="more cells") as raised:
        read_table(path)

    assert canary not in str(raised.value)


@pytest.mark.parametrize(
    ("suffix", "delimiter"),
    [(".csv", ","), (".tsv", "\t")],
)
def test_delimited_reader_rejects_underflow_cells(
    tmp_path: Path,
    suffix: str,
    delimiter: str,
) -> None:
    path = tmp_path / f"underflow{suffix}"
    path.write_text(
        f"age{delimiter}condition\n30\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fewer cells"):
        read_table(path)


@pytest.mark.parametrize(
    "payload",
    [
        '{"age":30,"age":40}\n',
        '{"profile":{"condition":"a","condition":"b"}}\n',
    ],
)
def test_jsonl_reader_rejects_duplicate_object_keys_without_echoing_them(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "duplicate.jsonl"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate object keys") as raised:
        read_table(path)

    assert "condition" not in str(raised.value)


@pytest.mark.parametrize(
    "payload",
    [
        '{"score":NaN}\n',
        '{"score":Infinity}\n',
        '{"score":1e9999}\n',
    ],
)
def test_jsonl_reader_rejects_non_finite_numbers(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "non-finite.jsonl"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="row 1"):
        read_table(path)


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".jsonl", ".ndjson"])
def test_text_writers_reject_non_finite_numbers_atomically(
    tmp_path: Path,
    suffix: str,
) -> None:
    path = tmp_path / f"release{suffix}"

    with pytest.raises(ValueError, match="non-finite"):
        write_table(path, [{"score": float("nan")}])

    assert not path.exists()


def test_jsonl_writer_rejects_nested_values_without_echoing_them(
    tmp_path: Path,
) -> None:
    path = tmp_path / "release.jsonl"
    canary = "sensitive-nested-canary"

    with pytest.raises(TypeError, match="unsupported scalar") as raised:
        write_table(path, [{"codes": [canary]}])

    assert canary not in str(raised.value)
    assert not path.exists()


def test_parquet_writer_uses_union_schema_for_later_row_columns(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "release.parquet"
    rows = [
        {"age": 30},
        {"age": 40, "condition": "example"},
    ]

    write_table(path, rows)

    assert pq.read_table(path).column_names == ["age", "condition"]
    assert read_table(path) == [
        {"age": 30, "condition": None},
        {"age": 40, "condition": "example"},
    ]


@pytest.mark.parametrize("temporal_kind", ["timestamp_ns", "time64_ns"])
def test_parquet_reader_rejects_submicrosecond_temporal_precision(
    temporal_kind: str,
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    data_type = (
        pa.timestamp("ns") if temporal_kind == "timestamp_ns" else pa.time64("ns")
    )
    path = tmp_path / f"{temporal_kind}.parquet"
    pq.write_table(
        pa.table({"event_time": pa.array([1, 2], type=data_type)}),
        path,
    )

    with pytest.raises(ValueError, match="sub-microsecond precision"):
        read_table(path)


def test_parquet_round_trip_canonical_arrow_clinical_scalars(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "clinical-scalars.parquet"
    recorded_at = datetime(
        2025,
        6,
        7,
        8,
        9,
        10,
        123456,
        tzinfo=timezone.utc,
    )
    rows = [
        {
            "service_date": date(2025, 6, 7),
            "recorded_at": recorded_at,
            "collection_time": time(8, 9, 10, 123456),
            "measurement": Decimal("12.3400"),
            "payload": b"\x00\xff",
        },
        {
            "service_date": date(2025, 6, 8),
            "recorded_at": recorded_at,
            "collection_time": time(9, 10),
            "measurement": Decimal("1.2"),
            "payload": b"example",
        },
    ]

    write_table(path, rows)
    restored = read_table(path)

    assert restored == [
        {
            **rows[0],
            "measurement": Decimal("12.34"),
        },
        {
            **rows[1],
            "measurement": Decimal("1.2"),
        },
    ]
    for row in restored:
        assert type(row["service_date"]) is date
        assert type(row["recorded_at"]) is datetime
        assert type(row["collection_time"]) is time
        assert type(row["measurement"]) is Decimal
        assert type(row["payload"]) is bytes


@pytest.mark.parametrize(
    "value",
    [
        float("inf"),
        Decimal("NaN"),
        Decimal("Infinity"),
    ],
)
def test_parquet_writer_rejects_non_finite_scalars_atomically(
    tmp_path: Path,
    value: object,
) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "release.parquet"

    with pytest.raises(ValueError, match="non-finite"):
        write_table(path, [{"measurement": value}])

    assert not path.exists()


def test_parquet_writer_rejects_ambiguous_mixed_scalar_types(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "release.parquet"

    with pytest.raises(TypeError, match="cannot mix scalar types"):
        write_table(path, [{"value": 1}, {"value": 1.5}])

    assert not path.exists()


def test_parquet_reader_rejects_nested_values_without_echoing_them(
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "nested.parquet"
    canary = "sensitive-parquet-canary"
    pq.write_table(pa.table({"codes": [[canary]]}), path)

    with pytest.raises(TypeError, match="unsupported scalar") as raised:
        read_table(path)

    assert canary not in str(raised.value)


def test_table_io_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported table format"):
        read_table(tmp_path / "release.xlsx")


def test_table_writer_refuses_empty_release(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty release"):
        write_table(tmp_path / "release.jsonl", [])
