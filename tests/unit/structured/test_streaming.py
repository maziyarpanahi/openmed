"""Tests for streaming, bounded-memory tabular de-identification."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest

import openmed.structured.streaming as streaming
from openmed.risk import KAnonymityEngine, kanon_report
from openmed.structured import read_table, write_table
from openmed.structured.streaming import (
    MemoryCeilingError,
    stream_deidentify_table,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _tag_deidentifier(text: str, **_kwargs: object) -> str:
    """Deterministic offline stand-in for ``deidentify`` on free-text cells."""

    return "[REDACTED]"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_rows(count: int, *, ages: int = 5, zips: int = 4) -> list[dict]:
    """Algorithmically generate synthetic clinical rows (no real PHI)."""

    rows = []
    for i in range(count):
        rows.append(
            {
                "mrn": f"MRN{i:06d}",
                "age": 30 + (i % ages),
                "zip": 10000 + (i % zips),
                "disease": "flu" if i % 2 else "cold",
                "note": f"Encounter number {i} recorded.",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Bounded memory
# ---------------------------------------------------------------------------


def test_working_set_does_not_scale_with_row_count(tmp_path: Path) -> None:
    """The class census stays constant while the row count grows 8x."""

    def run(count: int) -> tuple[dict, int]:
        source = tmp_path / f"in_{count}.csv"
        output = tmp_path / f"out_{count}.csv"
        _write_csv(source, _synthetic_rows(count))
        tracemalloc.start()
        report = stream_deidentify_table(
            source,
            output,
            quasi_identifiers=["age", "zip"],
            free_text_columns=["note"],
            target_k=2,
            chunk_size=256,
            deidentifier=_tag_deidentifier,
            overwrite=True,
        )
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return report, peak

    small, small_peak = run(1_000)
    big, big_peak = run(8_000)

    # Identical distinct-class census despite 8x more rows: the working set is
    # a function of quasi-identifier cardinality, not of the number of records.
    assert small["decision"]["class_count"] == big["decision"]["class_count"]
    assert small["decision"]["census_bytes"] == big["decision"]["census_bytes"]
    assert big["decision"]["record_count"] == 8_000
    # Peak allocation must not grow proportionally to the row count.
    assert big_peak < small_peak * 4


def test_memory_ceiling_rejects_high_cardinality_census(tmp_path: Path) -> None:
    """A quasi-identifier space that would blow the ceiling is refused."""

    source = tmp_path / "unique.csv"
    output = tmp_path / "unique_out.csv"
    rows = [{"age": 20 + i, "zip": 10_000 + i, "note": "x"} for i in range(500)]
    _write_csv(source, rows)

    with pytest.raises(MemoryCeilingError):
        stream_deidentify_table(
            source,
            output,
            quasi_identifiers=["age", "zip"],
            target_k=2,
            chunk_size=32,
            memory_ceiling=2_000,
            deidentifier=_tag_deidentifier,
            overwrite=True,
        )
    # No partial release is left behind when the first pass aborts.
    assert not output.exists()
    assert not list(tmp_path.glob(".unique_out.csv.*.tmp"))


def test_memory_ceiling_also_blocks_excess_process_rss(tmp_path: Path) -> None:
    """The guard measures process RSS, not only the class-count estimate."""

    source = tmp_path / "one_class.csv"
    output = tmp_path / "one_class_out.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["age", "zip", "note"])
        for _ in range(50_000):
            writer.writerow([40, 10_000, "synthetic"])

    script = """
from pathlib import Path
import sys
from openmed.structured.streaming import MemoryCeilingError, stream_deidentify_table

try:
    stream_deidentify_table(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        quasi_identifiers=["age", "zip"],
        target_k=2,
        chunk_size=50_000,
        memory_ceiling=2_048,
        overwrite=True,
    )
except MemoryCeilingError:
    print("blocked")
else:
    raise SystemExit("process RSS ceiling did not block the oversized batch")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(source), str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"memory guard subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert result.stdout.strip() == "blocked"
    assert not output.exists()
    assert not list(tmp_path.glob(".one_class_out.csv.*.tmp"))


def test_linux_rss_reader_uses_current_resident_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux RSS measurement reads current resident pages from procfs."""

    monkeypatch.setattr(streaming, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(
        streaming,
        "os",
        SimpleNamespace(name="posix", sysconf=lambda _name: 4_096),
    )
    monkeypatch.setattr(
        streaming,
        "Path",
        lambda _path: SimpleNamespace(read_text=lambda **_kwargs: "100 7 0 0 0 0 0\n"),
    )

    assert streaming._process_rss_bytes() == 7 * 4_096


def test_file_larger_than_memory_ceiling_streams_below_process_limit(
    tmp_path: Path,
) -> None:
    """A file larger than the ceiling succeeds when its live working set fits."""

    source = tmp_path / "larger_than_ceiling.csv"
    output = tmp_path / "larger_than_ceiling_out.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["age", "zip", "note"])
        for index in range(50_000):
            writer.writerow([30 + index % 5, 10_000 + index % 4, "x" * 200])
    ceiling = 8 * 1024 * 1024
    assert source.stat().st_size > ceiling

    report = stream_deidentify_table(
        source,
        output,
        quasi_identifiers=["age", "zip"],
        target_k=2,
        chunk_size=32,
        memory_ceiling=ceiling,
        overwrite=True,
    )

    assert report["decision"]["record_count"] == 50_000
    assert report["memory"]["rss_guard_available"] is True
    assert report["memory"]["peak_rss_delta_bytes"] <= ceiling


# ---------------------------------------------------------------------------
# Leakage equivalence and chunk invariance
# ---------------------------------------------------------------------------


def test_streamed_output_equals_in_memory_reference(tmp_path: Path) -> None:
    """Chunked global decisions match the materialized suppression engine."""

    source = tmp_path / "in.csv"
    _write_csv(
        source,
        [
            {"age": 30, "zip": 10_000, "note": "a"},
            {"age": 40, "zip": 20_000, "note": "b"},
            {"age": 30, "zip": 10_000, "note": "c"},
            {"age": 50, "zip": 30_000, "note": "d"},
            {"age": 40, "zip": 20_000, "note": "e"},
            {"age": 30, "zip": 10_000, "note": "f"},
        ],
    )

    chunked = tmp_path / "chunked.csv"
    reference = tmp_path / "reference.csv"
    chunked_report = stream_deidentify_table(
        source,
        chunked,
        quasi_identifiers=["age", "zip"],
        target_k=2,
        chunk_size=2,
        remove_direct_identifiers=False,
        overwrite=True,
    )
    source_rows = read_table(source)
    reference_rows = KAnonymityEngine(
        ["age", "zip"],
        target_k=2,
    ).suppress(source_rows)
    _write_csv(
        reference,
        reference_rows,
    )

    released_k = kanon_report(read_table(chunked), quasi_identifiers=["age", "zip"])[
        "k"
    ]
    reference_k = kanon_report(reference_rows, quasi_identifiers=["age", "zip"])["k"]
    assert released_k == reference_k
    assert released_k == chunked_report["decision"]["released_k"]
    assert released_k >= 2
    assert chunked_report["decision"]["suppressed_count"] == 1
    assert chunked.read_bytes() == reference.read_bytes()


def test_chunk_boundaries_produce_byte_identical_output(tmp_path: Path) -> None:
    """Row batching must not change the released bytes."""

    source = tmp_path / "in.csv"
    _write_csv(source, _synthetic_rows(300))

    outputs = []
    for chunk_size in (1, 13, 300, 5_000):
        target = tmp_path / f"out_{chunk_size}.csv"
        stream_deidentify_table(
            source,
            target,
            quasi_identifiers=["age", "zip"],
            free_text_columns=["note"],
            target_k=2,
            chunk_size=chunk_size,
            deidentifier=_tag_deidentifier,
            overwrite=True,
        )
        outputs.append(target.read_bytes())

    assert len(set(outputs)) == 1


# ---------------------------------------------------------------------------
# Transform semantics
# ---------------------------------------------------------------------------


def test_direct_identifiers_dropped_and_free_text_redacted(tmp_path: Path) -> None:
    """Direct identifiers are removed and free-text cells are de-identified."""

    calls: list[str] = []

    def recording_deidentifier(text: str, **_kwargs: object) -> str:
        calls.append(text)
        return "[REDACTED]"

    source = tmp_path / "in.csv"
    output = tmp_path / "out.csv"
    _write_csv(source, _synthetic_rows(40))

    report = stream_deidentify_table(
        source,
        output,
        quasi_identifiers=["age", "zip"],
        free_text_columns=["note"],
        target_k=2,
        chunk_size=8,
        deidentifier=recording_deidentifier,
        overwrite=True,
    )

    released = read_table(output)
    assert "mrn" not in released[0]
    assert "mrn" not in report["released_columns"]
    assert all(row["note"] == "[REDACTED]" for row in released)
    # One de-identify call per released row: the column is never buffered whole.
    assert len(calls) == report["decision"]["released_count"]


def test_generalization_merges_classes_and_lifts_k(tmp_path: Path) -> None:
    """A coarser generalization level merges singleton classes up to k."""

    source = tmp_path / "in.csv"
    output = tmp_path / "out.csv"
    rows = [{"age": 20 + i, "zip": 10_000, "note": "x"} for i in range(12)]
    _write_csv(source, rows)

    report = stream_deidentify_table(
        source,
        output,
        quasi_identifiers=["age"],
        generalization={"age": 3},  # 20-year band
        target_k=2,
        chunk_size=4,
        deidentifier=_tag_deidentifier,
        overwrite=True,
    )

    released = read_table(output)
    assert report["decision"]["released_count"] == 12
    assert report["decision"]["released_k"] >= 2
    assert {row["age"] for row in released} == {"20-39"}


def test_subthreshold_classes_are_suppressed(tmp_path: Path) -> None:
    """Classes below target_k are dropped from the release."""

    source = tmp_path / "in.csv"
    output = tmp_path / "out.csv"
    rows = (
        [{"age": 30, "zip": 10_000, "note": "x"} for _ in range(3)]
        + [{"age": 40, "zip": 20_000, "note": "x"}]  # singleton -> suppressed
    )
    _write_csv(source, rows)

    report = stream_deidentify_table(
        source,
        output,
        quasi_identifiers=["age", "zip"],
        target_k=2,
        chunk_size=2,
        deidentifier=_tag_deidentifier,
        overwrite=True,
    )

    released = read_table(output)
    assert report["decision"]["suppressed_count"] == 1
    assert len(released) == 3
    assert all(row["age"] == "30" for row in released)


# ---------------------------------------------------------------------------
# Parquet row-group streaming
# ---------------------------------------------------------------------------


def test_parquet_row_group_streaming_matches_csv_reference(tmp_path: Path) -> None:
    """Parquet row-group iteration yields the same release as CSV streaming."""

    rows = _synthetic_rows(120)
    parquet_source = tmp_path / "in.parquet"
    csv_source = tmp_path / "in.csv"
    write_table(parquet_source, rows)
    _write_csv(csv_source, rows)

    parquet_out = tmp_path / "out.parquet"
    csv_out = tmp_path / "out.csv"
    parquet_report = stream_deidentify_table(
        parquet_source,
        parquet_out,
        quasi_identifiers=["age", "zip"],
        free_text_columns=["note"],
        target_k=2,
        chunk_size=16,
        deidentifier=_tag_deidentifier,
        overwrite=True,
    )
    csv_report = stream_deidentify_table(
        csv_source,
        csv_out,
        quasi_identifiers=["age", "zip"],
        free_text_columns=["note"],
        target_k=2,
        chunk_size=16,
        deidentifier=_tag_deidentifier,
        overwrite=True,
    )

    assert (
        parquet_report["decision"]["released_count"]
        == (csv_report["decision"]["released_count"])
    )
    parquet_rows = read_table(parquet_out)
    # Parquet keeps native integer QIs; the CSV reference stringifies them.
    normalized = [
        {**row, "age": str(row["age"]), "zip": str(row["zip"])} for row in parquet_rows
    ]
    assert normalized == read_table(csv_out)
    assert kanon_report(parquet_rows, quasi_identifiers=["age", "zip"])["k"] >= 2


def test_parquet_boundaries_produce_byte_identical_output(tmp_path: Path) -> None:
    """Input row groups and processing chunks do not affect Parquet bytes."""

    rows = _synthetic_rows(streaming.DEFAULT_CHUNK_SIZE + 104)
    table = pa.Table.from_pylist(rows)
    outputs = []
    output_row_group_counts = []
    for source_row_group_size, chunk_size in (
        (17, 7),
        (511, 113),
        (len(rows), len(rows)),
    ):
        source = tmp_path / f"in_{source_row_group_size}.parquet"
        target = tmp_path / f"out_{chunk_size}.parquet"
        pq.write_table(table, source, row_group_size=source_row_group_size)
        stream_deidentify_table(
            source,
            target,
            quasi_identifiers=["age", "zip"],
            free_text_columns=["note"],
            target_k=2,
            chunk_size=chunk_size,
            deidentifier=_tag_deidentifier,
            overwrite=True,
        )
        outputs.append(target.read_bytes())
        output_row_group_counts.append(pq.ParquetFile(target).num_row_groups)

    assert len(set(outputs)) == 1
    assert set(output_row_group_counts) == {2}


# ---------------------------------------------------------------------------
# PHI containment
# ---------------------------------------------------------------------------


def test_report_and_workspace_expose_no_raw_cell_values(tmp_path: Path) -> None:
    """The aggregate report and the workspace never carry raw cell values."""

    source = tmp_path / "in.csv"
    output = tmp_path / "out.csv"
    _write_csv(source, _synthetic_rows(60))

    report = stream_deidentify_table(
        source,
        output,
        quasi_identifiers=["age", "zip"],
        free_text_columns=["note"],
        target_k=2,
        chunk_size=8,
        deidentifier=_tag_deidentifier,
        overwrite=True,
    )

    serialized = json.dumps(report)
    assert "MRN" not in serialized
    assert "Encounter number" not in serialized
    # Only schema names and aggregate counts are reported.
    assert set(report["decision"]) == {
        "record_count",
        "released_count",
        "suppressed_count",
        "released_k",
        "class_count",
        "suppressed_class_count",
        "census_bytes",
    }
    # No spill/temp files survive the two passes.
    assert not list(tmp_path.glob(".out.csv.*.tmp"))


def test_unsupported_format_is_rejected(tmp_path: Path) -> None:
    """Only CSV/TSV/Parquet are accepted for streaming."""

    source = tmp_path / "in.jsonl"
    source.write_text('{"age": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported streaming input"):
        stream_deidentify_table(
            source,
            tmp_path / "out.csv",
            quasi_identifiers=["age"],
            deidentifier=_tag_deidentifier,
        )
